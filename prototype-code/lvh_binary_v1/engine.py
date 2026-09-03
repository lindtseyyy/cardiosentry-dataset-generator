"""Train / evaluate loops - identical code to round 2's engine.py, single head.

The loop body, the LR schedule, the AMP handling, the gradient accumulation
and the data-wait instrumentation are byte-identical to what trained the
four-head reference. The only change is forced by the task: the criterion
receives the single LVH pos_weight (the same 2.5524 value) and validate()
tunes one threshold instead of four.
"""
from __future__ import annotations
import math, time

import numpy as np
import torch
import torch.nn as nn

import config as C
import data as D
import metrics


def to_channels_last(views: dict) -> dict:
    """NHWC for 4-D image tensors. Layout only - the numbers are unchanged."""
    return {k: v.contiguous(memory_format=torch.channels_last) if v.dim() == 4 else v
            for k, v in views.items()}


def prepare_inputs(model, views: dict, device) -> dict:
    v = D.to_device(views, device)
    return to_channels_last(v) if getattr(model, "channels_last", False) else v


def set_seed(seed: int = C.SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic algorithms are NOT forced: on some backbones it falls back to
    # far slower kernels. Seeding is identical to the reference run, which is
    # what the comparison needs; exact bitwise reproducibility is a separate goal.
    torch.backends.cudnn.benchmark = True


def build_criterion(pos_weight, device):
    if C.USE_POS_WEIGHT:
        pw = torch.tensor(pos_weight, dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    return nn.BCEWithLogitsLoss()


def build_optimizer(model, lr: float | None = None):
    """AdamW with no weight decay on norms and biases - the same rule, applied
    to the same parameter groups as the reference run."""
    lr = C.LR if lr is None else lr
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": C.WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=C.BETAS,
    )


def lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return base_lr
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def predict(model, loader, device, max_batches: int | None = None):
    model.eval()
    probs, trues, idxs = [], [], []
    amp = C.AMP and device.type == "cuda"
    for bi, (views, y, i) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        views = prepare_inputs(model, views, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(views)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        trues.append(y.numpy())
        idxs.append(i.numpy())
    return (np.concatenate(probs), np.concatenate(trues), np.concatenate(idxs))


def train_one_epoch(model, loader, optimizer, criterion, scaler, device,
                    epoch: int, total_epochs: int, accum: int, base_lr: float,
                    max_batches: int | None = None, log_every: int = 50):
    model.train()
    amp = C.AMP and device.type == "cuda"
    n_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    steps_per_epoch = math.ceil(n_batches / accum)
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * C.WARMUP_EPOCHS

    running, seen, t0 = 0.0, 0, time.time()
    optimizer.zero_grad(set_to_none=True)

    # Data-starvation instrumentation, same as round 2: reports data-wait
    # inline, because Colab runs one cell at a time and nvidia-smi is not
    # visible while the cell trains.
    t_data = 0.0
    t_mark = time.time()

    for bi, (views, y, _) in enumerate(loader):
        t_data += time.time() - t_mark
        if max_batches and bi >= max_batches:
            break
        views = prepare_inputs(model, views, device)
        y = y.to(device, non_blocking=True)
        bs = y.size(0)

        gstep = epoch * steps_per_epoch + bi // accum
        for pg in optimizer.param_groups:
            pg["lr"] = lr_at(gstep, total_steps, warmup_steps, base_lr)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = criterion(model(views), y) / accum

        scaler.scale(loss).backward()

        if (bi + 1) % accum == 0 or (bi + 1) == n_batches:
            if C.GRAD_CLIP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running += loss.item() * accum * bs
        seen += bs
        if log_every and (bi + 1) % log_every == 0:
            el = time.time() - t0
            mem = (f"  {torch.cuda.memory_reserved()/1024**3:.1f}/"
                   f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}G"
                   if device.type == "cuda" else "")
            print(f"    batch {bi+1}/{n_batches}  loss {running/max(1,seen):.4f}  "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}  "
                  f"{seen/el:.1f} img/s  data-wait {100*t_data/el:.0f}%{mem}",
                  flush=True)
        t_mark = time.time()

    total = time.time() - t0
    data_pct = 100 * t_data / max(total, 1e-9)
    if log_every:
        print(f"    epoch: {seen/max(total,1e-9):.1f} img/s   "
              f"data-wait {data_pct:.0f}%   {total/60:.1f} min", flush=True)
        if data_pct > 30:
            import os as _os
            cores = _os.cpu_count() or 2
            print(f"    NOTE: {data_pct:.0f}% of the epoch was spent WAITING FOR "
                  f"DATA, not computing.\n"
                  f"          Decoding {C.LOAD_SIZE}px JPEGs is CPU-bound; this "
                  f"machine has {cores} CPU core(s).\n"
                  f"          The only levers are a higher-CPU runtime or more "
                  f"workers (capped at the core count).", flush=True)
    return {"loss": running / max(1, seen), "seconds": total,
            "img_per_sec": seen / max(total, 1e-9), "data_wait_pct": data_pct}


def validate(model, loader, device, thresholds=None, max_batches: int | None = None):
    p, y, _ = predict(model, loader, device, max_batches=max_batches)
    if thresholds is None:
        thresholds = [metrics.tune_threshold(y, p)]
    return metrics.evaluate(y, p, thresholds), thresholds, p, y


# ------------------------------------------------------- complexity + speed
def model_complexity(model) -> dict:
    """Parameter counts and in-memory size - same definition as round 2."""
    p_total = sum(p.numel() for p in model.parameters())
    p_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    p_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    b_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "params_total": int(p_total),
        "params_total_m": p_total / 1e6,
        "params_trainable": int(p_train),
        "params_trainable_m": p_train / 1e6,
        "size_mb": (p_bytes + b_bytes) / 1024**2,
        "param_mb": p_bytes / 1024**2,
        "buffer_mb": b_bytes / 1024**2,
    }


@torch.no_grad()
def benchmark_inference(model, device, views: tuple[int, ...], batch_size: int,
                        reps: int = 30, warmup: int = 10, amp: bool | None = None) -> dict:
    """Inference speed on synthetic input at the model's own view geometry.

    Two numbers, because they answer different questions:
      * latency_ms_b1  - one ECG at a time. What a clinic would see.
      * throughput_ips - batched. What a screening backlog would see.
    """
    amp = C.AMP if amp is None else amp
    use_amp = amp and device.type == "cuda"
    model.eval()

    def _time(bs, n_warm, n_rep):
        x = {s: torch.randn(bs, 3, s, s, device=device) for s in views}
        if getattr(model, "channels_last", False):
            x = to_channels_last(x)
        for _ in range(n_warm):
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp):
                model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return np.array(times)

    t1 = _time(1, warmup, reps)
    tb = _time(batch_size, max(3, warmup // 2), max(10, reps // 2))

    return {
        "latency_ms_b1_mean": float(t1.mean() * 1000),
        "latency_ms_b1_std": float(t1.std() * 1000),
        "latency_ms_b1_p50": float(np.percentile(t1, 50) * 1000),
        "latency_ms_b1_p95": float(np.percentile(t1, 95) * 1000),
        "throughput_ips_b1": float(1.0 / t1.mean()),
        "batch_size": int(batch_size),
        "latency_ms_batch_mean": float(tb.mean() * 1000),
        "ms_per_image_batched": float(tb.mean() * 1000 / batch_size),
        "throughput_ips_batched": float(batch_size / tb.mean()),
        "amp": bool(use_amp),
        "device": (torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"),
        "views": list(views),
        "reps": reps,
    }
