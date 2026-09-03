"""Train / evaluate loops. Identical for every architecture."""
from __future__ import annotations
import math, time

import numpy as np
import torch
import torch.nn as nn

import config as C
import metrics


def set_seed(seed: int = C.SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic algorithms are NOT forced: on some backbones it falls back to
    # far slower kernels. Seeding is identical across models, which is what the
    # comparison needs; exact bitwise reproducibility is a separate goal.
    torch.backends.cudnn.benchmark = True


def build_criterion(pos_weight, device):
    if C.USE_POS_WEIGHT:
        pw = torch.tensor(pos_weight, dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    return nn.BCEWithLogitsLoss()


def build_optimizer(model, lr: float | None = None):
    """AdamW with no weight decay on norms and biases - standard, and applied
    identically to every architecture."""
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
    for bi, (x, y, i) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
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

    # Data-starvation instrumentation. Colab runs one cell at a time, so you
    # cannot watch nvidia-smi while training; this reports it inline instead.
    # t_data is time spent BLOCKED waiting for the loader to hand over a batch.
    # A high fraction means the GPU is idle waiting on JPEG decode, and a bigger
    # batch will not help - more workers or smaller source images will.
    t_data = 0.0
    t_mark = time.time()

    for bi, (x, y, _) in enumerate(loader):
        t_data += time.time() - t_mark
        if max_batches and bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        gstep = epoch * steps_per_epoch + bi // accum
        for pg in optimizer.param_groups:
            pg["lr"] = lr_at(gstep, total_steps, warmup_steps, base_lr)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = criterion(model(x), y) / accum

        scaler.scale(loss).backward()

        if (bi + 1) % accum == 0 or (bi + 1) == n_batches:
            if C.GRAD_CLIP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running += loss.item() * accum * x.size(0)
        seen += x.size(0)
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
            print(f"    NOTE: {data_pct:.0f}% of the epoch was spent WAITING FOR DATA,"
                  f" not computing.\n"
                  f"          The GPU is starved by JPEG decode of the "
                  f"1686x1311 source images.\n"
                  f"          A larger micro_batch will NOT help. Try --workers 4, "
                  f"or pre-resize\n"
                  f"          the dataset to {C.IMG_SIZE}px (see README, "
                  f"'dataloader-bound').", flush=True)
    return {"loss": running / max(1, seen), "seconds": total,
            "img_per_sec": seen / max(total, 1e-9), "data_wait_pct": data_pct}


def validate(model, loader, device, thresholds=None, max_batches: int | None = None):
    p, y, _ = predict(model, loader, device, max_batches=max_batches)
    if thresholds is None:
        thresholds = metrics.tune_thresholds(y, p)
    return metrics.evaluate(y, p, thresholds), thresholds, p, y


# ------------------------------------------------------- complexity + speed
def model_complexity(model) -> dict:
    """Parameter counts and in-memory size.

    size_mb is computed from the actual dtype of parameters + buffers, which is
    what the saved checkpoint will weigh. Reported alongside accuracy because a
    model that wins by 0.005 AUPRC at 5x the size is not obviously the winner.
    """
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
def benchmark_inference(model, device, batch_size: int, reps: int = 30,
                        warmup: int = 10, amp: bool | None = None) -> dict:
    """Measure inference speed on synthetic input of the configured size.

    Two numbers, because they answer different questions:
      * latency_ms_b1  - one ECG at a time. What a clinic would see.
      * throughput_ips - batched. What a screening backlog would see.

    Synthetic input is deliberate: it removes disk and JPEG-decode time so the
    figure reflects the ARCHITECTURE, not the storage medium. AMP is forced to
    the same setting for every model, since it is a shared condition.
    """
    amp = C.AMP if amp is None else amp
    use_amp = amp and device.type == "cuda"
    model.eval()

    def _time(bs, n_warm, n_rep):
        x = torch.randn(bs, 3, C.IMG_SIZE, C.IMG_SIZE, device=device)
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
        "img_size": C.IMG_SIZE,
        "reps": reps,
    }
