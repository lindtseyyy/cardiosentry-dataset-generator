"""Single-GPU continued FCMAE training, invoked by the Colab notebook."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from training_state import (save_atomic, load_resume, restore_rng, epoch_order,
                            ResumeBatchSampler, snapshot, train_epoch)

from data_utils import (ECGDataset, MEAN, STD, prepare_splits, sha256, targets,
                        uniform_mask, unpatchify)
from rect_fcmae import RectFCMAE, export_encoder, load_initial, masked_stem_features
from models.convnextv2 import convnextv2_tiny


def visualize(model, dataset, out, seed, device):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    image, valid, record = dataset[0]
    image, valid = image[None].to(device), valid[None].to(device)
    model.eval()
    with torch.no_grad():
        mask = uniform_mask(valid, generator=torch.Generator(device=device).manual_seed(seed))
        _, pred, _ = model(image, valid, mask)
        _, _, mean, var = targets(image, valid)
        pred = pred.float().flatten(2).transpose(1, 2)
        # Display only: unnormalize using target patch statistics. This is not
        # autonomous recovery of the unknown patch's brightness/contrast.
        pred = pred * (var+1e-6).sqrt() + mean
        grid = (image.shape[-2]//32, image.shape[-1]//32)
        recon = unpatchify(pred, grid)
        pixels = mask.reshape(1, 1, *grid).repeat_interleave(32, 2).repeat_interleave(32, 3)
        mu = torch.tensor(MEAN, device=device)[None, :, None, None]
        sd = torch.tensor(STD, device=device)[None, :, None, None]
        original = (image*sd+mu).clamp(0, 1)
        decoded = (recon*sd+mu).clamp(0, 1)
        composite = original*(1-pixels)+decoded*pixels
        panels = [original, original*(1-pixels)+.5*pixels, composite, valid.expand(-1,3,-1,-1)]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, panel, title in zip(axes, panels, ['Original', 'Masked input', 'Reconstruction (target stats)', 'Valid pixels']):
            ax.imshow(panel[0].cpu().permute(1, 2, 0)); ax.set_title(title); ax.axis('off')
        fig.suptitle(record); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def preflight(model, device):
    """Actual sparse execution: shape, hidden-content isolation, export parity."""
    model.eval()
    image = torch.randn(1, 3, 96, 160, device=device)
    valid = torch.ones(1, 1, 96, 160, device=device)
    mask = uniform_mask(valid, generator=torch.Generator(device=device).manual_seed(8))
    # Force outer cells masked to catch accidental sparse bounding-box cropping.
    mask[:, -5:] = 1
    with torch.no_grad():
        loss, pred, _ = model(image, valid, mask)
        assert pred.shape == (1, 3072, 3, 5) and torch.isfinite(loss)
        pixels = mask.reshape(1, 1, 3, 5).repeat_interleave(32, 2).repeat_interleave(32, 3)
        altered = image + pixels*torch.randn_like(image)*10
        # Compare at the deterministic boundary immediately before conversion
        # to a sparse tensor. MinkowskiEngine 0.5.4 CUDA convolutions use atomic
        # accumulation and can vary slightly between otherwise identical full
        # forwards, so decoder-output equality is not a valid leakage test.
        stem = masked_stem_features(model.encoder, image, mask, valid)
        changed_stem = masked_stem_features(model.encoder, altered, mask, valid)
        torch.testing.assert_close(stem, changed_stem, rtol=0, atol=0)
        dense = convnextv2_tiny().to(device).eval()
        missing, unexpected = dense.load_state_dict(export_encoder(model), strict=False)
        assert set(missing) == {'norm.weight','norm.bias','head.weight','head.bias'} and not unexpected
        sparse_feature = model.encoder(image, torch.zeros_like(mask), valid)
        dense_feature = image
        for i in range(4):
            dense_feature = dense.stages[i](dense.downsample_layers[i](dense_feature))
        torch.testing.assert_close(sparse_feature, dense_feature, rtol=2e-3, atol=2e-3)
    del dense
    model.train()
    loss, _, _ = model(image, valid, mask)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    model.zero_grad(set_to_none=True)


def main(cfg, pilot=False):
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU and the pinned MinkowskiEngine environment are required')
    device = 'cuda'
    seed = cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    out = Path(cfg['run_dir']) / ('pilot' if pilot else 'full')
    out.mkdir(parents=True, exist_ok=True)
    checkpoints, encoders, previews = out/'checkpoints', out/'encoders', out/'previews'
    for directory in (checkpoints, encoders, previews):
        directory.mkdir(parents=True, exist_ok=True)
    split_stage = Path(tempfile.mkdtemp(prefix='split_audit_', dir=out))
    train_frame, val_frame, audit = prepare_splits(cfg['index'], cfg['image_root'], split_stage,
                                                  seed, cfg.get('holdout_csv'), cfg['ssl_val_fraction'])
    if not pilot and not audit['holdout_sha256']:
        raise ValueError('Full run requires downstream validation/test exclusion CSV; pilot remains available')
    if pilot:
        train_frame = train_frame.iloc[:max(cfg['batch_size']*cfg['accumulation']*2, 8)]
        val_frame = val_frame.iloc[:4]
    train_ds = ECGDataset(train_frame, cfg['image_root'], cfg['input_size'], cfg.get('resize_cache'))
    val_ds = ECGDataset(val_frame, cfg['image_root'], cfg['input_size'], cfg.get('resize_cache'))
    # Dedicated loader generator prevents iterator construction on resume from
    # consuming the RNG stream used for masks/model computations.
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['workers'], pin_memory=True,
                            generator=torch.Generator().manual_seed(seed+500000))
    model = RectFCMAE().to(device)
    report = load_initial(model, cfg['checkpoint'])
    print('Initialization:', report, flush=True)
    preflight(model, device)
    print('Sparse preflight passed: rectangular shape, hidden-content isolation, export parity, gradients', flush=True)
    ckpt_hash = sha256(cfg['checkpoint'])
    source_hashes = {str(p.relative_to(Path(__file__).parent)): sha256(p)
                     for p in Path(__file__).parent.rglob('*.py')}
    identity = dict(config=cfg, split=audit, checkpoint_sha256=ckpt_hash, sources=source_hashes)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    provenance_path = out/'provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Existing run configuration/data/source mismatch: use a new run_dir')
    for path in split_stage.iterdir():
        os.replace(path, out/path.name)
    split_stage.rmdir()
    (out/'provenance.json').write_text(json.dumps(dict(**identity, fingerprint=fingerprint,
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        initialization=report), indent=2))
    # Matched downstream control: exactly the encoder before ECG updates.
    save_atomic({'model': export_encoder(model), 'kind': 'ImageNet-1K FCMAE control',
                 'input_size': cfg['input_size'], 'checkpoint_sha256': ckpt_hash}, encoders/'encoder_initial.pt')
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim <= 1 or name.endswith('bias') or 'grn' in name else decay).append(parameter)
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': .05},
                                   {'params': no_decay, 'weight_decay': 0.}],
                                  lr=cfg['learning_rate'], betas=(.9, .95))
    # FP16 is available on T4/L4/A100. Actual sparse AMP support is exercised
    # by the pilot; no automatic change of precision or batch size on failure.
    amp = cfg['amp']
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    epochs = 1 if pilot else cfg['epochs']
    total_batches = math.ceil(len(train_ds)/cfg['batch_size'])
    updates_per_epoch = math.ceil(total_batches/cfg['accumulation'])
    total_steps = epochs*updates_per_epoch
    warmup = min(total_steps-1, (0 if pilot else cfg['warmup_epochs'])*updates_per_epoch)
    start, step, best, history = 0, 0, float('inf'), []
    resume_path = checkpoints/'last.pt'
    resumed = load_resume(resume_path, fingerprint)
    if resumed is not None:
        model.load_state_dict(resumed['model'], strict=True)
        optimizer.load_state_dict(resumed['optimizer']); scaler.load_state_dict(resumed['scaler'])
        start = resumed['epoch'] + int(resumed['epoch_complete'])
        step, best, history = resumed['progress']['step'], resumed['best'], resumed['history']
        restore_rng(resumed['rng'])
        print(f"Resuming at epoch {start+1}, next training batch "
              f"{1 if resumed['epoch_complete'] else resumed['progress']['next_batch']+1}", flush=True)
        # Do not keep a second full CPU model/optimizer checkpoint in RAM.
        resumed = {k: resumed[k] for k in ['epoch_complete', 'order', 'progress']}
    for epoch in range(start, epochs):
        partial = resumed is not None and not resumed['epoch_complete'] and epoch == start
        order = resumed['order'] if partial else epoch_order(len(train_ds), seed, epoch)
        progress = dict(resumed['progress']) if partial else dict(
            next_batch=0, step=step, loss_sum=0., seen=0, seconds=0., peak_gpu_gib=0.)
        sampler = ResumeBatchSampler(order, cfg['batch_size'], progress['next_batch'], cfg['accumulation'])
        if len(order) != len(train_ds) or progress['seen'] != min(progress['next_batch']*cfg['batch_size'], len(train_ds)):
            raise ValueError('Saved cursor/sample count does not match dataset')
        train_loader = DataLoader(train_ds, batch_sampler=sampler,
                                  num_workers=cfg['workers'], pin_memory=True,
                                  generator=torch.Generator().manual_seed(seed+100000+epoch))
        def checkpoint_partial(current):
            state = snapshot(model, optimizer, scaler, fingerprint, epoch, False,
                             order, current, best, history)
            save_atomic(state, resume_path, keep_previous=True)
            print(f"Saved epoch {epoch+1}, {current['next_batch']}/{total_batches} batches, "
                  f"optimizer update {current['step']} to {resume_path}", flush=True)
        progress = train_epoch(model, train_loader, optimizer, scaler, cfg, device,
                               epoch, len(train_ds), progress, total_steps, warmup, checkpoint_partial)
        step = progress['step']
        resumed = None
        validation_started = time.monotonic()
        model.eval(); val_sum, val_seen = 0., 0
        val_rng = torch.Generator(device=device).manual_seed(seed+1000)
        with torch.no_grad():
            for images, valid, _ in val_loader:
                images, valid = images.to(device), valid.to(device)
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    loss, _, _ = model(images, valid, generator=val_rng)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite validation loss')
                val_sum += loss.item()*len(images); val_seen += len(images)
        val_loss = val_sum/val_seen
        row = dict(epoch=epoch+1, train_loss=progress['loss_sum']/progress['seen'], ssl_val_loss=val_loss,
                   lr=optimizer.param_groups[0]['lr'], seconds=progress['seconds']+time.monotonic()-validation_started,
                   peak_gpu_gib=max(progress['peak_gpu_gib'], torch.cuda.max_memory_allocated()/2**30))
        history.append(row); print(row, flush=True)
        improved = val_loss < best
        best = min(best, val_loss)
        pd.DataFrame(history).to_csv(out/'metrics.csv', index=False)
        if improved:
            save_atomic({'model': export_encoder(model), 'epoch': epoch+1,
                         'kind': 'best SSL reconstruction; not downstream-selected',
                         'input_size': cfg['input_size'], 'fingerprint': fingerprint}, encoders/'encoder_best_ssl.pt')
        if (epoch+1)%cfg['save_every'] == 0 or epoch+1 == epochs:
            # Preserve full resumable milestones, including epoch 20, even
            # after last.pt advances. Both checkpoint types live on Drive.
            save_atomic({'model': export_encoder(model), 'epoch': epoch+1,
                         'input_size': cfg['input_size'], 'fingerprint': fingerprint}, encoders/f'encoder_epoch_{epoch+1:04d}.pt')
            visualize(model, val_ds, previews/f'reconstruction_{epoch+1:04d}.png', seed+2000, device)
        # Publish completion after validation artifacts. An interruption during
        # validation/export resumes the pre-validation checkpoint and retries it.
        state = snapshot(model, optimizer, scaler, fingerprint, epoch, True,
                         order, progress, best, history)
        if (epoch+1)%cfg['save_every'] == 0 or epoch+1 == epochs:
            save_atomic(state, checkpoints/f'checkpoint_epoch_{epoch+1:04d}.pt')
        save_atomic(state, resume_path, keep_previous=True)
        print(f'Completed epoch {epoch+1}; checkpoint saved: {resume_path}', flush=True)
    (out/'completed.json').write_text(json.dumps(dict(epochs=epochs, fingerprint=fingerprint, pilot=pilot), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    main(json.loads(Path(args.config).read_text()), args.pilot)
