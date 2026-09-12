"""Optimizer-boundary checkpointing and deterministic resumable batch order.

The dataset must be deterministic per index (as ECGDataset currently is).
The trainer, not DataLoader prefetch, owns the committed batch cursor.
"""
import errno
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import Sampler


def save_atomic(value, path, keep_previous=False):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    previous = path.with_name('previous.pt')
    try:
        with open(tmp, 'wb') as stream:
            torch.save(value, stream)
            stream.flush()
            try:
                os.fsync(stream.fileno())
            except OSError as exc:
                # Some mounted filesystems lack fsync. Rename still prevents
                # an unfinished serialization from replacing the last file.
                if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS}:
                    raise
        if keep_previous and path.exists():
            os.replace(path, previous)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_resume(path, fingerprint):
    path = Path(path)
    candidates = [path, path.with_name('previous.pt')]
    found, errors = False, []
    for candidate in candidates:
        if not candidate.exists():
            continue
        found = True
        try:
            # These are this run's own files, including Python/NumPy RNG state.
            state = torch.load(candidate, map_location='cpu', weights_only=False)
        except Exception as exc:
            errors.append(f'{candidate.name}: {type(exc).__name__}: {exc}')
            continue
        if state.get('format_version') != 2:
            raise ValueError('Older checkpoint format: resume with its original notebook; no implicit migration')
        if state.get('fingerprint') != fingerprint:
            raise ValueError('Resume configuration/data/source mismatch: use the original notebook/settings or a new run_dir')
        if candidate != path:
            print(f'Recovering from {candidate}; latest checkpoint missing/unreadable. {errors}', flush=True)
            if path.exists():
                os.replace(path, path.with_name('last.unreadable.pt'))
        return state
    if found:
        raise RuntimeError('No readable resume checkpoint: '+'; '.join(errors))
    return None


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def epoch_order(size, seed, epoch):
    return torch.randperm(size, generator=torch.Generator().manual_seed(seed+epoch))


class ResumeBatchSampler(Sampler):
    def __init__(self, order, batch_size, start_batch=0, accumulation=1):
        self.order = torch.as_tensor(order, dtype=torch.long).cpu()
        self.batch_size, self.start_batch = batch_size, start_batch
        total = math.ceil(len(self.order)/batch_size)
        if not 0 <= start_batch <= total:
            raise ValueError('Invalid saved batch cursor')
        if start_batch != total and start_batch % accumulation:
            raise ValueError('Cannot resume within a gradient-accumulation group')
        if not torch.equal(self.order.sort().values, torch.arange(len(self.order))):
            raise ValueError('Saved image order is not a permutation of the dataset')

    def __iter__(self):
        for offset in range(self.start_batch*self.batch_size, len(self.order), self.batch_size):
            yield self.order[offset:offset+self.batch_size].tolist()

    def __len__(self):
        return math.ceil(len(self.order)/self.batch_size)-self.start_batch


def snapshot(model, optimizer, scaler, fingerprint, epoch, complete, order,
             progress, best, history):
    return dict(format_version=2, fingerprint=fingerprint, model=model.state_dict(),
                optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                epoch=epoch, epoch_complete=complete,
                order=None if complete else order.clone().cpu(),
                progress=dict(progress), best=best, history=list(history), rng=capture_rng())


def train_epoch(model, loader, optimizer, scaler, cfg, device, epoch,
                samples, progress, total_steps, warmup, checkpoint):
    """Checkpoint only at clean accumulation boundaries after zero_grad.

    Loader RNG must be separate from model RNG. Worker transforms must not draw
    randomness. Both conditions are enforced by construction in train.py.
    """
    progress = dict(progress)
    total_batches = math.ceil(samples/cfg['batch_size'])
    start_batch = progress['next_batch']
    if start_batch == total_batches:
        return progress  # Training is done; replay validation only.
    model.train(); optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    base_seconds = progress['seconds']
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    last_saved = started
    every_updates = cfg.get('checkpoint_every_updates', 100)
    every_seconds = cfg.get('checkpoint_every_minutes', 15)*60
    if every_updates < 0 or every_seconds < 0:
        raise ValueError('Checkpoint intervals must be nonnegative; zero disables an interval')
    for i, (images, valid, _) in enumerate(loader, start=start_batch):
        images, valid = images.to(device, non_blocking=True), valid.to(device, non_blocking=True)
        group_start = (i//cfg['accumulation'])*cfg['accumulation']
        group_samples = min(cfg['accumulation']*cfg['batch_size'], samples-group_start*cfg['batch_size'])
        step = progress['step']
        factor = (step+1)/max(1,warmup) if step < warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup-1)))
        for group in optimizer.param_groups:
            group['lr'] = cfg['learning_rate']*factor
        with torch.autocast(device, dtype=torch.float16 if device == 'cuda' else torch.bfloat16, enabled=cfg['amp']):
            loss, _, _ = model(images, valid)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss epoch={epoch} batch={i}')
        scaler.scale(loss*len(images)/group_samples).backward()
        progress['loss_sum'] += loss.item()*len(images)
        progress['seen'] += len(images)
        boundary = (i+1)%cfg['accumulation'] == 0 or i+1 == total_batches
        if boundary:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
            if not torch.isfinite(norm):
                if not cfg['amp']:
                    raise FloatingPointError('Nonfinite gradients with AMP disabled; reduce LR in a new run')
                # GradScaler records nonfinite gradients during unscale_. Let it
                # skip the optimizer update and reduce its scale instead of
                # turning an expected, recoverable FP16 overflow into a fatal
                # training error.
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                new_scale = scaler.get_scale()
                if not new_scale < old_scale:
                    raise FloatingPointError('Nonfinite AMP gradients but GradScaler did not reduce its scale')
                progress['amp_skipped_updates'] = progress.get('amp_skipped_updates', 0) + 1
                progress['next_batch'] = i+1
                progress['seconds'] = base_seconds + time.monotonic()-started
                if device == 'cuda':
                    progress['peak_gpu_gib'] = max(progress['peak_gpu_gib'], torch.cuda.max_memory_allocated()/2**30)
                print(f"Skipped nonfinite AMP update at epoch {epoch+1}, batch {i+1}; "
                      f"scale {old_scale:g} -> {new_scale:g}", flush=True)
                # Persist the reduced scale and consumed-batch cursor
                # immediately so a disconnect cannot recreate the overflow.
                checkpoint(progress)
                last_saved = time.monotonic()
                continue
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            progress['step'] += 1
            progress['next_batch'] = i+1
            progress['seconds'] = base_seconds + time.monotonic()-started
            if device == 'cuda':
                progress['peak_gpu_gib'] = max(progress['peak_gpu_gib'], torch.cuda.max_memory_allocated()/2**30)
            due = ((every_updates and progress['step']%every_updates == 0)
                   or (every_seconds and time.monotonic()-last_saved >= every_seconds)
                   or i+1 == total_batches)
            if due:
                checkpoint(progress)
                last_saved = time.monotonic()
        if i%20 == 0:
            print(f'epoch {epoch+1} batch {i+1}/{total_batches} loss={loss.item():.5f}', flush=True)
    assert progress['next_batch'] == total_batches and progress['seen'] == samples
    progress['seconds'] = base_seconds+time.monotonic()-started
    return progress
