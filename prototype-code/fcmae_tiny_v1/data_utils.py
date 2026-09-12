"""Data and rectangular reconstruction utilities; no MinkowskiEngine dependency."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def patchify(x, p=32):
    n, c, h, w = x.shape
    if h % p or w % p:
        raise ValueError('Both image dimensions must be divisible by patch size')
    return x.reshape(n, c, h//p, p, w//p, p).permute(0, 2, 4, 3, 5, 1).reshape(n, (h//p)*(w//p), p*p*c)


def unpatchify(x, grid, channels=3, p=32):
    gh, gw = grid
    return x.reshape(-1, gh, gw, p, p, channels).permute(0, 5, 1, 3, 2, 4).reshape(-1, channels, gh*p, gw*p)


def uniform_mask(valid, ratio=.6, generator=None, p=32):
    """Exactly round(ratio * eligible_count) masked patches, with visible context."""
    if not 0 < ratio < 1:
        raise ValueError('mask ratio must be between zero and one')
    eligible = patchify(valid, p).sum(-1) > 0
    mask = torch.zeros_like(eligible, dtype=valid.dtype)
    for i, row in enumerate(eligible):
        ids = row.nonzero().flatten()
        if len(ids) < 2:
            raise ValueError('Need at least two non-padding patches')
        count = max(1, min(len(ids)-1, round(len(ids)*ratio)))
        order = torch.randperm(len(ids), device=valid.device, generator=generator)
        mask[i, ids[order[:count]]] = 1
    return mask


def targets(images, valid, norm=True, p=32):
    raw = patchify(images.float(), p)
    weights = patchify(valid.float().expand(-1, images.shape[1], -1, -1), p)
    count = weights.sum(-1, keepdim=True)
    mean = (raw * weights).sum(-1, keepdim=True) / count.clamp_min(1)
    # Match official unbiased variance for fully valid patches.
    var = ((raw-mean).square()*weights).sum(-1, keepdim=True) / (count-1).clamp_min(1)
    target = (raw-mean) / (var+1e-6).sqrt() if norm else raw
    return target, weights, mean, var


def reconstruction_loss(images, pred, valid, mask, norm=True, p=32):
    target, weights, _, _ = targets(images, valid, norm, p)
    if pred.ndim == 4:
        pred = pred.flatten(2).transpose(1, 2)
    per_patch = ((pred.float()-target).square()*weights).sum(-1) / weights.sum(-1).clamp_min(1)
    selected = mask * (weights.sum(-1) > 0)
    if selected.sum() == 0:
        raise ValueError('No masked valid patches')
    return (per_patch*selected).sum()/selected.sum()


def letterbox(path, size, cache_path=None):
    h, w = size
    cached = cache_path is not None and Path(cache_path).is_file()
    with Image.open(cache_path if cached else path) as original:
        im = original.convert('RGB')
        scale = min(w/im.width, h/im.height)
        nw, nh = max(1, round(im.width*scale)), max(1, round(im.height*scale))
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        if cache_path is not None and not cached:
            cache_path = Path(cache_path); cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp = cache_path.with_suffix(f'.{os.getpid()}.tmp')
            # Cache the resized, UNPADDED image. Its dimensions retain the
            # exact valid rectangle; no per-pixel mask needs to be stored.
            im.save(temp, format='PNG'); os.replace(temp, cache_path)
        canvas = Image.new('RGB', (w, h), (255, 255, 255))
        left, top = (w-nw)//2, (h-nh)//2
        canvas.paste(im, (left, top))
    image = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float()/255
    valid = torch.zeros(1, h, w)
    valid[:, top:top+nh, left:left+nw] = 1
    image = (image-torch.tensor(MEAN)[:, None, None])/torch.tensor(STD)[:, None, None]
    return image, valid


class ECGDataset(Dataset):
    def __init__(self, frame, root, size, cache_root=None):
        self.frame, self.root, self.size = frame.reset_index(drop=True), Path(root).resolve(), size
        self.cache_root = Path(cache_root) if cache_root else None

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        path = (self.root / row.image_path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Image path escapes dataset root')
        cache_path = None
        if self.cache_root:
            identity = f'{row.image_path}:{row.get("image_sha256", "")}:{self.size}:lanczos-v1'
            cache_path = self.cache_root/(hashlib.sha256(identity.encode()).hexdigest()+'.png')
        image, valid = letterbox(path, self.size, cache_path)
        return image, valid, row.fcmae_record_id


def prepare_splits(index_path, root, out, seed, holdout_path=None, val_fraction=.05):
    """Canonical IDs must use the same source namespace in index and holdout CSV.

    A missing holdout CSV permits a diagnostic pilot only. Full training checks
    the returned audit. Unknown patient IDs are explicitly record-grouped.
    """
    df = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    required = {'image_path', 'fcmae_record_id', 'source_dataset', 'canonical_record_id', 'patient_id', 'patient_id_available'}
    if missing := required-set(df.columns):
        raise ValueError(f'Index missing columns: {sorted(missing)}; join raw manifest by fcmae_record_id first')
    if not df.fcmae_record_id.is_unique or not df.image_path.is_unique:
        raise ValueError('Expected exactly one image per source record')
    for col in ['fcmae_record_id', 'source_dataset', 'canonical_record_id', 'image_path']:
        if df[col].eq('').any():
            raise ValueError(f'Empty {col}')
    known = df.patient_id_available.str.lower().isin(['1', 'true']) & df.patient_id.ne('')
    df['record_key'] = df.source_dataset+'::'+df.canonical_record_id
    df['patient_key'] = np.where(known, df.source_dataset+'::'+df.patient_id, '')
    if df.record_key.duplicated().any():
        raise ValueError('Duplicate canonical source records')
    excluded = np.zeros(len(df), dtype=bool)
    holdout_hash = None
    if holdout_path:
        hd = pd.read_csv(holdout_path, dtype=str, keep_default_na=False)
        need = {'source_dataset', 'canonical_record_id', 'patient_id', 'patient_id_available'}
        if need-set(hd.columns) or hd.empty:
            raise ValueError('Holdout CSV must contain all downstream validation/test records and required ID columns')
        if hd.source_dataset.eq('').any() or hd.canonical_record_id.eq('').any():
            raise ValueError('Holdout source and canonical record IDs must not be empty')
        hk = hd.patient_id_available.str.lower().isin(['1', 'true']) & hd.patient_id.ne('')
        excluded = df.record_key.isin(hd.source_dataset+'::'+hd.canonical_record_id)
        excluded |= known & df.patient_key.isin((hd.source_dataset+'::'+hd.patient_id)[hk])
        holdout_hash = sha256(holdout_path)
    df['group_key'] = np.where(known, 'patient::'+df.patient_key, 'record::'+df.record_key)
    kept = df.loc[~excluded].copy()
    # Hash thresholds keep assignments stable if records are appended later.
    values = kept.group_key.map(lambda key: int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:16], 16)/2**64)
    kept['ssl_split'] = np.where(values < val_fraction, 'ssl_validation', 'ssl_train')
    train = kept.loc[kept.ssl_split.eq('ssl_train')]
    val = kept.loc[kept.ssl_split.eq('ssl_validation')]
    if train.empty or val.empty:
        raise ValueError('SSL split is empty; use a larger dataset or adjust split seed/fraction')
    assert not set(train.group_key) & set(val.group_key)
    for path in kept.image_path:
        resolved = (Path(root)/path).resolve()
        if not resolved.is_relative_to(Path(root).resolve()) or not resolved.is_file():
            raise ValueError(f'Missing or unsafe image path: {path}')
    if 'image_sha256' in kept:
        hashes = kept.loc[kept.image_sha256.ne('')]
        if hashes.image_sha256.duplicated().any():
            raise ValueError('Duplicate image hashes; resolve before pretraining')
    audit = dict(index_sha256=sha256(index_path), holdout_sha256=holdout_hash,
                 input_records=len(df), excluded_records=int(np.asarray(excluded).sum()),
                 ssl_train=len(train), ssl_validation=len(val),
                 unknown_patient_records=int((kept.patient_key == '').sum()),
                 limitation='Unknown patient IDs are grouped by record; cross-source patient linkage is not established.')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out/'ssl_train.csv', index=False); val.to_csv(out/'ssl_validation.csv', index=False)
    df.loc[excluded].to_csv(out/'excluded_holdouts.csv', index=False)
    (out/'split_audit.json').write_text(json.dumps(audit, indent=2))
    return train, val, audit
