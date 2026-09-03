"""
Dataset, transforms and loaders - single 768 view, LVH binary task.

WHY THIS IS A STRAIGHT COPY OF THE ROUND-2 LOADER

The four-head reference (`resnet50_hires`) was trained by
cardiosentry_hybrids_v1's data.py with views=(768,) on the SAME
prototype_1152_768 corpus. This module keeps the identical code path so the
binary model sees the same bytes in the same order:

  * same generator seed -> same shuffle order for train
  * same worker seeding, same pin_memory, same drop_last=False
  * same photometric jitter applied once on the PIL image
  * the 768 view is a straight ToTensor - no resampling - because the on-disk
    copy is already 768

The only difference is that labels are the single LVH column (C.LABELS).
"""
from __future__ import annotations
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import config as C
import manifest


# --------------------------------------------------------------------- transforms
def build_shared_transform(train: bool) -> T.Compose | None:
    """Photometric jitter only, applied to the PIL image before normalisation.

    See config.AUG for why flips, rotation and random-resized-crop are excluded:
    horizontal flip reverses time, vertical flip turns ST elevation into ST
    depression, and cropping amputates diagnostic leads. The corpus is already
    augmented (grid colour, tilt, perspective, creases, lighting, blur, noise,
    JPEG artifacts), so only a +/-10% brightness/contrast jitter remains.
    """
    if not train or not (C.AUG["brightness"] or C.AUG["contrast"]):
        return None
    return T.ColorJitter(brightness=C.AUG["brightness"],
                         contrast=C.AUG["contrast"],
                         saturation=C.AUG["saturation"],
                         hue=C.AUG["hue"])


def build_view_transform(size: int) -> T.Compose:
    """Resample to `size` and normalise. Deterministic for train and eval alike."""
    steps = []
    if size != C.LOAD_SIZE:
        steps.append(T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC))
    steps += [T.ToTensor(), T.Normalize(C.NORM_MEAN, C.NORM_STD)]
    return T.Compose(steps)


# ----------------------------------------------------------------------- dataset
class ECGSheetDataset(Dataset):
    """One synthetic paper-ECG sheet -> ({size: image}, 1 binary label)."""

    def __init__(self, df, train: bool, views: tuple[int, ...]):
        self.paths = df["abs_path"].tolist()
        self.labels = df[C.LABELS].values.astype("float32")
        self.views = tuple(views)
        self.shared = build_shared_transform(train)
        self.view_tf = {s: build_view_transform(s) for s in self.views}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
            if self.shared is not None:
                img = self.shared(img)
            out = {s: tf(img) for s, tf in self.view_tf.items()}
        return out, torch.from_numpy(self.labels[i]), i


# ----------------------------------------------------------------------- loaders
def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def to_device(views: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in views.items()}


def resolve_workers(requested: int | None = None) -> int:
    """How many loader processes to actually start.

    Decoding is CPU-bound, so more workers than cores buys nothing. Two cores
    are left for the main process, which is not idle: it runs the loop, copies
    into pinned memory and drives the GPU.
    """
    cores = os.cpu_count() or 2
    if requested:
        return requested
    if cores <= 4:
        return max(1, cores)
    return max(1, min(cores - 2, 12))


def build_loaders(root, micro_batch: int, views: tuple[int, ...], workers: int = 2):
    df = manifest.load(root)
    report = manifest.check_integrity(df)

    bad = [s for s in views if s > C.LOAD_SIZE]
    if bad:
        raise ValueError(
            f"model asks for view(s) {bad} but the dataset copy is stored at "
            f"{C.LOAD_SIZE}px. Upsampling a {C.LOAD_SIZE}px JPEG to {max(bad)}px "
            f"invents detail that is not there. Rebuild the corpus from the "
            f"original prototype_1152.tar, never from a resized copy."
        )

    g = torch.Generator()
    g.manual_seed(C.SEED)

    out = {}
    for split in ("train", "val", "test"):
        sub = df[df.split == split].reset_index(drop=True)
        is_train = split == "train"
        out[split] = DataLoader(
            ECGSheetDataset(sub, train=is_train, views=views),
            batch_size=micro_batch,
            shuffle=is_train,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=workers > 0,
            worker_init_fn=_seed_worker,
            generator=g if is_train else None,
        )
        out[split + "_df"] = sub
    out["report"] = report
    out["pos_weight"] = manifest.pos_weight(df)
    out["df"] = df
    out["views"] = tuple(views)
    return out
