"""
Datasets, transforms and loaders — multi-view.

WHAT IS DIFFERENT FROM ROUND 1

Round 1's dataset produced one tensor per image. This one produces a DICT of
tensors keyed by pixel size, e.g. {384: t, 768: t}, because dual-resolution
models need two views of the same sheet. Models declare the sizes they want in
hybrids.REGISTRY[...]["views"] and index the dict by size.

BOTH VIEWS COME FROM ONE DECODE OF ONE FILE.

That is the point, not an optimisation. If the 384 view and the 768 view were
loaded from two separately-written JPEGs, they would differ in quantisation
noise as well as resolution, and a dual-stream win could be an ensembling
artifact rather than a resolution effect. Decoding once and resampling twice
means the streams differ in exactly one property.

The on-disk copy is pre-resized to config.LOAD_SIZE (768) so no run ever decodes
a 1686x1311 sheet. The 768 view is then a straight ToTensor with no resampling
at all, and the 384 view is one bicubic step down from it.

COLOUR JITTER IS APPLIED ONCE, BEFORE THE SPLIT INTO VIEWS, so both views of one
sheet carry the same brightness and contrast. Jittering them independently would
hand the fusion head a trivial way to tell the streams apart.
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
    """Photometric jitter only, applied to the PIL image before any resampling.

    See config.AUG for why flips, rotation and random-resized-crop are excluded.
    The short version: horizontal flip reverses time, vertical flip turns ST
    elevation into ST depression, and cropping amputates diagnostic leads. All
    three corrupt the label without touching it.
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
    """One synthetic paper-ECG sheet -> ({size: image}, 4 independent labels)."""

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

    Decoding is CPU-bound, so more workers than cores buys nothing: they take
    turns on the same silicon and torch warns about it. Colab's standard VM
    reports 2 CPUs, which is the case this exists for.

    Two cores are left for the main process, which is not idle: it runs the
    training loop, copies into pinned memory and drives the GPU. Handing every
    core to the loader makes those compete with decode.

    Measured on a 12-vCPU L4, 8 workers already put data-wait at 1-2%, so the
    remaining headroom buys almost nothing - the cap exists to stop a 96-core
    machine spawning 96 processes, not because more never helps. Each worker
    holds a couple of prefetched batches, roughly 150 MB at micro-batch 8 with
    both views.

    Passing an explicit number overrides all of this.
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
            f"invents detail that is not there and would make the resolution "
            f"comparison meaningless. Re-run prepare_dataset.py at --size {max(bad)}."
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
