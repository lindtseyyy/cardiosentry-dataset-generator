"""Datasets, transforms and loaders. Identical for every architecture."""
from __future__ import annotations
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
def build_transforms(train: bool) -> T.Compose:
    """Deterministic at eval. At train, photometric jitter ONLY.

    See config.AUG for why flips, rotation and random-resized-crop are excluded.
    The short version: horizontal flip reverses time, vertical flip turns ST
    elevation into ST depression, and cropping amputates diagnostic leads. All
    three corrupt the label without touching it.
    """
    steps = [T.Resize((C.IMG_SIZE, C.IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC)]
    if train and (C.AUG["brightness"] or C.AUG["contrast"]):
        steps.append(T.ColorJitter(brightness=C.AUG["brightness"],
                                   contrast=C.AUG["contrast"],
                                   saturation=C.AUG["saturation"],
                                   hue=C.AUG["hue"]))
    steps += [T.ToTensor(), T.Normalize(C.NORM_MEAN, C.NORM_STD)]
    return T.Compose(steps)


# ----------------------------------------------------------------------- dataset
class ECGSheetDataset(Dataset):
    """One synthetic paper-ECG sheet -> (image, 4 independent binary labels)."""

    def __init__(self, df, train: bool):
        self.paths = df["abs_path"].tolist()
        self.labels = df[C.LABELS].values.astype("float32")
        self.sources = df["source"].tolist()
        self.records = df["record"].tolist()
        self.tf = build_transforms(train)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            img = self.tf(im.convert("RGB"))
        return img, torch.from_numpy(self.labels[i]), i


# ----------------------------------------------------------------------- loaders
def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def build_loaders(root, micro_batch: int, workers: int = 2):
    df = manifest.load(root)
    report = manifest.check_integrity(df)

    g = torch.Generator()
    g.manual_seed(C.SEED)

    out = {}
    for split in ("train", "val", "test"):
        sub = df[df.split == split].reset_index(drop=True)
        is_train = split == "train"
        out[split] = DataLoader(
            ECGSheetDataset(sub, train=is_train),
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
    return out
