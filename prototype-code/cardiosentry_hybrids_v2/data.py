"""Datasets, transforms and loaders. Identical for every architecture."""
from __future__ import annotations
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import config as C
import dimensions as DIM
import manifest


# --------------------------------------------------------------------- transforms
class RedactHeader:
    """Paint a fixed rectangle over the printed ID / Age / Sex block.

    THE ID IS A LABEL. Stage 1 numbered records by source in order, so in this
    corpus CHAPMAN and PTB-XL are <= ECG007796 and every STEMI record is
    >= ECG007797, with no overlap. A model that learns to read one digit of
    ~25 px type gets the STEMI head for free, and `source` with it. See
    dimensions.REDACT_BOX for the measurement behind the rectangle.

    WHY A PLAIN FILL. Erasing only the ink leaves a legible ghost; dilating the
    mask to catch the anti-aliased halo means erasing a text-SHAPED region, whose
    silhouette encodes the glyphs just as well. Only a fixed rectangle, identical
    on every sheet, removes the information.

    WHY THE LOCAL PAPER COLOUR AND NOT WHITE. Stage 4 bakes in lighting
    gradients; a constant fill would leave a bright patch on a shaded sheet,
    which is itself a feature. The 75th percentile inside the box is the paper
    level under whatever light this corner happens to be in. The percentile is
    taken on a 4x-subsampled grid: 0.56 ms instead of 1.31, and statistically
    identical over ~70k pixels.

    Applied AFTER the resize - the box is defined in fractions so it lands on the
    same region either way, but the smaller box is 2.7x cheaper and avoids
    bicubic smearing the rectangle's edges. The ~30-50 px of margin around the
    text is far beyond bicubic's +/-2 px support, so no glyph survives.
    """

    def __call__(self, im: Image.Image) -> Image.Image:
        box = DIM.redact_box(im.width, im.height)
        patch = np.asarray(im.crop(box), dtype=np.float32)[::4, ::4]
        paper = np.percentile(patch.reshape(-1, patch.shape[-1]), 75, axis=0)
        ImageDraw.Draw(im).rectangle(list(box),
                                     fill=tuple(int(v) for v in paper))
        return im

    def __repr__(self) -> str:
        return f"RedactHeader(box={DIM.REDACT_BOX})"


class HeaderCrop:
    """Cut the printed header off the top of the sheet, before any resize.

    Deterministic, applied identically to train / val / test, and expressed as a
    FRACTION of the image height so it survives a pre-resized copy of the corpus
    (resize_dataset.py bakes the crop in, but a run against the original folder
    must still produce the same window).

    It is not augmentation. It removes printed text only: measured over 150 random
    sheets the header block never reached past 0.199 H and the first trace ink
    never began before 0.220 H, so the 0.207 cut sits inside that gap on every
    sheet in the corpus, tilted ones included. See dimensions.CONTENT_CROP.

    Two things this buys, and one it costs:
      + the record ID, which separates STEMI EXACTLY (dimensions.py), is gone
      + ~24% of the vertical pixel budget stops being spent on blank paper
      - the printed Age and Sex go with it. Those are real clinical covariates,
        not leakage. Losing them is a genuine cost; say so in the writeup.
    """

    def __call__(self, im: Image.Image) -> Image.Image:
        return im.crop(DIM.crop_box(im.width, im.height))

    def __repr__(self) -> str:
        return f"HeaderCrop(y={DIM.CONTENT_CROP[0]:.3f}..{DIM.CONTENT_CROP[1]:.3f})"


def build_transforms(train: bool, apply_crop: bool | None = None,
                     apply_redact: bool | None = None) -> T.Compose:
    """Deterministic at eval. At train, photometric jitter ONLY.

    See config.AUG for why flips, rotation and random-resized-crop are excluded.
    The short version: horizontal flip reverses time, vertical flip turns ST
    elevation into ST depression, and cropping amputates diagnostic leads. All
    three corrupt the label without touching it.

    T.Resize takes (h, w) - the same order as config.IMG_SIZE, and the OPPOSITE
    of PIL's (w, h). Getting this backwards produces a portrait ECG that still
    trains, still converges, and is silently wrong, so it is asserted in
    selftest.py rather than trusted.
    """
    # apply_crop is decided by manifest.check_geometry, not by config alone: a
    # folder produced by resize_dataset.py already HAS the crop baked in, and
    # applying it a second time would cut the same fraction off an
    # already-cropped sheet - straight into the first trace row.
    apply_crop = C.CROP_HEADER if apply_crop is None else apply_crop
    apply_redact = C.REDACT_HEADER if apply_redact is None else apply_redact

    steps: list = []
    if apply_crop:
        steps.append(HeaderCrop())
    steps.append(T.Resize((C.IMG_H, C.IMG_W),
                          interpolation=T.InterpolationMode.BICUBIC))
    # After the resize, and never together with a crop: a crop* geometry has
    # already discarded the band the box sits in, so redacting there would paint
    # a rectangle over waveform.
    if apply_redact and not apply_crop:
        steps.append(RedactHeader())
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

    def __init__(self, df, train: bool, apply_crop: bool | None = None,
                 apply_redact: bool | None = None):
        self.paths = df["abs_path"].tolist()
        self.labels = df[C.LABELS].values.astype("float32")
        self.sources = df["source"].tolist()
        self.records = df["record"].tolist()
        self.tf = build_transforms(train, apply_crop, apply_redact)

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
    geo = manifest.check_geometry(root)

    g = torch.Generator()
    g.manual_seed(C.SEED)

    out = {}
    for split in ("train", "val", "test"):
        sub = df[df.split == split].reset_index(drop=True)
        is_train = split == "train"
        out[split] = DataLoader(
            ECGSheetDataset(sub, train=is_train, apply_crop=geo["apply_crop"],
                            apply_redact=geo["apply_redact"]),
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
    out["geometry_source"] = geo
    out["pos_weight"] = manifest.pos_weight(df)
    out["df"] = df
    return out
