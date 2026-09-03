"""
Frozen-model inference for the filter-bandwidth experiment.

Builds the EXACT reference architecture (trimmed from round 2's hybrids.py:
SingleStream + FusionHead, four heads), loads the frozen best.pt, verifies
the checkpoint's recorded identity, and runs images through the same eval
path the reference `predict()` used: stored 768px views, no resampling,
ImageNet normalisation, batch 8, no TTA, fp16 autocast on GPU.

Nothing here is trained or modified - weights load strict and stay read-only.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config as C


# ---------------------------------------------------------------- the model
class FusionHead(nn.Module):
    """Trimmed from round 2's hybrids.py - identical structure, no training
    machinery. Per-group LayerNorm -> concat -> MLP -> 4 logits."""

    def __init__(self, group_dims: "OrderedDict[str, int]", num_classes: int):
        super().__init__()
        self.group_names = list(group_dims)
        self.norms = nn.ModuleDict({g: nn.LayerNorm(d) for g, d in group_dims.items()})
        total = sum(group_dims.values())
        self.mlp = nn.Sequential(
            nn.Dropout(C.HEAD_DROPOUT),
            nn.Linear(total, C.HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(C.HEAD_DROPOUT),
            nn.Linear(C.HEAD_HIDDEN, num_classes),
        )
        self.in_features = total

    def forward(self, feats: "OrderedDict[str, object]"):
        parts = [self.norms[g](feats[g]) for g in self.group_names]
        return self.mlp(torch.cat(parts, dim=1))


class SingleStream(nn.Module):
    """Round 2's SingleStream, trimmed: one backbone at one resolution."""

    def __init__(self, tag: str, size: int, num_classes: int):
        import timm
        super().__init__()
        self.views = (size,)
        self.size = size
        self.backbone = timm.create_model(tag, pretrained=False, num_classes=0,
                                          global_pool="avg")
        dim = self.backbone.num_features
        self.head = FusionHead(OrderedDict([(f"r{size}", dim)]), num_classes)

    def forward(self, x):
        return self.head(OrderedDict([(f"r{self.size}", self.backbone(x))]))


def build_reference_model(tag: str = C.BACKBONE,
                          size: int = C.LOAD_SIZE,
                          num_classes: int = len(C.LABELS)):
    model = SingleStream(tag, size, num_classes)
    model.channels_last = C.CHANNELS_LAST
    if C.CHANNELS_LAST and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    return model


# ---------------------------------------------------------------- checkpoint
def load_checkpoint(path: Path) -> dict:
    """Load best.pt, tolerating the torch >= 2.6 weights_only default."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck:
        raise ValueError(f"{path} is not a round-2 checkpoint "
                         f"(missing 'model' state dict)")
    return ck


def check_checkpoint_identity(ck: dict) -> list[str]:
    """Recorded identity vs frozen transcription. Returns warning lines."""
    warns = []
    fp = ck.get("protocol_fingerprint")
    if fp != C.REFERENCE_PROTOCOL_FP:
        warns.append(f"protocol fingerprint {fp} != frozen "
                     f"{C.REFERENCE_PROTOCOL_FP}")
    views = ck.get("views")
    if views != C.REFERENCE_VIEWS:
        warns.append(f"views {views} != frozen {C.REFERENCE_VIEWS}")
    thr = ck.get("thresholds")
    if thr is not None and len(thr) == len(C.LABELS):
        if abs(float(thr[C.LVH_INDEX]) - C.LVH_THRESHOLD) > 1e-9:
            warns.append(f"LVH threshold {thr[C.LVH_INDEX]} != frozen "
                         f"{C.LVH_THRESHOLD}")
    else:
        warns.append("checkpoint carries no threshold vector")
    return warns


# ---------------------------------------------------------------- inference
class SheetSet:
    """Stored 768px sheets: decode -> ToTensor -> Normalize, NO resize.

    The stored view is exactly 768px; resampling anywhere in this path would
    break the 'same bytes, same math' comparison with the frozen file.
    """

    def __init__(self, paths: list[Path]):
        from torchvision import transforms as T
        self.abs = [str(p) for p in paths]
        self.tf = T.Compose([T.ToTensor(), T.Normalize(C.NORM_MEAN, C.NORM_STD)])

    def __len__(self):
        return len(self.abs)

    def __getitem__(self, i: int):
        from PIL import Image
        with Image.open(self.abs[i]) as im:
            img = im.convert("RGB")
        return self.tf(img), torch.tensor([i])


def _check_sizes(paths: list[Path]):
    from PIL import Image
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
        if w != C.LOAD_SIZE or h != C.LOAD_SIZE:
            raise SystemExit(
                f"stored image is {w}x{h}, not {C.LOAD_SIZE}x{C.LOAD_SIZE}:\n"
                f"  {p}\n"
                f"  This path must read the SAME 768px view the reference run "
                f"read. Build the views with prepare_views.py - resampling at "
                f"inference time would add a second resize and make the "
                f"probability comparison meaningless.")


def run_inference(model, paths: list[Path], batch_size: int = 8,
                  device: str = "cuda") -> np.ndarray:
    """Sigmoid probabilities (n, 4) for the given stored 768px sheets, in
    the given order. Same math as the reference predict()."""
    from torch.utils.data import DataLoader

    _check_sizes(paths)
    model = model.to(device)
    model.eval()
    ds = SheetSet(paths)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, drop_last=False)
    use_cl = getattr(model, "channels_last", False)
    amp = C.AMP and device == "cuda"

    probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            if use_cl:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device, dtype=torch.float16,
                                enabled=amp):
                logits = model(x)
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs, axis=0)
