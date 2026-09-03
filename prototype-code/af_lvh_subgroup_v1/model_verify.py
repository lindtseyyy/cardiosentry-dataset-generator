"""
Weights-reload verification: prove the frozen predictions are the model.

WHY THIS MODULE EXISTS

The subgroup analysis reads `test_predictions.csv` - the per-image
probabilities the reference run wrote. That file is the ONLY thing standing
between the analysis and the actual trained model, so this round carries a
verification path that makes the file trustworthy rather than assumed:

  1. rebuild the exact architecture (trimmed from round 2's hybrids.py -
     SingleStream + FusionHead, four heads),
  2. load `best.pt` from the reference run folder (the weights the user keeps
     on Drive),
  3. cross-check the checkpoint's recorded identity (protocol fingerprint,
     views, threshold vector) against the frozen transcription,
  4. re-run the 938-image test split through the same eval pipeline (768px
     stored view, no resampling, ImageNet normalisation, batch 8, no TTA,
     fp16 autocast on GPU - the same path `predict()` took),
  5. compare the recomputed probabilities against the frozen file, per label.

WHAT THE COMPARISON MEANS

  * max|delta| ~ 1e-6            -> same model, same inputs, same math. The
                                    frozen file is exactly this checkpoint.
  * max|delta| up to ~1e-3       -> floating-point/library drift (e.g. a
                                    different torch/cuDNN version than the
                                    reference L4 used). The frozen file stays
                                    authoritative; this cell reports the
                                    magnitude so it is known, not suspected.
  * large delta or flipped labels -> the file on Drive does not match the
                                    weights on Drive. STOP: do not trust the
                                    subgroup tables until this is explained.

The analysis numbers ALWAYS come from the frozen file. This module never
feeds its recomputed probabilities back into the verdict - it only measures
how far they are from the frozen ones.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as C

# ---------------------------------------------------------------- the model
HEAD_HIDDEN = 512
HEAD_DROPOUT = 0.1


class FusionHead(nn.Module):
    """Trimmed from round 2's hybrids.py - identical structure, no training
    machinery. Per-group LayerNorm -> concat -> MLP -> 4 logits."""

    def __init__(self, group_dims: "OrderedDict[str,int]", num_classes: int):
        super().__init__()
        self.group_names = list(group_dims)
        self.norms = nn.ModuleDict({g: nn.LayerNorm(d) for g, d in group_dims.items()})
        total = sum(group_dims.values())
        self.mlp = nn.Sequential(
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(total, HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(HEAD_HIDDEN, num_classes),
        )
        self.in_features = total

    def forward(self, feats: "OrderedDict[str, object]"):
        import torch
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


def build_reference_model(tag: str = "resnet50.tv_in1k",
                          size: int = C.LOAD_SIZE,
                          num_classes: int = len(C.LABELS)):
    import torch
    model = SingleStream(tag, size, num_classes)
    model.channels_last = C.CHANNELS_LAST
    if C.CHANNELS_LAST and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    return model


# ---------------------------------------------------------------- checkpoint
def load_checkpoint(path: Path) -> dict:
    """Load best.pt, tolerating the torch >= 2.6 weights_only default."""
    import torch
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
        warns.append(f"  protocol fingerprint {fp} != frozen "
                     f"{C.REFERENCE_PROTOCOL_FP}")
    views = ck.get("views")
    if views != C.REFERENCE_VIEWS:
        warns.append(f"  views {views} != frozen {C.REFERENCE_VIEWS}")
    thr = ck.get("thresholds")
    if thr is not None and len(thr) == len(C.LABELS):
        if abs(float(thr[C.LVH_INDEX]) - C.LVH_THRESHOLD) > 1e-9:
            warns.append(f"  LVH threshold {thr[C.LVH_INDEX]} != frozen "
                         f"{C.LVH_THRESHOLD}")
    else:
        warns.append("  checkpoint carries no threshold vector")
    return warns


# ---------------------------------------------------------------- inference
class TestSheets:
    """Minimal test-split dataset: decode -> ToTensor -> Normalize, NO resize.

    The stored copy is exactly 768px; resampling anywhere in this path would
    break the 'same bytes, same math' comparison with the frozen file.
    """

    def __init__(self, data_root: Path, rows: pd.DataFrame):
        import torch
        from PIL import Image
        from torchvision import transforms as T
        self.rows = rows
        self.abs = [str(data_root / p) for p in rows["rel_path"]]
        self.tf = T.Compose([T.ToTensor(), T.Normalize(C.NORM_MEAN, C.NORM_STD)])

    def __len__(self):
        return len(self.abs)

    def __getitem__(self, i: int):
        import torch
        from PIL import Image
        with Image.open(self.abs[i]) as im:
            img = im.convert("RGB")
        return self.tf(img), torch.tensor([i])


def run_test_inference(model, data_root: Path, rows: pd.DataFrame,
                       batch_size: int = 8, device: str = "cuda") -> np.ndarray:
    """Re-run the test split exactly like the reference `predict()` did.

    Returns (n, 4) sigmoid probabilities, rows in index order.
    """
    import torch
    from torch.utils.data import DataLoader

    first = str(data_root / rows["rel_path"].iloc[0])
    from PIL import Image
    with Image.open(first) as im:
        w, h = im.size
    if w != C.LOAD_SIZE or h != C.LOAD_SIZE:
        raise SystemExit(
            f"stored image is {w}x{h}, not {C.LOAD_SIZE}x{C.LOAD_SIZE}.\n"
            f"  This verification path must read the SAME 768px corpus the\n"
            f"  reference run read ({first}). Resampling from the original\n"
            f"  1686px sheets would add a second JPEG generation and make the\n"
            f"  probability comparison meaningless. Extract\n"
            f"  prototype_1152_768.tar and point --data at it.")

    model = model.to(device)
    model.eval()
    ds = TestSheets(data_root, rows)
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


# ---------------------------------------------------------------- comparison
def compare_with_frozen(frozen: pd.DataFrame, recomputed: np.ndarray,
                        order: list[str]) -> dict:
    """Per-label max|delta|, RMSE, and LVH decision flips at the frozen
    threshold. `order` is the row order of `recomputed` (index.csv order)."""
    frozen = frozen.set_index("record").loc[order].reset_index()
    out = {"n": int(len(frozen))}
    for k, lab in enumerate(C.LABELS):
        d = recomputed[:, k] - frozen[f"prob_{lab}"].values
        out[f"max_abs_{lab}"] = float(np.abs(d).max())
        out[f"rmse_{lab}"] = float(np.sqrt((d ** 2).mean()))
    # LVH decisions at the frozen threshold
    y = frozen[f"true_LVH"].values
    pred_new = (recomputed[:, C.LVH_INDEX] >= C.LVH_THRESHOLD).astype(int)
    pred_old = frozen[f"pred_LVH"].values
    flips = int((pred_new != pred_old).sum())
    out["lvh_decision_flips"] = flips
    out["lvh_flips_at_true_pos"] = int(((pred_new != pred_old) & (y == 1)).sum())
    out["lvh_flips_at_true_neg"] = int(((pred_new != pred_old) & (y == 0)).sum())
    return out


def format_verification(summary: dict, max_abs: float) -> str:
    L = ["recomputed-from-weights vs frozen predictions file",
         f"  n = {summary['n']} test images"]
    for lab in C.LABELS:
        L.append(f"  {lab:7}  max|delta| {summary[f'max_abs_{lab}']:.3e}   "
                 f"RMSE {summary[f'rmse_{lab}']:.3e}")
    L.append(f"  LVH decisions flipped at 0.47: {summary['lvh_decision_flips']}"
             f" ({summary['lvh_flips_at_true_pos']} of LVH+, "
             f"{summary['lvh_flips_at_true_neg']} of LVH-)")
    if max_abs < 1e-6:
        L.append("  VERDICT: identical to float precision - the frozen file is "
                 "this checkpoint.")
    elif max_abs < 1e-3:
        L.append("  VERDICT: within floating-point/library drift - the frozen "
                 "file matches this checkpoint.")
    else:
        L.append("  VERDICT: DO NOT TRUST THE SUBGROUP TABLES - the frozen "
                 "predictions do not match the saved weights.")
    return "\n".join(L)


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="path to best.pt")
    ap.add_argument("--data", help="768px corpus root (with index.csv)")
    ap.add_argument("--frozen", help="frozen test_predictions.csv")
    ap.add_argument("--out", help="write verification summary json here")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint-only", action="store_true",
                    help="build + load + identity check, no inference")
    args = ap.parse_args()

    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable - falling back to CPU (slower but identical math)")
        device = "cpu"

    print(f"building {C.REFERENCE_MODEL} (timm)")
    model = build_reference_model()
    print("loading checkpoint:", args.ckpt)
    ck = load_checkpoint(Path(args.ckpt))
    model.load_state_dict(ck["model"], strict=True)
    print("  state dict loaded (strict) -", len(ck["model"]), "keys")
    for w in check_checkpoint_identity(ck):
        print("  WARNING:" + w.replace("  ", " ", 1))

    summary = {
        "model": C.REFERENCE_MODEL,
        "protocol_fingerprint": ck.get("protocol_fingerprint"),
        "views": ck.get("views"),
        "thresholds": ck.get("thresholds"),
        "epoch_saved": ck.get("epoch"),
    }

    if args.checkpoint_only:
        print("checkpoint-only mode: build + load + identity OK")
        return summary

    if not args.data or not args.frozen:
        raise SystemExit("--data and --frozen are required without "
                         "--checkpoint-only")

    idx = pd.read_csv(Path(args.data) / "index.csv")
    test_rows = idx[idx.split == "test"].reset_index(drop=True)
    if len(test_rows) != C.TEST_N:
        raise SystemExit(f"test split has {len(test_rows)} rows, expected "
                         f"{C.TEST_N} - wrong corpus copy?")

    print(f"re-running {len(test_rows)} test images on {device} ...")
    probs = run_test_inference(model, Path(args.data), test_rows,
                               batch_size=8, device=device)
    frozen = pd.read_csv(args.frozen)
    cmp = compare_with_frozen(frozen, probs, test_rows["record"].tolist())
    summary.update(cmp)
    max_abs = max(summary[f"max_abs_{l}"] for l in C.LABELS)
    summary["verdict"] = ("identical" if max_abs < 1e-6 else
                          "within_drift" if max_abs < 1e-3 else "MISMATCH")
    print()
    print(format_verification(summary, max_abs))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, default=str))
        print("\nwrote:", out)
    return summary


if __name__ == "__main__":
    main()
