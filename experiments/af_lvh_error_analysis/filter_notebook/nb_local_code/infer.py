"""
Frozen-checkpoint inference over the 40 stored views (10 cases x 4 variants):
original 40Hz sheet + regenerated 40/100/150Hz sheets.

Builds the reference architecture, loads best.pt STRICT, verifies the
checkpoint identity, then runs the same eval math the reference predict()
used (768px stored views, no resampling, batch 8, no TTA, fp16 on GPU).
Writes live_probs.json for analyze.py - nothing is retrained or modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import config as C
import model as M


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="path to best.pt")
    ap.add_argument("--views", required=True,
                    help="dir of stored 768px views from prepare_views.py")
    ap.add_argument("--out-json", required=True, help="write live_probs.json")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable - falling back to CPU (slower but same math)")
        device = "cpu"

    views_root = Path(args.views)
    paths: list[Path] = []
    for case in C.CASES:
        for variant in C.VARIANTS:
            p = views_root / case / f"{case}__{variant}__view768.jpg"
            if not p.exists():
                raise SystemExit(f"missing stored view: {p}\n  run "
                                 f"prepare_views.py first")
            paths.append(p)

    print(f"building {C.BACKBONE} (timm) on {device}")
    model = M.build_reference_model()
    print("loading checkpoint:", args.ckpt)
    ck = M.load_checkpoint(Path(args.ckpt))
    model.load_state_dict(ck["model"], strict=True)
    print("  state dict loaded (strict) -", len(ck["model"]), "keys")
    warns = M.check_checkpoint_identity(ck)
    for w in warns:
        print("  WARNING:", w)
    if warns:
        print("\nSTOP: checkpoint identity does not match the frozen "
              "transcription - investigate before inference.")
        return 2

    print(f"running {len(paths)} stored views ({len(C.CASES)} cases x "
          f"{len(C.VARIANTS)} variants) ...")
    probs = M.run_inference(model, paths, batch_size=8, device=device)

    live = {"probs": {c: {v: None for v in C.VARIANTS} for c in C.CASES}}
    i = 0
    for c in C.CASES:
        for v in C.VARIANTS:
            live["probs"][c][v] = [float(x) for x in probs[i]]
            i += 1
    live["device"] = device
    live["model"] = C.REFERENCE_MODEL
    live["protocol_fingerprint"] = ck.get("protocol_fingerprint")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(live, indent=2))
    print("wrote:", out)

    print("\nLVH probabilities (threshold 0.47):")
    print(f"  {'case':10s} " + "  ".join(f"{v:>10s}" for v in C.VARIANTS))
    for c in C.CASES:
        row = "  ".join(f"{live['probs'][c][v][C.LVH_INDEX]:10.4f}"
                        for v in C.VARIANTS)
        print(f"  {c:10s} {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
