#!/usr/bin/env python3
"""Local end-to-end validation of the filter-notebook harness (CPU)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "filter_notebook"))

import numpy as np
import torch

import config as C
import model as M
import prepare_views as PV

ROOT = Path("/home/lindtseyxvii/Codes/cardiosentry")
CKPT = ROOT / "prototype-results/hybrid_v1/resnet50_hires/weights/best.pt"
FILTER_EXP = ROOT / "af_lvh_error_analysis/filter_experiment"
VIEWS = ROOT / "af_lvh_error_analysis/filter_notebook/local_views"
OUTJSON = ROOT / "af_lvh_error_analysis/filter_notebook/local_live_probs.json"

# 1. 768px views
sources = {}
for case in C.CASES:
    sources[(case, "orig_40")] = FILTER_EXP / "by_case" / case / "original_0.05-40Hz.jpg"
    for variant, band in (("regen_40", "0.05-40Hz"),
                          ("regen_100", "0.05-100Hz"),
                          ("regen_150", "0.05-150Hz")):
        sources[(case, variant)] = FILTER_EXP / "by_band" / band / f"{case}_a0.jpg"
views = PV.build_all(sources, VIEWS)
print(f"views: {len(views)}")

# 2. model + checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
print("building", C.BACKBONE, "on", device)
model = M.build_reference_model()
ck = M.load_checkpoint(CKPT)
model.load_state_dict(ck["model"], strict=True)
print("  state dict loaded (strict) -", len(ck["model"]), "keys")
for w in M.check_checkpoint_identity(ck):
    print("  WARNING:", w)

# 3. inference per variant (40 images)
paths = [views[(c, v)] for c in C.CASES for v in C.VARIANTS]
probs = M.run_inference(model, paths, batch_size=8, device=device)
print("inference done:", probs.shape)

live = {"probs": {c: {v: None for v in C.VARIANTS} for c in C.CASES}}
i = 0
for c in C.CASES:
    for v in C.VARIANTS:
        live["probs"][c][v] = [float(x) for x in probs[i]]
        i += 1
OUTJSON.write_text(json.dumps(live, indent=2))
print("wrote", OUTJSON)

# quick peek at LVH probs
for c in C.CASES:
    row = " ".join(f"{v}={live['probs'][c][v][C.LVH_INDEX]:.4f}" for v in C.VARIANTS)
    print(f"  {c}: {row}")
