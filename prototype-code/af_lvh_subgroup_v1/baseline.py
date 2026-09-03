"""
The four-head reference, frozen, for the AF x LVH subgroup analysis.

WHY THESE ARE HARD-CODED

This round measures the LVH head of `resnet50_hires` (round 2) inside
comorbidity strata. If the reference could be recomputed, re-thresholded or
re-predicted by anything in this harness, a "deterioration" conclusion could
be made to appear or disappear by changing how the reference was derived.
These numbers are transcribed ONCE, mechanically, from

    prototype-results/hybrid_v1/resnet50_hires/result.json

and from its frozen test_predictions.csv - the exact files the reference run
wrote on 2026-08-20 under protocol fingerprint 36988c5e0ba5d72d, input
fingerprint 6b9b1dd6f951bad0, micro-batch 8.

analyze.py reads the LIVE reference (result.json + test_predictions.csv from
the hybrid_v1 run folder on Drive) and cross-checks it against everything
frozen here, field by field. If they disagree, analyze.py stops - that would
mean the "frozen" reference moved.

The model-verification path (model_verify.py) adds a third cross-check: it
reloads the saved weights and re-runs the test split, then compares the
recomputed probabilities against the frozen predictions file. The analysis
itself always uses the frozen file; the re-run only proves the file is the
model it claims to be.
"""
from __future__ import annotations

import hashlib
import json

# ---------------------------------------------------------------- the reference
REFERENCE_MODEL = "resnet50_hires"
REFERENCE_TASK = "multi-label (STEMI, AF, LVH, NORMAL)"

# Run metadata, transcribed from prototype-results/hybrid_v1/resnet50_hires/.
REFERENCE_META = {
    "protocol_fingerprint": "36988c5e0ba5d72d",
    "input_fingerprint": "6b9b1dd6f951bad0",
    "views": [768],
    "micro_batch": 8,
    "accum_steps": 4,
    "lr_used": 1e-4,
    "timm_version": "1.0.28",
    "tags": ["resnet50.tv_in1k"],
    "pretrain": "ImageNet-1k",
    "channels_last": True,
}

# The four-head run's recorded protocol settings, verbatim from result.json.
REFERENCE_PROTOCOL_SETTINGS = {
    "SEED": 42,
    "LABELS": ["STEMI", "AF", "LVH", "NORMAL"],
    "NORM_MEAN": [0.485, 0.456, 0.406],
    "NORM_STD": [0.229, 0.224, 0.225],
    "AUG": {
        "hflip": 0.0,
        "vflip": 0.0,
        "random_resized_crop": False,
        "rotation_deg": 0.0,
        "brightness": 0.1,
        "contrast": 0.1,
        "saturation": 0.0,
        "hue": 0.0,
    },
    "EFFECTIVE_BATCH": 32,
    "EPOCHS": 20,
    "OPTIMIZER": "adamw",
    "LR": 1e-4,
    "WEIGHT_DECAY": 0.05,
    "BETAS": [0.9, 0.999],
    "SCHEDULE": "cosine",
    "WARMUP_EPOCHS": 2,
    "GRAD_CLIP": 1.0,
    "AMP": True,
    "USE_POS_WEIGHT": True,
    "MONITOR": "macro_auprc",
    "MONITOR_MODE": "max",
    "EARLY_STOP_PATIENCE": 5,
    "THRESHOLD_TUNING": "val_f1",
    "TEST_TIME_AUGMENTATION": False,
}


def protocol_fingerprint() -> str:
    """Hash of the frozen protocol settings, same hasher as the reference
    harness (json, sort_keys, default=str, sha256[:16])."""
    return hashlib.sha256(
        json.dumps(REFERENCE_PROTOCOL_SETTINGS, sort_keys=True, default=str)
        .encode()).hexdigest()[:16]


# The four-head model's LVH column, transcribed from the same result.json.
# Pooled test split (938 images, 255 LVH positives), val-tuned threshold 0.47.
# Every int here is re-derived from the live predictions file before any
# subgroup number is printed; any disagreement stops the analysis.
REFERENCE_LVH = {
    "pooled": {
        "auroc": 0.9626159102000975, "auprc": 0.9248250492461644,
        "sensitivity": 0.8470588235294118, "specificity": 0.9502196193265008,
        "precision": 0.864, "f1": 0.8554455445544554,
        "fpr": 0.04978038067349927,
        "threshold": 0.47, "support": 255, "n": 938,
        "tp": 216, "fp": 34, "fn": 39, "tn": 649,
    },
    # PTB-XL subgroup of test: 619 images, 234 positives. The stratum the
    # source shortcut cannot reach.
    "PTBXL": {
        "auroc": 0.9495615495615496, "auprc": 0.9314660395197524,
        "sensitivity": 0.8504273504273504, "specificity": 0.9272727272727272,
        "precision": 0.8766519823788547, "f1": 0.8633405639913233,
        "fpr": 0.07272727272727272,
        "support": 234, "n": 619,
    },
    "CHAPMAN": {
        "auroc": 0.9774436090225564, "auprc": 0.8947372686791442,
        "sensitivity": 0.8095238095238095, "specificity": 0.9671052631578947,
        "precision": 0.7727272727272727, "f1": 0.7906976744186046,
        "fpr": 0.03289473684210526,
        "support": 21, "n": 173,
    },
    # The STEMI database contributes 146 images and ZERO LVH positives, so it
    # can only contribute false positives: 1 of 146.
    "STEMI": {"fpr": 0.00684931506849315, "support": 0, "n": 146, "fp": 1},
}

# Frozen test-split geometry. Cross-checked on load.
TEST_N = 938
TEST_POSITIVES_LVH = 255
PTBXL_N = 619
PTBXL_POSITIVES_LVH = 234
CHAPMAN_N = 173
CHAPMAN_POSITIVES_LVH = 21
STEMI_N = 146

# The frozen val-tuned threshold vector of the reference run, in LABELS order.
# The LVH slot is the operating point every threshold-dependent number in this
# analysis uses. Cross-checked against result.json and the saved checkpoint.
REFERENCE_THRESHOLDS = [0.43999999999999995, 0.8099999999999999,
                        0.4699999999999999, 0.38999999999999996]


def check_frozen_counts() -> None:
    """Cross-check the transcribed counts the moment they are loaded.

    A transcription error here would silently bias every stratum table, so
    fail loudly instead.
    """
    p = REFERENCE_LVH["pooled"]
    assert p["tp"] + p["fn"] == TEST_POSITIVES_LVH, \
        f"pooled LVH positives: tp+fn={p['tp']+p['fn']} != {TEST_POSITIVES_LVH}"
    assert p["tp"] + p["fp"] + p["fn"] + p["tn"] == TEST_N, \
        f"pooled counts sum to {p['tp']+p['fp']+p['fn']+p['tn']} != {TEST_N}"
    q = REFERENCE_LVH["PTBXL"]
    assert q["support"] == PTBXL_POSITIVES_LVH and q["n"] == PTBXL_N, \
        "PTBXL subgroup geometry mismatch"
    s = REFERENCE_LVH["STEMI"]
    assert s["n"] == STEMI_N, "STEMI-db subgroup size mismatch"
    assert abs(REFERENCE_THRESHOLDS[2] - REFERENCE_LVH["pooled"]["threshold"]) < 1e-9, \
        "LVH threshold slot disagrees with the transcribed threshold"
    assert protocol_fingerprint() == REFERENCE_META["protocol_fingerprint"], \
        "frozen protocol settings hash disagrees with the recorded fingerprint"


check_frozen_counts()
