"""
The four-head reference, frozen, for the negative-transfer comparison.

WHY THESE ARE HARD-CODED

The comparison this round exists to make is binary ResNet-50@768 vs the shared
four-head ResNet-50@768 (`resnet50_hires`, round 2). If the reference could be
recomputed or re-thresholded by anything in this harness, a "negative transfer"
conclusion could be made to appear or disappear by changing how the reference
was derived. These numbers are transcribed ONCE, mechanically, from

    prototype-results/hybrid_v1/resnet50_hires/result.json

produced under protocol fingerprint 36988c5e0ba5d72d, input fingerprint
6b9b1dd6f951bad0, uniform micro-batch 8, on the SAME frozen 938-image test
split this round uses, with thresholds tuned on the same validation split.

compare.py reads the LIVE reference (result.json + test_predictions.csv from
the hybrid_v1 run folder) whenever it is given one, and cross-checks its
numbers against what is frozen here. If they disagree, compare.py stops -
that would mean the "frozen" reference moved.

WHAT THE CROSS-CHECK PROTECTS

The shared-protocol settings recorded here are the four-head run's
protocol_settings with the task fields removed. The binary harness computes
the same quantity (config.shared_settings / config.shared_fingerprint) and
compare.py refuses to compare when they differ. A changed learning rate,
augmentation or batch size anywhere in the pipeline cannot pass silently.
"""
from __future__ import annotations

import json

# ---------------------------------------------------------------- the reference
REFERENCE_MODEL = "resnet50_hires"          # the shared four-head ResNet-50@768
REFERENCE_TASK = "multi-label (STEMI, AF, LVH, NORMAL)"

# Run metadata, transcribed from prototype-results/hybrid_v1/resnet50_hires/.
REFERENCE_META = {
    "protocol_fingerprint": "36988c5e0ba5d72d",
    "input_fingerprint": "6b9b1dd6f951bad0",
    "views": [768],
    "micro_batch": 8,
    "accum_steps": 4,
    "lr_used": 1e-4,
    "epochs_run": 18,
    "train_min": 36.9,
    "params_total_m": 24.563268,
    "size_mb": 93.90447235107422,
    "latency_ms_b1": 8.472823466612075,
    "throughput_ips_batched": 123.58011965014343,
    "tags": ["resnet50.tv_in1k"],
    "pretrain": "ImageNet-1k",
    "channels_last": True,
    "timm_version": "1.0.28",
    "data_root": "/content/prototype_1152_768",
}

# The four-head run's recorded protocol settings, verbatim from result.json.
# The task fields (LABELS) are present so the shared-settings check can remove
# them the same way config.shared_settings() does.
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

# LVH pos_weight the reference run used. The binary run must compute the same
# value from the same train split (it does: 2.552437223042836).
REFERENCE_POS_WEIGHT_LVH = 2.552437223042836

# Frozen shared-protocol hash, computed once with the same hasher config.py
# uses over REFERENCE_PROTOCOL_SETTINGS minus the task fields. compare.py
# re-derives it from the live reference and from the binary run and demands
# equality in both directions.
def _shared_hash(protocol: dict) -> str:
    import hashlib
    task_keys = ("LABELS", "NUM_CLASSES")
    d = {k: v for k, v in protocol.items() if k not in task_keys}
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


REFERENCE_SHARED_FINGERPRINT = _shared_hash(REFERENCE_PROTOCOL_SETTINGS)

# The four-head model's LVH column, transcribed from the same result.json.
# Pooled test split (938 images, 255 LVH positives), val-tuned threshold 0.47.
REFERENCE_LVH = {
    "pooled": {
        "auroc": 0.9626159102000975, "auprc": 0.9248250492461644,
        "sensitivity": 0.8470588235294118, "specificity": 0.9502196193265008,
        "precision": 0.864, "f1": 0.8554455445544554,
        "fpr": 0.04978038067349927, "fnr": 0.15294117647058825,
        "accuracy": 0.9221748400852878, "balanced_accuracy": 0.8986392214279563,
        "threshold": 0.47, "support": 255, "n": 938,
        "tp": 216, "fp": 34, "fn": 39, "tn": 649,
    },
    # PTB-XL subgroup of test: 619 images, 234 positives. The column the
    # source shortcut cannot reach - the primary read of this round.
    "PTBXL": {
        "auroc": 0.9495615495615496, "auprc": 0.9314660395197524,
        "sensitivity": 0.8504273504273504, "specificity": 0.9272727272727272,
        "precision": 0.8766519823788547, "f1": 0.8633405639913233,
        "fpr": 0.07272727272727272, "accuracy": 0.8982229402261712,
        # sensitivity capped at FPR <= 0.08 (subgroup.MATCHED_FPR), transcribed
        # from prototype-results/hybrid_v1/comparison.txt - threshold-free, so
        # it compares DETECTORS rather than operating points.
        "sens_at_matched_fpr": 0.8675,
        "support": 234, "n": 619,
    },
    "CHAPMAN": {
        "auroc": 0.9774436090225564, "auprc": 0.8947372686791442,
        "sensitivity": 0.8095238095238095, "specificity": 0.9671052631578947,
        "precision": 0.7727272727272727, "f1": 0.7906976744186046,
        "fpr": 0.03289473684210526, "accuracy": 0.9479768786127167,
        "support": 21, "n": 173,
    },
    # The STEMI database contributes 146 images and ZERO LVH positives, so it
    # can only contribute false positives. 1 of 146 was called LVH by the
    # reference - the database-recognition effect the PTB-XL column exists to
    # see around.
    "STEMI": {
        "fpr": 0.00684931506849315, "support": 0, "n": 146, "fp": 1,
    },
}

# The frozen test-split geometry both runs share. Cross-checked on load.
TEST_N = 938
TEST_POSITIVES_LVH = 255
PTBXL_N = 619
PTBXL_POSITIVES_LVH = 234

# ---------------------------------------------------------------- context rows
# Round 1's ResNet-50 (384, single-Linear head) LVH column - where this thread
# started. From prototype-results/v1, fingerprint 936b42d988015ade.
ROUND1_RESNET50_LVH = {
    "auroc": 0.9560, "auprc": 0.8805, "sensitivity": 0.8078,
    "specificity": 0.9209, "precision": 0.7923, "f1": 0.8000, "fpr": 0.0791,
    "tp": 206, "fp": 54, "fn": 49, "tn": 629,
}

# Round 2's 384 four-head control (resnet50_base) pooled LVH - the reference
# point that separates "task change" from "resolution change". From
# prototype-results/hybrid_v1/resnet50_base/result.json.
ROUND2_BASE_384_LVH = {
    "auprc": 0.9036, "auroc": 0.9528, "sensitivity": 0.7961, "fpr": 0.0732,
    "PTBXL_auprc": 0.9270, "PTBXL_sensitivity": 0.7863, "PTBXL_fpr": 0.0779,
    "stemi_db_fp": 11,
}


def check_frozen_counts() -> None:
    """Cross-check the transcribed counts the moment they are loaded.

    A transcription error here would silently bias the whole comparison, so
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
    assert s["n"] == 146, "STEMI-db subgroup size mismatch"


check_frozen_counts()
