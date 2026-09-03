"""
Round-1 results, frozen, for the per-condition regression check.

WHY THESE ARE HARD-CODED

Round 1 is a finished experiment. Nothing in this round should be able to
recompute, re-tune or re-threshold it - if it could, a "regression" could be
made to disappear by changing how the reference was derived. These numbers are
transcribed once, mechanically, from

    prototype-results/cardiosentry_runs_v1/comparison.txt

which was produced under config fingerprint 936b42d988015ade on the SAME frozen
938-image test split this round uses, with thresholds tuned on the same
validation split. That shared split is what makes the comparison legitimate.

WHAT IS AND IS NOT COMPARABLE ACROSS THE TWO ROUNDS

Comparable: the test split, the label definitions, the seed, the optimiser, the
schedule, the epoch budget, the monitor, and the val-tuned-frozen threshold rule.
All identical by construction - see config.py.

NOT identical: round 1 used timm's default single Linear head; this round uses a
2-layer MLP for every model. So a round-1 number is a REFERENCE POINT, not a
control. `resnet50_base` is the control, and the regression check reports deltas
against BOTH so you can see which one a change is attributable to.
"""
from __future__ import annotations

# model -> metric -> {label: value, "macro": value}
ROUND1: dict[str, dict[str, dict[str, float]]] = {
    "resnet50": {
        "auroc": {"STEMI": 0.9861, "AF": 0.9863, "LVH": 0.9560, "NORMAL": 0.9796, "macro": 0.9770},
        "auprc": {"STEMI": 0.9560, "AF": 0.9777, "LVH": 0.8805, "NORMAL": 0.9534, "macro": 0.9419},
        "sensitivity": {"STEMI": 0.8904, "AF": 0.8667, "LVH": 0.8078, "NORMAL": 0.9179, "macro": 0.8707},
        "specificity": {"STEMI": 0.9672, "AF": 0.9847, "LVH": 0.9209, "NORMAL": 0.9590, "macro": 0.9579},
        "precision": {"STEMI": 0.8333, "AF": 0.9611, "LVH": 0.7923, "NORMAL": 0.9049, "macro": 0.8729},
        "f1": {"STEMI": 0.8609, "AF": 0.9114, "LVH": 0.8000, "NORMAL": 0.9113, "macro": 0.8709},
        "fpr": {"STEMI": 0.0328, "AF": 0.0153, "LVH": 0.0791, "NORMAL": 0.0410, "macro": 0.0421},
        "accuracy": {"STEMI": 0.9552, "AF": 0.9488, "LVH": 0.8902, "NORMAL": 0.9467, "macro": 0.9352},
    },
    "convnext_base": {
        "auroc": {"STEMI": 0.9815, "AF": 0.9761, "LVH": 0.9414, "NORMAL": 0.9784, "macro": 0.9694},
        "auprc": {"STEMI": 0.9463, "AF": 0.9639, "LVH": 0.8782, "NORMAL": 0.9386, "macro": 0.9317},
        "sensitivity": {"STEMI": 0.8767, "AF": 0.9158, "LVH": 0.7922, "NORMAL": 0.9179, "macro": 0.8756},
        "specificity": {"STEMI": 0.9773, "AF": 0.9602, "LVH": 0.9253, "NORMAL": 0.9407, "macro": 0.9509},
        "precision": {"STEMI": 0.8767, "AF": 0.9094, "LVH": 0.7984, "NORMAL": 0.8682, "macro": 0.8632},
        "f1": {"STEMI": 0.8767, "AF": 0.9126, "LVH": 0.7953, "NORMAL": 0.8924, "macro": 0.8692},
        "fpr": {"STEMI": 0.0227, "AF": 0.0398, "LVH": 0.0747, "NORMAL": 0.0593, "macro": 0.0491},
        "accuracy": {"STEMI": 0.9616, "AF": 0.9467, "LVH": 0.8891, "NORMAL": 0.9339, "macro": 0.9328},
    },
    "vgg16": {
        "auroc": {"STEMI": 0.9592, "AF": 0.9791, "LVH": 0.9493, "NORMAL": 0.9773, "macro": 0.9662},
        "auprc": {"STEMI": 0.9175, "AF": 0.9636, "LVH": 0.8566, "NORMAL": 0.9580, "macro": 0.9240},
        "sensitivity": {"STEMI": 0.9110, "AF": 0.9088, "LVH": 0.8784, "NORMAL": 0.9036, "macro": 0.9004},
        "specificity": {"STEMI": 0.9710, "AF": 0.9694, "LVH": 0.8902, "NORMAL": 0.9650, "macro": 0.9489},
        "precision": {"STEMI": 0.8526, "AF": 0.9283, "LVH": 0.7492, "NORMAL": 0.9167, "macro": 0.8617},
        "f1": {"STEMI": 0.8808, "AF": 0.9184, "LVH": 0.8087, "NORMAL": 0.9101, "macro": 0.8795},
        "fpr": {"STEMI": 0.0290, "AF": 0.0306, "LVH": 0.1098, "NORMAL": 0.0350, "macro": 0.0511},
        "accuracy": {"STEMI": 0.9616, "AF": 0.9510, "LVH": 0.8870, "NORMAL": 0.9467, "macro": 0.9366},
    },
    "vit_b_384": {
        "auroc": {"STEMI": 0.9558, "AF": 0.9766, "LVH": 0.9473, "NORMAL": 0.9730, "macro": 0.9632},
        "auprc": {"STEMI": 0.8659, "AF": 0.9538, "LVH": 0.8822, "NORMAL": 0.9340, "macro": 0.9090},
        "sensitivity": {"STEMI": 0.7808, "AF": 0.8772, "LVH": 0.8431, "NORMAL": 0.8964, "macro": 0.8494},
        "specificity": {"STEMI": 0.9545, "AF": 0.9495, "LVH": 0.8873, "NORMAL": 0.9438, "macro": 0.9338},
        "precision": {"STEMI": 0.7600, "AF": 0.8834, "LVH": 0.7363, "NORMAL": 0.8715, "macro": 0.8128},
        "f1": {"STEMI": 0.7703, "AF": 0.8803, "LVH": 0.7861, "NORMAL": 0.8838, "macro": 0.8301},
        "fpr": {"STEMI": 0.0455, "AF": 0.0505, "LVH": 0.1127, "NORMAL": 0.0562, "macro": 0.0662},
        "accuracy": {"STEMI": 0.9275, "AF": 0.9275, "LVH": 0.8753, "NORMAL": 0.9296, "macro": 0.9150},
    },
    "efficientnet_b3": {
        "auroc": {"STEMI": 0.9375, "AF": 0.9307, "LVH": 0.9178, "NORMAL": 0.9643, "macro": 0.9376},
        "auprc": {"STEMI": 0.8129, "AF": 0.8701, "LVH": 0.8242, "NORMAL": 0.9201, "macro": 0.8568},
        "sensitivity": {"STEMI": 0.7329, "AF": 0.8175, "LVH": 0.7804, "NORMAL": 0.8857, "macro": 0.8041},
        "specificity": {"STEMI": 0.9571, "AF": 0.9188, "LVH": 0.8887, "NORMAL": 0.9134, "macro": 0.9195},
        "precision": {"STEMI": 0.7589, "AF": 0.8147, "LVH": 0.7236, "NORMAL": 0.8131, "macro": 0.7776},
        "f1": {"STEMI": 0.7456, "AF": 0.8161, "LVH": 0.7509, "NORMAL": 0.8479, "macro": 0.7901},
        "fpr": {"STEMI": 0.0429, "AF": 0.0812, "LVH": 0.1113, "NORMAL": 0.0866, "macro": 0.0805},
        "accuracy": {"STEMI": 0.9222, "AF": 0.8881, "LVH": 0.8593, "NORMAL": 0.9051, "macro": 0.8937},
    },
}

# Round 1's overall winner and the model this round is trying to beat.
REFERENCE = "resnet50"

# Best value achieved by ANY round-1 backbone, per label per metric. Useful
# because the reference model is not the best on every head - ViT-B/384 beat it
# on {LVH} AUPRC, and VGG-16 beat it on {LVH} sensitivity by a wide margin.
def best(metric: str, label: str, lower_is_better: bool = False) -> tuple[str, float]:
    vals = [(m, ROUND1[m][metric][label]) for m in ROUND1]
    return (min if lower_is_better else max)(vals, key=lambda kv: kv[1])


LOWER_IS_BETTER = {"fpr"}


def reference(metric: str, label: str) -> float:
    return ROUND1[REFERENCE][metric][label]


# Round-1 confusion counts on the same 938-image test split, for the reference
# model. Transcribed from the "Confusion counts" table in EVALUATION_REPORT.md.
# Counts, not rates, because "16 missed infarcts" is the sentence that should
# appear in a flag - a sensitivity delta of -0.011 does not read as anything.
COUNTS: dict[str, dict[str, int]] = {
    "STEMI":  {"tp": 130, "fp": 26, "fn": 16, "tn": 766},
    "AF":     {"tp": 247, "fp": 10, "fn": 38, "tn": 643},
    "LVH":    {"tp": 206, "fp": 54, "fn": 49, "tn": 629},
    "NORMAL": {"tp": 257, "fp": 27, "fn": 23, "tn": 631},
}

# Positives per head on the test split, from the split table in the round-1
# evaluation report. 938 images total.
TEST_POSITIVES = {"STEMI": 146, "AF": 285, "LVH": 255, "NORMAL": 280}
TEST_N = 938



# ---------------------------------------------------------------- round A
#
# Notebook A (resnet50_base, resnet50_hires, resnet50_cbam, resnet50_fpn),
# frozen the same way and for the same reason as ROUND1 above: transcribed
# mechanically from prototype-results/hybrid_v1/, protocol fingerprint
# 36988c5e0ba5d72d, uniform micro-batch 8, same 938-image test split.
#
# BOTH columns are here because notebook A's finding was that they disagree.
# Pooled LVH AUPRC rose +0.0212 from 384 to 768; inside PTB-XL, which carries
# 234 of the 255 LVH positives, the same change was +0.0045 with a 95% bootstrap
# CI of [-0.0124, +0.0208]. Four fifths of the pooled gain came from the model
# making fewer LVH calls on STEMI-database images (11 -> 1), which is database
# recognition, not electrocardiography. See subgroup.py.
#
# resnet50_hires is THE BAR for notebook B. Not round 1's resnet50, and not
# resnet50_base: if a dual-resolution model cannot beat single-stream 768, the
# second branch bought nothing but parameters and latency.
ROUNDA: dict[str, dict[str, float]] = {
    "resnet50_base":  {"views": "384",     "pooled_auprc": 0.9036, "ptbxl_auprc": 0.9270,
                       "ptbxl_sens": 0.7863, "ptbxl_fpr": 0.0779, "stemi_db_fp": 11,
                       "macro_auprc": 0.9442, "params_m": 24.6, "train_min": 10.2},
    "resnet50_hires": {"views": "768",     "pooled_auprc": 0.9248, "ptbxl_auprc": 0.9315,
                       "ptbxl_sens": 0.8504, "ptbxl_fpr": 0.0727, "stemi_db_fp": 1,
                       "macro_auprc": 0.9644, "params_m": 24.6, "train_min": 36.9},
    "resnet50_cbam":  {"views": "384",     "pooled_auprc": 0.8901, "ptbxl_auprc": 0.9097,
                       "ptbxl_sens": 0.8162, "ptbxl_fpr": 0.0935, "stemi_db_fp": 7,
                       "macro_auprc": 0.9482, "params_m": 26.9, "train_min": 17.5},
    "resnet50_fpn":   {"views": "384",     "pooled_auprc": 0.8767, "ptbxl_auprc": 0.8976,
                       "ptbxl_sens": 0.8376, "ptbxl_fpr": 0.1273, "stemi_db_fp": 13,
                       "macro_auprc": 0.9381, "params_m": 27.4, "train_min": 14.3},
}

ROUNDA_BAR = "resnet50_hires"
ROUNDA_CONTROL = "resnet50_base"

# LVH positives per database in the test split. The reason PTB-XL is the only
# subgroup the focus label can be scored inside, and the reason the STEMI
# database can only ever contribute false positives to it.
FOCUS_POSITIVES_BY_SOURCE = {"PTBXL": 234, "CHAPMAN": 21, "STEMI": 0}


# ---------------------------------------------------------------- AF x LVH
#
# The frozen AF x LVH subgroup finding, transcribed once from
#
#     prototype-results/af_lvh_joint_v1/report.txt
#
# which scored resnet50_hires (768) on the SAME frozen 938-image test split
# this round uses, at its frozen thresholds (AF 0.81 / LVH 0.47). Verdict:
# SUPPORTED - the model correctly detects AF but disproportionately misses LVH
# on the AF+LVH cases (10 of 28), with LVH recall significantly below the
# LVH-only comparand (Fisher one-sided p = 0.0038) and AF recall not degraded.
#
# These are the numbers every new run in this round is scored against, by
# af_lvh.py. Frozen for the same reason as ROUND1 and ROUNDA: if a later run
# could recompute the reference, the bar would stop meaning anything.
#
# AUPRC values use the reference-pool design: the subgroup is the positive
# class and the negatives are every test ECG with true_<head> == 0. They are
# small-sample reads, reported as asked, not verdict reads.
AF_LVH_REFERENCE = "resnet50_hires"

AF_LVH_FROZEN = {
    "n": {"af_lvh": 28, "af_only": 257, "lvh_only": 227},
    "lvh": {
        "recall_af_lvh": 0.6429,     # TP 18 / 28
        "recall_lvh_only": 0.8722,   # TP 198 / 227
        "delta": -0.2294,
        "fisher_p": 0.0038,
        "auprc_af_lvh": 0.4535,
        "tp_af_lvh": 18, "fn_af_lvh": 10,
    },
    "af": {
        "recall_af_lvh": 0.9286,     # TP 26 / 28
        "recall_af_only": 0.9416,    # TP 242 / 257
        "delta": -0.0131,
        "fisher_p": 0.5142,
        "auprc_af_lvh": 0.8801,
        "tp_af_lvh": 26, "fn_af_lvh": 2,
    },
}
