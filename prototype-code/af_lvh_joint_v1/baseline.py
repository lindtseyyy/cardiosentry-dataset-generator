"""
The four-head reference, frozen, for the AF x LVH joint-head analysis.

WHY HARD-CODED

If the reference could be recomputed or re-thresholded by anything in this
harness, the head-by-head comparison could be made to shift by changing how
the reference was derived. These numbers are transcribed ONCE, mechanically,
from

    prototype-results/hybrid_v1/resnet50_hires/result.json

produced under protocol fingerprint 36988c5e0ba5d72d, on the SAME frozen
938-image test split the previous round (af_lvh_subgroup_v1) used, with
thresholds tuned on the same validation split.

analyze.py reads the LIVE reference (result.json + test_predictions.csv from
the hybrid_v1 run folder on Drive) and cross-checks every number below
against it before printing anything: the pooled AF and LVH rows, both
thresholds, and the subgroup sizes. Any disagreement stops the run - a
moving reference cannot anchor a verdict.
"""
from __future__ import annotations

# ---------------------------------------------------------------- the reference
REFERENCE_MODEL = "resnet50_hires"
REFERENCE_TASK = "multi-label (STEMI, AF, LVH, NORMAL)"
REFERENCE_PROTOCOL_FP = "36988c5e0ba5d72d"
REFERENCE_INPUT_FP = "6b9b1dd6f951bad0"
REFERENCE_VIEWS = [768]

# Frozen val-tuned thresholds, in LABELS order (STEMI, AF, LVH, NORMAL).
REFERENCE_THRESHOLDS = [0.43999999999999995, 0.8099999999999999,
                        0.4699999999999999, 0.38999999999999996]

# The four-head model's AF column on the frozen test split, transcribed from
# result.json. The pooled row analyze.py re-derives from the predictions file
# must equal this exactly.
REFERENCE_AF = {
    "auroc": 0.98873485398028, "auprc": 0.9825146769889157,
    "sensitivity": 0.9403508771929825, "specificity": 0.9770290964777948,
    "precision": 0.9469964664310954, "f1": 0.943661971830986,
    "fpr": 0.022970903522205207, "threshold": 0.8099999999999999,
    "support": 285, "n": 938, "tp": 268, "fp": 15, "fn": 17, "tn": 638,
}

# The four-head model's LVH column, same source. Equal to the transcription
# in the previous round's baseline.py by construction.
REFERENCE_LVH = {
    "auroc": 0.9626159102000975, "auprc": 0.9248250492461644,
    "sensitivity": 0.8470588235294118, "specificity": 0.9502196193265008,
    "precision": 0.864, "f1": 0.8554455445544554,
    "fpr": 0.04978038067349927, "threshold": 0.4699999999999999,
    "support": 255, "n": 938, "tp": 216, "fp": 34, "fn": 39, "tn": 649,
}

# Frozen test-split geometry - subgroup sizes, the same counts the previous
# round's counting tables reported.
TEST_N = 938
AF_LVH_N = 28          # the subgroup this round exists for
AF_ONLY_N = 257        # AF=1, LVH=0
LVH_ONLY_N = 227       # LVH=1, AF=0, STEMI=0
ALL_AF_N = 285
ALL_LVH_N = 255

# The negative pools the per-head rows evaluate against: every test ECG that
# is true-negative for that head.
AF_NEG_POOL_N = TEST_N - ALL_AF_N          # 653
LVH_NEG_POOL_N = TEST_N - ALL_LVH_N        # 683


def check_frozen_counts() -> None:
    """Cross-check the transcribed numbers the moment they are loaded."""
    for lab, ref, pos in (("AF", REFERENCE_AF, ALL_AF_N),
                          ("LVH", REFERENCE_LVH, ALL_LVH_N)):
        assert ref["tp"] + ref["fn"] == pos, f"{lab}: tp+fn != positives"
        assert ref["support"] == pos, f"{lab}: support != positives"
        assert ref["n"] == TEST_N, f"{lab}: n != {TEST_N}"
        assert ref["tp"] + ref["fp"] + ref["fn"] + ref["tn"] == TEST_N, \
            f"{lab}: counts do not sum to {TEST_N}"
    assert AF_LVH_N + AF_ONLY_N == ALL_AF_N, "AF+LVH + AF-only != all AF"
    # LVH positives outside the AF stratum are exactly the LVH-only stratum
    # (the STEMI database carries zero LVH positives - a corpus fact), so
    # the two strata partition all 255 LVH positives.
    assert AF_LVH_N + LVH_ONLY_N == ALL_LVH_N, "LVH strata do not partition all LVH"
    assert REFERENCE_THRESHOLDS[1] == REFERENCE_AF["threshold"], "AF threshold slot"
    assert REFERENCE_THRESHOLDS[2] == REFERENCE_LVH["threshold"], "LVH threshold slot"


check_frozen_counts()
