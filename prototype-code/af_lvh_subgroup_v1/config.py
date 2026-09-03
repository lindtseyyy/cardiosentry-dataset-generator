"""
AF x LVH subgroup analysis of the four-head reference - shared constants.

WHAT THIS ROUND IS

Rounds 1-2 trained multi-label models over (STEMI, AF, LVH, NORMAL); the best
LVH numbers so far are the four-head ResNet-50@768 `resnet50_hires` (PTB-XL
LVH AUPRC 0.9315, pooled 0.9248), and LVH remains the worst of the four heads
(STEMI AUPRC 0.9979, AF 0.9825, NORMAL 0.9523). This round asks a
stratification question about the SAME frozen model:

    does the LVH head deteriorate specifically on ECGs that carry AF?

It does not train anything. It reads the frozen per-image test predictions of
`resnet50_hires` (the exact file its reported numbers were computed from),
stratifies the 938-image test split by the GROUND-TRUTH comorbidity labels,
and measures the LVH head inside each stratum. Optionally - and only as a
verification - it reloads the saved weights and re-runs the test split to
prove the frozen predictions file is the model it claims to be.

WHY SUBGROUPS ARE DEFINED BY GROUND TRUTH, NOT BY PREDICTIONS

A subgroup analysis that used the model's own AF predictions to decide which
ECGs "are AF" would make a false AF call drag an LVH error into the AF
stratum, or a missed AF call hide one - the two heads' errors would bleed
into each other. The true labels in the frozen index are the only stratum
boundaries that do not contaminate the LVH measurement.

FROZEN NUMBERS, SAME DISCIPLINE AS lvh_binary_v1

The reference is frozen in baseline.py and cross-checked three ways before a
single delta is printed: against the live result.json on Drive, against the
live test_predictions.csv, and against the transcribed constants. Any
disagreement stops the analysis - a moving reference cannot anchor a verdict.
"""
from __future__ import annotations

# ------------------------------------------------------------------ the task
# Order matters: this is the order the four-head run used, and the LVH column
# of the saved threshold vector is index 2 in it. Do not reorder.
LABELS = ["STEMI", "AF", "LVH", "NORMAL"]
FOCUS_LABEL = "LVH"
LVH_INDEX = LABELS.index("LVH")            # == 2

# ------------------------------------------------------------------ the model
# The reference model this analysis exists for. Its weights, result.json and
# frozen predictions all live in the same Drive run folder.
REFERENCE_MODEL = "resnet50_hires"
REFERENCE_TASK = "multi-label (STEMI, AF, LVH, NORMAL) - the four-head ResNet-50@768"

# Frozen val-tuned threshold for the LVH head. This is the operating point the
# reference run reported; every threshold-dependent number in this analysis is
# computed AT this threshold and never re-tuned. baseline.py cross-checks it
# against the live result.json and refuses on disagreement.
LVH_THRESHOLD = 0.47

# Reference operating point for the threshold-free matched-FPR column, same
# definition as lvh_binary_v1/subgroup.py: sensitivity capped at the highest
# achievable FPR <= 0.08. Thresholds are tuned on the whole validation set, so
# raw sensitivity can differ across subgroups purely because the operating
# point sits differently there; this column compares DETECTORS instead.
MATCHED_FPR = 0.08

# ------------------------------------------------------------------ subgroups
# The four comorbidity strata. Every ECG in the corpus falls into exactly one,
# by GROUND-TRUTH labels. Names are the user-facing names of the LVH-positive
# part of each stratum (the analysis question is about LVH detection inside
# it), but the strata themselves contain LVH-negatives too - that is what
# makes sensitivity/specificity/AUPRC computable inside each.
#
#   clean       AF=0 and STEMI=0      -> "LVH only"        (LVH vs NORMAL, in test)
#   af          AF=1                  -> "AF + LVH"
#   stemi       STEMI=1               -> "STEMI + LVH"
#   af_stemi    AF=1 and STEMI=1      -> "AF + STEMI + LVH"
CONTEXTS = {
    "clean":     {"AF": 0, "STEMI": 0, "display": "LVH only (no AF, no STEMI)"},
    "af":        {"AF": 1,             "display": "AF + LVH (AF present)"},
    "stemi":     {"STEMI": 1,          "display": "STEMI + LVH (STEMI present)"},
    "af_stemi":  {"AF": 1, "STEMI": 1, "display": "AF + STEMI + LVH"},
}

# The strata the verdict compares: AF present vs clean. STEMI strata are
# degenerate for LVH (see subgroups.py - the corpus carries zero STEMI+LVH
# ECGs), so they contribute counts and false positives, not LVH scores.
COMPARISON = ("clean", "af")

# Sufficiency floors for scoring a stratum's LVH head. Below these the stratum
# is reported as counts only - a CI over 12 positives would be noise with a
# confidence band on it.
MIN_POSITIVES = 20
MIN_NEGATIVES = 50

# ------------------------------------------------------------------ bootstrap
# Same conventions as the round-1 evaluation report and lvh_binary_v1:
# 1,200 resamples, percentile 95% CIs, seed 42. NOTE the difference: a
# two-model comparison can resample the SAME images for both models (paired).
# A two-subgroup comparison cannot - the groups are disjoint image sets - so
# each group is resampled independently and the delta distribution is the
# per-draw difference of the two independent draws. This is stated in every
# table this round prints, not hidden in a footnote.
N_RESAMPLES = 1200
SEED = 42
ALPHA = 0.05

# ------------------------------------------------------------------ the corpus
# Frozen geometry of the reference test split, cross-checked at load time.
TEST_N = 938
TEST_POSITIVES_LVH = 255
CORPUS_N = 6670

# The reference run's recorded identity, cross-checked against the live
# result.json and the saved checkpoint.
REFERENCE_PROTOCOL_FP = "36988c5e0ba5d72d"
REFERENCE_INPUT_FP = "6b9b1dd6f951bad0"
REFERENCE_VIEWS = [768]
REFERENCE_TIMM_VERSION = "1.0.28"

# ------------------------------------------------------------------ input pipeline
# Only the model-verification path needs these; they must match the reference
# run's config byte for byte or the recomputed probabilities would not be
# comparable. Transcribed from cardiosentry_hybrids_v1/config.py.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
LOAD_SIZE = 768
CHANNELS_LAST = True
AMP = True


def context_name(af: int, stemi: int) -> str:
    """Map (true_AF, true_STEMI) to the stratum key. Exhaustive by construction."""
    return ("clean" if not af and not stemi else
            "af" if af and not stemi else
            "stemi" if not af and stemi else "af_stemi")
