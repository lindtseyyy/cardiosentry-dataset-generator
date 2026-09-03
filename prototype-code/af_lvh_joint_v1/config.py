"""
AF x LVH joint-head analysis - shared constants.

WHAT THIS ROUND IS

Round af_lvh_subgroup_v1 stratified the frozen test split by ground-truth
comorbidity and showed the LVH head deteriorates inside the AF stratum
(AUROC 0.9663 -> 0.8974, verdict SUPPORTED). This round zooms into the 28
ECGs where AF and LVH co-occur and asks the head-by-head question:

    when AF and LVH coexist, does the model correctly detect AF
    but miss LVH?

Same frozen everything: the SAME test_predictions.csv of `resnet50_hires`,
the SAME frozen val-tuned thresholds (AF 0.81, LVH 0.47), the SAME ground-
truth subgroup definition. Nothing is retrained, re-thresholded,
re-predicted or re-split. The AF+LVH subgroup of the frozen test split has
28 ECGs - a structural fact of the corpus, stated in every table.

DESIGN: how "AF specificity" and "AF AUPRC" exist for a subgroup

Sensitivity/recall is computable INSIDE the subgroup (all 28 are AF- and
LVH-positive). Specificity, precision, AUPRC, AUROC need negatives, and the
subgroup contains none. The reference-pool design: for each head, the
subgroup's ECGs are the positive class and the negatives are the frozen pool
of test ECGs that are true-negative FOR THAT HEAD (653 non-AF for the AF
head, 683 non-LVH for the LVH head). That makes every requested metric
computable per subgroup, at the frozen thresholds, with the same negative
pool for every subgroup row - so the rows differ ONLY in which positives
they count. Consequences, declared:

  * specificity and FP/TN are IDENTICAL across the rows of one head (same
    negatives, same decisions) - they are reported once per row for
    completeness, and the table says so;
  * precision and AUPRC with a 28-case positive class are mechanically
    small (fewer TP against the same FP pool) - they are reported as asked,
    but the verdict reads are recall, the joint breakdown and the exact
    tests, not these prevalence- and numerator-sensitive columns.

THE VERDICT RULE (frozen - stated here, applied in analyze.py, never
adjusted to fit the numbers)

HYPOTHESIS: "When AF and LVH coexist, the model correctly detects AF but
frequently misses LVH."

R1  WITHIN-CASE ASYMMETRY - McNemar's exact test on the joint breakdown of
    the 28 cases (discordant pairs: "AF correct + LVH missed" vs
    "AF missed + LVH correct"). Significant at p < 0.05 in the claimed
    direction = the two heads behave differently ON THE SAME ECGs.
R2  AF NOT DEGRADED - AF recall on AF+LVH vs AF-only, Fisher exact one-sided.
    "Correctly detects AF" is credible if AF recall on AF+LVH is not
    significantly below AF-only recall.
R3  LVH DEGRADED - LVH recall on AF+LVH vs LVH-only, Fisher exact one-sided.
    "Misses LVH" is a subgroup deficit only if LVH recall on AF+LVH is
    significantly below LVH-only recall.

  SUPPORTED       R1 significant in the claimed direction, R2 not
                  significant, R3 significant.
  PARTIALLY       R1 significant but R2 significant or R3 not significant.
  NOT SEPARATED   R1 not significant (directional evidence only).
  CONTRADICTED    R1 significant in the opposite direction.

SAMPLE-SIZE LIMITATION (stated in every report, regardless of verdict): the
subgroup has 28 ECGs (the previous round counted 25 distinct patients), so
recall CIs are wide and R2/R3 are low-power comparisons. The verdict grades
the evidence for the ASSOCIATION on this test split. No causal claim is made
- "AF causes the miss" cannot be concluded from these numbers, only that the
co-occurring cases behave this way in the frozen evaluation.
"""
from __future__ import annotations

LABELS = ["STEMI", "AF", "LVH", "NORMAL"]

# The two heads this round reads, and their frozen val-tuned thresholds -
# transcribed from result.json (AF 0.8099999999999999, LVH 0.4699999999999999)
# and cross-checked against the live file by analyze.py.
HEADS = ("AF", "LVH")
THRESHOLDS = {
    "AF": 0.8099999999999999,      # printed as 0.81 in the reference report
    "LVH": 0.4699999999999999,     # printed as 0.47
}

# The subgroups, defined by GROUND-TRUTH labels of the frozen test split.
# af_lvh is the round's subject; af_only and lvh_only are its comparands;
# all_af / all_lvh reproduce the frozen pooled reference rows (cross-check).
SUBGROUPS = {
    "af_lvh":   {"display": "AF + LVH (n=28)",     "AF": 1, "LVH": 1},
    "af_only":  {"display": "AF-only (AF, no LVH)", "AF": 1, "LVH": 0},
    "lvh_only": {"display": "LVH-only (no AF, no STEMI)",
                 "AF": 0, "LVH": 1, "STEMI": 0},
    "all_af":   {"display": "all AF (pooled reference)", "AF": 1},
    "all_lvh":  {"display": "all LVH (pooled reference)", "LVH": 1},
}

# Head -> the subgroup rows shown for it, in display order.
HEAD_ROWS = {
    "AF":  ("af_lvh", "af_only", "all_af"),
    "LVH": ("af_lvh", "lvh_only", "all_lvh"),
}

# Frozen subgroup sizes of the test split (also re-derived live and checked).
SUBGROUP_N = {"af_lvh": 28, "af_only": 257, "lvh_only": 227,
              "all_af": 285, "all_lvh": 255}
TEST_N = 938

# Bootstrap conventions - the project's, unchanged.
N_RESAMPLES = 1200
SEED = 42
ALPHA = 0.05

# Reference run identity, cross-checked against the live result.json.
REFERENCE_MODEL = "resnet50_hires"
REFERENCE_PROTOCOL_FP = "36988c5e0ba5d72d"
