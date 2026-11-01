"""Threshold policies compared in this study (all fitted on validation only).

`PRIMARY_POLICY` was chosen from validation evidence and the literature review
before any test-set metric was computed. Every other policy is a predeclared
comparison arm or a sensitivity-analysis scenario, not a candidate for
post-hoc promotion after inspecting test results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from common import LABELS


@dataclass(frozen=True)
class Rule:
    method: str                 # fixed | max_fbeta | youden_j | sens_constrained | spec_constrained
    params: dict = field(default_factory=dict)
    goal_category: str = ""     # guideline | published study protocol | observed study result | our experimental hypothesis
    rationale: str = ""


F1 = Rule("max_fbeta", {"beta": 1.0}, "published study protocol",
          "Current CardioSentry baseline: per-label validation-optimal F1.")
F2 = Rule("max_fbeta", {"beta": 2.0}, "our experimental hypothesis",
          "Recall-weighted F2: missing the finding is judged costlier than a false alert, "
          "but no defensible numeric sensitivity floor exists.")


def uniform(rule: Rule) -> dict[str, Rule]:
    return {label: rule for label in LABELS}


CLINICAL_V1 = {
    "NORM": Rule(
        "spec_constrained", {"min_specificity": 0.995, "use_wilson_lower": True},
        "our experimental hypothesis",
        "A 'Normal' output is reassurance; its harmful error is calling an abnormal ECG normal "
        "(NORM false positive). CODE-II (2026) argues qualitatively for a high-precision normal "
        "class; no numeric target exists. Maximise NORM sensitivity subject to the lower 95% "
        "Wilson bound of specificity (abnormal ECGs not called normal) >= 0.995, i.e. at most "
        "~5 false reassurances per 1,000 abnormal ECGs."),
    "SB": Rule(F1.method, F1.params, F1.goal_category,
               "Statement-defined rate finding, often physiological; urgency is context dependent. "
               "Balanced F1 retained."),
    "STACH": Rule(F1.method, F1.params, F1.goal_category,
                  "Usually secondary to another process; balanced F1 retained."),
    "LBBB": Rule(F1.method, F1.params, F1.goal_category,
                 "Detection itself is not time-critical (new LBBB alone is not STEMI-equivalent, "
                 "2025 ACC/AHA ACS); its relevance is as a STEMI confounder, analysed separately. "
                 "Balanced F1 retained."),
    "LQT": Rule(F1.method, F1.params, F1.goal_category,
                "Report-statement QT/QTc flag (not congenital LQTS); the model's weakest common label "
                "(validation AP 0.88). A recall-weighted rule would triple false alerts (F2 arm) on a "
                "measurement-defined finding that a clinician must measure anyway. Balanced F1 retained."),
    "AF": Rule(F2.method, F2.params, F2.goal_category,
               "Missed AF delays stroke-risk assessment; ESC 2024 (Class I) requires physician review "
               "of any ECG-based AF call, so false alerts are reviewable. Mildly recall-weighted F2."),
    "AFL": Rule(F2.method, F2.params, F2.goal_category,
                "Same anticoagulation consequence as AF (ESC 2024); mildly recall-weighted F2."),
    "3AVB": Rule(
        "sens_constrained", {"min_sensitivity": 0.90, "use_wilson_lower": True},
        "our experimental hypothesis",
        "Complete heart block can need urgent pacing; a miss is high-consequence. Maximise "
        "specificity subject to the lower 95% Wilson bound of validation sensitivity >= 0.90 "
        "(margin for transport to unseen data). The 0.90 level is a hypothesis, not a guideline."),
    "2AVB": Rule(F2.method, F2.params, F2.goal_category,
                 "Mobitz II / high-grade need pacing regardless of symptoms (2018 ACC/AHA/HRS), but the "
                 "label merges them with Mobitz I and 2:1 and has the weakest waveform support in the "
                 "signal audit (34%). A hard floor would chase weakly supported labels at a large "
                 "false-alert cost; recall-weighted F2 instead."),
    "1AVB": Rule(F1.method, F1.params, F1.goal_category,
                 "Usually benign, no urgent action; balanced F1 retained."),
    "STEMI": Rule(
        "sens_constrained", {"min_sensitivity": 0.90, "use_wilson_lower": True},
        "our experimental hypothesis",
        "Missing an ST-elevation MI pattern delays reperfusion; false alerts are triaged by a "
        "clinician. Maximise specificity subject to the lower 95% Wilson bound of validation "
        "sensitivity >= 0.90. The 0.90 level is a hypothesis, not a guideline."),
    "STE": Rule(F2.method, F2.params, F2.goal_category,
                "Morphological ST elevation (superset of STEMI: every STEMI record is also STE) with "
                "many mimics. Sensitivity-leaning but less aggressive than STEMI: recall-weighted F2."),
}

# Predeclared variant: same as CLINICAL_V1 but with a stricter 0.95 floor on the critical labels.
CLINICAL_V1_STRICT = {
    **CLINICAL_V1,
    **{label: Rule("sens_constrained", {"min_sensitivity": 0.95, "use_wilson_lower": True},
                   "our experimental hypothesis",
                   "Sensitivity analysis: 0.95 Wilson-lower floor on the critical labels.")
       for label in ("STEMI", "3AVB")},
}

POLICIES: dict[str, dict[str, Rule]] = {
    "fixed_0_5": uniform(Rule("fixed", {"threshold": 0.5}, "", "Uncalibrated engineering default.")),
    "max_f1": uniform(F1),
    "max_f2": uniform(F2),
    "youden_j": uniform(Rule("youden_j", {}, "our experimental hypothesis",
                             "Symmetric sensitivity + specificity baseline.")),
    "sens_0_90": uniform(Rule("sens_constrained", {"min_sensitivity": 0.90}, "our experimental hypothesis",
                              "Scenario: point-estimate sensitivity >= 0.90 for every label.")),
    "sens_0_95": uniform(Rule("sens_constrained", {"min_sensitivity": 0.95}, "our experimental hypothesis",
                              "Scenario: point-estimate sensitivity >= 0.95 for every label.")),
    "sens_0_98": uniform(Rule("sens_constrained", {"min_sensitivity": 0.98}, "our experimental hypothesis",
                              "Scenario: point-estimate sensitivity >= 0.98 for every label.")),
    "spec_0_95": uniform(Rule("spec_constrained", {"min_specificity": 0.95}, "observed study result",
                              "Comparator: sensitivity at specificity >= 0.95, the pooled specificity of "
                              "computer-assisted STEMI interpretation (JRC/JCS 2022 meta-analysis: 95.4%).")),
    "clinical_v1": CLINICAL_V1,
    "clinical_v1_strict": CLINICAL_V1_STRICT,
}
PRIMARY_POLICY = "clinical_v1"
BASELINE_POLICY = "max_f1"

for _name, _rules in POLICIES.items():
    assert tuple(_rules) == LABELS, f"{_name} does not cover the labels in checkpoint order"
