# CardioSentry — Clinical Threshold Research & Implementation Guide

**Document type:** Evidence-informed recommendations and coding-agent research brief  
**Status:** Proposed research/engineering plan — **not** an approved medical protocol  
**Prepared:** 2026-10-15  
**Audience:** Coding agent, ML developer, thesis researchers, and clinical adviser

> **How to use this document:** Treat the suggestions below as hypotheses and design options, **not rigid instructions**. First inspect the actual CardioSentry repository, validate the scientific claims and publication details against original sources, identify the available data and label definitions, then choose an implementable plan and explain deviations. The goal is a reliable, reproducible comparison of threshold-selection strategies — **not to force a particular threshold, numeric medical target, or algorithm to win**.

---

## 1. Context and motivation

CardioSentry is an **image-based, multi-label ECG classification system** using a trained ConvNeXtV1 model (per our discussion). A sigmoid-style output gives a score for each of 12 categories. An independent class-specific cutoff maps score to binary prediction:

```python
positive_for_label_k = (score_k >= threshold_k)
```

The current notebooks reportedly use a function named `tune_thresholds_on_validation` that:

1. Loads/runs the selected model checkpoint on validation ECG images.
2. Uses `sklearn.metrics.precision_recall_curve` per label.
3. Chooses the threshold with the best **F1** on validation, with 0.5 as a default/fallback for labels lacking both classes or candidate thresholds.
4. Saves the per-label thresholds within `weights/best.pt`.
5. Applies those saved thresholds to test data **without tuning on test**.

A notebook mentioned in conversation is `CardioSentry_ConvNeXtV1_Base_Clean12_EC2.ipynb`, with threshold logic reportedly around line 2755. **This document has not inspected that notebook or `AGENTS.md` directly.** Locate the real files and verify behavior before editing. Do not assume the exact filename, code location, framework API, checkpoint shape, or label order.

### Underlying question

A per-label F1-maximizing threshold is legitimate for a machine-learning benchmark, but F1 **does not encode the clinical consequences** of missing a STEMI-like ECG pattern versus overcalling a relatively low-acuity finding. Would a sensitivity-constrained or otherwise clinically informed selection strategy better match the intended use?

### Suggested research objective

> Compare conventional validation-selected thresholds with alternative, clinically motivated **class-specific** operating points for the 12-label CardioSentry model. Evaluate whether high-priority ECG findings can achieve improved sensitivity without an unacceptable specificity/false-alert trade-off, while preserving an untouched independent test assessment.

**Reason:** This framing is testable, honest about trade-offs, and does not assume the clinical strategy will outperform F1 on every metric.

---

## 2. Non-negotiable scientific boundaries

These are safeguards rather than algorithm prescriptions.

- **No retraining is required to change decision thresholds.** Use the existing trained model and saved probability outputs when valid and available. Recompute predictions if preprocessing, checkpoints, or cached data are incompatible. Retrain **only if separate evidence suggests it is needed**.
- **Never tune, choose, revise, calibrate, or rank a final deployment strategy based on the held-out test labels.** Validation chooses; test evaluates. It is fine to compare several strategies on test **if they were all fully specified first**, but do not choose the winner after inspecting test performance and then present that score as an unbiased final result.
- **Keep the original F1 approach as a baseline.** Do not erase previous threshold logic/results or overwrite trained weights merely to add a new threshold policy.
- **Retain 12 independent label scores.** This is multi-label, not softmax or argmax multiclass classification. Do not force predictions to sum to one or select exactly one ECG label.
- **Do not claim any arbitrary target (e.g., STEMI sensitivity ≥90% or ≥95%) is a universal clinical requirement.** Research whether the intended workflow justifies a target; document the source, context, intended use, and uncertainty. Treat illustrative percentages as candidate experimental scenarios only.
- **Distinguish an ECG-pattern label from a confirmed patient diagnosis.** An image classifier labeling “STEMI pattern” is not automatically validated to rule in/out infarction or acute coronary occlusion. Similarly, “QTc prolongation” is not synonymous with congenital long-QT syndrome.
- **Report infeasible constraints as infeasible.** Do not relabel fallback thresholds as clinically validated; do not silently relax constraints after looking at test results.
- **Respect existing patient-/record-level split boundaries.** Prevent repeated ECGs, derived images, or duplicate renderings from the same patient/record leaking across partitions.
- **Keep the system clinically assistive until appropriately validated.** A negative model result does not independently exclude ACS, arrhythmia, or other disease; positive model alerts should prompt appropriate clinician assessment, not autonomous treatment.

---

## 3. The 12 labels: clinical-priority hypotheses for investigation

**Treat this matrix as a suggested starting taxonomy**, subject to actual label definitions, prevalence, annotation quality, intended population, and clinical review. Priority indicates a *research direction*, not a verified severity ranking or required decision threshold.

| # | CardioSentry output label | Suggested focus to investigate | Reason / clinical caveat | Open questions for agent |
|---|---|---|---|---|
| 1 | Normal ECG | False-reassurance / *any-abnormal* detection analysis; investigate a dedicated normal policy | A positive “normal” alongside a high-confidence critical abnormality could be misleading. Absence of the 11 target findings does **not** prove a completely normal ECG. | Is “normal” mutually exclusive in annotations? Are all meaningful abnormal classes included? |
| 2 | Sinus bradycardia | F1, F2, or balanced criteria as baseline | Slow sinus rates can be physiological; urgency is context dependent. | What heart-rate and rhythm definition labels use? |
| 3 | Sinus tachycardia | F1, F2, or balanced criteria as baseline | Rate detection useful, but clinical significance depends on circumstances. | Are rate and sinus rhythm independently verified? |
| 4 | First-degree atrioventricular block | F1 or F2 with specificity checks | Generally different clinical consequences from complete AV block. | Is this based on PR interval >200 ms and 1:1 conduction? |
| 5 | Atrial fibrillation | Consider sensitivity-oriented selection, constrained by false alerts | Missed findings may affect further workup; a model-predicted AF label is not a definitive treatment decision. | What duration/ECG reference standard defines AF? |
| 6 | Atrial flutter | Consider sensitivity-oriented selection, constrained by false alerts | Similar screening issue; often confusable with AF or other rhythms. | Do co-occurring AF/flutter labels reflect multiple rhythms or noisy annotations? |
| 7 | Left bundle branch block | Investigate balanced metrics or moderately recall-weighted selection | Conduction abnormality warrants accurate detection; important to analyze alongside ischemic patterns. | Is the LBBB reference label verified and clearly defined? |
| 8 | Second-degree atrioventricular block | Higher-sensitivity investigation, **subtype-aware if possible** | Mobitz I versus Mobitz II/high-grade blocks can have very different risk. | Can Mobitz I, Mobitz II, 2:1, and high-grade be separated? |
| 9 | QT/QTc prolongation (Long QT) | Sensitivity analysis with robust label/measurement definition | QTc measurement depends on correction method, rate, sex/age, and QRS context; marked QTc deserves separate scrutiny. | Is this an ECG QTc finding, a measured interval threshold, or a clinical LQTS label? |
|10 | ST elevation (not labeled STEMI) | Balanced detection plus false-alert analysis | Morphologic ST elevation can have multiple causes and is not interchangeable with STEMI. | Does it mean generic ST elevation or explicitly non-STEMI ST elevation? |
|11 | STEMI pattern | Explore explicit high-sensitivity constraints, then maximize specificity or minimize false alarms | Missing potential acute ischemic patterns is consequential, but false alerts also cause burden. This is an ECG pattern, not automatically adjudicated MI. | Is outcome annotated by ECG report, clinical diagnosis, angiography, or another standard? |
|12 | Third-degree (complete) atrioventricular block | Explore explicit high-sensitivity constraints with strong uncertainty reporting | Complete block can be medically serious; label rarity may make apparent sensitivity unstable. | How many genuinely positive independent validation/test patients exist? |

**Additional recommendation:** Ask a qualified cardiology/clinical adviser to review the intent, label ontology, and high-priority class criteria. If none is available, describe the approach as **hypothesis-driven exploratory research**, not a clinically endorsed threshold policy.

---

## 4. Suggested literature review and evidence-gathering tasks

The coding agent should **research before locking implementation choices**. Prefer original studies, clinical practice guidelines, medical-device evaluation guidance, and research-reporting statements. Record full citation, publication date, exact study population, ECG modality, target definition, threshold-selection approach, test design, and limitations.

### 4.1 Literature search questions

Investigate at least:

1. How do multi-label ECG models choose cutoffs (F1, Youden index, target sensitivity, fixed specificity, cost-sensitive optimization, clinician-matched operating points)?
2. How are **STEMI/acute myocardial infarction**, **AF/flutter**, **complete/high-grade AV block**, **LBBB**, **QTc prolongation**, and **normal ECG/abnormal screening** assessed differently?
3. What clinically meaningful **false-negative** outcomes and **false-positive** burdens are associated with each finding in its *intended workflow*?
4. Are there published threshold targets explicitly specified for a comparable screening/triage workflow? Are they study-specific or genuinely guideline recommendations?
5. How do PPV/NPV change with class prevalence? Is the dataset population similar to CardioSentry's intended use?
6. How should image-based ECG models handle low-quality photos/scans, label noise, class imbalance, multiple recordings per patient, and external-site shift?
7. How do papers present confidence intervals and demonstrate that high sensitivity from rare positives is statistically credible?
8. Are probabilistic calibration or decision-curve analyses relevant to the intended output (score-only decision, clinical alert, triage, or actual action)?

### 4.2 Initial sources to verify and consult

These are **starting points, not automatic proof that particular clinical cutoffs are justified**:

| Source | URL | Why consult it / limit of applicability |
|---|---|---|
| Ribeiro et al., 2020, *Automatic diagnosis of the 12-lead ECG using a deep neural network* | https://doi.org/10.1038/s41467-020-15432-4 | Strong ECG-classification methodological benchmark. The labels and input modality may differ from image-based CardioSentry. |
| Hannun et al., 2019, *Cardiologist-level arrhythmia detection and classification* | https://doi.org/10.1038/s41591-018-0268-3 | Useful precedent for sensitivity comparisons at **matched specificity**; single-lead ambulatory ECG, not identical to 12-lead images. |
| CODE-II, 2026, *A large-scale dataset for artificial intelligence in ECG analysis* | https://doi.org/10.1038/s41746-026-02704-4 | Particularly relevant: per-class F1 threshold selection on validation and unchanged application to independent testing; multi-label ECG benchmark. |
| ROMIAE multicentre study, 2025 | https://doi.org/10.1093/eurheartj/ehaf004 | Demonstrates distinct sensitivity-oriented and specificity-oriented risk operating points for **AMI**. **AMI ≠ ECG-only STEMI-pattern label**. |
| Kim et al., 2022, *AI screening for early STEMI detection* | https://doi.org/10.3346/jkms.2022.37.e81 | Research example of evaluated STEMI screening and clinical comparison. Check reference standard and thresholds in original study. |
| 2025 ACC/AHA guideline for acute coronary syndromes | https://www.jacc.org/doi/10.1016/j.jacc.2024.11.009 | Clinical interpretation, urgency and ECG limitations; do **not** infer a universal AI model sensitivity cutoff. |
| 2018 ACC/AHA/HRS guideline on bradycardia and conduction delay | https://doi.org/10.1161/CIR.0000000000000628 | Clinical distinctions across Mobitz II, high-grade and complete AV block. |
| 2024 ESC atrial fibrillation guideline | https://academic.oup.com/eurheartj/article/45/36/3314/7738779 | ECG confirmation, AF clinical context; not a direct threshold table for a new AI classifier. |
| 2022 ESC ventricular arrhythmia / sudden death guideline | https://academic.oup.com/eurheartj/article/43/40/3997/6675633 | QTc and ventricular arrhythmia context; distinguish measured QTc from prediction score. |
| TRIPOD+AI reporting guideline, 2024 | https://www.bmj.com/content/385/bmj-2023-078378 | Transparency, model validation, calibration, and reporting. |
| STARD-AI reporting guideline, 2025 | https://doi.org/10.1038/s41591-025-03953-8 | Diagnostic-accuracy reporting, reference standard, selection bias, external applicability. |

#### Expected research output

Produce a short **evidence matrix** with fields:

`label_or_use_case | proposed_operating_goal | value_if_any | study_and_year | endpoint | ECG_type | source_population | threshold_selection_protocol | confidence_interval | applicability_to_CardioSentry | uncertainty | recommendation`.

Mark explicit categories: **guideline**, **published study protocol**, **observed study result**, **clinical-adviser decision**, or **our experimental hypothesis**. Never conflate them.

---

## 5. Candidate thresholding strategies: suggestion + rationale

Implementing every algorithm immediately may be unnecessary; the agent should determine available data, computational cost, and study scope first. These are reasonable **comparison arms**, with F1 as the essential baseline.

### 5.1 Fixed 0.5 baseline

- **Suggestion:** Evaluate cutoff `0.5` per label if scores are in `[0,1]`.
- **Reason:** Easy reference point and verifies the pipeline.
- **Caveat:** Sigmoid scores need not be well calibrated; 0.5 is not inherently clinically appropriate.

### 5.2 Per-label validation-optimal F1 — existing behavior

For class `k`, select `t` maximizing:

`F1 = 2 * precision * recall / (precision + recall)`.

- **Suggestion:** Preserve/reproduce original results exactly before introducing new policies.
- **Reason:** Controls for code changes and supports a fair experiment against current CardioSentry.
- **Caveat:** F1 ignores true negatives, disease prevalence varies, and it does not guarantee clinically desired sensitivity.

### 5.3 F-beta optimization (e.g., F2)

`F_beta = (1+beta²)*precision*recall / (beta²*precision+recall)`.

- **Suggestion:** Explore F2 as a recall-weighted alternative when missing positives is of greater concern.
- **Reason:** Simple extension to the existing precision–recall threshold workflow.
- **Caveat:** **F2 does not guarantee minimum sensitivity**, even when beta > 1.

### 5.4 Youden's J

`J(t) = sensitivity(t) + specificity(t) - 1`.

- **Suggestion:** Consider as a symmetric diagnostic-performance baseline.
- **Reason:** Includes true-negative behavior, unlike F1.
- **Caveat:** Equal weighting of sensitivity and specificity may not reflect clinical consequences. It is not a default clinical mandate.

### 5.5 Sensitivity-constrained selection (candidate for higher-priority labels)

A possible form:

```text
maximize    specificity_k(t)
subject to  sensitivity_k(t) >= min_sensitivity_k
optionally  specificity_k(t) >= min_specificity_k
```

- **Suggestion:** Explore for defined high-consequence labels such as STEMI pattern and complete AV block; possibly also AF/flutter and other clinically important findings depending on use case.
- **Reason:** Expresses a measurable minimum detection objective while controlling false alarms as much as possible.
- **Caveat:** A threshold satisfying an **empirical validation sensitivity** target does **not** guarantee that target on unseen data. Confidence intervals matter; testing may reveal the target was not achieved.
- **Candidate sensitivity levels for sensitivity analysis ONLY:** 0.90, 0.95, 0.98. **Do not encode these as medical facts or defaults without documented support.** Discuss acceptable false-positive burden and feasibility as well.

### 5.6 Specificity-constrained selection

```text
maximize    sensitivity_k(t)
subject to  specificity_k(t) >= min_specificity_k
```

- **Suggestion:** Explore for a high-alert or confirmation-oriented category only if the clinical workflow requires strong false-positive control.
- **Reason:** Prevents alert overload while maximizing detection within that operational constraint.
- **Caveat:** A specificity-oriented rule-in alert is not a clinical diagnosis and should not conceal the possibility of false negatives.

### 5.7 Expected-cost / clinical utility selection (optional research branch)

`loss(t) = C_FN * prevalence * (1-sensitivity) + C_FP * (1-prevalence) * (1-specificity)`.

- **Suggestion:** Explore only if defensible relative costs or clinical action thresholds can be specified, ideally with clinician input.
- **Reason:** Represents trade-offs more explicitly than F1 or Youden J.
- **Caveat:** Arbitrary FP/FN weights give only arbitrary optimums; results depend on population prevalence and cost assumptions. If reliable inputs are unavailable, keep as theoretical discussion.

### 5.8 Dual-threshold / abstention region (optional, later)

- **Suggestion:** Investigate separate low-alert/high-alert thresholds with an intermediate **review required** state for selected classes.
- **Reason:** Can better model uncertain cases and different clinical review priorities.
- **Caveat:** More complex product semantics and separate evaluation of coverage, referral burden, and missed events. Must **not** label low-alert results as definitive disease exclusion.

### 5.9 Calibration (separate question from threshold choice)

- **Suggestion:** Inspect score calibration; if visibly poor and data allow, investigate per-label Platt/sigmoid or isotonic calibration using data independent of threshold/test evaluation.
- **Reason:** A score of 0.8 is not necessarily an 80% observed probability; calibration matters for interpreting outputs and transporting thresholds.
- **Caveat:** Calibrating and tuning on the same small validation set may overfit; use split, cross-fitting, or another justified protocol. Only apply calibration mappings fitted without test data. Calibration is **not** obligatory for every threshold-based detector and must not become unplanned scope creep.

---

## 6. Proposed end-to-end experimental workflow

### Stage A — Repository and data audit (before coding)

**Suggestion:** Inspect `AGENTS.md`, actual notebook/scripts, config, labels, preprocessing, checkpoint, evaluation conventions, and available cached predictions.

**Reason:** A generic threshold script can introduce label-order or preprocessing mismatches while appearing to run correctly.

Questions the agent should answer:

- What is the authoritative ordered list of all 12 labels? Are the exact class names identical to this document?
- Does the checkpoint output 12 independent sigmoid probabilities, logits, or something else? Is sigmoid already applied?
- How was `best.pt` chosen, and what else does it store? Does inference load thresholds from the checkpoint or from another source?
- How are validation/test labels encoded: `0/1`, `-1` uncertain, missing/NaN, soft targets, etc.?
- Are train/validation/test splits defined by **patient**, ECG recording, image filename, or random image? Are there duplicate or derived ECG images?
- Have test-set results already influenced any earlier architecture, hyperparameter, or threshold decisions? If yes, describe that limitation honestly; an independently held-out dataset may be required for fresh unbiased claims.
- Are validation predictions already cached? Do they belong to the same trained checkpoint, image transforms, and class order?
- How many independent positive and negative cases exist **per class** in validation and test?

**Output:** A short audit report and a safe change plan **before modifying existing notebooks**.

### Stage B — Reproduce the existing F1 results

**Suggestion:** Run/cache validation inference **without training**, compare recovered original F1 thresholds and validation metrics with existing saved results.

**Reason:** Creates a trustworthy baseline and catches score/label/order or float-handling errors before other methods are added.

Cache only if privacy rules permit. Prefer deterministic CSV/Parquet/NPZ with a manifest and only minimally necessary identifiers, e.g.:

```text
record_id | split | y_true[12] | y_score[12] | model_id | preprocessing_id
```

If identifiers could expose sensitive medical data, use de-identified or pseudonymous IDs, and protect the data according to project policy.

### Stage C — Develop or select threshold policies on validation only

**Suggestion:** Run candidate methods per class; record *all* operating points, feasibility, and data support. Make threshold policy decisions using prior clinical rationale and validation evidence, then freeze the plan.

**Reason:** Enables meaningful comparison while keeping the test dataset independent.

For a small validation set, consider separate checkpoint-selection vs threshold-development subsets, cross-validation, cross-fitting, or bootstrap stability diagnostics rather than blindly subdividing into unusably small sets. **Do not allow threshold selection to leak test labels.**

### Stage D — Freeze experiments before touching test labels

**Suggestion:** Write an experiment manifest specifying:

- checkpoint hash/version and preprocessing hash/version;
- exact label order and label definitions;
- dataset partition version;
- chosen strategies to evaluate (e.g., fixed0.5, F1, F2, Youden, clinical policy);
- per-label threshold or per-label selection rule **and already-selected value**;
- target sensitivity/specificity (if any), with evidence/justification;
- whether probability calibration is enabled and the mapping/version;
- tie-breaking rule and comparison operator (`>=`);
- metrics, confidence intervals, and planned subgroup/error analyses;
- timestamp, random seeds, and code version.

**Reason:** Prevents inadvertent retuning and makes results reproducible.

### Stage E — Independent test evaluation

**Suggestion:** Apply every pre-frozen candidate threshold set to untouched test scores. Compute all metrics; report **test performance even when the validation goal fails**.

**Reason:** The critical research result is generalization, not how well you optimized the validation subset.

Example only:

```text
Validation: STEMI threshold 0.21, sensitivity 96%, specificity 85%.
Test:       SAME threshold 0.21, sensitivity 93%, specificity 83%.
Conclusion: test sensitivity did not meet the predeclared 95% target.
Action: report honestly; do not change 0.21 based on test results.
```

The numbers above are fictional illustration, **not a CardioSentry observation**.

### Stage F — Interpret feasibility and generalization

**Suggestion:** Compare methods and report whether the clinical goals appear feasible at an acceptable false-alert burden; quantify missed cases, uncertainty, and subgroup variation.

**Reason:** Threshold tuning cannot make an intrinsically weak classifier safe. A new threshold may improve recall while making false alarms untenable.

Only after the study should researchers decide whether model improvements, better annotations, a new dataset, a new validation split, or retraining are warranted. Any subsequent refit of a final production model needs its own appropriately independent evaluation strategy.

---

## 7. Metrics the agent should consider and why

### Per-label confusion matrix and metrics

For each class, compute `TP, FP, TN, FN` plus:

| Metric | Formula / role | Reason |
|---|---|---|
| Sensitivity / recall | `TP / (TP + FN)` | Captures missed positive conditions. |
| Specificity | `TN / (TN + FP)` | Captures false alarms among true negatives. |
| Precision / PPV | `TP / (TP + FP)` | How credible a positive alert is; highly prevalence dependent. |
| NPV | `TN / (TN + FN)` | How credible a negative output is; highly prevalence dependent. |
| F1 | Harmonic mean of precision/recall | Existing study baseline. |
| F2 | Recall-weighted F-score | Useful comparison without claiming fixed sensitivity. |
| False-negative rate | `FN / (TP + FN)` | Readable risk-related complement of sensitivity. |
| False-positive rate | `FP / (TN + FP)` | Alert burden. |
| False alarms per 1,000 ECGs | `FP / N * 1000` | Operationally interpretable if dataset prevalence is representative. |
| AUROC | Rank discrimination | Threshold-independent comparison; can hide imbalanced-class behavior. |
| Average precision / PR-AUC | PR discrimination | Better visibility for rare positives. Distinguish AP from trapezoidal PR-AUC. |
| 95% CI for sensitivity/specificity | Wilson/exact binomial or justified alternative | High point estimate from few cases is uncertain. |
| 95% CI for PPV/NPV/F1 | Bootstrap where appropriate | Quantifies instability in common thesis metrics. |

**Undefined denominators:** If no positive ground-truth cases, sensitivity is **undefined**, not 0 or 100%. If no negative cases, specificity undefined. If no predicted positives, PPV is undefined unless a deliberate reporting convention is used and clearly labeled. Do not hide these as meaningful zeros. Report support (`n_pos`, `n_neg`).

### Multi-label aggregate metrics

Consider reporting:

- macro F1, macro F2, macro recall, macro AUROC / average precision **where defined**;
- micro precision/recall/F1 (pool all label-level decisions);
- per-label raw performance rather than only an aggregate score;
- any-target-abnormal sensitivity / false-normal rate **if the dataset labels permit such a calculation**;
- number of abnormal ECGs incorrectly shown as normal;
- per-label and clinically relevant co-occurrence/confusion analysis (e.g., LBBB + ischemic patterns, AF vs flutter);
- sensitivity-specificity and PR curves, with selected operating points marked;
- calibration plots if clinically relevant.

**Reason:** Good macro/micro scores may still obscure serious rare-class false negatives. Labels are correlated, and interpretation may depend on combinations.

### Prevalence and alert burden

**Suggestion:** Present PPV/NPV at observed dataset prevalence and, when justified, sensitivity analysis at plausible intended-use prevalences using sensitivity/specificity assumptions. Do not misrepresent prevalence-adjusted values as directly observed test results.

**Reason:** Even high sensitivity and specificity can yield poor PPV in a low-prevalence environment. Alert burden affects downstream review and acceptability.

### Statistical comparisons

**Suggestion:** Prefer paired comparisons on identical test ECGs, with patient-level clustered bootstrap where repeated recordings exist. If confidence intervals or formal tests are included, prespecify them. Correct or qualify extensive exploratory multi-method comparisons.

**Reason:** Each thresholding method processes the **same** records; observations may not be independent. Confidence intervals and study design matter more than a small numerical improvement.

---

## 8. Technical design suggestions, not required architecture

The agent should **adapt to actual repository conventions**. Example components (rename/reorganize as needed):

```text
threshold_research/
  README.md                       # how to reproduce, scientific limitations
  research_notes.md              # sources, evidence-to-policy decisions
  policies.example.yaml          # human-reviewable strategies and options
  collect_predictions.py         # inference/caching on validation/test
  threshold_methods.py           # pure threshold optimizers
  evaluate_thresholds.py         # metrics and confidence intervals
  compare_experiments.py         # tables/figures from frozen outputs
  tests/
    test_threshold_methods.py
    test_no_test_leakage.py
    test_multilabel_normal.py
  outputs/
    validation_scores.*
    threshold_candidates.csv
    frozen_thresholds.json
    results_validation.csv
    results_test.csv
    figures/
```

**Reason:** Separating model inference from threshold search and evaluation makes experimentation faster, clearer, and less risky than repeatedly altering the original training notebook.

### Interface / data contract to investigate

Possible canonical shapes:

```python
scores.shape  == (n_ecgs, 12)  # probabilities in [0, 1]
labels.shape  == (n_ecgs, 12)  # ground-truth 0/1, except documented missing values
thresholds.shape == (12,)
predictions = (scores >= thresholds[None, :]).astype(int)
```

**Important:** Explicitly check whether the model outputs **logits** and whether `sigmoid` must be applied once (not twice). Maintain exact class order from checkpoint/data dictionary; never silently sort names alphabetically. Use explicit missing-label handling and record-level metadata as needed.

### Suggested per-policy configuration (research placeholders)

This is **illustrative structure** rather than a recommended final set of clinical cutoff values. Class names are human-readable placeholders; substitute the repository's exact names. Blank targets mean **undecided pending evidence review**, not zero or auto-pass.

```yaml
experiment: clinical_thresholds_research_v1
pretrained_checkpoint: "DISCOVER_FROM_REPOSITORY"
label_order_source: "CHECKPOINT_OR_TRAINING_METADATA"
comparison_operator: ">="
validation_split: "DISCOVER_FROM_REPOSITORY"
test_split: "DISCOVER_FROM_REPOSITORY"
method_comparisons:
  - fixed_0_5
  - max_f1
  - max_f2
  - youden_j
  - clinical_policy
clinical_policy_draft:
  "Normal ECG":
    proposal: normal_safety_analysis
    rationale: investigate_false_reassurance
  "Sinus bradycardia":
    proposal: max_f1
    rationale: general_benchmark
  "Sinus tachycardia":
    proposal: max_f1
    rationale: general_benchmark
  "First-degree atrioventricular block":
    proposal: max_f1_or_f2
    rationale: explore_balanced_vs_recall_weighted
  "Atrial fibrillation":
    proposal: sensitivity_constrained_candidate
    minimum_sensitivity: null   # set only after research/review
    rationale: screening_sensitivity_vs_alert_burden
  "Atrial flutter":
    proposal: sensitivity_constrained_candidate
    minimum_sensitivity: null
    rationale: screening_sensitivity_vs_alert_burden
  "Left bundle branch block":
    proposal: balanced_or_f2
    rationale: detect_conduction_pattern_with_specificity
  "Second-degree atrioventricular block":
    proposal: subtype_review_then_sensitivity_candidate
    minimum_sensitivity: null
    rationale: clinically_heterogeneous_label
  "QT/QTc prolongation (Long QT)":
    proposal: validate_label_then_sensitivity_candidate
    minimum_sensitivity: null
    rationale: ECG_QTc_not_diagnosed_congenital_LQTS
  "ST elevation (not labeled STEMI)":
    proposal: balanced_with_false_alert_review
    rationale: morphological_finding_not_infarction_diagnosis
  "ST-elevation myocardial infarction pattern":
    proposal: sensitivity_constrained_candidate
    minimum_sensitivity: null
    minimum_specificity: null
    rationale: high_consequence_false_negatives
  "Third-degree (complete) atrioventricular block":
    proposal: sensitivity_constrained_candidate
    minimum_sensitivity: null
    minimum_specificity: null
    rationale: high_consequence_false_negatives
notes: |
  Draft research options, NOT clinically prescribed targets.
  Agent should choose evidence-supported numeric study targets or
  explicitly label tested targets as exploratory hypotheses.
```

**Why avoid filling in 0.95 automatically?** Because this document is asking the agent to **research and justify** goals. It is entirely appropriate to test 0.90/0.95/0.98 in a *sensitivity analysis*, but those are **not** universal STEMI or AV block requirements.

### How to persist output

**Suggestion:** Prefer a versioned external threshold artifact (JSON/YAML + manifest) linked to the original model checkpoint, unless the existing app requires storing cutoffs in `best.pt`. If updating `best.pt` is necessary, make a copy and avoid changing model tensors.

**Reason:** Keeps immutable learned weights separate from post-training clinical policies, simplifies rollback and method comparisons, and avoids misleading checkpoint provenance.

Suggested metadata:

```json
{
  "schema_version": 1,
  "model_checkpoint_sha256": "TO_BE_COMPUTED",
  "source_split": "validation",
  "label_order": ["REPLACE_WITH_ALL_12_REAL_LABELS"],
  "score_definition": "post_sigmoid_probability_like_score",
  "comparison_operator": ">=",
  "policy_version": "research_v1",
  "calibration_version": null,
  "selected_thresholds": "ARRAY_OF_12_FINITE_VALUES",
  "selection_methods": "ARRAY_OF_12_METHOD_NAMES",
  "status_per_label": "ARRAY_OF_12_STATUSES",
  "selection_evidence": "PATH_TO_EVIDENCE_MATRIX_AND_REPORT"
}
```

The JSON above is a **schema sketch**, not a usable artifact until fields/types are completed and validated.

---

## 9. Algorithm details and edge cases to handle carefully

### 9.1 Correct handling of `precision_recall_curve`

In scikit-learn, `precision` and `recall` have **one more element** than the `thresholds` array. The final PR endpoint has no corresponding threshold.

**Suggestion:** When scoring thresholds, match `precision[:-1]`, `recall[:-1]` to `thresholds` and test against a tiny synthetic example.

**Reason:** An off-by-one can silently choose the wrong cutoff.

### 9.2 Enumerating sensitivity-constrained candidates

Possible approach: enumerate all valid score change-points (plus documented edge rules), compute class-wise confusion counts with `score >= t`, and select the candidate maximizing specificity among those meeting sensitivity (and optional specificity) constraints. A `roc_curve(..., drop_intermediate=False)` implementation can also work, but check edge thresholds (`inf`) and exact equality semantics.

**Suggestion:** Define a deterministic tie-break and test tied scores carefully.

**Reason:** No continuous optimizer is required for piecewise-constant metrics; the behavior changes only at score boundaries, but floating-point and equality semantics can change which ECGs are included.

### 9.3 Infeasibility and missing-class support

Suggested statuses:

```text
ok
infeasible_constraint
insufficient_positive_support
insufficient_negative_support
undefined_metric
invalid_input
needs_clinical_definition
```

**Suggestion:** Do not silently use 0.5 and mark a clinical constraint met when a class is absent or a constraint is infeasible. A 0.5 operational fallback may exist for engineering compatibility, but must be explicitly marked **unvalidated** and excluded from safety claims.

**Reason:** A threshold that could not be estimated is not medical evidence.

### 9.4 Confidence-interval constraints (optional advanced exploration)

**Suggestion:** Investigate whether the lower 95% confidence limit for sensitivity should be required to meet the predeclared goal rather than only the point estimate, **if the study size makes this feasible**.

**Reason:** 19/20 correctly detected positives is 95% observed sensitivity but provides substantial uncertainty. Such an approach may be impossible at small sample sizes — that is informative and should be reported.

### 9.5 Validity of score distributions

Check all scores are finite and within the expected domain; confirm no NaNs and no accidental logits/softmax confusion. Consider the effects of repeated scores (`0`/`1`), all-identical predictions, and numeric precision.

### 9.6 Missing / uncertain ground-truth labels

**Suggestion:** Audit dataset label conventions and either mask unknown entries per label (with counts) or follow the original dataset's documented interpretation. Do not convert unknown labels to negatives without evidence.

**Reason:** Misclassified unknowns distort sensitivity and specificity and may especially harm rare conditions.

### 9.7 Normal label and prediction compatibility

Do **not** assume `normal = NOT(any of other 11 predicted labels)` represents a medically normal ECG.

**Suggestion:** Analyze current training/ground-truth semantics and consider a presentation-level safety policy (e.g., don't display an unconditional normal reassurance beside a critical alert) as a **separate evaluated rule**.

**Reason:** The model cannot detect abnormalities outside its trained scope. Hard overrides may affect reported performance, so compare pre- and post-processing transparently.

### 9.8 Two related ST-elevation labels

**Suggestion:** Verify whether the “ST elevation (not labeled STEMI)” class is a distinct mutually exclusive dataset annotation or a general feature that may co-occur with STEMI pattern.

**Reason:** Naively forcing exclusivity could erase valid multi-label predictions or hide annotation problems.

---

## 10. Proposed tests / quality gates for the coding agent

The exact test framework should match the project. These **behavioral** checks matter more than file naming:

1. **Model untouched:** Loading the same checkpoint before/after threshold experiments yields byte-identical model parameters or a verified equal checkpoint hash; no training loop, optimizer step, or gradient update occurs.
2. **Reproducible baseline:** New F1 results match old validated notebook behavior (or differences are explained by a documented bug fix).
3. **Multi-label shape:** `(N,12)` scores produce `(N,12)` binary outputs; multiple labels may be simultaneously positive.
4. **Exact threshold semantics:** A score **equal** to threshold is positive (`>=`), for every method, for val and test.
5. **PR curve indexing:** F1/F2 selection uses threshold-aligned precision/recall values; tested with small hand-computable arrays.
6. **Sensitivity-constrained correctness:** For small synthetic data, independently enumerate thresholds and confirm chosen candidate satisfies constraint and is maximally specific among feasible candidates.
7. **Constraint infeasibility:** Both sensitivity and specificity floors cannot be met -> explicit infeasible status, no disguised success.
8. **No positives/no negatives:** Undefined class-specific rates and threshold-estimation limitations are visible; engineering fallback not described as validated.
9. **All-tied scores:** Method remains deterministic and documents inability to separate classes at any usable cutoff.
10. **Missing labels:** Per-label masking counts and behavior match documented annotation rules.
11. **Label-order consistency:** Deliberately permuting label order without metadata validation must fail, not silently yield wrong disease mappings.
12. **Test immutability:** Threshold-fitting APIs reject test split or have a clearly enforced read-only/frozen manifest; test labels cannot flow into threshold search/calibration.
13. **Test reproducibility:** Re-running evaluation with same scores, thresholds, and seeds returns identical point metrics and deterministic CI results when seeded.
14. **Calibration isolation:** If implemented, calibration fits strictly on development-only data and carries versioned parameters to evaluation.
15. **Normal label safety:** Test contradictory normal + critical-abnormal prediction examples and ensure UI semantics are explicit; do not silently alter benchmark outputs without reporting both.
16. **Patient-level leakage audit:** Repeated ECG images/records and patient IDs are checked for partition intersections where metadata permits.
17. **Privacy:** No raw patient-identifiable data, paths, or medical information is unintentionally committed to version control or report artifacts.
18. **Clinical claim discipline:** Generated report states intended use, reference standard, ground-truth support, and whether any clinical objective was actually met.

---

## 11. Expected deliverables (suggested)

The coding agent should propose implementation scope appropriate to the repository, but a good research-ready result would include:

### Research/documentation

- **Repository audit:** Actual current implementation, data splits, scores/checkpoint design, no-retraining plan.
- **Evidence review:** Citations, exact study endpoint, threshold target (if one exists), why it is or isn't applicable.
- **Policy rationale:** Decision log for each of 12 labels, alternative considered, assumptions, clinical-review status, and unresolved uncertainty.
- **Limitations:** Reference standard, missing labels, class imbalance, image modality, overlap between train/val/test, lack of external validation, and no clinical deployment claim.

### Code and data outputs

- Reusable threshold-search methods and input validation.
- Reusable per-label evaluation with confusion matrices and 95% CIs.
- A fully reproducible **validation-only** comparison pipeline, ideally using cached inference probabilities.
- One frozen versioned threshold artifact per candidate strategy (or equivalent manifest design).
- Independent test evaluation from pre-frozen artifacts, without threshold refitting.
- Tests for leakage and corner cases.
- Documentation of how existing CardioSentry inference loads whichever **chosen** threshold policy is approved for a research demo (without silently promoting exploratory thresholds to clinical defaults).

### Suggested result tables and figures

1. Validation positives/negatives per label, threshold, method, sensitivity, specificity, PPV, NPV, F1, F2, infeasibility status.
2. Independent test table per label and method with the **same frozen threshold**, counts, metrics, and 95% CIs.
3. Macro/micro aggregate table with caveats for undefined or rare labels.
4. Sensitivity-specificity trade-off plot per high-priority class and PR curve per class.
5. False negatives and false alerts per 1,000, clearly labeled as **observed at this dataset prevalence**.
6. Image source/subgroup and calibration plots if data support them.
7. A short `Conclusions` section that compares sensitivity gains **against** specificity/PPV losses, instead of assuming F1 is or is not the winner.

---

## 12. Decision questions the agent should answer explicitly

These are questions for the research and engineering report — not reasons to stop work if answers are initially missing. Start with an audit and state reasonable assumptions.

1. **Intended use:** Is CardioSentry a thesis benchmark, clinician-facing screening aid, patient-facing educational UI, or eventual clinical product? Different uses imply different acceptable FP/FN consequences.
2. **Ground truth:** Are the labels ECG report statements, expert ECG findings, machine-generated statements, or clinically adjudicated diagnoses? Do we have reliable measurements for QTc and rate-based labels?
3. **Dataset counts:** Are there enough validation positives for meaningful 90–98% sensitivity constraints in STEMI/third-degree AV block?
4. **Data separation:** Is the test set still truly untouched for this research question? Were duplicates and repeat patient ECGs partitioned together?
5. **Evidence:** Which clinical priority rules are supported by peer-reviewed studies versus design hypotheses?
6. **Operational costs:** Can the downstream user handle the number of extra false alerts associated with a sensitivity-focused cutoff?
7. **Calibration:** Does calibration help interpretability or affect threshold generalization; can it be studied without leakage?
8. **Normal-class safety:** Are the dataset labels sufficiently comprehensive to evaluate “any abnormal ECG,” and how does the UI avoid false reassurance?
9. **Threshold stability:** How much do cutoffs move across patient-level validation folds or bootstrap resamples?
10. **Deployment scope:** Which components should remain experimental and which should affect the actual app's outputs after verification?

---

## 13. Suggested implementation sequence and stop/go criteria

| Phase | Suggestion | Reason / completion criterion |
|---|---|---|
| 1 — Discover | Inspect actual code, splits, checkpoint, data dictionary | No changes until model-output semantics and label order are known. |
| 2 — Reproduce | Recover current validation-tuned F1 cutoff vector | Baseline matches the existing notebook, or discrepancy is explained. |
| 3 — Research | Build source-backed label-specific clinical-priority evidence matrix | Every numeric target is identified as published protocol, clinical recommendation, or hypothesis. |
| 4 — Prototype | Add candidate threshold methods and unit tests | Correctness proven on synthetic arrays, without retraining. |
| 5 — Validate | Compare methods **on validation**, choose primary policy | Explicit feasibility and uncertainty; threshold/policy manifest frozen. |
| 6 — Test | Run one independent final evaluation of predeclared methods | No refitting; report results even where targets fail. |
| 7 — Interpret | Produce report and discuss trade-offs with clinical adviser | Avoid unsupported safety or regulatory claims. |
| 8 — Optional integration | Add versioned threshold config to CardioSentry UI/inference | Only integrate policies clearly approved for research/educational use and accompanied by limitations. |

If the project cannot meet the clinical objective at any threshold, **stop calling the goal achieved** and document that the current model needs further work. Do not automatically retrain within this threshold-only task.

---

## 14. Handoff message for the coding agent

> **Task:** Investigate and improve the **post-training threshold evaluation** of CardioSentry's existing 12-label ConvNeXtV1 image-based ECG model. You are being given an **evidence-informed suggestion document, not a fixed implementation order**. First inspect the repository's `AGENTS.md`, training/evaluation notebooks, class-order metadata, split definitions, and checkpoint structures. Reproduce the current per-label F1-on-validation threshold selection and frozen-test behavior. **Do not retrain the model unless new evidence separately establishes that retraining is necessary.**
>
> Independently research established methods and clinical rationale for alternative threshold policies. Consider fixed 0.5, F1, F2, Youden's J, and class-specific sensitivity constraints; consider calibration, cost-sensitive selection, or dual thresholds only when scientifically and technically justified. Keep the project multi-label. **Do not assume a universal medically mandated 90% or 95% sensitivity for STEMI or other ECG findings.** Document how any proposed numeric objective was selected and whether it is a published target, an adviser-approved target, or an experimental scenario.
>
> Use **validation/development data** for model-choice/threshold-choice decisions and an **untouched test dataset** exclusively for evaluating frozen candidate policies. You may compare all predeclared methods on test, but you must not pick the final method by optimizing test performance. Include per-label sensitivity, specificity, precision/PPV, NPV, F1, false positives/negatives, 95% CIs, aggregate multi-label metrics, normal-label false-reassurance analysis, and support counts; mark impossible or unsupported constraints explicitly.
>
> Before changing the existing training notebook, present a brief repo audit, research findings, and implementation proposal with reasons and uncertainties. Prefer small reversible changes, immutable weights, versioned threshold manifests, unit tests, reproducible results, and a clear explanation of every design choice. If evidence is absent, make a well-labeled exploratory recommendation rather than inventing medical authority.

---

## 15. Summary of the original discussion

- Current CardioSentry practice: **per-class thresholds maximizing validation F1** and then frozen for held-out test evaluation.
- Main concern: F1 alone may trade away clinically important sensitivity for serious findings without making that trade-off explicit.
- Proposed direction: **compare** F1 with more clinically motivated, class-specific strategies; research clinical goals rather than assuming fixed numeric cutoffs.
- Metrics beyond sensitivity: specificity, precision/PPV, NPV, F1/F2, false-positive/false-negative counts, AUROC, AP/PR-AUC, calibration where appropriate, and **confidence intervals**.
- Multi-label implication: one independent decision threshold per label is appropriate as a starting point; investigate label overlap and special normal-class behavior separately.
- Validation versus test: **choose/tune on validation; freeze; evaluate on test.** The test set is not for choosing thresholds or a winning policy.
- Retraining: **not necessary to experiment with threshold postprocessing**. Retain your existing checkpoint and consider retraining only for separate performance/data problems.
- Clinical interpretation: treat this work as a **research operating-point comparison** until supported by appropriate clinical review and prospective/external validation.

---

**End of research and implementation guide.**
