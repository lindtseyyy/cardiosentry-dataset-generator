# CardioSentry — External STEMI Validation

**Does the STEMI head detect ST elevation, or does it detect the Chongqing scanner?** The training corpus could not tell the difference. This one can.

| | |
|---|---|
| **Sheets** | 413 · one canonical image per ECG |
| **Gold STEMI** | 8 (LUDB — asserted, unseen source) |
| **Proxy STEMI** | 58 (PTB-XL acute MI — seen source, never labelled STEMI) |
| **Hard negatives** | 59 abnormal sheets carrying no target class |
| **Config fingerprint** | `e0fef6cd72607f35` |
| **Bootstrap** | 2000 resamples, percentile CIs |

---

## Verdict

Read this table row by row. The STEMI column alone means nothing — it is only interpretable against the control column beside it, because a corpus shift depresses every head at once and that is not a confound.

| Model | STEMI AUROC — internal (confounded) | STEMI AUROC — external gold | Drop | Control AUROC (AF/LVH/NORMAL, same sheets) | Reading |
|---|---:|---:|---:|---:|---|
| efficientnetv2_s_cbam@nat768 | 0.9884 | 0.8913 | 0.0971 | 0.7644 | **TRANSFERS** |
| resnet50_cbam@nat768 | 0.9913 | 0.9076 | 0.0837 | 0.8176 | **TRANSFERS** |

The *internal* column is the number the prototype reported on its own held-out test set — where every STEMI positive was a Chongqing sheet and every Chongqing sheet was a STEMI positive. It is the figure this corpus was built to re-take, not a target to be matched.

- **efficientnetv2_s_cbam@nat768** — STEMI ranks external positives above negatives at least as well as the control heads do; the confound existed in the data but this model did not rely on it (note: one control head is itself at 0.55, so the control baseline is not uniformly healthy).
- **resnet50_cbam@nat768** — STEMI ranks external positives above negatives at least as well as the control heads do; the confound existed in the data but this model did not rely on it (note: one control head is itself at 0.65, so the control baseline is not uniformly healthy).

**Across all 2 models:** 2 transfers.

---

## 1. The STEMI head, cohort by cohort

Three cohorts, three different questions.

- **LUDB** (8 positives / 200 sheets) — a source seen in no class, with STEMI asserted in words by the reporting cardiologist. The gold standard, and far too small to carry a precise estimate. Always read the interval, never the point.
- **PTB-XL acute MI** (58 positives) — a source seen extensively in training but **never once labelled STEMI**. A model keying on paper cannot fire here; a model reading ST morphology can. The label is a proxy (`mi` + Stadium I, i.e. acute), not an asserted ST-elevation label.
- **PTB-XL normal** (155 sheets) — negative controls from the same familiar source, which is what keeps source from being a clean proxy for the label inside the PTB-XL arm.

| Model | Cohort | n | pos | AUROC (95% CI) | AUPRC (95% CI) | Sens @ frozen thr | Spec | mean p(pos) | mean p(neg) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| efficientnetv2_s_cbam@nat768 | LUDB | 200 | 8 | 0.8913 [0.760, 0.979] | 0.5265 [0.168, 0.844] | 0.875 | 0.776 | 0.814 | 0.257 |
| efficientnetv2_s_cbam@nat768 | PTBXL_MI | 58 | 58 | n/a | n/a | 0.603 | n/a | 0.571 | n/a |
| efficientnetv2_s_cbam@nat768 | PTBXL_ALL | 213 | 58 | 0.8044 [0.707, 0.886] | 0.8067 [0.710, 0.882] | 0.603 | 1.000 | 0.571 | 0.017 |
| efficientnetv2_s_cbam@nat768 | ALL | 413 | 66 | 0.7372 [0.649, 0.821] | 0.5261 [0.403, 0.652] | 0.636 | 0.876 | 0.601 | 0.150 |
| resnet50_cbam@nat768 | LUDB | 200 | 8 | 0.9076 [0.770, 0.992] | 0.5306 [0.212, 0.852] | 1.000 | 0.474 | 0.916 | 0.435 |
| resnet50_cbam@nat768 | PTBXL_MI | 58 | 58 | n/a | n/a | 0.483 | n/a | 0.473 | n/a |
| resnet50_cbam@nat768 | PTBXL_ALL | 213 | 58 | 0.9142 [0.863, 0.958] | 0.8809 [0.810, 0.937] | 0.483 | 1.000 | 0.473 | 0.005 |
| resnet50_cbam@nat768 | ALL | 413 | 66 | 0.7298 [0.662, 0.795] | 0.4716 [0.352, 0.582] | 0.545 | 0.709 | 0.527 | 0.243 |

> **`PTBXL_MI` shows `n/a` for AUROC and AUPRC, and that is correct rather than broken.** Every sheet in that cohort is a STEMI positive, so there is nothing to rank it against and both metrics are mathematically undefined — the same degeneracy that made the prototype's internal per-source probe return `nan`. Read **sensitivity** and **mean p(pos)** on that row, and use **`PTBXL_ALL`** (the MI arm plus its 155 same-source normal controls) when you want a ranking metric from PTB-XL.

> **`mean p(pos)` is the column to read when sensitivity is zero.** Sensitivity depends on a threshold frozen from the original validation split; a model can rank external positives correctly and still fire on none of them because its threshold sits above every probability it assigned. Mean probability separates *cannot see it* from *sees it but will not commit*. Those need different fixes — retraining versus recalibration.

## 2. The control heads

AF, LVH and NORMAL had multi-source training — PTB-XL and Chapman both contribute all three — so they carry no source confound. They ride on the same sheets through the same renderer. Whatever they lose here is the price of the corpus, not of a confound, and it is the baseline the STEMI drop has to be measured against.

Scored **on LUDB only** (200 sheets: 15 AF, 108 LVH, 25 NORMAL), so the control heads and the gold STEMI head are measured on the identical sheets — same cardiograph, same population, same render. Anything else would let a difference in cohort masquerade as a difference between heads.

| Model | AF AUROC | LVH AUROC | NORMAL AUROC | Control mean | STEMI (gold) | Gap |
|---|---:|---:|---:|---:|---:|---:|
| efficientnetv2_s_cbam@nat768 | 0.9852 | 0.5533 | 0.7545 | 0.7644 | 0.8913 | -0.1269 |
| resnet50_cbam@nat768 | 0.9841 | 0.6477 | 0.8208 | 0.8176 | 0.9076 | -0.0900 |

For reference, the same heads across the whole corpus (LUDB + PTB-XL):

| Model | AF AUROC | LVH AUROC | NORMAL AUROC | Control mean |
|---|---:|---:|---:|---:|
| efficientnetv2_s_cbam@nat768 | 0.9857 | 0.4023 | 0.8116 | 0.7332 |
| resnet50_cbam@nat768 | 0.9748 | 0.3793 | 0.8579 | 0.7373 |

A large positive gap is the signature the experiment was built to detect: the heads that were trained across sources transferred, and the one head whose positives came from a single source did not.

## 3. Specificity on STEMI mimics

These are the sheets that separate *reads ST morphology* from *reads paper*, and they need reading carefully in the right direction.

- **Early repolarisation syndrome** — benign ST elevation. Genuinely elevated ST segments on a patient who is not infarcting. (9 sheets)
- **Non-specific repolarisation abnormality** — ST-T change without infarction. (49 sheets)
- **Ischemia / scar / suspected NSTEMI** — coronary disease that is specifically *not* ST-elevation MI. (46 sheets)

| Model | early_repol FP rate | repol_abnorm FP rate | ischemia_scar FP rate | hard-neg FP rate |
|---|---:|---:|---:|---:|
| efficientnetv2_s_cbam@nat768 | 0.000 | 0.143 | 0.535 | 0.220 |
| resnet50_cbam@nat768 | 0.222 | 0.510 | 0.721 | 0.542 |

> **A perfect score here is not automatically good news.** A model that has learnt "Chongqing paper ⇒ STEMI" scores 0.000 on every column of this table, because none of these sheets are on Chongqing paper. Zero false positives is only meaningful from a model that also *fired on the true positives* in section 1. Read the two tables together or not at all.

## 3b. What the STEMI head actually fires on

Mean predicted STEMI probability on LUDB sheets, grouped by what the cardiologist's report says. This is the most diagnostic table in the report, because the competing explanations predict *different orderings* rather than merely different magnitudes.

- A model reading **the Chongqing scanner**, or the record ID printed on the sheet, scores every LUDB sheet alike — true positives included — because none of them carry the cue. Flat row.
- A model reading **raw ST elevation** also fires on early repolarisation: that finding *is* ST elevation, in a patient who is not infarcting.
- A model reading **infarct morphology** separates the two: high on STEMI, low on early repolarisation, intermediate on ischemia and scar, which share ST-T changes without the injury pattern.

| Model | STEMI+ | ischemia / scar / NSTEMI | non-specific repol. | other negatives | **early repol (benign STE)** |
|---|---:|---:|---:|---:|---:|
| efficientnetv2_s_cbam@nat768 | **0.814** | 0.495 | 0.210 | 0.186 | 0.130 |
| resnet50_cbam@nat768 | **0.916** | 0.608 | 0.415 | 0.374 | 0.313 |

Group sizes: 8 STEMI positives, 43 ischemia/scar, 49 non-specific repolarisation, 93 other negatives, 9 early repolarisation.

## 4. What this corpus cannot tell you

1. **Eight gold positives.** LUDB contributes 8 asserted STEMI records and that is the entire supply in the database. Every LUDB STEMI interval in this report is correspondingly wide. It is enough to detect a collapse — a head scoring near 0.5 with 8 positives is still clearly not working — and nowhere near enough to certify that a head works.
2. **The PTB-XL arm is a proxy, and it is not a new source.** `dgedon/nstemi_prediction` publishes a curated selection *of PTB-XL*, which is already a training source. Its value is one-directional: PTB-XL contributed zero STEMI labels, so firing here cannot be explained by paper. But the label is `mi` + Stadium I (acute), and acute MI still admits NSTEMI. Do not merge it with the LUDB figure into a single sensitivity.
3. **Rendering is synthetic on both sides.** These sheets and the training sheets came from the same `ecg-image-kit` at the same DPI with the same augmentation. That is deliberate — it isolates *source of signal* from *style of paper* — but it means this measures generalisation across ECG databases, not across real cardiograph printouts or phone photographs.
4. **Prevalence is not clinical.** LVH sits near 54% in LUDB against a far lower rate in the training corpus. AUROC is prevalence-independent; AUPRC, F1 and precision are not, and should not be compared across cohorts.
5. **NORMAL in LUDB is an assertion by absence.** LUDB publishes no "normal ECG" statement, so a record is called NORMAL when its full cardiologist report lists a sinus rhythm, an axis, and nothing else. That is weaker than PTB-XL's explicit `NORM` and yields 25 records.

## 5. Provenance

- Test corpus: `/home/lindtseyxvii/Codes/cardiosentry/experiments/external-validation/data/external_testset`
- Weights: `/home/lindtseyxvii/Codes/cardiosentry/prototype-results`
- Controlled-config fingerprint: `e0fef6cd72607f35` — matches the training runs, so transforms, normalisation and image size are identical.
- Thresholds: frozen per model on the **original validation split** and applied unchanged. Never re-tuned here; doing so would be tuning on test.
- Per-model checkpoint epochs: efficientnetv2_s_cbam@nat768 = 5, resnet50_cbam@nat768 = 7
