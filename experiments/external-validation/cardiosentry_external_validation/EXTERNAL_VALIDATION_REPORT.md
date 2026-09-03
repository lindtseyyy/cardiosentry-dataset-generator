# CardioSentry — External STEMI Validation

**Does the STEMI head detect ST elevation, or does it detect the Chongqing scanner?** The training corpus could not tell the difference. This one can.

| | |
|---|---|
| **Sheets** | 413 · one canonical image per ECG |
| **Gold STEMI** | 8 (LUDB — asserted, unseen source) |
| **Proxy STEMI** | 58 (PTB-XL acute MI — seen source, never labelled STEMI) |
| **Hard negatives** | 59 abnormal sheets carrying no target class |
| **Config fingerprint** | `936b42d988015ade` |
| **Bootstrap** | 2000 resamples, percentile CIs |

---

## Verdict

Read this table row by row. The STEMI column alone means nothing — it is only interpretable against the control column beside it, because a corpus shift depresses every head at once and that is not a confound.

| Model | STEMI AUROC — internal (confounded) | STEMI AUROC — external gold | Drop | Control AUROC (AF/LVH/NORMAL, same sheets) | Reading |
|---|---:|---:|---:|---:|---|
| vgg16 | 0.9592 | 0.9466 | 0.0126 | 0.8089 | **TRANSFERS** |
| resnet50 | 0.9861 | 0.9004 | 0.0857 | 0.7993 | **TRANSFERS** |
| efficientnet_b3 | 0.9375 | 0.8086 | 0.1289 | 0.7433 | **PARTIAL** |
| convnext_base | 0.9815 | 0.9342 | 0.0473 | 0.8079 | **TRANSFERS** |
| vit_b_384 | 0.9558 | 0.8659 | 0.0899 | 0.8326 | **TRANSFERS** |

The *internal* column is the number the prototype reported on its own held-out test set — where every STEMI positive was a Chongqing sheet and every Chongqing sheet was a STEMI positive. It is the figure this corpus was built to re-take, not a target to be matched.

- **vgg16** — STEMI ranks external positives above negatives at least as well as the control heads do; the confound existed in the data but this model did not rely on it.
- **resnet50** — STEMI ranks external positives above negatives at least as well as the control heads do; the confound existed in the data but this model did not rely on it (note: one control head is itself at 0.59, so the control baseline is not uniformly healthy).
- **efficientnet_b3** — STEMI degraded well beyond the control heads without collapsing — consistent with a model using both ST morphology and source cues (note: one control head is itself at 0.57, so the control baseline is not uniformly healthy).
- **convnext_base** — STEMI ranks external positives above negatives at least as well as the control heads do; the confound existed in the data but this model did not rely on it (note: one control head is itself at 0.59, so the control baseline is not uniformly healthy).
- **vit_b_384** — STEMI ranks external positives above negatives at least as well as the control heads do; the confound existed in the data but this model did not rely on it.

**Across all 5 models:** 4 transfers, 1 partial.

---

## 1. The STEMI head, cohort by cohort

Three cohorts, three different questions.

- **LUDB** (8 positives / 200 sheets) — a source seen in no class, with STEMI asserted in words by the reporting cardiologist. The gold standard, and far too small to carry a precise estimate. Always read the interval, never the point.
- **PTB-XL acute MI** (58 positives) — a source seen extensively in training but **never once labelled STEMI**. A model keying on paper cannot fire here; a model reading ST morphology can. The label is a proxy (`mi` + Stadium I, i.e. acute), not an asserted ST-elevation label.
- **PTB-XL normal** (155 sheets) — negative controls from the same familiar source, which is what keeps source from being a clean proxy for the label inside the PTB-XL arm.

| Model | Cohort | n | pos | AUROC (95% CI) | AUPRC (95% CI) | Sens @ frozen thr | Spec | mean p(pos) | mean p(neg) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vgg16 | LUDB | 200 | 8 | 0.9466 [0.905, 0.977] | 0.3097 [0.136, 0.551] | 1.000 | 0.604 | 1.000 | 0.306 |
| vgg16 | PTBXL_MI | 58 | 58 | n/a | n/a | 0.603 | n/a | 0.583 | n/a |
| vgg16 | PTBXL_ALL | 213 | 58 | 0.8385 [0.751, 0.910] | 0.8193 [0.728, 0.891] | 0.603 | 0.968 | 0.583 | 0.011 |
| vgg16 | ALL | 413 | 66 | 0.7695 [0.696, 0.841] | 0.4928 [0.374, 0.620] | 0.652 | 0.767 | 0.633 | 0.174 |
| resnet50 | LUDB | 200 | 8 | 0.9004 [0.828, 0.958] | 0.3137 [0.097, 0.620] | 1.000 | 0.620 | 0.973 | 0.352 |
| resnet50 | PTBXL_MI | 58 | 58 | n/a | n/a | 0.672 | n/a | 0.669 | n/a |
| resnet50 | PTBXL_ALL | 213 | 58 | 0.9532 [0.914, 0.982] | 0.9245 [0.867, 0.966] | 0.672 | 0.987 | 0.669 | 0.013 |
| resnet50 | ALL | 413 | 66 | 0.8395 [0.789, 0.885] | 0.5068 [0.390, 0.631] | 0.712 | 0.784 | 0.706 | 0.201 |
| efficientnet_b3 | LUDB | 200 | 8 | 0.8086 [0.675, 0.922] | 0.1543 [0.055, 0.438] | 0.750 | 0.646 | 0.848 | 0.421 |
| efficientnet_b3 | PTBXL_MI | 58 | 58 | n/a | n/a | 0.603 | n/a | 0.624 | n/a |
| efficientnet_b3 | PTBXL_ALL | 213 | 58 | 0.9137 [0.867, 0.950] | 0.8528 [0.775, 0.914] | 0.603 | 0.974 | 0.624 | 0.038 |
| efficientnet_b3 | ALL | 413 | 66 | 0.7895 [0.729, 0.846] | 0.5051 [0.392, 0.620] | 0.621 | 0.793 | 0.651 | 0.250 |
| convnext_base | LUDB | 200 | 8 | 0.9342 [0.882, 0.973] | 0.2835 [0.136, 0.594] | 1.000 | 0.677 | 0.990 | 0.446 |
| convnext_base | PTBXL_MI | 58 | 58 | n/a | n/a | 0.690 | n/a | 0.777 | n/a |
| convnext_base | PTBXL_ALL | 213 | 58 | 0.9644 [0.934, 0.986] | 0.9373 [0.888, 0.973] | 0.690 | 1.000 | 0.777 | 0.076 |
| convnext_base | ALL | 413 | 66 | 0.8704 [0.825, 0.912] | 0.5798 [0.465, 0.708] | 0.727 | 0.821 | 0.802 | 0.281 |
| vit_b_384 | LUDB | 200 | 8 | 0.8659 [0.747, 0.941] | 0.1661 [0.076, 0.352] | 0.875 | 0.771 | 0.875 | 0.219 |
| vit_b_384 | PTBXL_MI | 58 | 58 | n/a | n/a | 0.586 | n/a | 0.554 | n/a |
| vit_b_384 | PTBXL_ALL | 213 | 58 | 0.9514 [0.918, 0.976] | 0.9029 [0.836, 0.952] | 0.586 | 0.994 | 0.554 | 0.008 |
| vit_b_384 | ALL | 413 | 66 | 0.8523 [0.804, 0.894] | 0.5636 [0.446, 0.681] | 0.621 | 0.870 | 0.593 | 0.125 |

> **`PTBXL_MI` shows `n/a` for AUROC and AUPRC, and that is correct rather than broken.** Every sheet in that cohort is a STEMI positive, so there is nothing to rank it against and both metrics are mathematically undefined — the same degeneracy that made the prototype's internal per-source probe return `nan`. Read **sensitivity** and **mean p(pos)** on that row, and use **`PTBXL_ALL`** (the MI arm plus its 155 same-source normal controls) when you want a ranking metric from PTB-XL.

> **`mean p(pos)` is the column to read when sensitivity is zero.** Sensitivity depends on a threshold frozen from the original validation split; a model can rank external positives correctly and still fire on none of them because its threshold sits above every probability it assigned. Mean probability separates *cannot see it* from *sees it but will not commit*. Those need different fixes — retraining versus recalibration.

## 2. The control heads

AF, LVH and NORMAL had multi-source training — PTB-XL and Chapman both contribute all three — so they carry no source confound. They ride on the same sheets through the same renderer. Whatever they lose here is the price of the corpus, not of a confound, and it is the baseline the STEMI drop has to be measured against.

Scored **on LUDB only** (200 sheets: 15 AF, 108 LVH, 25 NORMAL), so the control heads and the gold STEMI head are measured on the identical sheets — same cardiograph, same population, same render. Anything else would let a difference in cohort masquerade as a difference between heads.

| Model | AF AUROC | LVH AUROC | NORMAL AUROC | Control mean | STEMI (gold) | Gap |
|---|---:|---:|---:|---:|---:|---:|
| vgg16 | 0.9683 | 0.6573 | 0.8010 | 0.8089 | 0.9466 | -0.1377 |
| resnet50 | 0.9849 | 0.5905 | 0.8226 | 0.7993 | 0.9004 | -0.1010 |
| efficientnet_b3 | 0.8719 | 0.5726 | 0.7854 | 0.7433 | 0.8086 | -0.0653 |
| convnext_base | 0.9870 | 0.5950 | 0.8417 | 0.8079 | 0.9342 | -0.1263 |
| vit_b_384 | 0.9582 | 0.7086 | 0.8311 | 0.8326 | 0.8659 | -0.0333 |

For reference, the same heads across the whole corpus (LUDB + PTB-XL):

| Model | AF AUROC | LVH AUROC | NORMAL AUROC | Control mean |
|---|---:|---:|---:|---:|
| vgg16 | 0.9674 | 0.5563 | 0.8215 | 0.7817 |
| resnet50 | 0.9792 | 0.4964 | 0.8395 | 0.7717 |
| efficientnet_b3 | 0.8912 | 0.5002 | 0.8399 | 0.7438 |
| convnext_base | 0.9835 | 0.4545 | 0.8847 | 0.7742 |
| vit_b_384 | 0.9621 | 0.5808 | 0.8236 | 0.7888 |

A large positive gap is the signature the experiment was built to detect: the heads that were trained across sources transferred, and the one head whose positives came from a single source did not.

## 3. Specificity on STEMI mimics

These are the sheets that separate *reads ST morphology* from *reads paper*, and they need reading carefully in the right direction.

- **Early repolarisation syndrome** — benign ST elevation. Genuinely elevated ST segments on a patient who is not infarcting. (9 sheets)
- **Non-specific repolarisation abnormality** — ST-T change without infarction. (49 sheets)
- **Ischemia / scar / suspected NSTEMI** — coronary disease that is specifically *not* ST-elevation MI. (46 sheets)

| Model | early_repol FP rate | repol_abnorm FP rate | ischemia_scar FP rate | hard-neg FP rate |
|---|---:|---:|---:|---:|
| vgg16 | 0.000 | 0.490 | 0.698 | 0.356 |
| resnet50 | 0.222 | 0.429 | 0.628 | 0.390 |
| efficientnet_b3 | 0.111 | 0.286 | 0.628 | 0.339 |
| convnext_base | 0.000 | 0.327 | 0.651 | 0.322 |
| vit_b_384 | 0.000 | 0.224 | 0.372 | 0.271 |

> **A perfect score here is not automatically good news.** A model that has learnt "Chongqing paper ⇒ STEMI" scores 0.000 on every column of this table, because none of these sheets are on Chongqing paper. Zero false positives is only meaningful from a model that also *fired on the true positives* in section 1. Read the two tables together or not at all.

## 3b. What the STEMI head actually fires on

Mean predicted STEMI probability on LUDB sheets, grouped by what the cardiologist's report says. This is the most diagnostic table in the report, because the competing explanations predict *different orderings* rather than merely different magnitudes.

- A model reading **the Chongqing scanner**, or the record ID printed on the sheet, scores every LUDB sheet alike — true positives included — because none of them carry the cue. Flat row.
- A model reading **raw ST elevation** also fires on early repolarisation: that finding *is* ST elevation, in a patient who is not infarcting.
- A model reading **infarct morphology** separates the two: high on STEMI, low on early repolarisation, intermediate on ischemia and scar, which share ST-T changes without the injury pattern.

| Model | STEMI+ | ischemia / scar / NSTEMI | non-specific repol. | other negatives | **early repol (benign STE)** |
|---|---:|---:|---:|---:|---:|
| vgg16 | **1.000** | 0.610 | 0.310 | 0.197 | 0.001 |
| resnet50 | **0.973** | 0.595 | 0.387 | 0.244 | 0.164 |
| efficientnet_b3 | **0.848** | 0.681 | 0.380 | 0.353 | 0.131 |
| convnext_base | **0.990** | 0.721 | 0.479 | 0.343 | 0.086 |
| vit_b_384 | **0.875** | 0.368 | 0.225 | 0.175 | 0.001 |

Group sizes: 8 STEMI positives, 43 ischemia/scar, 49 non-specific repolarisation, 93 other negatives, 9 early repolarisation.

## 4. What this corpus cannot tell you

1. **Eight gold positives.** LUDB contributes 8 asserted STEMI records and that is the entire supply in the database. Every LUDB STEMI interval in this report is correspondingly wide. It is enough to detect a collapse — a head scoring near 0.5 with 8 positives is still clearly not working — and nowhere near enough to certify that a head works.
2. **The PTB-XL arm is a proxy, and it is not a new source.** `dgedon/nstemi_prediction` publishes a curated selection *of PTB-XL*, which is already a training source. Its value is one-directional: PTB-XL contributed zero STEMI labels, so firing here cannot be explained by paper. But the label is `mi` + Stadium I (acute), and acute MI still admits NSTEMI. Do not merge it with the LUDB figure into a single sensitivity.
3. **Rendering is synthetic on both sides.** These sheets and the training sheets came from the same `ecg-image-kit` at the same DPI with the same augmentation. That is deliberate — it isolates *source of signal* from *style of paper* — but it means this measures generalisation across ECG databases, not across real cardiograph printouts or phone photographs.
4. **Prevalence is not clinical.** LVH sits near 54% in LUDB against a far lower rate in the training corpus. AUROC is prevalence-independent; AUPRC, F1 and precision are not, and should not be compared across cohorts.
5. **NORMAL in LUDB is an assertion by absence.** LUDB publishes no "normal ECG" statement, so a record is called NORMAL when its full cardiologist report lists a sinus rhythm, an axis, and nothing else. That is weaker than PTB-XL's explicit `NORM` and yields 25 records.

## 5. Provenance

- Test corpus: `/content/data/external_testset`
- Weights: `/content/drive/MyDrive/CardioSentry/cardiosentry_runs/v1`
- Controlled-config fingerprint: `936b42d988015ade` — matches the training runs, so transforms, normalisation and image size are identical.
- Thresholds: frozen per model on the **original validation split** and applied unchanged. Never re-tuned here; doing so would be tuning on test.
- Per-model checkpoint epochs: vgg16 = 11, resnet50 = 14, efficientnet_b3 = 15, convnext_base = 3, vit_b_384 = 12
