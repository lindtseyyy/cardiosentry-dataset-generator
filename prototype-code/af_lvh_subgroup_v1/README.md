# AF × LVH subgroup analysis of the four-head reference

Round 2 (`cardiosentry_hybrids_v1`) produced the best LVH numbers so far with
the four-head **ResNet-50@768** (`resnet50_hires`): PTB-XL LVH AUPRC 0.9315,
pooled 0.9248 — and LVH remains the worst of the four heads (STEMI AUPRC
0.9979, AF 0.9825, NORMAL 0.9523). This round asks a stratification question
about the SAME frozen model:

**does the LVH head deteriorate specifically on ECGs that carry AF, and is
the AF+LVH subgroup contributing to the pooled LVH deficit?**

It trains nothing, re-thresholds nothing and re-predicts nothing. It reads
the frozen per-image test predictions of `resnet50_hires`, stratifies the
938-image test split by GROUND-TRUTH comorbidity labels, and measures the LVH
head inside each stratum. An optional cell reloads the saved weights and
re-runs the test split, purely to prove the frozen file matches the saved
checkpoint — it cannot change an analysis number.

---

## The four strata (ground truth, never model predictions)

| stratum | definition | user-facing name |
| --- | --- | --- |
| `clean` | AF=0 and STEMI=0 | LVH only |
| `af` | AF=1 | AF + LVH |
| `stemi` | STEMI=1 | STEMI + LVH |
| `af_stemi` | AF=1 and STEMI=1 | AF + STEMI + LVH |

A stratum contains LVH-negatives as well — those are what make specificity,
FPR, precision and AUPRC computable *inside* it.

**Structural zeros (corpus facts):** STEMI is exactly `source == "STEMI"` and
that database carries zero LVH positives, so STEMI+LVH and AF+STEMI+LVH ECGs
do not exist. AF and STEMI never co-occur on any row of the corpus. Those
strata contribute LVH false-positive counts only.

## What the numbers are

Frozen test split (938 images, 255 LVH+), threshold **0.47** (val-tuned by the
reference run, frozen — never re-tuned inside a stratum). Per stratum:
n, LVH+, prevalence, AUPRC, AUROC, sensitivity, specificity, FPR, precision,
F1, and sensitivity at FPR ≤ 0.08 (threshold-free), plus TP/FP/FN/TN.
Companion rows: POOLED, pooled-minus-AF-stratum, and PTB-XL-only clean/AF
(source-stratification check).

## Statistics

1,200 resamples, percentile 95% CIs, seed 42 — the project convention.
Resampling is **independent per group**, not paired: `clean` and `af` are
disjoint image sets, so the two-model paired-bootstrap convention does not
apply. The delta distribution is the per-draw difference of the two
independent resamples.

## The verdict rule (stated before the numbers, in the notebook and report)

- **Sufficiency**: the AF stratum is scored only with ≥ 20 LVH+ and ≥ 50 LVH−.
- **Detector read**: AUROC delta and sensitivity@FPR≤0.08 delta (both
  prevalence-independent) decide the verdict:
  - **SUPPORTED** — both point estimates negative, ≥ 1 CI entirely below 0,
    neither CI entirely above 0
  - **PARTIALLY** — exactly one CI entirely below 0
  - **NOT SEPARATED** — negative point estimates, CIs include 0
  - **NO EVIDENCE** — point estimates ≥ 0
- AUPRC/precision/F1/threshold metrics are reported with CIs but do not
  decide: AUPRC's floor is the stratum prevalence (clean 44.8% vs AF 9.8% in
  test).
- The verdict is an **association on the frozen test split**, never a causal
  claim. The confound ledger (prevalence, negative-class composition, source
  composition, support, patient structure) is printed beside every verdict.

## The fairness contract

Everything below is frozen, and `analyze.py` verifies it before printing a
single number, refusing on any disagreement:

| frozen | value |
| --- | --- |
| the model | `resnet50_hires`, protocol fp `36988c5e0ba5d72d`, input fp `6b9b1dd6f951bad0` |
| the numbers | `baseline.py` — the four-head run's LVH column, transcribed mechanically from `result.json` and re-derived from `test_predictions.csv` |
| the threshold | 0.47 (LVH slot of the saved threshold vector) |
| the splits | the frozen `index.csv`, patient-grouped, never recomputed |
| the predictions | `hybrid_v1/resnet50_hires/predictions/test_predictions.csv` — the exact file the reference numbers were computed from |

The weights path (`model_verify.py`) adds the third cross-check: rebuild the
architecture, load `best.pt`, re-run the test split through the same eval
pipeline, compare per-image probabilities against the frozen file. Any
max|Δ| beyond ~1e-3 is reported as a red flag; the frozen file stays
authoritative for the analysis either way.

## Files

| file | role |
| --- | --- |
| `config.py` | strata, frozen threshold, bootstrap conventions |
| `baseline.py` | the frozen reference, self-checked on import |
| `subgroups.py` | strata, count tables, prevalence, patient structure |
| `metrics.py` | LVH metrics inside one stratum + vectorised ranking metrics |
| `bootstrap.py` | independent two-group bootstrap for the clean-vs-AF delta |
| `model_verify.py` | weights reload + re-inference drift check (optional) |
| `analyze.py` | the pipeline: cross-checks → counts → metrics → CIs → verdict |
| `selftest.py` | local smoke test against the repo's own result files |
| `make_notebook.py` | builds the self-contained Colab notebook |
| `AF_LVH_Subgroup_Analysis.ipynb` | the deliverable — one file, code embedded |

## Runtime

The analysis itself is pure pandas/numpy over 938 rows — it runs on any
Colab CPU runtime in under a minute. Only the optional weights-verification
cell wants a GPU (any card; the reference used an L4, but the re-run is a
938-image forward pass, ~10 s on a T4 once the 768px corpus is extracted).

## Drive layout (unchanged from round 2)

```
MyDrive/CardioSentry/
  cardiosentry_dataset/prototype_1152_768.tar   <- the frozen 768px corpus
  cardiosentry_runs/
    hybrid_v1/resnet50_hires/
      result.json, weights/best.pt, predictions/test_predictions.csv
    af_lvh_subgroup_v1/                          <- THIS round writes here
```

## Prototype caveats, unchanged

These numbers are the prototype corpus (`prototype_1152`): AUROC/AUPRC/CIs
and the comparison transfer to the full corpus; thresholds and absolute
scores do not. The subgroup stratification itself is a property of the
labels, not of the prototype subsampling, but re-check the stratum counts on
the full corpus before quoting the prevalence numbers in a paper.
