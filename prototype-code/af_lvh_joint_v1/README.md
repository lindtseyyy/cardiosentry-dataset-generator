# AF × LVH joint-head analysis of the frozen four-head reference

Round `af_lvh_subgroup_v1` showed the LVH head deteriorates inside the AF
stratum (AUROC 0.9663 → 0.8974; verdict SUPPORTED). This round zooms into
the **28 test ECGs where AF and LVH co-occur** and asks the head-by-head
question:

**"When AF and LVH coexist, does the model correctly detect AF but
frequently miss LVH?"**

Same frozen everything — the SAME `test_predictions.csv` of `resnet50_hires`,
the SAME frozen val-tuned thresholds (AF 0.81, LVH 0.47), the SAME ground-
truth subgroup. Nothing is retrained, re-thresholded, re-predicted or
re-split; the frozen results folders are read-only for this round.

## The reference-pool design

Sensitivity is computable inside the subgroup (all 28 ECGs are AF- and
LVH-positive). Specificity, precision, AUPRC and AUROC need negatives, and
the subgroup contains none — so for each head, the subgroup is the positive
class and the negatives are the frozen pool of test ECGs that are
true-negative for that head (653 non-AF for the AF head, 683 non-LVH for the
LVH head). Consequences, declared:

- specificity and FP/TN are **identical across the rows of one head** (same
  negative pool, same frozen decisions) — reported once per row, and the
  table says so;
- precision/AUPRC/AUROC with the 28-case positive class are **small-sample
  reads** — reported as asked, but the verdict reads are recall, the joint
  breakdown and the exact tests.

## The verdict rule (stated before the numbers)

| read | test | decides |
| --- | --- | --- |
| R1 within-case asymmetry | McNemar exact on the joint breakdown (discordant pairs "AF correct + LVH missed" vs "AF missed + LVH correct") | the hypothesis itself |
| R2 AF not degraded | AF recall on AF+LVH vs AF-only, Fisher exact one-sided | "correctly detects AF" |
| R3 LVH degraded | LVH recall on AF+LVH vs LVH-only, Fisher exact one-sided | "frequently misses LVH" |

- **SUPPORTED** — R1 significant in the claimed direction, R2 not
  significant, R3 significant.
- **PARTIALLY** — R1 significant but R2 significant or R3 not significant.
- **NOT SEPARATED** — R1 not significant (directional evidence only).
- **CONTRADICTED** — R1 significant in the opposite direction.

**Sample-size limitation, stated regardless of verdict:** the subgroup has
28 ECGs (25 distinct patients per the previous round), so recall CIs are
wide and R2/R3 are low-power. The verdict grades the evidence for the
**association** on this test split; no causal claim is made — "AF causes the
miss" is not decidable from these numbers.

## Statistics

1,200 resamples, seed 42, percentile 95% CIs (the project convention).
Recall CIs resample each group's decisions independently (disjoint ECGs, no
pairing); joint-category CIs resample the 28 cases. McNemar and Fisher are
implemented exactly (small combinatorial sums), so the notebook needs zero
installs.

## The fairness contract

`analyze.py` refuses to print anything unless, from the live files on Drive:

1. thresholding `prob_AF`/`prob_LVH` at the frozen thresholds reproduces
   every saved `pred_` entry (0 mismatches);
2. subgroup sizes re-derived from ground truth match the frozen counts
   (28 / 257 / 227);
3. the pooled AF row equals the frozen transcription (268/15/17/638,
   AUPRC 0.9825) and the pooled LVH row equals it (216/34/39/649,
   AUPRC 0.9248);
4. the live `result.json` agrees (protocol fingerprint + counts).

## Files

| file | role |
| --- | --- |
| `config.py` | thresholds, subgroups, verdict rule (documented) |
| `baseline.py` | the frozen reference rows, self-checked on import |
| `metrics.py` | per-head subgroup rows + joint breakdown |
| `stats.py` | exact McNemar/Fisher + bootstrap CIs |
| `analyze.py` | the pipeline: cross-checks → rows → joint table → tests → verdict |
| `selftest.py` | local smoke test against the repo's own result files |
| `make_notebook.py` | builds the self-contained Colab notebook |
| `AF_LVH_Joint_Analysis.ipynb` | the deliverable — one file, code embedded |

## Drive layout (unchanged)

```
MyDrive/CardioSentry/
  cardiosentry_runs/
    hybrid_v1/resnet50_hires/          read-only: result.json, predictions/
    af_lvh_joint_v1/                   THIS round writes here
```

## Prototype caveats, unchanged

These numbers are the prototype corpus (`prototype_1152`): the subgroup
counts are a property of the frozen labels, and AUROC/AUPRC/CIs transfer;
thresholds and absolute scores do not. Re-check subgroup counts on the full
corpus before quoting the prevalence numbers in a paper.
