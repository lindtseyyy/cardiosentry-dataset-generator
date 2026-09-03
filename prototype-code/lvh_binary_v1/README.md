# LVH binary — negative-transfer test

Round 2 (`cardiosentry_hybrids_v1`) found the best LVH numbers so far with a
**four-head ResNet-50@768** (`resnet50_hires`): PTB-XL LVH AUPRC **0.9315**
(pooled 0.9248). This round asks a different question about the same drop:

**is the shared multi-label objective itself holding LVH back?**

The test: train a ResNet-50 at the same 768×768, on **LVH only**, as a binary
task — keeping the dataset split, preprocessing, augmentation, optimiser,
training protocol and evaluation procedure as close to the four-head run as
possible. If the binary model beats `resnet50_hires` on PTB-XL LVH, that is
evidence of **negative transfer** from the multi-label setup.

---

## The fairness contract

Everything below is byte-identical to the round-2 harness that trained the
four-head reference, and `compare.py` verifies it field by field before it
prints a single delta:

| held constant | value |
| --- | --- |
| data partitions (frozen, patient-grouped) | the same `index.csv`, never recomputed |
| stored corpus | the same `prototype_1152_768` — same bytes, same JPEG generation |
| input view | 768×768, no resampling (the stored size) |
| ImageNet normalisation | `config.NORM_MEAN/STD` |
| augmentation (photometric only) | `config.AUG` — the three label-corrupting flips/crop stay disabled |
| loss | `BCEWithLogitsLoss` + `pos_weight`; LVH weight = 2.5524, the same number |
| optimiser | AdamW, β=(0.9, 0.999), weight decay 0.05, same no-decay rule |
| LR, warmup, cosine schedule | 1e-4, 2 warmup epochs, same `lr_at` |
| effective batch 32 | micro-batch **8** × accum 4 — the same micro-batch the reference used (BatchNorm normalises over the micro-batch, so this is not a free knob) |
| epochs, early stopping, monitor | 20, patience 5, monitor name `macro_auprc` |
| seed | 42 |
| threshold rule | F1-optimal on val, frozen for test |

**The one declared difference — the experimental variable — is the task:**
four sigmoid heads → one. Its forced consequences, all declared:

- `LABELS = ["LVH"]` instead of `["STEMI", "AF", "LVH", "NORMAL"]`
- the shared 2-layer MLP head keeps its exact structure
  (`LayerNorm → Dropout(0.1) → 2048×512 → GELU → Dropout(0.1) → Linear`) and
  only the final width changes: 512×4 → 512×1
- over one head, the monitor `macro_auprc` **is** the LVH head's val AUPRC, so
  the checkpoint-selection rule reduces to the same quantity the four-head run
  would call val LVH AUPRC (the four-head run selected on the *macro* over four
  heads — that asymmetry is inherent to the task change and is stated in every
  report, not hidden)

Fingerprints are two-tier, like round 2's: a **protocol** fingerprint and a
**task** fingerprint (expected to differ — that is the experiment). There is
also a **shared** fingerprint: the protocol with task fields removed. It must
equal the same quantity computed from the frozen reference settings in
`baseline.py`; if it does not, some shared value changed by accident and
`compare.py` refuses.

## Why the same 768 corpus, and not a new one

The four-head reference read `prototype_1152_768` (built once by round 2's
`0_Prepare` notebook from the original 1686×1311 sheets). The binary model
reads **the same folder**: same JPEGs, same decode, same loader order (same
generator seed → same shuffle), same transforms. Anything else would add a
second JPEG generation or a resample difference between the two runs and give
the delta a place to hide.

## The reference is frozen — and cross-checked

`baseline.py` holds `resnet50_hires`'s LVH numbers, transcribed mechanically
from `prototype-results/hybrid_v1/resnet50_hires/result.json` (protocol fp
`36988c5e0ba5d72d`, input fp `6b9b1dd6f951bad0`, micro-batch 8). `compare.py`
reads the **live** reference (result.json + test predictions) whenever the
hybrid_v1 run folder is available on Drive, and refuses if the live numbers
disagree with the frozen transcription. A moving reference cannot anchor a
delta.

Reference LVH numbers on the frozen 938-image test split:

| subset | AUPRC | AUROC | Sens | Spec | Prec | F1 | FPR | TP/FP/FN/TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pooled (255 pos) | 0.9248 | 0.9626 | 0.8471 | 0.9502 | 0.8640 | 0.8554 | 0.0498 | 216/34/39/649 |
| **PTB-XL (234 pos)** | **0.9315** | 0.9496 | 0.8504 | 0.9273 | 0.8767 | 0.8633 | 0.0727 | — |
| Chapman (21 pos) | 0.8947 | 0.9774 | 0.8095 | 0.9671 | 0.7727 | 0.7907 | 0.0329 | — |
| STEMI db (0 pos) | — | — | — | — | — | — | 1/146 FP | — |

## The primary read — and why the pooled number cannot be

**PTB-XL carries 234 of the 255 LVH positives; the STEMI database carries
none** — all 146 of its images are LVH-negative. A model that merely learns to
recognise that database stops calling LVH on those images and its **pooled**
AUPRC rises without it reading a single QRS complex. Round 2 already measured
this: most of the 384→768 pooled gain (+0.0212) evaporated inside PTB-XL
(+0.0045).

The binary model can play the same game — nothing about one head stops it from
keying on grid colour or JPEG texture. So the same discipline applies:

1. **PTB-XL LVH AUPRC** — the number that goes in the headline claim.
2. `sens@.08` — PTB-XL sensitivity at FPR ≤ 0.08, threshold-free, so operating
   points stop masquerading as detectors.
3. **STEMI-db false positives** — the concrete count behind any shortcut.
4. The pooled delta, labelled with its shortcut share.

## Statistics — paired bootstrap, not point estimates

`compare.py` computes 95% CIs for every requested metric delta (AUPRC, AUROC,
sensitivity, specificity, FPR, precision, F1) with **1,200 paired bootstrap
resamples** — the same indices for both models on every draw, from the saved
`test_predictions.csv` files. This is the same methodology the round-1
evaluation report used. Threshold metrics are bootstrapped at each model's
**own val-tuned frozen threshold**, exactly as reported.

## The verdict rule, stated before the numbers arrive

| PTB-XL LVH AUPRC delta (binary − four-head) | verdict |
| --- | --- |
| 95% CI entirely above **+0.005** | **supports negative transfer** — the multi-label objective was hurting LVH |
| point estimate positive, CI includes 0 | **not separated** — directional evidence only |
| point estimate ≤ 0 | **no evidence** — removing STEMI/AF/NORMAL did not help LVH |

Two companion readings keep the verdict honest:

- **FPR delta**: round 1's LVH failure was false-positive dominated. A binary
  model that gains sensitivity *and* FPR has moved its operating point, not
  improved — the AUPRC delta and `sens@.08` answer that; F1 at frozen
  thresholds does not.
- **Shortcut share**: how much of the pooled gain does not reproduce inside
  PTB-XL. High = the binary model got better at telling the databases apart,
  which is a corpus win, not an LVH win.

## Runtime

One model, micro-batch 8. Measured on the identical harness in round 2:

| runtime | vCPU | expected |
| --- | ---: | --- |
| T4 standard | 2 | ~1.5–2 h (decode-bound at ~27 img/s) |
| **L4 high-RAM** | 12 | **~40 min** (resnet50_hires took 36.9 min) |

Same rule as round 2: pick the runtime on units-per-job — L4 is ~1.8× faster
for 1.24× the hourly rate. Do **not** change `MICRO_BATCH` to buy speed: it
would change BatchNorm statistics and break comparability with the reference.

## Files

| file | role |
| --- | --- |
| `config.py` | the shared protocol (byte-identical values) + two-tier fingerprints |
| `baseline.py` | the four-head reference, frozen, cross-checked on load |
| `bootstrap.py` | paired percentile bootstrap over saved predictions |
| `models.py` | FusionHead + SingleStream — round 2's model with 512×1 head |
| `data.py` | the round-2 loader, single 768 view |
| `engine.py` | the round-2 train/eval loops, single head |
| `metrics.py` | the round-2 metrics, single label |
| `manifest.py` | index loading, integrity, binary-label checks |
| `subgroup.py` | source-stratified focus analysis (verbatim from round 2) |
| `artifacts.py` | live-flushed logs, status, history, LVH summary |
| `train.py` | train the binary model |
| `compare.py` | verify comparability, then tabulate binary vs four-head with CIs |
| `preflight.py` | GPU / data / loader / model VRAM check |
| `run_lvh_binary.sh` | the batch (one model), does not clobber the summary on failure |
| `make_notebooks.py` | build the self-contained Colab notebook |

## Prototype caveats, unchanged

These runs use `prototype_1152`: a subsample of train at one augmentation
variant per record. **AUROC, AUPRC, CIs and the comparison transfer** to the
full corpus. **Thresholds, calibration and absolute scores do not** — retune
thresholds on the full validation set before reporting final numbers.

## Requirements

Colab already has torch, torchvision, numpy, pandas, pillow, scikit-learn.
Only `timm` needs installing: `pip install -q timm`.
