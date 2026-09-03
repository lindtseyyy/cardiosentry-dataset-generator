# CardioSentry — controlled architecture comparison

A training harness for comparing **VGG-16, ResNet-50, EfficientNet-B3,
ConvNeXt-Base and ViT-B/384** on the synthetic paper-ECG corpus, built so that
the *only* thing differing between runs is the backbone.

Four independent sigmoid heads: **STEMI, AF, LVH, NORMAL**.

---

## The fairness contract

Everything below is defined once in `config.py` and applied identically to every
architecture:

| held constant | where |
| --- | --- |
| data partitions (frozen, patient-grouped) | read from `index.csv`, never recomputed |
| input resolution 384×384, bicubic | `config.IMG_SIZE` |
| ImageNet normalisation | `config.NORM_MEAN/STD` |
| augmentation (photometric only) | `config.AUG` |
| loss `BCEWithLogitsLoss` + `pos_weight` | `engine.build_criterion` |
| optimiser AdamW, β, weight decay | `engine.build_optimizer` |
| LR, warmup, cosine schedule | `config.LR`, `engine.lr_at` |
| **effective** batch size 32 | grad accumulation, `models.accumulation_steps` |
| epochs, early-stopping rule, monitor | `config.EPOCHS`, `config.MONITOR` |
| seed 42 | `engine.set_seed` |
| threshold tuning on val, frozen for test | `metrics.tune_thresholds` |

`config.fingerprint()` hashes all of it. Every run records its fingerprint, and
**`compare.py` refuses to rank runs whose fingerprints disagree.** You cannot
accidentally compare a model trained under changed conditions — that is the one
mistake this harness exists to prevent.

Micro-batch is the only per-model number, and it is a VRAM accommodation, not an
experimental variable: gradient accumulation restores the same effective batch of
32 everywhere, so the optimisation trajectory is unchanged.

## Three augmentations that would silently corrupt your labels

Standard image recipes include these. On ECG paper, each one changes the
diagnosis while leaving the label untouched — they are **disabled in
`config.AUG` and must stay disabled**:

- **Horizontal flip** — the x axis is *time*. Flipping runs the heartbeat
  backwards and puts the lead columns (I/aVR/V1/V4 …) in the wrong order.
- **Vertical flip** — the y axis is *voltage*, and its sign is diagnostic.
  Flipping turns **ST elevation into ST depression**: a STEMI becomes its
  clinical opposite, still labelled STEMI. This is the most dangerous default in
  the entire pipeline.
- **Random resized crop** — leads sit at fixed sheet positions. A crop that
  removes V1–V3 removes the only place anterior ST elevation is visible.

The corpus is *already* augmented — stage 4 baked in grid colour, tilt,
perspective, creases, lighting, shadow, blur, noise and JPEG artifacts. Adding
more geometric augmentation double-counts it. What remains here is a ±10%
brightness/contrast jitter, which moves no pixel and flips no sign.

## Quick start (Colab)

**You do not paste these modules into notebook cells.** They are files. The
notebook only issues commands that run them.

Run `python make_notebook.py` to build `CardioSentry_Colab.ipynb`, which carries
the entire package inside cell 1 as a base64 tar.gz. Upload **that one file** to
Colab and run the cells top to bottom — 13 code cells, no folder upload:

| cell | what it does | time |
| ---: | --- | --- |
| 1 | unpack the harness to `/content/code`, `pip install timm` | ~30 s |
| 2 | mount Drive, extract `prototype_1152.tar` to local disk | ~2 min |
| 3 | **preflight** — GPU, dataset integrity, all 5 timm tags, peak VRAM | ~2 min |
| 4 | **smoke test** — 20 batches, proves the path runs | ~2 min |
| 5 | **train all five** (`run_all.sh`) | hours |
| 6 | morning check — `_SUMMARY.txt`, any failures | instant |
| 7 | comparison (verifies comparability first) | instant |
| 8–11 | results table, curves, per-model report, source-confound check | instant |

Re-run `make_notebook.py` after editing any module to refresh the embedded copy.

**Alternative:** upload the `prototype-code/` folder to Drive and point the
notebook at it instead — useful if you want to edit modules between runs without
regenerating the notebook.

```python
!cp -r /content/drive/MyDrive/prototype-code /content/code
%cd /content/code
```

## Colab disconnects

Every epoch writes `runs/<model>/last.pt`. Re-running the same command resumes
automatically, so a disconnect costs at most one epoch. The checkpoint stores the
config fingerprint and refuses to resume into changed conditions — it restarts
instead, rather than producing a run that is half one experiment and half another.

To survive a **runtime reset**, point `--out` at Drive:

```bash
python train.py --model vit_b_384 --data /content/prototype_1152 \
                --out /content/drive/MyDrive/cardiosentry_runs
```

Or the whole batch, which is what you want overnight:

```bash
./run_all.sh /content/prototype_1152 /content/drive/MyDrive/cardiosentry_runs
```

## What you get

`runs/<model>/result.json` — full config, resolved pretrained tag, per-epoch
history, val and test metrics, per-source breakdown, timing, environment.

`compare.py` prints:

1. **comparability check** — fingerprints, dataset identity, row counts, smoke flags
2. **headline test table** — macro AUROC/AUPRC/F1, STEMI AUPRC and sensitivity, params, minutes
3. **per-label** AUROC and AUPRC
4. **per-source macro AUROC and its spread**

## Metrics

Every metric below is computed **per head** and **macro-averaged** across the
four heads, on both val and test.

| | |
| --- | --- |
| Sensitivity (recall) | `sensitivity` / `macro_sensitivity` |
| Specificity | `specificity` / `macro_specificity` |
| Precision (PPV) | `precision` / `macro_precision` |
| F1 | `f1` / `macro_f1` |
| AUROC | `auroc` / `macro_auroc` |
| AUPRC | `auprc` / `macro_auprc` |
| False positive rate | `fpr` / `macro_fpr` |
| Accuracy | `accuracy` / `macro_accuracy` |
| Parameters | `params_total`, `params_trainable` |
| Model size (MB) | `size_mb` |
| Inference time | `latency_ms_b1_mean`, `throughput_ips_batched` |

Also included: NPV, FNR, balanced accuracy, micro-averages, and full confusion
counts (TP/FP/FN/TN) per head.

**Three things worth knowing about these numbers.**

*Accuracy is ambiguous for a multi-label model*, so both readings are reported
and neither is called "accuracy" on its own:

- `exact_match_accuracy` — all four heads correct on the same ECG. Strict, and
  the honest number to quote.
- `hamming_accuracy` — fraction of individual label slots correct. Higher, and
  what `macro_accuracy` averages per head.

*Macro vs micro.* Macro is the unweighted mean over the four heads, so STEMI
counts as much as NORMAL — that is what you want when the minority class is the
clinically important one. Micro pools the confusion matrices, so common classes
dominate. Both are reported; quote macro.

*Threshold dependence.* Sensitivity, specificity, precision, F1, FPR and
accuracy all move when the threshold moves. Thresholds are tuned to maximise F1
**on validation only**, then frozen for test. AUROC and AUPRC are
threshold-free, which is why `macro_auprc` is the model-selection monitor.

*Inference timing* is measured on synthetic input after warmup, with CUDA
synchronisation, at batch=1 (single-ECG latency, what a clinic sees) and at the
model's micro-batch (throughput, what a screening backlog sees). Synthetic input
deliberately excludes disk and JPEG-decode time so the number reflects the
architecture rather than the storage medium. AMP is forced identical across
models.

## Artifacts — built for unattended overnight runs

Point `--out` at Drive and everything lands there as it happens. **Nothing is
held in memory until the end**: the log and per-epoch history are flushed as
they are produced, so a disconnect, an OOM at 4am or a runtime reset still
leaves a complete record up to that moment.

```
<out>/
├── _SUMMARY.txt                all models at a glance, rewritten per model
├── _batch_log.txt              run_all.sh timeline
├── comparison.txt / .csv       written by compare.py
└── <model>/
    ├── STATUS.txt              RUNNING | DONE | FAILED  + elapsed
    ├── report.txt              full human-readable metrics  <- read this
    ├── train_log.txt           complete console output, flushed live
    ├── history.csv             one row per epoch, appended live
    ├── metrics.csv             flat per-label + macro
    ├── result.json             machine-readable everything
    ├── config_snapshot.json    controlled settings + fingerprint
    ├── ERROR.txt               traceback, only if it failed
    ├── weights/best.pt         best-on-val weights + tuned thresholds
    ├── weights/last.pt         resume checkpoint
    └── predictions/{val,test}_predictions.csv
```

**In the morning, read `_SUMMARY.txt` first** — one line per model with status
and headline metrics, so you know what finished before opening anything else.

`predictions/*.csv` hold per-image probabilities. They let you redo threshold
analysis, error review or subgroup breakdowns later **without retraining**.

`run_all.sh` deliberately does **not** abort on failure. If ViT OOMs at 3am the
remaining models still run — losing four models because the fifth failed is the
worst possible outcome for an overnight batch. Failures are recorded in
`STATUS.txt` and `ERROR.txt` and summarised at the end.

## Two asymmetries the harness cannot remove — declare both

**1. ViT pretraining.** Four backbones load clean ImageNet-1k weights. ViT-B/16
at 384 has no realistic in1k-only checkpoint in timm; the usable ones are
ImageNet-**21k** pretrained then fine-tuned to 1k. ViT therefore starts with more
pretraining data than the CNNs.

The harness records the resolved tag for every run and `compare.py` prints them
side by side. If ViT wins, the defensible claim is *"ViT-B/384 with in21k
pretraining outperformed these CNNs with in1k pretraining"* — a real finding, but
not the same as *"ViT is the better architecture"*. Check for an in1k tag with:

```bash
python preflight.py --list-tags vit_b_384
```

**2. One learning rate for everyone.** `config.LR = 1e-4` is applied to all five.
That is maximally controlled, but architectures do not share an optimum — ViTs
usually prefer lower, VGG can prefer lower still. A single LR risks measuring
"which architecture likes 1e-4" instead of "which architecture is better".

The fix, when you move past the prototype, is **equal tuning budget**:

```bash
python train.py --model <m> --data <d> --sweep     # same 3 LRs for every model
```

`--sweep` runs `config.LR_GRID = [3e-5, 1e-4, 3e-4]` and selects on val. Use it
for *every* model or *none* — `compare.py` warns if LRs differ without it.

## The source confound — always report it

All 1,442 STEMI records come from a single database, and stage 5 of the image
pipeline measured residual source signal (probe accuracy 0.515 vs 0.372 chance).
A high STEMI score cannot, on its own, distinguish *recognising infarction* from
*recognising the Chongqing scanner*.

`config.REPORT_PER_SOURCE = True` makes per-source metrics unconditional. Both
`train.py` and `compare.py` print them, and `train.py` warns when macro AUROC
varies by more than 0.10 across sources. Put these numbers in your results
chapter next to the headline, whatever they say.

## Prototype caveats

These runs use `prototype_1152` (or `_800` / `_500`): a subsample of train at
**one augmentation variant per record** instead of four.

**Transfers to the full corpus** — AUROC, AUPRC, and the *ranking* of
architectures. That ranking is the deliverable.

**Does not transfer** — decision thresholds, calibration, accuracy at a fixed
0.5 cutoff, and absolute scores. Expect the full corpus to score higher; retune
thresholds on the full validation set before reporting final numbers.

## Files

| file | role |
| --- | --- |
| `config.py` | every controlled setting + `fingerprint()` |
| `artifacts.py` | live-flushed logs, status, history, report writing |
| `manifest.py` | index loading, integrity enforcement, `pos_weight` |
| `data.py` | dataset, transforms, loaders |
| `models.py` | the five backbones behind one factory |
| `metrics.py` | multi-label + per-source metrics, threshold tuning |
| `engine.py` | train/eval loops, LR schedule, optimiser |
| `train.py` | train one architecture |
| `compare.py` | verify comparability, then tabulate |
| `preflight.py` | check GPU / data / models before spending hours |
| `run_all.sh` | all five, then compare (does not abort on failure) |
| `make_notebook.py` | build the self-contained Colab notebook |

## Requirements

Colab already has torch, torchvision, numpy, pandas, pillow, scikit-learn.
In practice only `timm` needs installing:

```bash
pip install -q timm
```
