# CardioSentry v1_768 — controlled architecture comparison @ 768×768

This is the **768×768 variant** of `prototype-code/cardiosentry_runs_v1` (384×384).
Its only purpose is the controlled resolution comparison **384 vs 768** for the
same five architectures — **VGG-16, ResNet-50, EfficientNet-B3, ConvNeXt-Base and
ViT-B/16** — under the identical evaluation framework.

Four independent sigmoid heads: **STEMI, AF, LVH, NORMAL**.

---

## What changed vs cardiosentry_runs_v1 — the complete list

The v1 harness was copied, not rewritten. Only the changes below were made, and
each is either the resolution change itself or a direct mechanical consequence of
it. Everything else — dataset splits, preprocessing, augmentation, optimizer,
LR, epochs, loss, seed, threshold tuning, model-selection rule and **every
evaluation metric** — is byte-identical to v1.

| change | file | why |
| --- | --- | --- |
| `IMG_SIZE` 384 → **768** | `config.py` | the experiment |
| micro-batch ÷ 2 from v1 (4/8/8/4/2), tuned for the Colab Pro **L4 22.5 GB** | `models.py` | **VRAM accommodation only.** Activations scale with pixel count (768² = 4×384²), but the L4 has ~1.5× the T4's VRAM and the previous hires runs proved resnet50 at micro 8 at 768 (4.7/22G reserved). Effective batch stays 32 via gradient accumulation, so the optimisation trajectory is unchanged. `preflight.py` verifies real peaks before the run. |
| ViT registry key `vit_b_384` → `vit_b_768` | `models.py`, `run_all.sh`, notebook | the old key embeds the resolution; a 768 run named `vit_b_384` would be misleading |
| ViT position embeddings interpolated 384 → 768 on load | `models.py` | technically unavoidable: `vit_base_patch16_384` ships fixed 384 position embeddings; timm's `resample_abs_pos_embed` (bicubic) resizes them when `img_size=768` is requested. Same pretrained checkpoint as v1. Verified locally (pos_embed 577 → 2305, forward passes at 768). |
| `resize_dataset.py --size` default 768, docstring | `resize_dataset.py` | pre-resize target matches training resolution |
| output dir `CardioSentry/runs/v1_768` on Drive | notebook cell 2, `run_all.sh` | finished runs land in the CardioSentry `runs/` folder, one subfolder per run (see Drive layout below) |

**Byte-identical to v1 (diff them to confirm):** `data.py`, `manifest.py`,
`metrics.py`, `engine.py`, `train.py`, `compare.py`, `artifacts.py`,
`requirements.txt`.

### Why the micro-batches are not a silent change

micro_batch is *the only per-model knob* in the v1 design, and it was already a
VRAM accommodation there: gradient accumulation restores the same effective batch
of 32 everywhere, so the optimisation trajectory is unchanged.

At 768 the per-sample activation memory is ~4× (and ViT's attention sequence
length is 4× too), but this batch targets **Colab Pro's L4 (22.5 GB)** rather
than the 16 GB T4 the v1 values were tuned for. Every v1 micro-batch was
therefore divided by **2** (not 4):

| model | v1 @384 (T4) | v1_768 @768 (L4) | batches/epoch @768 |
| --- | ---: | ---: | ---: |
| vgg16 | 8 | **4** | 1203 |
| resnet50 | 16 | **8** | **602** |
| efficientnet_b3 | 16 | **8** | **602** |
| convnext_base | 8 | **4** | 1203 |
| vit_b_768 | 4 | **2** | 2405 |

The anchor is empirical: the previous `cardiosentry_hybrids_v1` hires runs on
this exact L4 trained resnet50 at 768 with **micro 8 × accum 4** (602
batches/epoch, 4.7/22G reserved) — the value shipped here. The other four keep
v1's relative ratios anchored to resnet50. `preflight.py` measures the real
peak VRAM per model (weights + grads + AdamW state + activations) before
training and will suggest raising any of these further; raising is a pure
speedup and changes no results. If you ever run on a 16 GB T4, halve these
again (2/4/4/2/1). **The effective batch of 32, and therefore the optimisation,
is untouched.**

---

## The fairness contract

Everything below is defined once in `config.py` and applied identically to every
architecture, exactly as in v1:

| held constant | where |
| --- | --- |
| data partitions (frozen, patient-grouped) | read from `index.csv`, never recomputed |
| input resolution **768×768**, bicubic | `config.IMG_SIZE` |
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
**`compare.py` refuses to rank runs whose fingerprints disagree.** Because
`IMG_SIZE` is part of the fingerprint, the 768 fingerprint differs from the 384
fingerprint — which is exactly right: `compare.py` ranks runs *within* a
resolution; the 384-vs-768 comparison is done across the two experiments (see
below).

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

## Metrics — identical to the 384 experiment

The metric code is byte-identical to v1 (`metrics.py`). Every metric below is
computed **per head** and **macro-averaged** across the four heads, on both val
and test.

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
**on validation only** (grid 0.05–0.95, step 0.01), then frozen for test. AUROC
and AUPRC are threshold-free, which is why `macro_auprc` is the model-selection
monitor.

*Inference timing* is measured on synthetic input after warmup, with CUDA
synchronisation, at batch=1 (single-ECG latency, what a clinic sees) and at the
model's micro-batch (throughput, what a screening backlog sees). Synthetic input
deliberately excludes disk and JPEG-decode time so the number reflects the
architecture rather than the storage medium. AMP is forced identical across
models.

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
| 2b | pre-resize images to 768² (optional but recommended) | ~10 min |
| 3 | **preflight** — GPU, dataset integrity, all 5 timm tags, peak VRAM at 768 | ~2 min |
| 4 | **smoke test** — 20 batches, proves the path runs | ~2 min |
| 5 | **train all five** (`run_all.sh`) | hours |
| 6 | morning check — `_SUMMARY.txt`, any failures | instant |
| 7 | comparison (verifies comparability first) | instant |
| 8–11 | results table, curves, per-model report, source-confound check | instant |

Re-run `make_notebook.py` after editing any module to refresh the embedded copy.

**Alternative:** upload the `v1_768/` folder to Drive and point the notebook at
it instead — useful if you want to edit modules between runs without
regenerating the notebook. Code lives in the `CardioSentry/codes/` folder:

```python
!cp -r /content/drive/MyDrive/CardioSentry/codes/v1_768 /content/code
%cd /content/code
```

### Drive layout (per the CardioSentry Drive directory map)

The CardioSentry project folder (`MyDrive/CardioSentry`) has exactly five
top-level folders, and every artifact this run touches goes in one of them:

| folder | this run uses it for |
| --- | --- |
| `CardioSentry/codes/` | the `v1_768` harness / notebook (`codes/v1_768/`) |
| `CardioSentry/datasets/` | raw archive `prototype_1152.tar` (read-only; extracted to local disk) |
| `CardioSentry/experiments/` | — (no experiment-local intermediates needed) |
| `CardioSentry/hf_cache/` | `HF_HOME` for timm/Hugging Face downloads |
| `CardioSentry/runs/` | **all finished results:** `runs/v1_768/<model>/…` |

On Colab these are plain folder names under `/content/drive/MyDrive/CardioSentry`.
If the CardioSentry folder is mounted through Linux GVfs instead, its five
subfolders appear as opaque Drive IDs — `runs` is
`1BkqB_sn-6bN6jQxgf8ue9SBjiZqFphaY`, so `runs/v1_768` is
`<gvfs-mount>/1BkqB_sn-6bN6jQxgf8ue9SBjiZqFphaY/v1_768`. If the mount is
read-only, save locally and upload into `CardioSentry/runs/` afterwards — never
scatter files at the CardioSentry root.

### Pre-resize the dataset (strongly recommended at 768)

```bash
python resize_dataset.py --src /content/prototype_1152 --size 768
```

This makes one `prototype_1152_768/` copy once, shrinking decode work ~4× for
every epoch of every model. All five models must use the same copy — `compare.py`
checks the dataset root.

## Colab disconnects

Every epoch writes `runs/<model>/last.pt`. Re-running the same command resumes
automatically, so a disconnect costs at most one epoch. The checkpoint stores the
config fingerprint and refuses to resume into changed conditions — it restarts
instead, rather than producing a run that is half one experiment and half another.

To survive a **runtime reset**, point `--out` at Drive:

```bash
python train.py --model vit_b_768 --data /content/prototype_1152_768 \
                --out /content/drive/MyDrive/CardioSentry/runs/v1_768
```

Or the whole batch, which is what you want overnight:

```bash
./run_all.sh /content/prototype_1152_768 /content/drive/MyDrive/CardioSentry/runs/v1_768
```

## Memory at 768 — what to expect and what to do

Per-sample activation memory scales ~4× from 384 to 768. The shipped
micro-batches (4/8/8/4/2) are the v1 values ÷ 2, tuned for the **Colab Pro L4
(22.5 GB)** this batch runs on — anchored to the previous hires runs, which
trained resnet50 at 768 with micro 8 on this exact GPU (4.7/22G reserved).
**Cell 3 (preflight) is mandatory**: it runs a real forward+backward with the
AdamW optimizer state for each model and reports peak VRAM, so an OOM shows up
in 2 minutes instead of at 3am. It will also suggest raising any micro-batch
that has headroom — apply those suggestions; it is a pure speedup.

- If a model still OOMs: lower only its `micro_batch` in `models.REGISTRY` to the
  next divisor of 32 (e.g. 8 → 4). Record the change; the effective batch and
  the comparison stay intact.
- If you switch to a 16 GB T4: halve all shipped values (2/4/4/2/1).

## Comparing 384 vs 768

Within each experiment `compare.py` refuses to mix fingerprints (and thus
resolutions) — that is the harness working as designed. For the cross-resolution
comparison:

1. `python compare.py --runs <v1 runs> --csv v1_comparison.csv` (or reuse the
   existing `comparison.csv` from the 384 run)
2. `python compare.py --runs <v1_768 runs> --csv v1_768_comparison.csv`
3. Join the two CSVs on `model` (the ViT appears as `vit_b_384` in v1 and
   `vit_b_768` here) and add a `resolution` column. Every column name and
   definition is identical, so rows are directly comparable.

Both CSVs carry the full metric set, `fingerprint`, `lr`, `epochs`,
`latency_ms_b1`, etc., so the join also exposes exactly what differed.

## What you get

`runs/<model>/result.json` — full config, resolved pretrained tag, per-epoch
history, val and test metrics, per-source breakdown, timing, environment.

`compare.py` prints:

1. **comparability check** — fingerprints, dataset identity, row counts, smoke flags
2. **headline test table** — macro AUROC/AUPRC/F1, STEMI AUPRC and sensitivity, params, minutes
3. **per-label** AUROC and AUPRC
4. **per-source macro AUROC and its spread**

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
has no realistic in1k-only checkpoint in timm; the usable ones are
ImageNet-**21k** pretrained then fine-tuned to 1k. ViT therefore starts with more
pretraining data than the CNNs.

The harness records the resolved tag for every run and `compare.py` prints them
side by side. If ViT wins, the defensible claim is *"ViT-B/16 with in21k
pretraining outperformed these CNNs with in1k pretraining"* — a real finding, but
not the same as *"ViT is the better architecture"*. Check for an in1k tag with:

```bash
python preflight.py --list-tags vit_b_768
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
| `config.py` | every controlled setting + `fingerprint()` (IMG_SIZE = 768) |
| `artifacts.py` | live-flushed logs, status, history, report writing |
| `manifest.py` | index loading, integrity enforcement, `pos_weight` |
| `data.py` | dataset, transforms, loaders |
| `models.py` | the five backbones behind one factory (768 micro-batches, ViT pos-embed interpolation) |
| `metrics.py` | multi-label + per-source metrics, threshold tuning (identical to v1) |
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
