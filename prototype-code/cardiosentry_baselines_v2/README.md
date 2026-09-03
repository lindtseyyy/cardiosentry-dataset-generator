# CardioSentry — baselines v2

Nine-architecture baseline comparison on the synthetic paper-ECG corpus, plus the
input-geometry study that decides what shape to feed them.

Same harness contract as [`cardiosentry_runs_v1`](../cardiosentry_runs_v1):
everything except the one variable under test is held constant and hashed, and
`compare.py` refuses to tabulate runs that did not share conditions. Same metrics,
same threshold rule, same per-source confound reporting — so v1, v1_768 and these
runs sit in one table.

---

## The two experiments, one command

| | question | what varies | what is fixed | runs |
|---|---|---|---|---|
| **A. geometry** | what shape should the input be? | input `(H, W)` and crop | ResNet-50, protocol | 6 |
| **B. architecture** | which backbone is best? | the backbone | geometry, protocol | 9 |

A is not a second model comparison — it is *one* model at six input shapes, and
it exists only to decide the shape B runs at. Nine architectures × seven
geometries would be 63 runs and roughly a fortnight of L4 time; it is also the
wrong experiment, because the geometry question is "how should the pixel budget
be spent" and one architecture answers it.

```bash
./run_study.sh /content/prototype_1152 "$RUNS"            # A, pick, then B
./run_study.sh /content/prototype_1152 "$RUNS" crop608    # skip A, go straight to B
```

`run_study.sh` hands A's answer to B automatically via `pick_dim.py`, so the whole
study runs unattended. **The selection is on VALIDATION macro AUPRC**, never test:
the geometry is a hyperparameter, and picking it on test would make B's test
numbers the maximum of six draws rather than a held-out estimate. Ties inside
0.005 val AUPRC break on cost, since B pays for the geometry nine more times.

`run_dims.sh` and `run_all.sh` still exist and run the two halves separately, if
you want to drive them by hand.

### How long

About **16 hours** end to end on an L4 — ≈4 h for A, ≈11.5 h for B, ≈1 h of
one-time pre-resizing — scaled from the measured `v1_768` timings. **That is two
Colab sessions, not one.** Re-run the identical command to resume: finished runs
are skipped, an interrupted run continues from its last completed epoch, and
pre-resized corpora are reused. Passing a fixed geometry skips A and saves ~4 h.

### The notebook

`CardioSentry_Baselines_v2.ipynb` — one file, the whole harness embedded as
base64 in cell 1. Cell 6 is the overnight cell. Regenerate with
`python make_notebook.py` after editing any module.

---

## The nine architectures

| model | timm tag | params | pretraining |
|---|---|---:|---|
| VGG-16 | `vgg16.tv_in1k` | 134 M | in1k supervised |
| ResNet-50 | `resnet50.tv_in1k` | 24 M | in1k supervised |
| DenseNet-121 | `densenet121.ra_in1k` | 7 M | in1k supervised |
| EfficientNet-B3 | `efficientnet_b3.ra2_in1k` | 11 M | in1k supervised |
| EfficientNetV2-S | `tf_efficientnetv2_s.in1k` | 20 M | in1k supervised |
| ConvNeXt-Base | `convnext_base.fb_in1k` | 88 M | in1k supervised |
| ConvNeXt-V2-Base | `convnextv2_base.fcmae_ft_in1k` | 88 M | in1k **FCMAE self-supervised** → in1k |
| ViT-B/16 | `vit_base_patch16_384.augreg_in21k_ft_in1k` | 88 M | **in21k** → in1k |
| Swin Transformer V2-Base | `swinv2_base_window8_256.ms_in1k` | 87 M | in1k supervised |

### Two asymmetries to declare in the writeup

**Pretraining is not uniform.** ViT-B/16 has no usable in1k-only checkpoint at
384 in timm, so it enters with strictly more pretraining data. ConvNeXt-V2's
in1k weights come from a self-supervised FCMAE stage rather than a supervised
one — same corpus, different recipe. Neither is fixable from inside the harness,
so it records `pretrain` per run and `compare.py` prints the column. If ViT wins,
the honest claim is *"ViT-B/16 with in21k pretraining beat these CNNs with in1k
pretraining"*.

**Swin-V2 is padded, not resized.** Window attention needs H and W divisible by
256, which none of the aspect-corrected geometries are. Distorting the geometry
to suit one model would break the comparison being run, so `models.PadToMultiple`
replicate-pads the input to the next legal size instead. Swin sees the waveform
at exactly the same px/mm as the other eight; it just pays ~1.26× the compute for
it at `crop608`. **Its accuracy numbers are comparable, its latency and VRAM are
not** — `compare.py` prints a `px fed` column and a warning. If you would rather
have no padding, swap in `models.ALTERNATES["swinv2_cr_small"]`, which takes any
size natively (but is Small, with timm's weights rather than Microsoft's).

---

## The geometry question

`cardiosentry_runs_v1` ran 384², `v1_768` ran 768², and 768 won on every head.
That reads as "more pixels is better", but both runs threw pixels away in the
same two places, and there is a third problem that is not about pixels at all.

### 1. A square resize squeezes time

The sheet is 1686×1311 (**1.286:1**). At 768×768 the horizontal scale is
768/1686 = 0.456 and the vertical is 768/1311 = 0.586 — **22% of the time axis is
gone before any model sees the sheet**, and the layout is 3×4 leads plus a rhythm
strip, so time is the axis carrying twelve columns of information.

### 2. A quarter of the vertical budget is blank paper

Measured over 150 sheets (`aspect_study.py`, reproduced in `profile.txt`):

| | fraction of sheet height |
|---|---|
| printed header block ends | never past **0.199** |
| first trace ink begins | never before **0.220** |
| last trace ink ends | never past **0.949** |

So `dimensions.CONTENT_CROP = (0.207, 0.964)` sits inside that gap on **every**
sheet in the corpus — tilted ones included, whose rows shift ±37 px at the 2.5°
end of the rotation range. It removes printed text and blank paper, never
waveform. `selftest.py --data <DATA>` re-measures and asserts this rather than
trusting it.

### 3. The printed record ID is a STEMI label

Every sheet carries `ID: ECG######`, and stage 1 assigned those numbers by
iterating sources in order:

```
CHAPMAN   ECG000002 .. ECG003241
PTBXL     ECG003242 .. ECG007796
STEMI     ECG007797 .. ECG009238     <- every STEMI record, and only those
```

**Perfectly separable.** A model that learns to read one digit of ~25 px type gets
the STEMI head for free — and gets `source` for free too, which is exactly the
confound stage 5's leakage probe flagged (0.515 vs 0.372 chance). See
`dim_samples/header_leak.png` for what it looks like at full resolution.

**The fix is redaction, not cropping.** `dimensions.REDACT_BOX = (0.10, 0.02,
0.30, 0.18)` paints a fixed rectangle over the printed block, filled with the
*local* paper colour so it does not become a bright patch on a shaded sheet. It
removes the ID without touching geometry, resolution or aspect, and it is baked
into `resize_dataset.py` so training pays nothing per epoch.

`crop_audit.py` verified the box against all 9,238 clean renders: printed text
and the sheet's 1 px border, nothing else, on 9,230 of them. Eight STEMI sheets
have a sliver of lead I/II/III waveform intruding (worst 454 ink px — the tip of
one deflection).

Cropping the band was the first answer and it was **wrong**. The same audit found
a full-width cut clips waveform on the sheets whose R waves run tallest — LVH and
STEMI, the classes defined by amplitude — and that no bottom cut clears the
rhythm strip at all. `CONTENT_CROP` is corrected to `(0.19, 1.0)` and the `crop*`
geometries are kept only for the ablation.

Redaction also removes the printed Age and Sex, which are genuine clinical
covariates rather than leakage — a real cost, though it buys the stronger claim
that classification came from the waveform alone.

**Two corrupt records** (`ECG009191`, `ECG009005`) carry a saturation artifact
spanning the full sheet height; one was in **test**. `manifest.load()` drops them.

### The candidates

```
dim               HxW  crop  aspect  distort    Mpx  x sq768  px/mm x  px/mm y  px/1mm ST  px/QRS  Nyq Hz
---------------------------------------------------------------------------------------------------------
sq384     384x384        no   1.000    0.778  0.147     0.25    1.345    1.730       1.73     3.4    16.8
sq768     768x768        no   1.000    0.778  0.590     1.00    2.690    3.460       3.46     6.7    33.6
nat672    672x864        no   1.286    1.000  0.581     0.98    3.026    3.027       3.03     7.6    37.8
nat768    768x1024       no   1.333    1.037  0.786     1.33    3.587    3.460       3.46     9.0    44.8
wide576   576x1024       no   1.778    1.382  0.590     1.00    3.587    2.595       2.59     9.0    44.8
crop608   608x1024      yes   1.684    0.991  0.623     1.06    3.587    3.618       3.62     9.0    44.8
crop512   512x864       yes   1.688    0.993  0.442     0.75    3.026    3.047       3.05     7.6    37.8
```

* **distort** — 1.000 preserves the sheet's own aspect. `sq768` sits at 0.78.
* **px/1mm ST** — 1 mm of voltage = 0.1 mV = the ST-elevation decision threshold,
  the finest measurement on the page.
* **Nyq Hz** — the highest temporal frequency the resized image can represent.
  The rendered signal band is 0.05–40 Hz, so **384 (16.8 Hz) and 768 (33.6 Hz)
  are both below it** and the three widest candidates are the first to clear it.
  That is a mechanism for the 384→768 gain, and a prediction that clearing 40 Hz
  gains again.

`crop608` dominates `sq768` on every physical axis at 1.06× the cost: +33% px/mm
on time, +5% on voltage, and it is the cheapest candidate that clears the signal
band. `crop512` is the "can we go smaller" probe — it matches `nat672`'s physical
resolution for 75% of `sq768`'s pixels. It is the default in `dimensions.py`, but
**it is a prediction, not a result** — run notebook A.

`wide576` is the deliberate wrong-direction control: same pixel count as `sq768`,
spent on time at the expense of voltage. If it wins anyway, the time axis matters
more than the ST-step arithmetic says it should.

---

## Files

```
dimensions.py       candidate geometries, the crop, and the physical-units arithmetic
config.py           controlled conditions + the two fingerprints
models.py           the nine backbones, per-model input adaptation, Swin padding
data.py             transforms (header crop, rectangular resize), datasets, loaders
manifest.py         index.csv loading + integrity + geometry-marker checks
metrics.py          multi-label metrics  (byte-identical to v1)
engine.py           train/eval loops, complexity, inference benchmark
artifacts.py        crash-safe run artifacts
train.py            train one architecture at one geometry
compare.py          rank architectures, or rank geometries (--across-dims)
pick_dim.py         pick the winning geometry, on VALIDATION only
preflight.py        GPU / data / tags / real fwd+bwd + VRAM suggestions
selftest.py         asserts the things that would otherwise be wrong SILENTLY
aspect_study.py     render the candidates; contact + zoom sheets; ink profile
crop_audit.py       the authoritative geometry audit - measures the CLEAN
                    renders and maps them through stage 4's parameter range
resize_dataset.py   bake one geometry into a pre-resized corpus copy
prepare_corpus.sh   pre-resize to local disk, cached as a tar on Drive
run_study.sh        A -> pick -> B, unattended
run_dims.sh         experiment A alone
run_all.sh          experiment B alone
make_notebook.py    regenerate the Colab notebook (embeds this whole folder)
```

### Two fingerprints, not one

```python
fingerprint()           # protocol AND geometry  -> must match to rank ARCHITECTURES
protocol_fingerprint()  # protocol alone         -> must match to rank GEOMETRIES
```

Without the split you cannot run the geometry study without disabling the check
that makes the architecture study trustworthy. `compare.py` enforces whichever
rule matches its mode, and when the default mode sees runs that differ *only* in
geometry it says so and points at `--across-dims` rather than ranking a
resolution sweep as an architecture comparison.

### Run naming

Results land in `<out>/<model>@<dim>/`, so both experiments can share one output
folder and `ls` tells you what you have.

---

## Before you train

```bash
python selftest.py --data <DATA>     # ~1 min, no GPU, no weights
python preflight.py --data <DATA>    # ~4 min, downloads the nine checkpoints
```

`selftest.py` is not a smoke test. Every check in it is a bug whose symptom is a
model that trains normally, converges normally, and reports a number meaning
something other than what you think:

* `(H, W)` vs `(W, H)` — torchvision takes `(h, w)`, PIL takes `(w, h)`. Swap them
  and you train on a portrait ECG. Loss goes down. Nothing warns you. This is
  invisible at 768×768, which is why v1_768 could not have caught it.
* **cropping twice** — `resize_dataset.py` bakes the crop in and writes a
  `.cs_geometry.json` marker; `manifest.check_geometry` reads it and turns the
  runtime crop off. Without that the second cut lands in the first trace row.
* the crop window drifting out of the header/trace gap as the corpus changes
* a `micro_batch` that does not divide `EFFECTIVE_BATCH`, so two models train at
  different effective batches and "controlled" is not
* the two fingerprints failing to separate protocol from geometry

## Pre-resizing, and caching it on Drive

The corpus stores 1686×1311 (2.2 Mpx). At `nat768` that is 2.8× more pixels
decoded per image per epoch than used. Measured: **41.3 ms/image on the fly vs
6.2 ms from a pre-resized copy** — a 6.6× decode saving, and the difference
between GPU-starved and GPU-bound on Colab's 2 cores (the v1 logs show ~21 img/s
where the GPU could do 70–100). The **ID redaction is baked in at the same time**,
so it costs nothing per epoch.

Colab wipes `/content` on every reset and the study needs two sessions, so
`prepare_corpus.sh` caches the result on Drive:

```bash
CS_DIM=nat768 ./prepare_corpus.sh \
    /content/prototype_1152 \
    /content/prototype_nat768 \
    "$DRIVE/experiments/baselines_v2/data"
```

* **first session** — builds it (~12 min) and archives it (~2.2 GB)
* **every session after** — restores it in a couple of minutes

A **tar**, not a folder: Drive is mounted over the network, so 6,670 individual
files means 6,670 round trips, which is slower than the resize it replaces. One
sequential stream each way. Not gzipped — JPEG does not compress.

**Training always reads from local disk.** The Drive copy exists to survive a
restart, never to train off. The cached tar carries the `.cs_geometry.json`
marker and its geometry is checked *before* extraction, so a tar built at another
geometry is refused rather than silently used. The cache is named
`<corpus>_<dim>.tar`, so caches for different geometries cannot overwrite each
other.

By hand, without the cache:

```bash
CS_DIM=nat768 python resize_dataset.py --src /content/prototype_1152
```

Fair for the comparison — geometry, interpolation and normalisation are unchanged
and all nine models read the same copy. **Do not resize from an already-resized
folder**; the marker check refuses it, because cropping twice cuts waveform.

## VRAM

`models.REGISTRY[*]["micro_batch"]` is anchored to the measured v1_768 L4 runs and
scaled by pixel count. It is a starting point: `preflight.py` measures the real
peak on your GPU and prints the largest micro-batch that still fits. Raising it is
a pure speedup — accumulation steps fall to compensate and the effective batch of
32 never moves. **On a 16 GB T4, halve every value.**

---

## Reading the results honestly

Three things bound what these numbers mean, and all three should appear next to
the headline:

1. **Prototype subset.** Train is 1 augmentation variant per record instead of 4.
   AUROC/AUPRC and the *ranking* transfer to the full corpus; thresholds,
   calibration and absolute scores do not.
2. **Single-source STEMI.** All 1,442 STEMI records come from one database.
   `REPORT_PER_SOURCE` is mandatory for this reason — a large per-source spread is
   the signature of a model keying on acquisition artifacts.
3. **The printed ID.** On a full-sheet geometry it is in the image and it is a
   perfect STEMI label. The header-crop ablation in notebook A bounds the damage;
   report the bound whatever it turns out to be.

If the `crop*` geometries win *mostly on the STEMI head* while AF, LVH and NORMAL
stay flat, that is the leak being removed rather than the crop being better — and
it means part of the v1/v1_768 STEMI score was the digit. That is a finding, not
a failure.
