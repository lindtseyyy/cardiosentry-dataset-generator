# CardioSentry hybrids — round 2

Round 1 compared five backbones and produced one unanimous finding: **LVH is
substantially harder than the other three conditions.** 562 errors against 255
positives — 2.20 per positive case, versus 1.11 for AF — lowest F1 for all five
models, highest false-positive rate for all five.

This round tests why, and whether architecture fixes it.

---

## The hypothesis, and the arithmetic behind it

Sheets render at 150 DPI = **5.91 px/mm**, at 10 mm/mV and 25 mm/s. Round 1
squashed the 1686×1311 sheet to 384×384:

| view | px/mm vertical | px per mV | px/mm horizontal |
|---:|---:|---:|---:|
| 384 (round 1) | 1.73 | 17.3 | 1.35 |
| **768 (round 2)** | **3.46** | **34.6** | **2.71** |

Sokolow-Lyon asks whether S<sub>V1</sub> + R<sub>V5</sub> exceeds 35 mm; Cornell
whether R<sub>aVL</sub> + S<sub>V3</sub> exceeds 20/28 mm. Both are decided by
differences of a few millimetres. At 1.73 px/mm a 3 mm difference is 5 pixels,
and it has been through JPEG twice.

**But resolution is not the only candidate, and the round is built to tell them
apart.** Two things argue against a pure resolution story:

- The failure is **false-positive dominated** (333 FP vs 229 FN across five
  models). Models over-call LVH on NORMAL and AF tracings. Losing measurement
  precision would predict a mixed error profile, not FP-dominance.
- The labels are **diagnostic-level LVH**, not voltage-criteria LVH. PTB-XL's
  `VCLVH` (a FORM statement, voltage measurement only) was deliberately excluded
  in the label audit, as was Chapman's undefined `55827005`. Diagnostic LVH also
  carries strain pattern — lateral ST depression and T inversion — and left axis
  deviation, which are *shape* features that 384px already resolves fine.

So the round runs a resolution arm and a receptive-field arm, and a control that
separates them.

---

## The nine runs

Ordered as `run_hybrids.sh` runs them. **The order is the point:** the control
pair comes first, because if it settles the question you do not need the rest,
and each architecture is followed immediately by its own 768 twin.

| run | views | params | what it isolates |
|---|---|---:|---|
| `resnet50_base` | 384 | 24.6 M | the in-family baseline |
| `resnet50_hires` | **768** | 24.6 M | **THE CONTROL — resolution alone** |
| `resnet50_cbam` | 384 | 26.9 M | attention reweighting, no extra pixels |
| `resnet50_cbam_hires` | **768** | 26.9 M | does attention help once detail survives? |
| `resnet50_fpn` | 384 | 27.4 M | multi-scale fusion, no extra pixels |
| `resnet50_fpn_hires` | **768** | 27.4 M | does multi-scale need the extra pixels? |
| `resnet50_dual` | 384+768 | 49.1 M | the headline hybrid |
| `resnet50_vit` | 768+768 | 111.0 M | local CNN detail + global attention, both at 768 |
| `resnet50_effb3_dual` | 384+768 | 36.0 M | heterogeneous dual-res (lowest priority) |

### Why CBAM and FPN are run twice

Round A ran both at 384 only, and both lost to a plain ResNet-50 at 768 — which
compared an *architecture* change against a *resolution* change and concluded the
architectures were worthless. That conclusion is not available from that
experiment.

At 384 the sheet is **1.73 px/mm** vertically, so a 0.5 mV LVH voltage difference
spans under 9 pixels. Attention cannot reweight detail the resampling already
destroyed, and an FPN's stride-4 level is only finer than its stride-8 level if
the pixels underneath it differ. The `_hires` twins hold the architecture fixed
and move only the resample, so `compare.py`'s **384 vs 768** section can separate
the two effects.

### Why the control exists

`resnet50_base` and `resnet50_hires` are the same model, the same head, the same
protocol, the same source files. **Only the input resolution differs.**

Without it, a dual-resolution win is uninterpretable: you cannot tell whether it
was the extra pixels or the two-stream fusion. With it, and with the stream
ablation below, you can.

And if `resnet50_hires` alone closes the LVH gap, the finding is *"preprocess at
higher resolution"* — cheaper to deploy, easier to defend, and a better result
than any hybrid.

### Why `resnet50_vit` runs at 768+768, and what question it answers

Round 1's ViT-B/384 placed **4th overall** and had the **worst STEMI AUPRC**
(0.8659) — but the **best LVH AUPRC of all five backbones (0.8822)**, edging out
ResNet-50's 0.8805. Sokolow-Lyon compares S in V1 against R in V5; on a 3×4
layout those leads sit in different columns, hundreds of pixels apart. Every
ViT block attends across that distance directly; a ResNet only bridges it at
stage 5, by which point the amplitudes have been pooled away.

The AF × LVH subgroup analyses (`af_lvh_subgroup_v1`, `af_lvh_joint_v1`) then
scored the frozen 768 reference and returned a **SUPPORTED** verdict: at 768,
`resnet50_hires` detects AF correctly but disproportionately misses LVH when
the two coexist — LVH recall **0.6429** on the 28 AF+LVH cases vs **0.8722** on
LVH-only (Fisher one-sided p = 0.0038), while AF recall held (0.9286 vs 0.9416,
p = 0.5142).

So this run fuses the two findings: **both branches at 768** — the exact view
the round's best LVH model saw — so any gain over `resnet50_hires` is
attributable to the ViT branch's attention, not to extra pixels. The question:
does a 768 representation enriched with global self-attention preserve LVH
information when AF is present? Every run in the round is scored for this
automatically (`af_lvh.py`: LVH recall inside AF+LVH, the LVH-only comparand,
Fisher test + bootstrap CI, reference-pool AUPRCs for both heads), with the
frozen numbers in `baseline.AF_LVH_FROZEN` as the bar.

Caveats that travel with the result: ViT-B/16-384 has no in1k-only checkpoint
in timm (in21k pretraining asymmetry), its pos-embed is interpolated from the
24×24 grid at first forward, and the 28-case subgroup is small — the Fisher p
and CI decide how much of any recall delta is real.

### Why `resnet50_effb3_dual` runs last

In round 1 EfficientNet-B3 finished last on all nine headline metrics. But its
**epoch-1 validation AUPRC was 0.4467 against 0.77–0.83 for every other
backbone** — the signature of a learning rate mismatched to the architecture,
not a weak architecture. No LR sweep was run (`sweep = false` in every round-1
`result.json`).

Fusing a possibly-mis-optimised branch under the shared 1e-4 risks re-measuring
that confound. **Treat a poor result here as uninterpretable** until an
equal-budget sweep has been run. A good result is still informative.

---

## What is held constant, and what is not

Round 1 hashed everything, including resolution, into one fingerprint and refused
to compare runs that disagreed. Resolution is now the experimental variable, so
the fingerprint is two-tier:

| tier | contents | enforced? |
|---|---|---|
| **protocol** | seed, labels, normalisation, augmentation, optimiser, effective batch 32, 20 epochs, cosine + warmup, macro-AUPRC monitor, early stopping, val-tuned frozen thresholds | **yes** — `compare.py` refuses to rank mismatches |
| **input** | which resolutions the model sees | no — printed as a column |

Every protocol value is **byte-identical to round 1**, deliberately: same seed 42,
same AdamW at 1e-4, same weight decay 0.05, same 20 epochs, same monitor. Round-1
numbers therefore remain a legitimate reference point, and `compare.py` prints
them beneath the round-2 table.

Micro-batch is the only per-model knob and it is a VRAM accommodation, not an
experimental variable: gradient accumulation restores the effective batch of 32
everywhere.

**One thing did change and it matters:** every model in this round uses the same
2-layer MLP head, where round 1 used a single Linear. That is why `resnet50_base`
must be re-trained rather than copied from round 1 — it, not round 1's
`resnet50`, is the control.

## The three augmentations that would silently corrupt your labels

Unchanged from round 1, and still disabled in `config.AUG`:

- **Horizontal flip** — the x axis is *time*. Reverses the heartbeat and reorders
  the lead columns (I/aVR/V1/V4 …).
- **Vertical flip** — the y axis is *voltage*, and its sign is diagnostic. Turns
  **ST elevation into ST depression**: a STEMI becomes its clinical opposite,
  still labelled STEMI.
- **Random resized crop** — leads sit at fixed sheet positions. A crop removing
  V1–V3 removes the only place anterior ST elevation is visible.

**There is a new temptation this round specifically.** Now that a 768 view
exists, cropping to "just the precordial leads" looks attractive for LVH. Do not.
Stage 4 baked in tilt and perspective, so a fixed fractional crop does not land
on the same leads in every image, and the `.json` lead boxes are not shipped with
the prototype subset. A crop that silently amputates V5 on tilted sheets would
make LVH worse while looking like an improvement on the sheets it happened to get
right.

---

## Runtime, and the micro-batch confound that comes with it

**Pick the runtime on units-per-job, not units-per-hour.** Measured on
`resnet50_base` epoch 2, identical code and micro-batch:

| runtime | vCPU | steady rate | whole round | units/hr | **total units** |
|---|---:|---:|---:|---:|---:|
| T4 standard | 2 | 25 img/s | ~11 h | 1.0 | ~11 |
| T4 high-RAM | 8 | 84.5 img/s | ~8.0 h | 1.24 | ~10.0 |
| **L4** | 12 | **149.7 img/s** | **~4.5 h** | 1.54 | **~7.0** |

**L4 is 1.77x faster for 1.24x the hourly rate, so it costs ~30% FEWER units
and half the wall clock.** The higher hourly figure makes it look expensive; it
is not, because you rent by the hour and need fewer of them. Break-even would
need L4 to be under 1.24x faster, and it is well past that.

The 22 GB matters too: it protects the uniform micro-batch of 8. On a 14.6 GB T4
the heaviest models are tight at micro 8, and an OOM there forces *every* run
down to 4 to keep BatchNorm comparable — including the ones already finished.

### High-RAM: only for the vCPUs

**RAM is not a constraint here.** Peak usage is about 5 GB — 8 workers holding
two prefetched batches each is ~1.1 GB, and the rest is the model, CUDA context
and pinned buffers. Any runtime with 8 cores has far more memory than that.

Turn High-RAM on **only if it raises the vCPU count** on your runtime, because
vCPUs are what decode scales with. On some Colab configurations the High-RAM
toggle is the only way to get more cores; on others it grants memory alone, in
which case it burns compute units for nothing. Cell 1 of every notebook prints
`vCPU / RAM / free disk` and the projected decode ceiling, so check it in both
modes and keep whichever reports more vCPU.

`resolve_workers()` scales to `min(cpu_count, 8)` — the cap is 8, not 4, so a
high-CPU runtime is actually used. Override with `--workers` if you want more.

### Two optimisations, and why neither is a confound

**NHWC (`channels_last`)** is on for every convolutional model. It is a memory
*layout* change, not a maths change — the same convolutions on the same numbers,
arranged the way Ada/Ampere tensor cores read them. Worth 10-30% on
ResNet/EfficientNet under AMP. Measured difference against NCHW: **1.79e-07**,
pure float accumulation order, orders of magnitude below the bootstrap CIs. It
is deliberately kept **out of the protocol fingerprint** and recorded per run in
`result.json`. Disabled for `resnet50_vit`, where a LayerNorm-and-attention
model gains nothing and can pick up transposes around the patch embedding.

**Worker count** scales to `min(cpu_count - 2, 12)`, leaving two cores for the
main process, which is not idle: it runs the loop, copies into pinned memory and
drives the GPU. On a 12-vCPU L4 that is 10 workers, with data-wait at 1-2%.

### Micro-batch is NOT just a VRAM knob

Round 1's harness claimed micro-batch was "a VRAM accommodation, not an
experimental variable", because gradient accumulation restores the effective
batch of 32. **That is true of the gradient and false of BatchNorm.** BN
normalises over whatever tensor it is handed — the micro-batch — and
accumulation never touches it.

Every backbone this round contains BN. The T4-sized defaults are *not* uniform:

```
resnet50_base 16   resnet50_cbam 16   resnet50_fpn 8
resnet50_hires 4   resnet50_dual 4    resnet50_vit 4   resnet50_effb3_dual 4
```

So on a T4, `resnet50_base` (16) vs `resnet50_hires` (4) — **the control pair** —
differ in BN statistics as well as resolution, and "only the resolution differs"
is not true of them.

On a 24 GB card, set one value for every run:

```bash
MICRO_BATCH=8 ./run_hybrids.sh <data> <out>
```

`compare.py` warns whenever compared runs disagree on micro-batch, and calls out
the control pair by name when it is affected. A slower clean comparison beats a
fast confounded one; if you must run confounded, declare it.

## Watching the other three conditions

**This round targets LVH, but it is scored on all four heads, everywhere.**
`comparison.txt` carries eight per-label tables (AUROC, AUPRC, sensitivity,
specificity, precision, F1, FPR, accuracy) across STEMI/AF/LVH/NORMAL plus
macro; `comparison.csv` has 56 per-condition columns; every `report.txt` prints
the full four-head table for validation and test.

That is not enough on its own, because **macro averages dilute**. A model that
gains 0.04 on LVH and loses 0.04 on STEMI has an unchanged macro AUPRC and looks
like a clean win in every headline table. It is not one — an over-called LVH
earns a follow-up echo, a missed STEMI does not. So the round adds an explicit
per-condition regression check in four places:

| where | what it shows |
|---|---|
| `_SUMMARY.txt` | per-head test AUPRC for all four, plus a `regressed` column — the morning glance already answers "did anything break" |
| `<model>/report.txt` | per-head AUPRC / sensitivity / FPR deltas vs round 1, with false-negative counts, and explicit FLAGS |
| `history.csv` + report | **every head's AUPRC, sensitivity and FPR every epoch**, so a head degrading mid-run is visible while it is still running |
| `compare.py` | three delta tables (AUPRC, sensitivity, FPR) × four heads × macro, then a FLAGS block |

### The flag thresholds, stated so they can be argued with

- **any head losing more than 0.02 AUPRC** against round 1's ResNet-50 — roughly
  the run-to-run noise floor on a 938-image test split.
- **STEMI sensitivity losing more than 0.01** — a tighter bar on purpose. This is
  the failure that harms a patient, so it gets the tighter one.
- **the focus head not improving by at least 0.005** — a run that did not do what
  the round asked, flagged so it cannot quietly pad the table.

Round 1's ResNet-50 missed **16 of 146 STEMIs** and over-called LVH **54 times**.
Those two numbers are what "better" has to beat, and they are printed next to
every flag block.

FPR deltas are **sign-flipped** in `compare.py` so that positive always means
"better" in all three tables — otherwise the FPR column reads backwards against
the two beside it, which is exactly how a regression gets missed at 2am.

### The reference numbers are frozen

`baseline.py` holds round 1's full per-condition results, transcribed once from
`prototype-results/cardiosentry_runs_v1/comparison.txt` (fingerprint
`936b42d988015ade`, same frozen 938-image test split, same validation-tuned
thresholds). Nothing in this round can recompute or re-threshold them — if it
could, a regression could be made to disappear by changing how the reference was
derived. The confusion counts are cross-checked on load: `tp + fn` equals the
known positives per head and `tp + fp + fn + tn` equals 938.

Round 1 used a single Linear head and this round uses a 2-layer MLP, so a
round-1 number is a **reference point**, not a control. `resnet50_base` is the
control, and both appear in the tables so you can see which one a change is
attributable to.

## Two measurements this round adds

### Stream ablation

Every model produces one or more **named feature groups**, each LayerNorm'd
*independently* before concatenation. That per-group normalisation is not
decoration — it is what makes ablation valid. Zeroing one group after its own
LayerNorm leaves every other group's statistics untouched; normalising after the
concat would have made the measurement meaningless.

After training, `evaluate_streams` re-scores test once per group with that group
zeroed, using the **same val-frozen thresholds** (an ablated model is a degraded
version of this model, not a new model entitled to its own operating point).

> **A dual-resolution model whose score barely moves when the 768 group is zeroed
> did not use the extra pixels — it ensembled two backbones.**

The ablation table is broken out **per head, not just LVH** — a stream can carry
LVH while simultaneously being what was costing STEMI, and a table showing only
LVH would present that as an unqualified win.

For `resnet50_fpn` the groups are pyramid levels, so the same table says which
*scale* the diagnosis was read at. P2 is stride 4 (≈2.9 mm per cell at 384); P5
is stride 32 (≈23 mm), wider than the whole QRS complex.

### Learned CBAM gate strengths

Dropping a stock CBAM into a pretrained ResNet multiplies every feature map by a
freshly-initialised sigmoid, whose expected value at init is 0.5 — halving the
activations of a network whose downstream weights were tuned for the unhalved
ones. On 4,810 training images that cost is not recoverable.

This version wraps it as `x + gamma * (cbam(x) - x)` with `gamma` initialised to
**zero**, so the module is *exactly* the identity at step 0 (verified: max
absolute difference 0.000e+00 against a plain ResNet-50 with the same weights).
The learned gammas are written to `report.txt`.

**Gammas that stay near zero are a real finding**, not a bug: attention was
offered and declined, which is evidence that reweighting existing features was
not the bottleneck.

---

## Drive layout

Everything derives from one root. `DRIVE_ROOT` and `RUN_NAME` are defined in
**cell 2 of every notebook** — edit those two lines, nothing else.

```
MyDrive/CardioSentry/
├── cardiosentry_code/            the harness, for reference only
├── cardiosentry_dataset/
│   ├── prototype_1152.tar        ORIGINAL 1686x1311 - never delete, cannot be rebuilt
│   └── prototype_1152_768.tar    built once by notebook 0
├── cardiosentry_runs/
│   ├── v1/                       round 1, five backbones
│   └── hybrid_v1/                THIS round
└── hf_cache/                     pretrained weights, downloaded once
```

Run folders are named by round alone — the parent is already
`cardiosentry_runs`, so a `cardiosentry_` prefix on the child says nothing. To
start a fresh round, change `RUN_NAME` and the notebooks write somewhere new
without touching what is there.

`cardiosentry_code/` is **not read by the notebooks** — each one carries the
whole harness embedded in cell 1, which is what makes them self-contained. Keep
a copy there for reference if you like.

The dataset only ever lives on Drive as a tar. Training always reads
`/content/prototype_1152_768` on local disk: reading images off the mounted
Drive turns every load into a network round trip and starves the GPU. `/content`
is wiped on each runtime reset, which is why notebooks A/B/C re-extract (about a
minute) at the start of every session.

## Quick start (Colab)

You do not paste modules into notebook cells. `python make_notebooks.py` builds
four self-contained notebooks, each carrying the whole harness as a base64
tar.gz in cell 1.

| notebook | what it does | time |
|---|---|---|
| `0_Prepare` | build the 768px dataset once, tar it to Drive, preflight, smoke test | ~15 min |
| `A_Controls` | `resnet50_base`, `resnet50_hires`, `cbam`, `fpn` | **~5 h** (~3.5 h on 12 vCPU) |
| `B_Hybrids` | `resnet50_dual`, `cbam_hires`, `fpn_hires`, `effb3_dual` | **~4.5 h** (~3.3 h on 12 vCPU) |
| `C_ViT` | `resnet50_vit` (768+768), then the full comparison and analysis | ~2-3 h (L4) |

The higher figure assumes a 2-vCPU VM, where decode is the bottleneck. On a
12-vCPU high-RAM VM the loader keeps up (data-wait under 2%) and the GPU becomes
the limit.

Notebooks A and B may outlast a free-tier session. Run them **one or two models
at a time** with their per-model cells; every run resumes from its last completed
epoch, a finished model is skipped, and `run_hybrids.sh` does not abort when one
model fails.

B runs in the order `dual` → `cbam_hires` → `fpn_hires` → `effb3_dual`. The
dual is first because it is the round's headline question and because
`effb3_dual` is uninterpretable without it; `effb3_dual` is last because it is
the lowest-priority run, so a dead session costs the least.

**Run `0_Prepare` first.** After that every notebook is independent — run them in
any order, across nights, or in parallel accounts.

### One run folder, not one per notebook

Every notebook writes to `cardiosentry_runs/hybrid_v1/<model>/`. **Do not give
each notebook its own folder.** `compare.py` reads a single directory and ranks
whatever it finds there, so a shared folder is the only thing that lets a table
put an A model next to a B one — and "does any architecture beat plain
ResNet-50 at 768?" spans both. Organisation comes from the
per-model subdirectory names, which are unique and sort sensibly. A model already
finished is skipped, so re-running a notebook after a disconnect costs a minute.

**Run A before committing to B and C.** It contains the control.

### Do not try to make this faster with PARALLEL or MICRO_BATCH

Both look like free speed on a 22 GB L4. Neither is.

**`PARALLEL=2`** (two models training at once) was tried and **measured slower**:

| model | PARALLEL=2 | alone |
|---|---:|---:|
| `resnet50_dual` | 19.5 img/s | ~34 img/s |
| `resnet50_cbam_hires` | 11.7 img/s | ~30 img/s |
| aggregate | **31.2 img/s** | |

Two models together move fewer images per second than `resnet50_dual` alone —
~137 min for that group against ~95 min sequential. The card looks idle on paper
(~6 TFLOP/s against a ~121 TFLOP/s fp16 peak, 4.1 G of 22 G) but an L4 has only
~300 GB/s of memory bandwidth and 768px convolutions are **bandwidth-bound, not
occupancy-bound**. The giveaway is in round A's own numbers: 159.4 → 43.7 img/s
from 384 to 768 is **3.65× for exactly 4× the pixels**. Near-linear scaling with
pixel count means the device is already saturated; idle SMs would have shown
*sublinear* scaling. It also corrupts `train_seconds` and the inference
benchmark, which is half the cost comparison. Worth retrying only on an
A100/H100.

**Raising `MICRO_BATCH`** to fill the VRAM changes BatchNorm statistics — BN
normalises over the micro-batch and gradient accumulation does not touch it — so
it breaks comparability with every finished run. Never do it to buy speed.

Neither affects accuracy: same seeds, same micro-batch, same batch order, same
arithmetic. A run part-done under `PARALLEL=2` is scientifically fine; only its
timings are junk.

### You cannot reuse round 1's dataset copy

Round 1 trained off `prototype_1152_384`. Resizing that to 768 would *upsample* a
384px JPEG and invent detail that is not in it — the resolution comparison would
measure nothing. `prepare_dataset.py` builds from the original
`prototype_1152.tar` and refuses to upsample; `data.py` and `preflight.py` both
check independently.

### Dataloader workers, and the throughput this round actually costs

`--workers` defaults to **auto** = `min(cpu_count, 4)`. Decoding is CPU-bound,
so more workers than cores buys nothing — they take turns on the same silicon,
torch warns about it, and an oversubscribed pool with `persistent_workers=True`
is a known source of dataloader hangs. Colab's standard VM reports 2 CPUs.

**Measured on a 2-CPU T4 runtime**, the 768px corpus decodes at about
**27 img/s**. That is the ceiling for every run, and it is *below the GPU rate
for every 384-only model*:

| run | GPU could do | actual | bound by | min/epoch | total |
|---|---:|---:|---|---:|---:|
| `resnet50_base` | 72 | 27 | **decode** | 3.0 | ~65 min |
| `resnet50_cbam` | 65 | 27 | **decode** | 3.0 | ~65 min |
| `resnet50_fpn` | 61 | 27 | **decode** | 3.0 | ~65 min |
| `resnet50_vit` | 6-10 | 6-10 | GPU | ~10 | ~3-4 h (T4) / ~2-3 h (L4) |
| `resnet50_hires` | 18 | 18 | GPU | 4.5 | ~100 min |
| `resnet50_dual` | 14 | 14 | GPU | 5.6 | ~2 h |

The `resnet50_vit` row is an estimate for its 768+768 configuration (the old
384+384 figure was 20 img/s measured; the ViT's attention cost grows ~16× from
384 to 768). Epoch 1's logged img/s is the authoritative rate.
| `resnet50_effb3_dual` | 12 | 12 | GPU | 6.7 | ~2.5 h |

Round 1's ResNet-50 took 22 minutes; `resnet50_base` takes about 65. **Total for
the round is ~11 hours, not the ~8 originally estimated**, and notebook A alone
is ~5 h.

Note where the cost lands: the **768px runs are GPU-bound anyway**, so the
expensive decode is nearly free for them. It is the cheap 384-only runs that pay.

### Why this is not "fixed" by a 384px copy

The obvious speedup — give the four 384-only runs their own 384px JPEG copy —
would save about 2 hours and is deliberately **not** done, because the current
arrangement is already the maximally consistent one:

- `resnet50_base` loads the 768 file and bicubic-resizes to 384.
- `resnet50_dual`'s low-res stream loads the same 768 file and applies the
  *same* bicubic resize to 384.

Those two 384 views are **identical operations on identical bytes**. Introducing
a separate 384 JPEG would put an extra JPEG generation between `resnet50_base`
and `resnet50_dual`'s low stream — landing precisely on the comparison the round
exists to make ("did the second stream add anything?"). Trading a clean control
for two hours of wall clock is the wrong trade here.

The only cost-free lever is **a higher-CPU runtime**. Colab Pro's high-RAM
runtimes report more cores, and workers scale with them automatically.

If you do decide to accept the JPEG-generation difference for speed, declare it
in the write-up — do not let it in silently.

### Colab disconnects

Every epoch writes `runs/<model>/last.pt`. Re-running the same command resumes
automatically, so a disconnect costs at most one epoch. The checkpoint stores the
protocol fingerprint **and the view geometry**, and refuses to resume into either
a changed protocol or a changed resolution — half a run at 384 and half at 768
would be silently invalid.

---

## What to read, in what order

1. **`_SUMMARY.txt`** — every run at a glance, all four heads plus a
   `regressed` column.
2. **The control pair.** `resnet50_base` vs `resnet50_hires`. Does resolution
   alone explain it?
3. **The stream ablation.** Did the models that got extra pixels actually use
   them?
4. **The per-condition regression check.** Did improving LVH cost STEMI, AF or
   NORMAL? Read this *before* the macro table, not after it.
5. **The source confound** — see below.
6. Only then, the headline table.

### Read FPR before AUPRC

Round 1's LVH failure was false-positive dominated. A model that lifts LVH
sensitivity while lifting FPR has moved its operating point, not solved anything.
`compare.py` prints the FPR delta next to the AUPRC delta for exactly this
reason.

### The source confound — and why this round raises the stakes

All 1,442 STEMI records come from a single database, and stage 5 of the image
pipeline measured residual source signal (probe 0.515 vs 0.372 chance). Round 1
macro-AUROC spreads: resnet50 0.0225, convnext 0.0287, vgg16 0.0188, vit 0.0096,
effnet 0.0347.

**The 768px and FPN models can see fine texture the 384px models could not** —
grid colour, paper wear, JPEG signature — and that texture identifies the source
database. `compare.py` prints round 1's spreads alongside this round's and flags
anything wider than all of them.

> A resolution gain that arrives together with a widening source spread is a
> leakage gain, not a diagnostic one.

---

## Files

| file | role |
| --- | --- |
| `config.py` | protocol + input settings, two-tier fingerprint |
| `baseline.py` | round-1 results, frozen, for the regression check |
| `hybrids.py` | the seven architectures, the shared fusion head, CBAM, FPN |
| `data.py` | multi-view dataset — one decode, several resolutions |
| `engine.py` | train/eval loops, stream ablation, LR schedule, benchmarking |
| `metrics.py` | multi-label + per-source metrics, threshold tuning |
| `manifest.py` | index loading, integrity enforcement, `pos_weight` |
| `artifacts.py` | live-flushed logs, status, history, LVH-first summary |
| `train.py` | train one architecture |
| `compare.py` | verify protocol match, then tabulate — LVH first |
| `preflight.py` | GPU / data / loader / per-model VRAM check |
| `prepare_dataset.py` | build the 768px copy, refusing to upsample |
| `run_hybrids.sh` | the batch, control pair first, does not abort on failure |
| `make_notebooks.py` | build the four Colab notebooks |

## Prototype caveats, unchanged

These runs use `prototype_1152`: a subsample of train at one augmentation
variant per record. **AUROC, AUPRC and the ranking transfer** to the full corpus.
**Thresholds, calibration and absolute scores do not** — retune thresholds on the
full validation set before reporting final numbers.

## Requirements

Colab already has torch, torchvision, numpy, pandas, pillow, scikit-learn.
Only `timm` needs installing: `pip install -q timm`.
