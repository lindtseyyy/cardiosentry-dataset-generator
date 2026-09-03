# CardioSentry — hybrids v3 (CBAM across five more backbones)

**One component, five backbones.** Every run here is a
[`cardiosentry_baselines_v2`](../cardiosentry_baselines_v2) backbone with **CBAM
inserted and nothing else changed**, under conditions identical to the baseline
runs, so every result is reported as a **delta against the bare backbone**.

|  | CBAM |
|---|:--:|
| **DenseNet-121** | ✓ |
| **EfficientNet-B3** | ✓ |
| **ConvNeXt-Base** | ✓ |
| **ViT-B/16** | ✓ |
| **SwinV2-Base** | ✓ |

Four independent sigmoid heads: **STEMI, AF, LVH, NORMAL**. Geometry `nat768`
(768×1024, full sheet, printed ID redacted). Same metrics, same threshold rule,
same per-source reporting as v1, v1_768, baselines_v2 and hybrids_v2 — so all
five result sets sit in one table.

---

## Why this package exists next to `hybrids_v2`

`hybrids_v2` is 7 components × 3 backbones. It answers *"which component"*, and it
answers it entirely inside one model family: ResNet-50, EfficientNetV2-S and
VGG-16 are all plain convolutional networks whose `forward_features` returns an
NCHW map at stride 32.

This package is 1 component × 5 backbones, and the five are chosen to leave that
family:

| backbone | feature layout from `forward_features` | stride |
|---|---|:--:|
| DenseNet-121 | `(B, 1024, H/32, W/32)` — NCHW | 32 |
| EfficientNet-B3 | `(B, 1536, H/32, W/32)` — NCHW | 32 |
| ConvNeXt-Base | `(B, 1024, H/32, W/32)` — NCHW | 32 |
| SwinV2-Base | `(B, H/32, W/32, 1024)` — **channels last** | 32 |
| ViT-B/16 | `(B, 1 + H/16·W/16, 768)` — **tokens, with a class token** | **16** |

Together the two packages give **eight backbones across three feature layouts**,
which is what turns *"CBAM helped"* from an observation about one model family
into a claim that either replicates outside it or does not.

These five are exactly the `baselines_v2` backbones `hybrids_v2` did not cover,
minus ConvNeXt-V2 — whose FCMAE pretraining is a second pretraining asymmetry on
top of ViT's, for no additional architectural variety. So **every run here has a
baseline row already trained**, and `import_baselines.py` copies it in.

---

## The one design rule (unchanged from v2)

    hybrid(backbone, attn=None)  IS the baseline.

```
x -> backbone.forward_features(x)          -> F   (whatever layout it uses)
  -> FORMAT.to_map(F)                      -> (B, C, gh, gw)
  -> CBAM(.)                               -> same shape, identity at step 0
  -> FORMAT.from_map(.)                    -> F' in F's own layout
  -> backbone.forward_head(F', pre_logits=True)   -> v
  -> nn.Linear(dim(v), 4)
```

With no attention the wrapper is **parameter-for-parameter** the model
`baselines_v2` trained — verified on all five real networks:

```
densenet121       6,957,956  vs   6,957,956
efficientnet_b3  10,702,380  vs  10,702,380
convnext_base    87,570,564  vs  87,570,564
vit_b16          88,010,500  vs  88,010,500     (at 768x1024; the pos-embed is bigger than at 384)
swinv2_base      86,897,916  vs  86,897,916
```

`config.py` here is **byte-identical** to baselines_v2's (and to hybrids_v2's), so
`fingerprint()` matches and `compare.py` ranks a baseline run, a v2 hybrid and a
v3 hybrid in one table without being forced to.

---

## The one thing that is new: `FeatureFormat`

CBAM's channel gate and 7×7 spatial gate are written against an NCHW map. Two of
these five backbones do not produce one. The alternative to an adapter would be a
second CBAM written for tokens — at which point *"CBAM on ViT"* and *"CBAM on
DenseNet"* would be two different modules and this comparison would not exist.
So the **features** move, not the module.

| format | backbones | what it does |
|---|---|---|
| `nchw` | DenseNet-121, EfficientNet-B3, ConvNeXt-Base | nothing |
| `nhwc` | SwinV2-Base | `permute(0,3,1,2)` and back |
| `tokens` | ViT-B/16 | splits the prefix tokens off, reshapes the patch tokens to `(B, C, gh, gw)`, and concatenates the prefix back |

Two properties are load-bearing, and `selftest.py` checks both **on the real
backbones' real feature tensors**, not on stand-ins:

1. **The round trip is exact.** `from_map(to_map(f)) == f` bit for bit — no
   interpolation, no reordering, no dtype change. That is what keeps *"CBAM is
   the identity at step 0"* true on all five rather than on the three where it is
   easy. A near-exact round trip would make the ViT and Swin deltas contain the
   conversion error as well as the component, and nothing in the metrics would
   separate them.

2. **ViT's class token never enters the grid.** It has no position on the sheet,
   and timm's ViT head *pools that token* — `forward_head` reads `x[:, 0]`. Fold
   it into a 7×7 spatial convolution and you quietly damage the exact vector the
   classifier reads, while everything still trains.

The patch tokens must also map **row-major** onto the grid. The transposed
alternative runs perfectly and has the spatial gate reading a 7×7 window across
TIME believing it is reading across VOLTAGE; `selftest.py` asserts a specific
token lands in a specific cell rather than trusting the reshape.

### The grids are not the same size, and that is a real caveat

At `nat768`, stride 32 gives CBAM a **24×32 = 768-cell** grid — one cell is
8.9 mm of time (357 ms at 25 mm/s) and 9.3 mm of voltage (0.93 mV at 10 mm/mV).
ViT-B/16 is stride 16, so it gets **48×64 = 3072 cells** at ~4.5 mm each.

Same module, four times the cells, a quarter of the millimetres per cell. A CBAM
win on ViT is therefore not automatically the same finding as a CBAM win on
DenseNet, and the writeup should say so rather than averaging the two.

### Why CBAM is the identity at step 0

Unchanged from v2, and it is the reason the deltas are readable at all: every
attention module is wrapped as `x + γ·(refine(x) − x)` with **γ initialised to
exactly 0**. Dropping a fresh sigmoid gate into a pretrained backbone otherwise
multiplies every feature map by something whose expected value is 0.5, and the
first epochs go on undoing that rather than learning — which on 4,810 training
images is not recoverable and shows up as *"attention hurt"*.

**The learned γ is a result.** It is recorded per run and printed by
`compare.py`. A γ that never left zero says the backbone did not want the
reweighting — a cleaner finding than a noisy ±0.004 AUPRC — and a row with a
positive delta and γ ≈ 0 is *"a differently-initialised head helped"*, not
*"attention helped"*. γ is also the one quantity directly comparable across all
five, because the wrapper is identical on every one of them.

---

## Running it

Two groups, which is exactly what each notebook does:

```bash
export G=cnn  RUNS=runs/hybrids_v3/$G
python selftest.py  --data <DATA>                 # ~1 min, no GPU, no weights
python preflight.py --data <DATA> --group $G      # this group only
python import_baselines.py --from <baselines_v2 runs> --to $RUNS \
    --backbone densenet121,efficientnet_b3,convnext_base
./run_all.sh <DATA> $RUNS --group $G              # ~8 h
python compare.py --runs $RUNS --csv $RUNS/comparison.csv
```

Then, once both groups have finished — the only five-backbone table:

```bash
python compare.py --runs runs/hybrids_v3/{cnn,transformer} \
    --out runs/hybrids_v3/_combined/comparison.txt \
    --csv runs/hybrids_v3/_combined/comparison.csv
```

`run_all.sh` also takes a single backbone, an explicit model list, and a
micro-batch override:

```bash
./run_all.sh <DATA> <RUNS> --family convnext_base
./run_all.sh <DATA> <RUNS> vit_b16_cbam
./run_all.sh <DATA> <RUNS> --group transformer --micro-batch 4
```

The override is gated by `check_micro_batch.py`, which **refuses** it on a
BatchNorm backbone rather than warning — see the VRAM section below.

### The two notebooks

`python make_notebook.py` emits **one notebook per group**, each with the whole
harness embedded as base64 in cell 1:

| notebook | group | runs | ~time |
|---|---|:--:|---|
| `CardioSentry_Hybrids_v3_A_CBAM_CNNs.ipynb` | densenet121, efficientnet_b3, convnext_base | 3 | ~8 h |
| `CardioSentry_Hybrids_v3_B_CBAM_Transformers.ipynb` | swinv2_base, vit_b16 | 2 | ~10 h |

**They are meant to run at the same time.** The groups are completely independent
— nothing in one reads anything the other produces — so two parallel Colab
sessions turn ~18 h of wall clock into **~10 h**, the longer group.

> The hour figures are **estimates** scaled from the hybrids_v2 runs at this
> geometry, not measurements: three of these five backbones have never been
> trained through this harness. `preflight.py` measures the real per-step cost
> and peak VRAM before you commit the night — read it before starting cell 8.

> **Colab caps concurrent GPU sessions per account** — typically one on the free
> tier, two to three on Pro/Pro+. If you can only run one, start **B** first: it
> is the long pole.

The split is on the **feature format**, which is also the cost boundary (2 samples
per micro-batch for the transformers against 4–8 for the CNNs) and the interesting
scientific boundary. Each notebook is then a self-contained claim — *"CBAM across
three convolutional backbones"* and *"CBAM across two attention backbones"* —
rather than a fragment. Splitting three-and-two by cost alone would leave each
session holding pieces of both claims and nothing reportable until the other
finished too.

Regenerate both with `python make_notebook.py` after editing any module.

### Two sessions, one Drive — what had to be made safe

Each notebook writes **only** to `runs/hybrids_v3/<group>/`, so `_SUMMARY.txt`,
`_batch_log.txt` and `comparison.csv` have exactly one writer each. Two things are
genuinely shared, and both were handled rather than assumed safe:

* **the pre-resized corpus tar.** `prepare_corpus.sh` stages to a PID-suffixed
  `.part`, so two simultaneous cold builds cannot interleave into one file and
  then publish it as the cache. It is keyed `<corpus>_<dim>.tar` with its geometry
  marker checked before extraction, and it is deliberately pointed at
  `experiments/baselines_v2/data` — **the same tar baselines_v2 and hybrids_v2
  already built**, so each session pays a two-minute restore instead of a
  twelve-minute rebuild. If the cache is cold, run cell 4 in one notebook alone
  first.
* **the cross-group table.** `compare.py --runs` takes several folders, so it is
  built at the end by pointing it at both. **It is the only place the pattern
  should be read from** — a component that helps one backbone is one draw.

The HF cache is shared too, and needs no fix: the five backbones are different
files and `huggingface_hub` takes per-file locks.

### VRAM, and when micro-batch is yours to change

Every value is the one `baselines_v2` trained that backbone at, because the delta
is only clean if the two runs computed the same thing.

| backbone | micro-batch | accum | BatchNorm layers | may you change it? |
|---|:--:|:--:|:--:|---|
| DenseNet-121 | 4 | 8 | **121** | **no** |
| EfficientNet-B3 | 8 | 4 | **78** | **no** |
| ConvNeXt-Base | 4 | 8 | 0 | yes |
| SwinV2-Base | 2 | 16 | 0 | yes |
| ViT-B/16 | 2 | 16 | 0 | yes |

`steps_per_epoch` is **151** at any of 2 / 4 / 8 / 16, so the cosine schedule and
the 2-epoch warmup are identical whatever you pick, and the effective batch of 32
never moves. What is *not* identical is BatchNorm.

**BatchNorm normalises over the MICRO-batch, and accumulation sums gradients — it
does not merge batch statistics.** Change micro-batch on a BN network and you
change what the network computes, not just how fast. Every hybrid here is scored
as a delta against the `baselines_v2` run of its backbone, trained at the value
above, so a hybrid trained at a different micro-batch measures the normalisation
as well as CBAM — and no metric in the report would reveal it.

So **DenseNet-121 and EfficientNet-B3 are locked.** `preflight.py` counts BN
layers on the real model and refuses to suggest raising them however much VRAM is
free; `check_micro_batch.py` refuses the flag outright. The other three are
LayerNorm networks, where raising or halving is a pure throughput/VRAM trade with
an identical accumulated gradient — **on a 16 GB T4 that is the knob to use.**

---

## Files

```
hybrids.py          CBAM / ECA / multi-scale attention, the FEATURE FORMAT
                    adapter, the transformer, the fusion rules, and the
                    HybridModel wrapper.  Shared with hybrids_v2 except for
                    FeatureFormat - only the CBAM rung is registered here
models.py           the 5-entry registry, the two groups, and the factory
config.py           IDENTICAL to baselines_v2 - do not edit, it is the thing
                    that makes the delta legitimate
dimensions.py       the geometry and its physical-units arithmetic
data.py             transforms (redaction, rectangular resize), datasets, loaders
manifest.py         index.csv loading + integrity + geometry-marker checks
metrics.py          multi-label metrics  (byte-identical to v1)
engine.py           train/eval loops, complexity, inference benchmark
artifacts.py        crash-safe run artifacts
train.py            train one hybrid
compare.py          rank everything, and emit the delta-against-baseline tables.
                    --runs takes SEVERAL folders, which is how the two parallel
                    groups are combined into one table
import_baselines.py copy the baseline rows in, fingerprint-checked
                    (--backbone takes a comma-separated list)
preflight.py        GPU / data / tags / real fwd+bwd + VRAM suggestions
                    (--group checks only one notebook's runs)
selftest.py         asserts the things that would otherwise be wrong SILENTLY
check_micro_batch.py the BatchNorm gate on --micro-batch
aspect_study.py     render sample sheets through the training code path
resize_dataset.py   bake the geometry into a pre-resized corpus copy
prepare_corpus.sh   pre-resize to local disk, cached as a tar on Drive
run_all.sh          the batch: all 5, one --group, one --family, or a list
make_notebook.py    regenerate the two Colab notebooks (embeds this folder)
```

**There is no `ablate.py` here.** Every run in this package is single-stream, so
there is nothing to ablate. If you add a two-stream rung to `models.LADDER`, copy
`ablate.py` back in from `hybrids_v2` — the fusion code it needs is already in
`hybrids.py`.

---

## Reading the results honestly

Five things bound what these numbers mean. All five belong next to the headline.

1. **A delta under ~0.01 macro AUPRC is a tie.** Test is ~660 records at one
   canonical image each, one seed.
2. **One seed per cell.** Nothing here estimates run-to-run variance, so no
   confidence interval can be attached to any single delta. What stands in for it
   is the **shape** across five very different backbones:
   * helps almost everywhere → a real, backbone-independent effect
   * helps the CNNs but not the transformers → CBAM is supplying something
     self-attention already has
   * scattered inside ±0.01 → one seed of noise, and say so
3. **Pretraining is not uniform.** ViT-B/16's usable timm checkpoint at 384 is
   in21k-pretrained then fine-tuned to in1k; the other four are supervised in1k.
   This does **not** contaminate any delta — each hybrid is measured against a
   baseline with the same weights — but it does contaminate any statement about
   absolute scores across backbones. `compare.py` prints the `pretrain` column.
4. **Prototype subset.** Train is 1 augmentation variant per record instead of 4.
   AUROC/AUPRC and the *ranking* transfer to the full corpus; thresholds,
   calibration and absolute scores do not.
5. **Single-source STEMI.** All 1,442 STEMI records come from one database, and
   stage 5 measured residual source signal (0.515 vs 0.372 chance).
   `REPORT_PER_SOURCE` is mandatory for this reason, and it matters **more** for
   CBAM than for a bare backbone: a spatial gate learns *which region of the
   sheet matters*, and an acquisition artifact is a region. Watch specifically
   for a hybrid that **widens** the per-source spread relative to its baseline
   while its headline AUPRC goes up.

### The added parameters are not uniform, and cannot be

CBAM's cost is set by the backbone's feature width, so the same component is a
different bargain on each:

| backbone | C | CBAM adds | as % of the backbone |
|---|:--:|---:|---:|
| ViT-B/16 | 768 | 0.078 M | +0.09% |
| ConvNeXt-Base | 1024 | 0.136 M | +0.16% |
| SwinV2-Base | 1024 | 0.136 M | +0.16% |
| DenseNet-121 | 1024 | 0.136 M | +1.96% |
| EfficientNet-B3 | 1536 | 0.303 M | +2.83% |

`result.json` records the per-component breakdown and `compare.py` prints
`added M` / `added %`, because *"won by 0.004 AUPRC for +0.09% parameters"* and
*"won by 0.004 AUPRC for +2.8% parameters"* are different findings and only one
of them is interesting.
