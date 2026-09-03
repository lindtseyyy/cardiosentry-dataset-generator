# CardioSentry — hybrids v2

Twenty-one hybrid architectures over the **top three backbones from
[`cardiosentry_baselines_v2`](../cardiosentry_baselines_v2)**, under conditions
identical to the baseline runs, so every result is reported as a **delta against
the bare backbone**.

|  | CBAM | ECA | MSA | LT | LT + concat | LT + gated | MSA + LT + gated |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **ResNet-50** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **EfficientNetV2-S** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **VGG-16** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**The grid is complete — 3 × 7, every cell filled.** That is load-bearing, not
tidiness: with one seed per cell, agreement across the three ladders is the only
evidence here that separates a real effect from noise.

Four independent sigmoid heads: **STEMI, AF, LVH, NORMAL**. Geometry `nat768`
(768×1024, full sheet, printed ID redacted). Same metrics, same threshold rule,
same per-source confound reporting as v1, v1_768 and baselines_v2 — so all four
result sets sit in one table.

---

## The one design rule

    hybrid(backbone, attn=None, lt=False, fusion="none")  IS the baseline.

Every hybrid is the baselines_v2 model with components **inserted**, never with
the model rebuilt around them:

```
x -> backbone.forward_features(x)              -> F   (B, C, H/32, W/32)
  -> ATTENTION(F)                              -> F'  (same shape, optional)
  -> backbone.forward_head(F', pre_logits=True)      -> v_cnn   (B, D)
  -> LightweightTransformer(F')                      -> v_lt    (B, 256)
  -> FUSION(v_cnn, v_lt)                             -> v
  -> nn.Linear(dim(v), 4)
```

`forward_head(..., pre_logits=True)` is timm's own pooling stage — VGG-16's
4096-d ConvMlp, ResNet-50's global average pool, EfficientNetV2-S's `conv_head` —
and the classifier is a single `nn.Linear`, which is what
`timm.create_model(tag, num_classes=4)` builds. With no components the wrapper is
**parameter-for-parameter the model baselines_v2 trained**:

```
vgg16             134,276,932  vs  134,276,932
resnet50           23,516,228  vs   23,516,228
efficientnetv2_s   20,182,612  vs   20,182,612
```

`selftest.py` asserts that rather than asking you to believe it.

`config.py` here is **byte-identical** to baselines_v2's, so `fingerprint()`
matches and `compare.py` ranks a baseline run and a hybrid run in one table
without being forced to. That is what makes "+0.006 macro AUPRC over the
ResNet-50 baseline" a statement about the component rather than about two
experiments that happened to be run near each other.

**There are no `*_base` runs in this package.** The baseline for every row
already exists in `runs/baselines_v2/<backbone>@nat768/`; `import_baselines.py`
copies the three `result.json` files in (and refuses any whose fingerprint
differs). Re-training them would cost four GPU hours to reproduce numbers you
already have.

---

## The components

Read `hybrids.py` for the full reasoning. In short:

| | what it is | added params (ResNet-50) |
|---|---|---:|
| **CBAM** | channel gate (shared MLP over avg+max pool) then spatial gate (7×7 conv). Woo et al. 2018. | 0.54 M |
| **ECA** | one 1-D conv of width *k* over the pooled channel descriptor — no bottleneck. *k* from the paper's rule: 512→5, 1280→5, 2048→7. Wang et al. 2020. | **7** |
| **MSA** | depthwise 5×5 plus separable strip convolutions at k = 3, 7, 11, mixed by a 1×1. The MSCA construction, Guo et al. 2022. | 4.36 M |
| **LT** | 2 pre-norm transformer blocks, width 256, 4 heads, over the 24×32 = 768-token feature grid, with a **2-D** position embedding that resamples. | 1.78 M |
| **concat** | per-stream LayerNorm, then concatenate. | +0.01 M over LT |
| **gated** | both streams projected to 512, `g = σ(W[h_cnn; h_lt])`, `out = g·h_cnn + (1−g)·h_lt`. | 3.49 M |

### Why the strip convolutions belong on an ECG sheet

At `nat768` one feature cell is 32 source pixels — **8.9 mm of TIME** (357 ms at
25 mm/s) and **9.3 mm of VOLTAGE** (0.93 mV at 10 mm/mV). So the horizontal
strips span ~1.1 s (one cardiac cycle), ~2.5 s (one lead panel) and ~3.9 s
(several beats of the rhythm strip — the scale AF is *defined* at), and the
vertical strips span 2.8 mV, 6.5 mV and 10 mV (two to three lead rows — the span
a Sokolow-Lyon S(V1)+R(V5) comparison has to reach across). A square kernel that
large costs *k²*; separable strips cost *2k* and keep the two axes independent,
which is the right prior when one axis is time and the other is voltage.

### Why every attention module is the identity at step 0

Dropping a freshly-initialised sigmoid gate into a pretrained backbone multiplies
every feature map by something whose expected value at init is 0.5. That halves
activations the ImageNet weights were tuned for, and the first epochs go on
undoing the damage rather than learning — on 4,810 training images that cost is
not recoverable, and it shows up as *"attention hurt"* when what happened is
*"attention was initialised badly"*.

So every module is wrapped as `x + γ·(refine(x) − x)` with **γ initialised to
exactly 0**. The same wrapper is used for all three, so "which attention is best"
is not secretly "which attention was initialised least destructively".

**The learned γ is a result.** It is recorded per run and printed by
`compare.py`. A γ that never left zero says the backbone did not want the
reweighting — a cleaner finding than a noisy ±0.004 AUPRC, and a row with a
positive delta and γ ≈ 0 is *"a differently-initialised head helped"*, not
*"attention helped"*.

The **fusion gate** is treated the same way: its mean is measured on **test**
(reset before the test pass, so the inference benchmark's 50 batches of
`torch.randn` cannot contaminate it) and reported, so *"the model learned to
ignore the transformer"* is something you can write down.

### The transformer ladder, and what each rung isolates

| run | the question |
|---|---|
| `+ lt` | does global self-attention over 768 tokens beat the backbone's own pooling? The transformer **replaces** it, so a loss is ambiguous between "attention did not help" and "256 dims is not enough". |
| `+ lt_concat` | the rung that **disambiguates** the one above: if concat recovers what LT alone lost, the loss was the discarded pooling. |
| `+ lt_gated` | same information as concat, but the model must commit to a mixture — and the mixture is readable. |
| `+ msa_lt_gated` | do multi-scale context and global attention compose, or overlap? |

Per-stream LayerNorm is not decoration: it is what makes `ablate.py` valid.
Zeroing one stream **after its own LayerNorm** leaves the other's statistics
untouched, so "what did the transformer contribute" is answerable on a finished
run without retraining. Normalising after the concat would make the ablation
measure a distribution shift instead.

Two exceptions, both deliberate and both asserted in `selftest.py`:

* **The attention-only rungs have no LayerNorm before the head.** Adding one
  would be internally tidy and would also mean `resnet50_cbam = baseline + CBAM +
  a LayerNorm`, so a win could be either component. Passing through instead makes
  those runs differ from their baseline by **exactly** the inserted module.
* **On the LT-only rung the backbone's pooling stage is deleted**, not merely
  unused — that rung's design is that the transformer replaces it. For VGG-16
  that removes a 4096-d ConvMlp, so `vgg16_lt` is a **16 M** parameter network
  against `vgg16_lt_concat`'s 136 M. Real, and worth saying out loud in the
  writeup.

---

## Running it

One family — which is exactly what each notebook does. `F` is `resnet50`,
`efficientnetv2_s` or `vgg16`:

```bash
export F=resnet50  RUNS=runs/hybrids_v2/$F
python selftest.py  --data <DATA>                  # ~1 min, no GPU, no weights
python preflight.py --data <DATA> --family $F      # ~2 min, this family's 7
python import_baselines.py --from <baselines_v2 runs> --to $RUNS --backbone $F
./run_all.sh <DATA> $RUNS --family $F              # 9-13 h
python compare.py --runs $RUNS --csv $RUNS/comparison.csv
python ablate.py   --runs $RUNS --data <DATA>
```

Then, once all three families have finished — the only cross-backbone table:

```bash
python compare.py --runs runs/hybrids_v2/{resnet50,efficientnetv2_s,vgg16} \
    --out runs/hybrids_v2/_combined/comparison.txt \
    --csv runs/hybrids_v2/_combined/comparison.csv
```

`run_all.sh` also takes an explicit model list, and a micro-batch override:

```bash
./run_all.sh <DATA> <RUNS> resnet50_eca vgg16_lt
./run_all.sh <DATA> <RUNS> --family vgg16 --micro-batch 8    # VRAM headroom
```

The override is gated by `check_micro_batch.py`, which **refuses** it on a
BatchNorm backbone rather than warning — see the VRAM section below.

### The three notebooks

`python make_notebook.py` emits **one notebook per backbone family**, each with
the whole harness embedded as base64 in cell 1:

| notebook | family | runs | ~time |
|---|---|:--:|---|
| `CardioSentry_Hybrids_v2_A_ResNet50.ipynb` | ResNet-50 | 7 | 9 h |
| `CardioSentry_Hybrids_v2_B_EfficientNetV2S.ipynb` | EfficientNetV2-S | 7 | 9 h |
| `CardioSentry_Hybrids_v2_C_VGG16.ipynb` | VGG-16 | 7 | 13 h |

**They are meant to run at the same time.** The families are completely
independent — nothing in one reads anything another produces — so three parallel
Colab sessions turn ~31 h of wall clock into **~13 h**, the longest family.

> **Colab caps concurrent GPU sessions per account** — typically one on the free
> tier, two to three on Pro/Pro+. All three at once needs that headroom, or
> separate accounts. If you can only run two, start **C (VGG-16)** first: it is
> the long pole.

Splitting on the *backbone* rather than on run count is what makes each notebook
self-contained. A family notebook trains one backbone's whole ladder and imports
that one backbone's baseline row, so it produces its own complete delta table
without waiting for the others. Any other split would leave all three holding
fragments of three ladders and nothing reportable until every one had finished.

Regenerate all three with `python make_notebook.py` after editing any module.

### Three sessions, one Drive — what had to be made safe

Each notebook writes **only** to `runs/hybrids_v2/<family>/`, so `_SUMMARY.txt`,
`_batch_log.txt` and `comparison.csv` have exactly one writer each. Two things
are genuinely shared, and both were fixed rather than assumed safe:

* **the pre-resized corpus tar.** `prepare_corpus.sh` now stages to a
  PID-suffixed `.part`, so three simultaneous cold builds cannot interleave into
  one file and then publish it as the cache. It is still three times the work for
  one result, so **if the cache is cold, run cell 4 in one notebook alone first**
  and start the other two once it finishes. If you ran `baselines_v2` the tar is
  already there — same corpus, same geometry, keyed `<corpus>_<dim>.tar` with its
  geometry marker checked before extraction — and each session pays a two-minute
  restore instead of a twelve-minute rebuild.
* **the cross-backbone table.** `compare.py --runs` takes several folders, so it
  is built at the end by pointing at all three. That is cell 18, present in all
  three notebooks and safe to re-run. **It is the only place a ranking should be
  read from** — a component that helps one backbone is one draw.

The HF cache is shared too, and that one needs no fix: the three backbones are
different files and `huggingface_hub` takes per-file locks.

### If you would rather run it in one stream

`./run_all.sh <DATA> <RUNS>` with no family argument still trains all 21 in
order, family by family (ResNet-50 → EfficientNetV2-S → VGG-16), into one
folder — about **31 h**, three or four sessions. Re-run the identical command to
resume: finished runs are skipped and an interrupted run continues from its last
completed epoch, and the family ordering means a session cut short leaves
*complete ladders* rather than three partial ones.

### VRAM, and when micro-batch is yours to change

| backbone | micro-batch | accum | BatchNorm layers | measured peak on an L4 |
|---|:--:|:--:|:--:|---|
| VGG-16 | **16** | 2 | **0** | 13.0 GB |
| ResNet-50 | 8 | 4 | 53 | ~6–7 GB |
| EfficientNetV2-S | 8 | 4 | 110 | ~8 GB |

`steps_per_epoch` is **151 at any of 4 / 8 / 16**, so the cosine schedule and the
2-epoch warmup are identical whatever you pick, and the effective batch of 32
never moves. What is *not* identical is BatchNorm.

**BatchNorm normalises over the MICRO-batch, and accumulation sums gradients — it
does not merge batch statistics.** Change micro-batch on a BN network and you
change what the network computes, not just how fast. Every hybrid here is scored
as a delta against the baselines_v2 run of its backbone, which was trained at the
value in the table, so a hybrid trained at a different micro-batch measures the
normalisation as well as the component — and no metric in the report would reveal
it.

So: **ResNet-50 and EfficientNetV2-S are locked at 8.** `preflight.py` counts BN
layers on the real model and refuses to suggest raising them however much VRAM is
free. VGG-16 has none (`vgg16.tv_in1k`, not `vgg16_bn`), which is why it is the
one backbone raised — from the 4 baselines_v2 carried over, which had been scaled
from the v1_768 runs rather than measured at this geometry and left a 22 GB card
holding 4.8 GB.

**On a 16 GB T4**, halving is a legitimate trade for VGG-16. For the other two it
breaks the nesting the delta depends on unless you also re-train their baselines
here at the new value.

---

## Files

```
hybrids.py          CBAM / ECA / multi-scale attention, the transformer,
                    the fusion rules, and the HybridModel wrapper
models.py           the 21-entry registry, run order, and the factory
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
                    --runs takes SEVERAL folders, which is how the three
                    parallel families are combined into one table
ablate.py           stream ablation on FINISHED runs - no retraining
import_baselines.py copy the baseline rows in, fingerprint-checked
                    (--backbone imports just one family's)
preflight.py        GPU / data / tags / real fwd+bwd + VRAM suggestions
                    (--family checks only one backbone's seven rungs)
selftest.py         asserts the things that would otherwise be wrong SILENTLY
aspect_study.py     render sample sheets through the training code path
resize_dataset.py   bake the geometry into a pre-resized corpus copy
prepare_corpus.sh   pre-resize to local disk, cached as a tar on Drive
run_all.sh          the batch: all 21, one --family, or an explicit list
make_notebook.py    regenerate the three Colab notebooks (embeds this folder)
```

---

## Reading the results honestly

Four things bound what these numbers mean. All four belong next to the headline.

1. **A delta under ~0.01 macro AUPRC is a tie.** Test is ~660 records at one
   canonical image each, one seed. Rank on differences that **repeat across all
   three backbones**, never on the ordering within one — that is why the ladder
   is run three times rather than once on the best backbone.
2. **One seed per cell.** Twenty-one runs at three seeds would be four days.
   Nothing here estimates run-to-run variance, so no confidence interval can be
   attached to any single delta; the complete 3 × 7 grid is what stands in for
   it. A component that helps on one backbone is one draw. A component that
   helps on all three is a pattern.
3. **Prototype subset.** Train is 1 augmentation variant per record instead of 4.
   AUROC/AUPRC and the *ranking* transfer to the full corpus; thresholds,
   calibration and absolute scores do not.
4. **Single-source STEMI.** All 1,442 STEMI records come from one database, and
   stage 5 measured residual source signal (0.515 vs 0.372 chance).
   `REPORT_PER_SOURCE` is mandatory for this reason. Watch in particular for a
   hybrid that **widens** the per-source spread relative to its baseline: a
   spatial attention gate that learned to look at an acquisition artifact would
   do exactly that while the headline AUPRC went up.

### Added parameters are not uniform, and cannot be

ECA adds **7** parameters to ResNet-50; the multi-scale block adds 4.36 M
(+18.5%); the full `msa_lt_gated` stack adds 7.84 M (+33%). `result.json` records
a per-component breakdown and `compare.py` prints `added M` / `added %`, because
*"won by 0.004 AUPRC for +0.03% parameters"* and *"won by 0.004 AUPRC for +33%
parameters"* are different findings and only one of them is interesting.
