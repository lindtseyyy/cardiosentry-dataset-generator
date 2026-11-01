# Choosing the explanation method: Grad-CAM vs Grad-CAM++ vs Score-CAM (clean12, Google Colab)

Open **CardioSentry_XAI_CAM_Comparison_Clean12_Colab.ipynb** in Colab, choose
an **L4** runtime (A100 and T4 also work), then Runtime → Run all. Upload only
the `.ipynb`; the method code (`xai_cam.py`) is embedded in it.

The notebook measures the three numbers for the xAI-choice section:

| Metric | Definition | Better |
|---|---|---|
| Deletion AUC | Area under p(label) as the map's top pixels are replaced by blank paper (0 → 100% of pixels, `CS_CURVE_STEPS` + 1 points) | lower |
| Insertion AUC | Area under p(label) as the map's top pixels are restored onto blank paper | higher |
| Explanation Time | Seconds for one map for one (image, label), on the GPU and on CPU, plus the forward-pass count | lower |

Deletion and insertion follow Petsiuk et al., *RISE*, BMVC 2018.

## What is compared, and on what

**Model.** The clean12 ConvNeXt-V1 Base run that the app serves:

- run folder: `new_runs/convnextv1_base_clean12_a100_seed42/best.pt`
- the EMA weights the run selected
- SHA-256 `7e2583a6…`, which is the hash pinned in the app's
  `models/active.yaml`; the notebook refuses any other checkpoint

**Layer.** `stages.3`, 1024 × 24 × 32 at 768 × 1024: the same layer the app's
Grad-CAM uses.

**Pairs.** The pairs are drawn from that run's own `test_predictions.csv`, which
is the frozen patient split's **test** set (11,323 records).

- For each label except NORM, take 20 **true positives**: the label is present
  and scored above its validation-tuned threshold.
- The 20 are the first 20 in SHA-256(`seed:label:record`) order.
- That is about 220 pairs over about 210 images.
- Train images would give near-1.0 probabilities and flat curves. Validation was
  already used for checkpoint and threshold selection.

**Methods.** All three compute ReLU(Σ_c w_c A_c) at 24 × 32, then bilinearly
upsample to 768 × 1024. Only the weights differ:

| Method | w_c | Cost per map |
|---|---|---|
| Grad-CAM | mean gradient of the label's logit over channel c | 1 forward + 1 head backward |
| Grad-CAM++ | Chattopadhay et al. 2018 closed-form α, as in pytorch-grad-cam | same |
| Score-CAM | sigmoid score of the label on the image masked by channel c (upsampled, min-max scaled) | 1 + 1,024 forwards |

**Score-CAM weighting.** Score-CAM follows the authors' reference code: each
channel is weighted by the class score, here the sigmoid. pytorch-grad-cam
instead applies a softmax over channels. That is why the library cross-check
covers only Grad-CAM and Grad-CAM++.

**Random map.** A seeded uniform 24 × 32 map, upsampled the same way, is scored
alongside the three methods as the floor.

**Fairness controls.**

- **Same tie-break.** Ties between pixels (Grad-CAM's zero regions) are broken
  by one seeded permutation per pair, shared by every method.
- **Same baseline.** Every method uses the same baseline image.
- **Same precision.** bf16 on L4/A100, fp16 on T4. The gradient path differentiates
  only the head, in fp32.

**Baseline.** By default (`CS_BASELINE=paper`), a deleted pixel becomes the
sheet's own **blank-paper colour**. It is read from the render's 15-px side pad,
so it is plain paper with no grid and no trace. A black or mean-grey image would
be far outside the training distribution, and the curves would then measure that
shock rather than the map. `blur` (Petsiuk's insertion canvas) and `zero` are
available for a sensitivity run, under a new `CS_RUN_NAME`.

## Statistics

`summary.md` / `summary.csv` report each method's mean Deletion AUC, Insertion
AUC and Insertion − Deletion with **95% CIs**.

- **Bootstrap by image.** The CIs come from 2,000 rounds that resample **images**,
  not rows, because one image can be several pairs.
- **Paired comparison** (`paired_vs_gradcam.csv`). For each method vs Grad-CAM on
  the same pairs:
  - the mean difference, with its CI
  - the share of pairs where the method beats Grad-CAM
  - a Wilcoxon signed-rank test over per-image mean differences
- **Per label** (`per_label.csv`). Rhythm labels (AF, SB, STACH) are whole-sheet
  findings; STEMI, STE and LBBB are local. The methods can rank differently on
  the two groups.
- **Explanation Time** (`timing_summary.csv`): median, IQR, × Grad-CAM, and the
  forward-pass count. The count is the hardware-independent part of the claim.

Report the outcome as found. If Score-CAM matches or beats Grad-CAM on the AUCs,
the case for Grad-CAM is **comparable faithfulness at a small fraction of the
cost**, not "better everywhere".

## Running on an A100 without wasting compute units

1. **CPU runtime: stage the images.** Uncomment
   `os.environ["CS_STAGE_IMAGES_ONLY"] = "1"` in cell 1b and choose Run all.
   - It streams the 104.5 GiB archive (pure I/O, 20–60 min).
   - It caches the ~210 test images as `eval_images.tar` in the run folder.
   - It stops with an intended `SystemExit`.
   - It writes no run design, so the GPU runtime's precision decides the
     fingerprint.
2. **A100 runtime: evaluate.** Comment the line out again and choose Run all.
   The images come from the cache in seconds.

What the GPU runtime does, in order:

- **Batch size.** It picks the largest batch that fits, by running the real
  Score-CAM and curve code (up to 64).
- **Compile check.** It compiles the model once with `torch.compile` at that
  fixed batch shape. `FixedBatch` pads the last chunk, so the model never
  recompiles.
- **Keeps the faster path.** It times compiled against eager on the same batch
  and keeps the faster one, but only if the outputs match within
  `CS_COMPILE_TOLERANCE`. `benchmark_<gpu>.json` records both speeds.
  - On this repo's laptop CPU, compiled was slower (107 vs 130 images/s), so
    eager was kept.
  - On an A100, ConvNeXt inference usually gains something; the notebook measures
    it rather than assuming it.
- **Already in place:** bf16 and TF32, channels-last, cuDNN autotuning, one
  Score-CAM pass shared by all labels of an image, and gradients only through the
  head.
- **What stays eager.** The gradient methods and the Explanation Time
  measurements stay eager, so the reported seconds are those of a plain deployed
  model.

The last cell releases the runtime when everything has finished.

## Run time and resume

- **Archive stream (first run only).** The ~210 PNGs are extracted from the
  104.5 GiB Drive archive, SHA-256 checked. A TAR has no index, so the whole
  archive streams past. This takes roughly 20–60 min.
- **Image cache.** The images are then cached as `eval_images.tar` (~0.2 GB) in
  the run folder. Every later runtime extracts that instead.
- **Evaluation.** About 215k Score-CAM forwards plus about 90k curve forwards at
  768 × 1024. Section 5c measures this GPU's throughput and prints the projected
  hours before the loop starts. My unmeasured guess is ~1–2 h on an A100 and
  ~3–6 h on an L4.
- **Resume.** Each finished image is on Drive. After a disconnect, choose Run all
  again with the same `CS_RUN_NAME`; finished images are skipped.
- **Changed design.** The run fingerprint covers pairs, steps, baseline,
  precision and checkpoint. A changed design is refused under the same
  `CS_RUN_NAME`. A T4 (fp16) run therefore needs a different `CS_RUN_NAME` from an
  L4 (bf16) run.
- **Cheaper run.** `CS_PAIRS_PER_LABEL=10` halves the work.
  `CS_CURVE_STEPS=25` cuts the curve part in half.

## Explanation Time on the deployment laptop

The notebook times this GPU and Colab's CPU. The app runs on a laptop, so time
it there too, with the same code:

```bash
cd prototype-code/xai_cam_comparison_clean12_colab_v1
../../.venv/bin/python cpu_timing.py --images <folder with a few clean12 PNGs> --out timing_laptop.csv
```

- It uses the app's own `models/weights/convnext_v1_base_clean12_best.pt` and
  the top-scoring label of each image.
- Grad-CAM and Grad-CAM++ are measured.
- Score-CAM is extrapolated from 16 masked forwards: the activation pass plus
  channels × seconds per masked forward. `--scorecam-full` runs one full map,
  which takes hours on a laptop.
- For PNGs, extract a few from `eval_images.tar` in the run folder.
- Close the browser first. On 2026-10-05 this i3 laptop measured 36–78 s per
  forward with ~3 GB free, which is mostly memory pressure.

## Outputs

`MyDrive/CardioSentry/new_runs/xai_cam_comparison_clean12_v1/`:

- **Results:**
  - `summary.md`, `summary.csv`
  - `paired_vs_gradcam.csv`, `per_label.csv`
  - `timing_summary.csv`, `timing/*.csv`
- **Raw data:**
  - `per_pair_metrics.csv`: one row per pair × method, with both AUCs,
    p(full), p(baseline), all-zero-map flag, and Score-CAM forward count
  - `maps/<image>.npz`: 24 × 32 maps
  - `curves/<image>.npz`: the full curves
- **Figures:** `figures/mean_curves.png`, `figures/example_maps.png`
- **Provenance:**
  - `config.json` (design + fingerprint)
  - `pairs.csv`, `pair_availability.csv`
  - `parity.csv`: our probabilities vs the run's stored ones
  - `library_check.csv`: Grad-CAM/Grad-CAM++ vs pytorch-grad-cam
  - `benchmark_<gpu>.json`
  - `eval_images.tar` (+ `.sha256`, `.json`)

## Files

| File | Role |
|---|---|
| `xai_cam.py` | The methods and metrics. Embedded verbatim as notebook section 3; imported by `cpu_timing.py` |
| `make_notebook.py` | Generates the notebook: `python3 make_notebook.py` |
| `cpu_timing.py` | Explanation Time on a local CPU |
| `validate_notebook.py` | CPU validation (below) |

## Validation

```bash
../../.venv/bin/python validate_notebook.py
```

`CS_GRADCAM_PATH=<dir with pytorch_grad_cam>` also runs the library comparison
locally. The validator runs on CPU with a tiny ConvNeXt (`convnext_atto`).

**Part 1: the method code.**

- the model split equals the model
- Grad-CAM equals a from-scratch full-graph Grad-CAM
- Grad-CAM and Grad-CAM++ equal pytorch-grad-cam 1.5.7 (r = 1.000000)
- Score-CAM is batch-size invariant and equals a naive per-channel loop;
  constant channels are skipped
- ranks are a seeded permutation in saliency order
- each curve starts and ends at p(image) and p(baseline)
- curves are batch-size invariant
- the trapezoid AUC is correct
- on a toy model whose score lives in one corner, an oracle map beats a random
  map on both metrics

**Part 2: every notebook cell against a temporary "Drive".** The fake Drive holds
a raw+EMA checkpoint, synthetic sheets in a real TAR with a SHA-256 sidecar, and
reference predictions from the EMA model. It checks:

- a full run, with outputs and the parity check
- a rerun on an empty runtime with the big archive removed: the Drive image
  cache and resume are used, and the metrics are unchanged
- refusal of a changed design
- a parity failure on wrong reference probabilities
- `cpu_timing.py` end to end

Later additions:

- `FixedBatch` padding (Score-CAM and curves equal eager)
- stage-only mode, followed by an evaluation runtime that uses the cache with
  the archive removed
- the compiled-vs-eager selection, with a forced `torch.compile` on CPU

**Result:** 57/57 checks passed on 2026-10-05.

**Not covered:** GPU speed and memory, bf16/fp16 numerics, the real Drive API and
mount, Colab sign-in, and the real model.
