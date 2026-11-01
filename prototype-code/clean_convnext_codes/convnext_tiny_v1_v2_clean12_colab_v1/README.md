# ConvNeXt Tiny (V1 and V2) on HEEDB clean12 (Google Colab, A100)

Open **CardioSentry_ConvNeXt_Tiny_V1_V2_Clean12_Colab.ipynb** in Colab. Its
metadata asks for an A100 with High-RAM; check Runtime → Change runtime type.
Then choose Runtime → Run all. The notebook is self-contained: upload only the
`.ipynb`.

The two models train **one after another in one session**:

1. ConvNeXt-V1 Tiny, `convnext_tiny.fb_in22k_ft_in1k_384`
2. ConvNeXt-V2 Tiny, `convnextv2_tiny.fcmae_ft_in22k_in1k_384`

These are the same pretrained tags as the V1 and V2 Base clean12 runs, at Tiny
size. The slow shared work runs once:

- the package install and Drive mount
- the streamed, SHA-256-checked ~104.5 GiB extraction from Google Drive
- the manifest audit and the patient split (must reproduce `4eba80c7…`)
- the decoder-worker tuning

Each model then gets its own batch probe, eager-vs-compiled benchmark, training
run with resume, and final evaluation.

The generator started as a copy of `../../cnn_baselines_clean12_colab_v1/`,
because that notebook already reads the dataset from Drive and runs several
models in sequence. Its plain CNN recipe was replaced by the ConvNeXt recipes
of `../convnext_v1_base_clean12_v1/` and `../convnextv2_base_clean12_colab_v1/`,
including layer decay, the head-init scale, `MixedPrecisionGRN`, and
rare-label sampling.

## Recipe: each Tiny model reuses its Base model's recipe

| | V1 Tiny (= V1 Base) | V2 Tiny (= V2 Base) |
|---|---|---|
| peak LR (head) | 2e-4 | 1e-4 |
| weight decay | 0.02 | 0.05 |
| warm-up | 2 epochs | 3 epochs |
| layer decay (timm, per block) | **0.86** (Base 0.925) | **0.86** (Base 0.925) |
| drop-path | **0.1** (Base 0.2) | **0.1** (Base 0.2) |
| head init scale | 0.001 | 0.001 |
| rare-label sampling | off | on, threshold 0.10 |
| GRN | — | mixed-precision GRN |

Shared by both models: 768×1024 full sheet, sqrt-pos_weight BCE (on the counts
each sampler feeds), effective batch 64, EMA 0.9998, cosine to 1% over at most
20 epochs, early stopping patience 5 from epoch 10, collapse guard,
raw-vs-EMA selection on validation pathology Macro AP, validation-tuned
thresholds, and 1,000-round patient-bootstrap CIs.

Only two values differ from Base, both because Tiny is shallower (3/3/9/3
blocks, not 3/3/27/3):

- **Layer decay.** timm gives each block its own level: 20 levels for Tiny,
  38 for Base. Base's 0.925 puts the stem at 0.056× the head LR. On Tiny,
  0.925 would give 0.23×, so 0.86 is used instead (0.86¹⁹ = 0.057×). The
  validator checks this.
- **Drop-path.** 0.1 is ConvNeXt-T's ImageNet rate.

What the comparisons mean:

- **Tiny vs Base, same version:** only model size differs (plus those two
  depth-driven values).
- **V1 Tiny vs V2 Tiny:** like the Base pair, the recipes differ (LR, weight
  decay, warm-up, sampling), so this does not isolate the architecture.

Compute per 768×1024 sheet, measured by `validate_notebook.py`:

| model | parameters | GMACs |
|---|---|---|
| ConvNeXt-V1 Tiny | 27.8 M | 69.8 |
| ConvNeXt-V2 Tiny | 27.9 M | 69.8 |
| ConvNeXt-V1 Base | 87.6 M | 240.6 |

## Outputs

`MyDrive/CardioSentry/new_runs/convnext_tiny_v1_v2_clean12_colab_a100_seed42/`:

- the shared split files
- one complete run folder per model: `convnext_tiny/`, `convnextv2_tiny/`
- `comparison.csv`, `comparison_per_label_ap.csv`,
  `comparison_paired_bootstrap.csv`, `comparison.md`, `comparison_curves.png`

**Base references.** Section 11 also reads the Base runs from
`new_runs/convnextv1_base_clean12_a100_seed42/` and
`new_runs/convnextv2_base_clean12_colab_a100_seed42/` (`CS_REFERENCE_RUNS`
changes the list).

- It uses their `test_predictions.csv`, after checking that the test records
  and labels are exactly this split's.
- It recomputes their patient bootstrap with the same function, seed and
  patients, so the comparison has paired **Tiny − Base** rows. (The Base
  notebooks did not save their draws.)
- A missing or mismatched reference is skipped with a message.

## Resume, subsets, overrides, failures

- **Resume:** after a disconnect, choose Run all again with the same `CS_RUN_GROUP`.
  - A finished model is skipped.
  - The interrupted model resumes from its last finished epoch.
  - The archive is streamed from Drive again on a new runtime. This costs time
    only, no S3 egress.
- **Subsets:** `CS_ARCHITECTURES=convnextv2_tiny` runs one model.
- **Per-model overrides:**
  - `CS_BASE_LR_<ARCH>`, `CS_WEIGHT_DECAY_<ARCH>`, `CS_WARMUP_EPOCHS_<ARCH>`,
    `CS_LAYER_DECAY_<ARCH>`, `CS_DROP_PATH_<ARCH>`, `CS_HEAD_INIT_SCALE_<ARCH>`,
    `CS_RARE_SAMPLING_<ARCH>`.
  - `<ARCH>` is `CONVNEXT_TINY` or `CONVNEXTV2_TINY`.
  - Each override is part of that model's fingerprint, so a changed recipe is
    refused under the same group name. Use a new `CS_RUN_GROUP`.
  - Example, matched V1/V2 recipe: set `CS_RARE_SAMPLING_CONVNEXTV2_TINY=0`
    and the same LR, weight decay and warm-up for both.
- **Failures:** a failing model writes `<arch>/failure.json` and the next one
  still runs. `CS_CONTINUE_ON_ERROR=0` stops instead. After a collapse-guard
  stop, retry with half the LR under a new group.
- **Benchmark only:** `CS_BENCHMARK_ONLY=1` (optionally with
  `CS_COMPUTE_UNITS_PER_HOUR`) runs the probes and the eager/compiled benchmark
  for each model. Section 10c then writes the cost table to
  `new_runs/convnext_tiny_benchmark_<gpu>/`.

Dataset options are the same as the CNN baselines notebook:

- `CS_DATASET_READ_MODE=api|mount`
- `CS_DATASET_DRIVE_DIR`
- `CS_DATASET_DRIVE_FILE_ID`
- `CS_DATASET_DOWNLOAD_WORKERS`
- `CS_DATASET_DOWNLOAD_CHUNK_MIB`

The archive must be at
`MyDrive/CardioSentry/new_dataset/clean_ecg_signal_images_clean12.tar`, with
its `.sha256` next to it.

## Regeneration and local verification

```bash
python3 make_notebook.py
../../../.venv/bin/python validate_notebook.py   # a few minutes on CPU
```

The validator runs the notebook's own code on CPU, with random weights, tiny
synthetic PNGs, a fake Drive API and a temporary "Drive". It checks:

- **Recipes and overrides:** the per-model recipes and the
  `CS_<SETTING>_<ARCH>` overrides.
- **Section 3 (dataset staging), in full:**
  - API mode, with injected 503/429/dropped/short reads
  - the skip on re-run
  - mount mode
  - a SHA-256 mismatch
  - a damaged Drive copy
- **Section 6 (models and optimizer):**
  - 20 layer-decay levels, with the stem at 0.86¹⁹ of the peak LR
  - the head-init scale
  - 18 `MixedPrecisionGRN` modules in V2 and none in V1
  - V2 output equal to timm's GRN in FP32, with identical state-dict keys
- **Section 7:** worker tuning.
- **`run_architecture()` end to end for both models:**
  - a full V1 run, export and Drive sync
  - the skip after a wiped runtime
  - a failing V2 that does not stop the notebook
  - V2 interrupted in epoch 2 and resumed from Drive's `last.pt`, with its
    rare-label sampler lengthening the epoch and its export loading into a
    plain timm model
  - refusal of a changed recipe
- **Section 11 (comparison):**
  - a Base reference re-scored from a copy of V1's predictions pairs with V1
    at exactly 0
  - a reference with other test records is skipped
  - a missing reference is skipped
- **Benchmark-only mode.**
- **Native 768×1024 forward,** for the MAC counts.

All checks passed (2026-10-04). They do not cover A100 speed and memory,
`torch.compile`, the real Drive API and mount, or full-corpus training.
