# ConvNeXt-V2 Tiny master-corpus fine-tuning

Open `CardioSentry_ConvNeXtV2_Tiny_Master_EC2.ipynb` on the EC2 instance and
run it from top to bottom. The notebook is self-contained; `make_notebook.py`
only regenerates the checked-in notebook after edits.

For Google Colab with an L4 and High-RAM runtime, use
`CardioSentry_ConvNeXtV2_Tiny_Master_Colab_L4_High_RAM.ipynb` and follow
`README_COLAB.md`. That copy keeps the training/evaluation configuration but
uses Google Drive persistence and fast ephemeral `/content` extraction.

## Required input

Place the following together in the directory configured by
`CS_ARCHIVE_DIR`:

- `images_train_00.tar` through `images_train_05.tar`
- `images_val.tar`
- `images_test.tar`
- one `<archive>.sha256` file for each archive
- either the generator's `runs/synthetic_images_v1/index.csv`
- or the exact v3 `harmonized_ecg_metadata.csv` used for image generation

If `index.csv` remains elsewhere, set `CS_INDEX_CSV` to its full path. If the
metadata CSV remains elsewhere, set `CS_MASTER_METADATA_CSV`. When the original
index is absent, the notebook reconstructs it under `CS_EXTRACT_ROOT`, then
writes `index_reconstruction_report.json` alongside it.

The image filenames use deterministic anonymous identifiers such as
`ECG000001_a0.jpg`; they are not global record IDs. The fallback reproduces the
generator's exact accepted-row ordinal mapping and refuses to run unless the
metadata has this SHA-256:

```text
b2434ab7da801f00d338cbab628cb7dae42d76d28bb8c74221537766d9423ac0
```

It also verifies every archive hash, all known generator record/image/class/
source totals, every record's expected variant set, split agreement, binary
labels, NORMAL semantics, and bidirectional index/file coverage. This prevents
a reordered or different metadata CSV from silently assigning the wrong label
to an anonymous image.

## Reuse a completed V1 extraction

The EC2 notebook includes a separate **Cell 1a** for machines where the
ConvNeXt V1 notebook already prepared the corpus. By default it detects V1's
standard location:

```text
/mnt/cardiosentry/master-dataset/extracted/
```

The cell requires the reconstructed index and all eight extraction markers,
then assigns the shared data paths to V2. The later archive cell still verifies
checksums but skips unpacking every archive. V2 results remain isolated at:

```text
/mnt/cardiosentry/runs/convnextv2_tiny_master_v1/convnextv2_tiny_master_seed42/
s3://cardiosentry-fcmae-758821198776-us-east-1-an/CardioSentry/runs/convnextv2_tiny_master_v1/convnextv2_tiny_master_seed42/
```

Use `CS_V2_RUN_NAME` for another V2 run name. For a fresh machine without the
V1 extraction, set `CS_REUSE_V1_PREPARED_DATA=0` before starting the kernel.
Non-default V1 paths can be supplied through `CS_V1_EXTRACT_ROOT`,
`CS_V1_ARCHIVE_DIR`, and `CS_V1_INDEX_CSV`.

Example environment configuration before starting Jupyter:

```bash
export CARDIOSENTRY_PROJECT_ROOT=/mnt/cardiosentry
export CS_ARCHIVE_DIR=/mnt/cardiosentry/datasets/synthetic_images_v1
export CS_MASTER_METADATA_CSV=/mnt/cardiosentry/datasets/synthetic_images_v1/harmonized_ecg_metadata.csv
export CS_EXTRACT_ROOT=/mnt/cardiosentry/experiments/convnextv2_tiny_master_v1/data
export CS_RUN_ROOT=/mnt/cardiosentry/runs/convnextv2_tiny_master_v1
```

Set `CS_INDEX_CSV` instead when the original index is available. The fallback
output defaults to:

```text
<CS_EXTRACT_ROOT>/reconstructed_index.csv
<CS_EXTRACT_ROOT>/index_reconstruction_report.json
```

## Automatic S3 persistence

S3 upload is enabled by default for this bucket:

```text
s3://cardiosentry-fcmae-758821198776-us-east-1-an/
```

The default destinations are:

```text
CardioSentry/experiments/convnextv2_tiny_master_v1/data/
CardioSentry/runs/convnextv2_tiny_master_v1/<CS_RUN_NAME>/
```

The notebook performs a real `PutObject` + `HeadObject` preflight before doing
GPU work. The EC2 instance profile therefore needs `s3:PutObject` and
`s3:GetObject` on these prefixes.

Recovery-critical artifacts are uploaded after every epoch by default:

- `last.pt`
- `best.pt` whenever validation Macro Average Precision improves
- `training_history.csv`, including all-label and pathology-only Macro AP,
  per-condition AP, macro AUROC, and the fixed-0.5 NORMAL/pathology
  contradiction rate for every epoch
- `config_snapshot.json`

Before `config_snapshot.json` is created or refreshed, the notebook validates
the fingerprints of any existing snapshot, `best.pt`, and `last.pt`. A reused
run directory with incompatible or internally inconsistent metadata fails
without overwriting those artifacts; select a new `CS_RUN_NAME` to start a
different configuration.

After final evaluation, all checkpoints, predictions, metrics, reports, and
plots are uploaded. `s3_upload_manifest.json` records each artifact's S3 URI,
byte count, and local SHA-256. The remote byte count is checked immediately
after every upload.

Optional overrides, set before starting the Jupyter kernel:

```bash
export CS_S3_BUCKET=cardiosentry-fcmae-758821198776-us-east-1-an
export AWS_REGION=us-east-1
export CS_S3_UPLOAD_EVERY_N_EPOCHS=1
# export CS_NUM_WORKERS=6
# export CS_DATALOADER_START_METHOD=fork  # Linux EC2 default; safe for notebook-defined datasets
# export CS_PROGRESS_UPDATE_EVERY_N_BATCHES=50
# export CS_S3_UPLOAD=0  # only for an intentional local-only run
```

Use a CUDA-ready AWS Deep Learning AMI. If archives and extracted files share
one filesystem, a 300-GiB EBS volume is the practical minimum; 350 GiB gives
safer working room.

## Training settings

- ConvNeXt-V2 Tiny: `convnextv2_tiny.fcmae_ft_in22k_in1k_384`
- Weight provenance: FCMAE pretraining on ImageNet-1K, followed by supervised
  ImageNet-22K and ImageNet-1K fine-tuning at 384 px
- Four independent sigmoid outputs ordered `NORMAL`, `STEMI`, `AF`, `LVH`
- Input: 768×1024, bicubic, ImageNet normalization
- Loss: `BCEWithLogitsLoss` with per-label, training-only `pos_weight`
- Additional penalty:
  `0.20 × p(NORMAL) × (1 - product(1 - p(pathology)))`
- The penalty has no pathology–pathology terms
- AdamW, peak classifier LR `1e-4`, layer-wise LR decay `0.9`, weight decay
  `0.05`, effective batch 32
- Drop-path rate `0.20`
- Automatic largest-safe micro-batch selection on the actual GPU
- Global Python, NumPy, CPU, and CUDA RNGs are reseeded after the
  hardware-dependent micro-batch probe
- Three warm-up epochs followed by cosine decay
- The learning-rate scheduler advances only after GradScaler completes an
  optimizer update; AMP-skipped updates are counted in `training_history.csv`
- Maximum 30 epochs; early stopping patience 6
- Best checkpoint selected only by validation Macro Average Precision (Macro AP)
- `config_snapshot.json` and best-checkpoint metadata record the exact RGB conversion,
  forced `768×1024` bicubic resize, antialiasing, tensor scaling, and ImageNet
  normalization contract required for inference
- Per-label and per-source validation/test reports at both fixed 0.5 and
  validation-tuned thresholds
- Exact-combination metrics for all valid ground-truth patterns, always with
  support and an explicit low-support flag below 30 samples
- Invalid predicted patterns retained and counted, including NORMAL+pathology
  contradictions and `NO_LABEL`

The generated image corpus already carries geometric and photometric variants,
so online augmentation is deliberately disabled. ECG-unsafe horizontal/vertical
flips and random crops are never used.

Training uses four rendered image variants for each training ECG and one for
each validation/test ECG. Splits and leakage checks operate at the ECG/patient
identity level; the notebook also writes `source_label_support_by_split.csv`
using unique ECG records so repeated renders do not inflate source support.

Final combination/source outputs include:

- `combination_metrics.csv` and `combination_summary.csv`
- `invalid_prediction_patterns.csv`
- `source_condition_metrics.csv`, with undefined AP/AUROC left missing and a
  reason such as `N/A — no positive samples`
- `per_source_combination_metrics.csv`, restricted to combinations with at
  least 30 samples within that source
- `per_source_combination_summary.csv`

The raw source identifier `STEMI` corresponds to the ACS/STEMI dataset. Dataset
source remains reporting metadata only and is never passed to the model.

The default is seeded, performance-oriented CUDA execution and is not guaranteed
to be bitwise deterministic. Set `CS_DETERMINISTIC=1` before launching the
kernel for stricter repeatability (with lower throughput), and use independently
named seeds for claims that depend on run-to-run stability.

The selected pretrained weights are licensed CC-BY-NC-4.0; verify suitability
before commercial use.

Official references:

- [timm checkpoint model card](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384)
- [Meta ConvNeXt-V2 training recipes](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/TRAINING.md)
