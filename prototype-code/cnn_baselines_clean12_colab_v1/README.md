# CNN baselines on HEEDB clean12: ResNet-50, DenseNet-121, EfficientNet-B3 (Google Colab, A100)

Open **CardioSentry_CNN_Baselines_Clean12_Colab.ipynb** in Colab. Its metadata
asks for an A100 with High-RAM; check Runtime → Change runtime type. Then
choose Runtime → Run all. The notebook is self-contained: upload only the
`.ipynb`.

The three models train **one after another in one session**. The slow shared
work runs once:

- the package install and Drive mount
- the streamed, SHA-256-checked ~104.5 GiB extraction from Google Drive
- the manifest audit and the patient split (must reproduce `4eba80c7…`)
- the decoder-worker tuning

Each model then gets its own:

- batch probe
- eager-vs-compiled benchmark
- training run with resume
- final evaluation

`torch.compile` cannot be shared, because each network is a different graph.
Each model compiles once, in the `default` mode (no `max-autotune` kernel
search), so this takes minutes per model.

## Recipe: raw performance, the same for every model

The ConvNeXt notebooks in `../clean_convnext_codes/` supplied the plumbing (data,
split, Drive, harness, metrics), not their hyperparameters.

| | |
|---|---|
| weights | timm defaults: `resnet50.a1_in1k`, `densenet121.ra_in1k`, `efficientnet_b3.ra2_in1k`; new 12-way head |
| input | 768×1024 full sheet, each model's own normalization (all three are ImageNet mean/std) |
| optimizer | AdamW, one LR `3e-4` for all layers, weight decay `0.01` on conv/linear weights only |
| schedule | 1-epoch warm-up, cosine to 1%, at most 20 epochs, early stop patience 5 from epoch 10 |
| regularization | none (no dropout, no drop-path); ±6% brightness/contrast |
| batch | effective 64; the **largest** micro-batch under 90% VRAM, because of BatchNorm |
| loss / sampling | sqrt-pos_weight BCE, uniform epochs (as in the ConvNeXt V1 runs) |
| selection / eval | raw-vs-EMA on validation pathology Macro AP; validation-tuned thresholds; one test pass; patient-bootstrap CIs |

Compute per 768×1024 sheet (measured by `validate_notebook.py`): ResNet-50 64.1
GMACs, DenseNet-121 44.4, EfficientNet-B3 15.0. ConvNeXt-V1 Base needs 240.6.

## Outputs

`MyDrive/CardioSentry/new_runs/cnn_baselines_clean12_colab_a100_seed42/`:

- the shared split files
- one complete run folder per model: `resnet50/`, `densenet121/`, `efficientnet_b3/`
- `comparison.csv`, `comparison_per_label_ap.csv`, `comparison_paired_bootstrap.csv`,
  `comparison.md`, `comparison_curves.png`

Every model uses the same 1,000 patient resamples, so the paired file gives the
per-pair difference in test pathology Macro AP with a 95% interval. The same
seed (42) and test patients were used by the ConvNeXt runs, so their
`test_bootstrap_ci.csv` can be compared directly.

## Resume, subsets, failures

- **Resume:** after a disconnect, choose Run all again with the same `CS_RUN_GROUP`.
  - Finished models are skipped. Only their small result files are copied back from Drive.
  - The interrupted model resumes from its last finished epoch.
  - The archive is streamed from Drive again on a new runtime (no cost, only time).
- **Subsets:** `CS_ARCHITECTURES=densenet121,efficientnet_b3` runs only those models.
- **Failures:** a failing model writes `<arch>/failure.json`, and the next model
  still runs. `CS_CONTINUE_ON_ERROR=0` stops instead.
- **Changing the recipe:** a changed recipe is refused under the same group name.
  Use a new `CS_RUN_GROUP`.

Other overrides: `CS_BASE_LR`, `CS_MAX_EPOCHS`, `CS_SEED`,
`CS_MODEL_RESNET50` / `CS_MODEL_DENSENET121` / `CS_MODEL_EFFICIENTNET_B3`,
`CS_TORCH_COMPILE=0`, `CS_TORCH_COMPILE_MODE`, `CS_RARE_SAMPLING=1`,
`CS_RELEASE_RUNTIME=0`, `CS_SPLIT_FILE`.

## Benchmark only: cost of a full run on this GPU

Before the config cell set `os.environ["CS_BENCHMARK_ONLY"] = "1"` (and
optionally `CS_COMPUTE_UNITS_PER_HOUR`, from Colab's Resources panel), then Run
all. The shared stages run as usual. Each model then runs only its batch probe
and eager-vs-compiled benchmark, with no training. Cell 10d writes
`new_runs/cnn_baselines_benchmark_<gpu>/benchmark_comparison.csv` with, per model:

- the micro-batch and images per second
- minutes per epoch
- projected hours (and compute units) at 10 and at 20 epochs, including setup and final evaluation

A TOTAL row adds the runtime's dataset staging. The comparison cell is skipped.

## Dataset: read from Google Drive, not S3

No AWS key or Colab Secret is needed. The archive and its checksum must be at:

```text
MyDrive/CardioSentry/new_dataset/clean_ecg_signal_images_clean12.tar          (112,181,524,480 bytes)
MyDrive/CardioSentry/new_dataset/clean_ecg_signal_images_clean12.tar.sha256   (7ecd013a…)
```

Section 3 streams it into `tar` (never stored on the Colab disk) and checks the
SHA-256 against the sidecar after the last byte.

- `CS_DATASET_READ_MODE=api` (default): 16 parallel ranged downloads through
  the Drive API. Colab asks for **one extra Google sign-in** at section 3;
  choose the account that owns `CardioSentry`. Drive's own SHA-256 of the file
  is compared with the sidecar before downloading.
- `CS_DATASET_READ_MODE=mount`: parallel reads through `/content/drive`, with
  no extra sign-in. Drive's FUSE cache may use extra local disk.
- `CS_DATASET_DRIVE_DIR` (another folder), `CS_DATASET_DRIVE_FILE_ID` (a file
  outside MyDrive), `CS_DATASET_DOWNLOAD_WORKERS`, `CS_DATASET_DOWNLOAD_CHUNK_MIB`.

The split is computed in the notebook and must reproduce `4eba80c7…` (the
shared S3 copy of the split is no longer read).

## Regeneration and local verification

```bash
python3 make_notebook.py
../../.venv/bin/python validate_notebook.py   # ~5 min on CPU
```

The validator runs the notebook's own code on CPU. It covers:

- **Dataset staging:** section 3 as a whole cell, on a small tar placed on a
  fake Drive. It covers API mode (path resolved folder by folder; injected
  503/429/dropped/short ranged reads retried), the skip on re-run with no API
  call, mount mode, a SHA-256 mismatch deleting the partial tree, and refusal
  of a Drive copy whose Drive-side SHA-256 disagrees.
- **Shared cells:** sections 6 and 7 as whole cells (weights, MACs,
  optimizer groups, worker tuning).
- **The real pipeline:** `run_architecture()` end to end for all three real
  architectures, with random weights, on tiny synthetic PNGs and a temporary
  "Drive". It checks these paths:
  - a full run, export and Drive sync
  - the skip after a wiped runtime
  - an interrupted model resuming from Drive's `last.pt`
  - a failing model that does not stop the next one
  - the comparison cell with paired differences
  - benchmark-only mode: no training, and the 10d cost table with TOTAL and units
  - refusal of a changed recipe
- **Native size:** the three models at 768×1024, for the MAC counts.

All checks passed (2026-10-04). They do not cover A100 speed and memory,
`torch.compile`, the real Drive API and mount, or full-corpus training.
