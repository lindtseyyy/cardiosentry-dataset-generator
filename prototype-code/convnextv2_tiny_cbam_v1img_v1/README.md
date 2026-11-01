# ConvNeXt-V2 Tiny + CBAM fine-tuning on the V1 ECG image corpus

Controlled architectural ablation of
`../convnextv2_tiny_v1img_v1/CardioSentry_ConvNeXtV2_Tiny_V1Img_EC2.ipynb`
(the baseline — do not modify it). The only meaningful experimental change is
one standard CBAM module (channel attention → spatial attention) inserted
after the fourth ConvNeXt-V2 feature stage and before the model's final
normalization/head. Dataset, splits, labels, input size, loss, optimizer
recipe, schedule, evaluation protocol, thresholding, checkpoint selection, and
seed are all identical to the baseline.

Open `CardioSentry_ConvNeXtV2_Tiny_CBAM_V1Img_EC2.ipynb` on the EC2 instance
and run it from top to bottom. The notebook is self-contained;
`make_notebook.py` only regenerates the checked-in notebook after edits:

```bash
python3 make_notebook.py
```

Target machine: AWS `g5.2xlarge` (8 vCPUs, 32 GiB RAM, one NVIDIA A10G).
Use a CUDA-ready AWS Deep Learning AMI and a **150 GiB EBS volume**.

## Dataset

Same V1 clean-view corpus as the baseline, published by
`prototype-code/dataset_generation_code/`:

```text
s3://cardiosentry-fcmae-758821198776-us-east-1-an/datasets/V1/
├── cardiosentry_v1_10s_768x1024_clean_only.tar
└── cardiosentry_v1_10s_768x1024_clean_only.tar.sha256
```

The notebook downloads the archive when it is not already on disk, verifies
its SHA-256 against the sidecar, and extracts the metadata plus all clean
images. Expected corpus: 24,881 / 3,115 / 3,113 train/val/test records with
one clean image each (31,109 images total) and 12 labels
(`NORM AF IAVB LBBB RBBB PAC PVC LAFB LAE TInv LQT PRWP`).

There is a single dataset mode — clean views only — so every run trains and
evaluates on the same data identity. The legacy `CS_USE_AUGMENTATIONS` /
`CS_EVAL_STRESS` flags were removed; the notebook fails loudly if they are
still set.

Suggested workflow with limited time:

1. Run with the default run name `convnextv2_tiny_cbam_v1img_clean_seed42`.
2. Read the benchmark cell output (time per batch/epoch, total estimate,
   GPU memory) before starting full training.
3. If the estimate fits, run the training cell.

## What differs from the baseline

- Architecture: `ConvNeXtV2WithCBAM` wraps the same
  `convnextv2_tiny.fcmae_ft_in22k_in1k_384` backbone (same upstream
  pretrained weights; CBAM newly initialized). Forward pass: stem → stages
  → CBAM → `norm_pre` → head. The timm backbone is retained untouched as
  `model.backbone`, so Grad-CAM hooks on `model.backbone.stages[-1]` work
  exactly as on the baseline.
- CBAM: exactly one module. Channel attention from average- and
  max-pooled descriptors through a shared two-layer 1×1-conv MLP
  (reduction 16, ReLU, no normalization); spatial attention from
  concatenated channel mean/max through one 7×7 convolution (padding 3,
  no BatchNorm — safe for small micro-batches); sigmoid gating;
  channel → spatial order.
- Wrapper-parity test: before training, the notebook proves the manual
  stem → stages → `norm_pre` → head path reproduces the original timm
  forward exactly when CBAM is bypassed (`torch.allclose`).
- Environment control: installed torch/torchvision/timm versions are
  recorded in `config_snapshot.json` (part of the fingerprint) and
  `result.json`. Optional `CS_EXPECT_*_VERSION` variables fail the run
  before training when the environment differs from the baseline run;
  `CS_EXPECT_MICRO_BATCH` warns when physical batching differs.
- Optimizer: the backbone keeps the exact baseline layer-decay grouping
  (the timm helper runs on the underlying backbone); CBAM parameters are
  appended explicitly at the final-stage LR scale with matching
  decay/no-decay treatment. Coverage is asserted: every trainable parameter
  exactly once, backbone + CBAM + classifier verified separately.
- Run identity: local default
  `runs/convnextv2_tiny_cbam_v1img_v1/convnextv2_tiny_cbam_v1img_clean_seed42`
  and S3 destination
  `runs/convnext_v2_tiny_V1/convnextv2_tiny_cbam_v1img_clean_seed42/`.
  The config fingerprint includes the attention settings, so baseline and
  CBAM checkpoints/configs are mutually incompatible and can never resume
  each other.
- Pre-training sanity checks: full architecture print, CBAM parameter
  count, programmatic stage-4 feature-shape check, output-shape/finiteness
  checks, CBAM dimension-preservation check, per-parameter CBAM gradient
  smoke test, and optimizer coverage verification. The micro-batch probe
  builds the CBAM model, so its VRAM estimate reflects this architecture.
- Extra outputs: attention settings, CBAM/total parameter counts, and
  training-efficiency summaries in `result.json`/`report.md`, plus a
  qualitative `attention_viz/` overlay set saved after all metrics are
  frozen (sanity check only — not causal evidence). Overlays use
  primary-metric **validation** images for hypothesis-relevant labels
  (PAC, PVC, LAE, TInv, PRWP, LBBB, RBBB), are labeled as a shared
  **spatial gate** (never class-specific), and ship with raw
  min/max/mean/std gate statistics.
- Logging fix: `learning_rate_head` is read from the classifier's
  optimizer group explicitly (the appended CBAM groups moved
  `param_groups[-1]`), keeping the column semantically identical to
  the baseline.

Everything else — including the validation pathology-only Macro AP
checkpoint selection, validation-only threshold tuning, and the single
post-selection test evaluation — is unchanged.

## Benchmark

The benchmark cell runs before full training and uses the actual loaders, on
a throwaway CBAM-model copy that leaves training state untouched. It reports
time per training/validation batch, projected time per epoch, worst-case total
for 30 epochs, throughput, and peak GPU memory, and saves `benchmark.json`
to the run directory (also uploaded to S3).

## Environment overrides

All local paths default under `/mnt/cardiosentry` (the g5.2xlarge 450 GB
NVMe instance store): `datasets/V1`, `experiments/.../data`, `runs/...`, and
`hf_cache`. Mount the instance store there before starting the kernel:

```bash
sudo mkfs -t ext4 /dev/nvme1n1   # confirm the device name in lsblk first
sudo mkdir -p /mnt/cardiosentry
sudo mount /dev/nvme1n1 /mnt/cardiosentry
sudo chown -R $USER /mnt/cardiosentry
```

The notebook refuses to start when `/mnt/cardiosentry` is not a mounted
volume (so an unmounted default can never silently fill the root disk).
Override with one variable to run anywhere else; per-directory overrides
remain available too. Set any of these before starting the Jupyter kernel:

```bash
# export CARDIOSENTRY_PROJECT_ROOT=/mnt/cardiosentry   # the default
# export CS_ARCHIVE_DIR=/mnt/cardiosentry/datasets/V1
# export CS_EXTRACT_ROOT=/mnt/cardiosentry/experiments/convnextv2_tiny_v1img_v1/data
# export CS_RUN_ROOT=/mnt/cardiosentry/runs/convnextv2_tiny_cbam_v1img_v1
export CS_RUN_NAME=convnextv2_tiny_cbam_v1img_clean_seed42
export CS_KEEP_LOCAL_TAR=0           # 1 to keep the ~28 GiB archive after extraction
export CS_S3_BUCKET=cardiosentry-fcmae-758821198776-us-east-1-an
export AWS_REGION=us-east-1
export CS_S3_UPLOAD_EVERY_N_EPOCHS=1
# export CS_NUM_WORKERS=6
# export CS_BENCHMARK_TRAIN_BATCHES=30
# export CS_BENCHMARK_VAL_BATCHES=10
# export CS_S3_UPLOAD=0  # only for an intentional local-only run
# Optional ablation controls (copy from the baseline run's artifacts):
# export CS_EXPECT_TORCH_VERSION=<baseline result.json environment.pytorch>
# export CS_EXPECT_TORCHVISION_VERSION=<baseline torchvision, e.g. via pip show>
# export CS_EXPECT_TIMM_VERSION=<baseline result.json environment.timm>
# export CS_EXPECT_MICRO_BATCH=<baseline benchmark.json micro_batch_size>
```

## Aligning the environment with the baseline

The install specification is intentionally identical to the baseline
notebook's, so both runs resolve dependencies the same way. After the
baseline run, copy its torch/timm versions from its `result.json`
`environment` section (plus its torchvision version, e.g. via
`pip show torchvision` on the baseline machine) into the
`CS_EXPECT_*_VERSION` variables above before starting the CBAM run: any
framework mismatch then fails loudly before training instead of silently
weakening the ablation. (This notebook additionally records all three
installed versions in its own `config_snapshot.json` fingerprint and
`result.json`, so future runs can compare directly.) If versions differ,
recreate the baseline environment (same AMI + kernel + install cell)
rather than overriding the check. `CS_EXPECT_MICRO_BATCH` (from the
baseline's `benchmark.json`) warns when the auto-selected physical
micro-batch differs; identical physical batching is preferable, but a
smaller CBAM micro-batch is acceptable if VRAM requires it — just
document it when comparing results.

## After the first run

Seed 42 vs 42 is the correct first screening comparison. If the
pathology-only macro AP gap is small, repeat **both** architectures over
multiple seeds (e.g. 42, 123, 456 — new `CS_RUN_NAME`s) and report
mean ± standard deviation before claiming an architectural effect, since
a small single-run difference can be stochastic training variation.

The dataset extraction directory defaults to the baseline's path, so a
machine that already extracted the V1 corpus reuses it instead of
re-extracting. The EC2 IAM role needs `s3:GetObject` on `datasets/V1/*`
(archive download plus checksum sidecar) and `s3:PutObject` + `s3:GetObject`
on `runs/convnext_v2_tiny_V1/*` (run artifacts).

## Checkpoints and resume

After every completed epoch the notebook uploads to
`s3://cardiosentry-fcmae-758821198776-us-east-1-an/runs/convnext_v2_tiny_V1/<CS_RUN_NAME>/`:

- `last.pt` — full training state after every epoch
- `best.pt` — full training state at the best validation pathology-only Macro AP so far
- `training_history.csv` — train/validation metrics, LR, and timing per epoch
- `config_snapshot.json` — run configuration and data identity

Both checkpoints contain everything needed to resume: model, optimizer,
scheduler and scaler states, current epoch, best validation metric,
best/bad-epoch counters, history, and all RNG states. If training is
interrupted or the instance is stopped, just re-run the notebook with the
same `CS_RUN_NAME`: missing run files are pulled back from S3 automatically
and training continues from the newest checkpoint's next epoch. A reused run
name with a different configuration still fails loudly instead of mixing
results. Baseline (non-CBAM) checkpoints can never resume this run: the
config fingerprint differs and the checkpoint state keys differ
(`backbone.*` / `cbam.*`).

## Training settings

Identical to `convnextv2_tiny_v1img_v1` except the CBAM architecture:

- ConvNeXt-V2 Tiny: `convnextv2_tiny.fcmae_ft_in22k_in1k_384`
- One final-stage CBAM: channel attention (reduction 16) → spatial
  attention (7×7); CBAM channels taken from the backbone `num_features`
- Twelve independent sigmoid outputs in `LABELS` order
- Input 768×1024, bicubic, ImageNet normalization; no online augmentation
- Loss: per-label `BCEWithLogitsLoss` with training-only `pos_weight` and
  no auxiliary penalty (`CONTRADICTION_LAMBDA = 0.0`; the noisy-OR term is
  reserved for a later ablation)
- AdamW, peak head LR `1e-4`, layer-wise LR decay `0.9`, weight decay
  `0.05`, gradient clip `1.0`, effective batch 32, drop-path `0.20`
- Automatic largest-safe micro-batch probe with RNG reseed afterwards
- 3 warm-up epochs, cosine decay, at most 30 epochs, patience 6
- Checkpoint selection and early stopping use only the **primary-subset**
  validation pathology-only Macro AP (conflicting-duplicate rows held out
  of the metric but still predicted and saved); the 12-label Macro AP is
  tracked as a secondary metric
- Fixed-0.5 plus validation-tuned per-label F1 thresholds; per-source and
  source × condition reports; top exact-combination table

The selected pretrained weights are licensed CC-BY-NC-4.0; verify
suitability before commercial use.
