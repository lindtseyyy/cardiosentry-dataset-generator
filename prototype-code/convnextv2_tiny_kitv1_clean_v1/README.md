# ConvNeXt-V2 Tiny fine-tuning on the kit-v1 clean ECG image corpus

Open `CardioSentry_ConvNeXtV2_Tiny_KitV1_Clean_EC2.ipynb` on the EC2 instance and
run it from top to bottom. The notebook is self-contained;
`make_notebook.py` only regenerates the checked-in notebook after edits:

```bash
python3 make_notebook.py
```

Target machines: AWS `g5.2xlarge` (8 vCPUs, 32 GiB RAM, one NVIDIA A10G 24 GiB)
or `g6e.2xlarge` (8 vCPU, 32 GiB RAM, one NVIDIA L40S 48 GiB). Use a CUDA-ready
AWS Deep Learning AMI. Bulk data lives on the **NVMe instance-store volume**
(detected and mounted automatically); only a small EBS root volume is needed
for code — see Storage below.

## Architecture: S3 → NVMe → DataLoader → GPU

```text
S3 (persistent)                        EC2 (ephemeral NVMe)              GPU
─────────────────                      ───────────────────              ───
generated_images/  ── one-time copy ──▶ /mnt/nvme/dataset  ──▶ DataLoader ──▶ train
   kit-v1 TAR        download+extract    193,924 PNGs, never
   (source of         (markers make       re-downloaded
    truth)             re-runs free)      unless wiped
                                            │
runs/convnext_v2_tiny_kitv1/ ◀── sync ── /mnt/nvme/runs/...            │
   last/best/config/history/log                              local-first saves,
   + final reports/eval outputs                              then S3 upload
```

Training reads **only** from NVMe — images are never streamed from S3 or EBS
during epochs. Checkpoints save locally first (NVMe-fast), then sync to S3:

- `last.pt` — written at every epoch boundary and uploaded then; this is the
  resume point.
- `last_intra.pt` — snapshotted locally and uploaded by a **background thread
  every `CS_INTRA_EPOCH_SYNC_MINUTES` (default 15) mid-epoch** for
  Spot-interruption visibility. Same-key uploads are serialized (an in-flight
  upload blocks a duplicate), and it never drives resume, so a Spot
  interruption redoes at most one epoch from a clean state.
- `best.pt` — uploaded **synchronously the moment a new best validation
  metric lands**, so it survives even an immediate interruption.
- `config_snapshot.json`, `training_history.csv`, `run.log` (JSONL event log),
  benchmark, QA grid, and all final evaluation outputs sync every epoch / at
  completion. No TensorBoard: `training_history.csv` + `run.log` are the logs.
- The dataset is **never** synced back to S3.

Resume order on every (re-)run: 1. detect/mount NVMe → 2. stage dataset from S3
only if missing → 3. pull missing checkpoints/config from S3 → 4. resume from
the newest checkpoint. NVMe may vanish at any stop/terminate/interrupt —
nothing irreplaceable exists only there.

## Dataset

The kit-v1 clean-view corpus is a single archive published by
`prototype-code/clean_ecg_signal_images_ec2/` (real ECG-image-kit render:
3×4 layout + full 10 s Lead II rhythm strip, 25 mm/s, 10 mm/mV, clean red
grid, **no printed ID header**):

```text
s3://ecg-thesis-share/generated_images/
├── clean_ecg_signal_images_kit_v1.tar
└── clean_ecg_signal_images_kit_v1.tar.sha256   (when published; otherwise
    the notebook verifies byte count plus post-extract manifests)
```

Archive top level (`clean_ecg_signal_images_kit_v1/`):

```text
images/train/  images/validation/  images/test/   # one 1024×768 PNG per record
manifest.csv  record_manifest.csv  patient_assignments.csv
split_audit.json  summary.json  README.md
```

Labels live in `record_manifest.csv` (semicolon-separated physician targets).
Expected corpus: **155,183 / 19,405 / 19,336 train/validation/test records**
across 158,796 patients (patient-disjoint 80/10/10), one clean image each
(~193k PNGs, ~0.94 MiB each), and 21 labels:

`Normal ECG, Atrial Fibrillation, Atrial Flutter, Strict STEMI, ST elevation,
ST depression, T-wave inversion, Prior infarct pattern, LVH, RVH, LBBB, RBBB,
First-degree AV block, 2AVB, Third-degree / complete AV block,
QT/QTc prolongation, Sinus bradycardia, Sinus tachycardia, LAFB, LAD, RAD`.

Records with no target finding (~2.5%) are audited and excluded from
training/evaluation, so every training row carries at least one positive.
Normal ECG is exclusive with all pathologies (verified in the audit cell).

There is a single dataset mode — clean views only — so every run trains and
evaluates on the same data identity. The legacy `CS_USE_AUGMENTATIONS` /
`CS_EVAL_STRESS` flags were removed; the notebook fails loudly if they are
still set.

### Why ConvNeXt-V2 Tiny

Tiny (28 M params, `convnextv2_tiny.fcmae_ft_in22k_in1k_384`, drop-path `0.20`)
fits a large micro-batch at 768×1024 on a single 24 GiB A10G with few
accumulation steps, so epochs run ~1.5–2× faster than Base and Spot-hour costs
stay low. The 21-label classifier head is initialized with
`head_init_scale=0.001` per Meta's ConvNeXt-V2 fine-tuning recipe (verified
against `timm` 1.0.28: default head std ≈ 0.02, scaled head std ≈ 2e-5), so
large early gradients don't thrash the pretrained backbone. The trade-off is
capacity on fine-grained ST/T morphology across 21 labels — if validation
plateaus below target, the valid upgrade ladder is Nano → Tiny → Base
(`convnextv2_nano`, `convnextv2_base`, e.g.
`convnextv2_base.fcmae_ft_in22k_in1k_384`). Note `timm` also lists a
`convnextv2_small` architecture name, but it ships no pretrained weights, so
it is not a usable upgrade. All other hyperparameters reuse the tested
`convnextv2_tiny_v1img_v1` harness.

Leak note: unlike the older v1/v2 synthetic sheets, this kit prints no
record-ID header, so the full-sheet geometry is safe (the ID→STEMI leak does
not apply) and no content crop is used.

Suggested workflow with limited time:

1. Run with the default run name `convnextv2_tiny_kitv1_clean_seed42`.
2. **Review checkpoint:** after the download/audit cells, the per-label QA
   cell renders `sample_grid_per_label.png` — one training sheet per label
   (21 images) — saves it to the run directory, and uploads it to S3.
   Inspect the grid and approve it before starting full training.
3. Read the benchmark cell output (time per batch/epoch, total estimate,
   GPU memory) before starting full training.
4. If the estimate fits, run the training cell.

## Benchmark

The benchmark cell runs before full training and uses the actual loaders, on
a throwaway model copy that leaves training state untouched. It reports time
per training/validation batch, projected time per epoch, worst-case total
for 30 epochs, throughput, and peak GPU memory, and saves `benchmark.json`
to the run directory (also uploaded to S3).

## Storage

Staging footprint on scratch: ~178 GiB TAR + ~182 GiB extracted PNGs
concurrently (~360 GiB peak; the archive is deleted after verified extraction
by default). Before copying, the notebook estimates only the *remaining* need
— archive size via S3 HEAD when the tar is absent, plus exact manifest byte
counts for PNGs still missing when manifests are staged (conservative
per-class estimates otherwise) — plus a margin (`CS_SCRATCH_MARGIN_GIB`,
default 15 GiB), and refuses the copy if scratch is short. NVMe
instance-store volumes on g5/g6e families cover this; on an EBS-only machine, provision a **600 GiB gp3 EBS volume** (same-region
EC2 keeps the download fast) and the notebook falls back to it with a warning.
By default the local archive is deleted after verified extraction to free room;
set `KEEP_LOCAL_TAR_AFTER_EXTRACT = True` to keep it.

No manual mounting: the setup cell identifies the instance-store disk by its
`EC2 NVMe Instance Storage` model string (never by device position, so an
attached EBS volume can't be mistaken for scratch), formats it on first use,
and mounts it at `/mnt/nvme`. The setup prints the detected device, mount
point, free space, dataset-copy time, and final dataset path before training.

## Environment overrides

Set any of these before starting the Jupyter kernel:

```bash
# export CARDIOSENTRY_PROJECT_ROOT=/mnt/cardiosentry   # EBS home for code-side files
# export CS_NVME_MOUNT=/mnt/nvme                       # scratch mount point
# export CS_NVME_DEVICE=/dev/nvme1n1                   # explicit disk (skips model-string match; never root)
# export CS_ARCHIVE_DIR=<scratch>/archive              # defaults under scratch
# export CS_EXTRACT_ROOT=<scratch>/dataset
# export CS_RUN_ROOT=<scratch>/runs/convnextv2_tiny_kitv1_clean_v1
export CS_RUN_NAME=convnextv2_tiny_kitv1_clean_seed42
export CS_KEEP_LOCAL_TAR=0           # 1 to keep the ~178 GiB archive after extraction
export CS_REQUIRE_TAR_SHA256=0       # 1 to fail loudly when the archive .sha256 sidecar is missing (research-run strictness; default 0 because no sidecar is published yet)
export CS_SCRATCH_MARGIN_GIB=15      # safety margin on top of the estimated remaining copy need
export CS_INTRA_EPOCH_SYNC_MINUTES=15  # background last.pt sync cadence
export CS_POS_WEIGHT_MODE=full       # full | sqrt | none (ablation; fingerprinted, needs a new run name)
export CS_HEAD_INIT_SCALE=0.001      # classifier init scale (Meta V2 recipe; fingerprinted)
export CS_S3_BUCKET=ecg-thesis-share
export AWS_REGION=us-east-1
export CS_S3_UPLOAD_EVERY_N_EPOCHS=1
# export CS_NUM_WORKERS=6             # 6 leaves 2 vCPUs for loop/S3 sync on 8-vCPU boxes
# export CS_BENCHMARK_TRAIN_BATCHES=30
# export CS_BENCHMARK_VAL_BATCHES=10
# export CS_S3_UPLOAD=0  # only for an intentional local-only run
```

The EC2 IAM role needs `s3:GetObject` on `generated_images/*` (archive
download plus checksum sidecar) and `s3:PutObject` + `s3:GetObject` on
`runs/convnext_v2_tiny_kitv1/*` (run artifacts). Spot instances are safe:
per-epoch + intra-epoch S3 sync plus automatic resume make interruptions cheap.

## Checkpoints and resume

`last.pt` / `best.pt` carry everything needed to resume: model, optimizer,
scheduler and scaler states, current epoch, best validation metric,
best/bad-epoch counters, history, and all RNG states. Each checkpoint records
`epoch_complete`, and resume trusts only epoch-boundary states: a mid-epoch
`last_intra.pt` snapshot is uploaded for Spot visibility but never drives
resume (reloading a partially stepped optimizer/scheduler and repeating the
epoch would double-count batches and shift the LR schedule). If training is
interrupted, the instance stops, or the NVMe volume is wiped, just re-run the
notebook with the same `CS_RUN_NAME`: scratch is re-mounted, the dataset is
re-staged from S3 only if missing (dataset download needs only S3 GetObject
and works with `CS_S3_UPLOAD=0`), missing run files are pulled back from S3
automatically, and training continues from the newest epoch-boundary checkpoint.
Only a genuine absent S3 object counts as "fresh" — permission, network, or
throttling errors raise loudly instead of restarting over existing checkpoints.
`config_snapshot.json` splits training identity (fingerprinted: model, head
init, labels, data hashes — archive SHA plus both `record_manifest.csv` and
`manifest.csv` — optimizer, LR, batch, loss incl. `pos_weight_mode`,
seed, …) from runtime metadata (micro-batch/accumulation, benchmark timings,
S3 layout — recorded, not fingerprinted), so a run can resume across hardware
(e.g. an A10G micro-batch 4×8 checkpoint continuing on an L40S at 8×4 with the
same effective batch 32). A reused run name with a different identity still
fails loudly instead of mixing results. `sample_grid_per_label.png` is uploaded at QA time so the
approval artifact is on S3 before training starts.

## Training settings

Reused from `convnextv2_tiny_v1img_v1`, adapted to 21 labels:

- ConvNeXt-V2 Tiny: `convnextv2_tiny.fcmae_ft_in22k_in1k_384`
- Twenty-one independent sigmoid outputs in `LABELS` order, classifier head
  init scale `0.001` (`CS_HEAD_INIT_SCALE`)
- Input 768×1024 (H×W), bicubic, ImageNet normalization; no online augmentation
- Loss: per-label `BCEWithLogitsLoss` with training-only `pos_weight`
  (`CS_POS_WEIGHT_MODE=full`: negatives/positives; `sqrt` and `none`
  available as fingerprinted ablations under new run names) and
  no auxiliary penalty (`CONTRADICTION_LAMBDA = 0.0`; the noisy-OR term is
  reserved for a later ablation)
- AdamW, peak head LR `1e-4`, layer-wise LR decay `0.9`, weight decay
  `0.05`, gradient clip `1.0`, effective batch 32, drop-path `0.20`
- Automatic largest-safe micro-batch probe with RNG reseed afterwards
- `timm==1.0.28` pinned (the verified implementation); exact `torch` /
  `torchvision` / dependency versions are recorded in the run config with a
  drift warning on Spot resume, so a replacement machine never trains
  silently on different code
- 3 warm-up epochs, cosine decay, at most 30 epochs, patience 6
- Checkpoint selection and early stopping use only the validation
  pathology-only Macro AP (20 pathologies; Normal ECG detected but never
  selecting); the 21-label Macro AP is tracked as a secondary metric
- Fixed-0.5 plus validation-tuned per-label F1 thresholds; per-source and
  source × condition reports; top exact-combination table

The selected pretrained weights are licensed CC-BY-NC-4.0; verify
suitability before commercial use.
