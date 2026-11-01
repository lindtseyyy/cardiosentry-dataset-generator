# ConvNeXt-V1 Base training on kit-v1 clean ECG sheets — g6e.2xlarge

Open **CardioSentry_ConvNeXtV1_Base_KitV1_Clean_EC2.ipynb** on a CUDA-ready EC2
Jupyter kernel and run the cells in order. The notebook is self-contained;
neither the reference folder nor these companion Python files are needed on
the training machine.

This is adapted from `../convnextv2_tiny_kitv1_clean_v1/`, retaining its dataset
staging, patient audits, 21-label order, source reports, and checkpoint resume.
The V1 model and training recipe are new. The current notebook specifically
tunes the g6e.2xlarge input pipeline and GPU execution. No training results or
measured L40S speedups are claimed from the local CPU checks.

## Model choice and recipe

Use `convnext_base.fb_in22k_ft_in1k_384`: the original ConvNeXt-V1 Base,
supervised-pretrained on ImageNet-22K and fine-tuned on ImageNet-1K at 384 px.
Its approximately 88 M parameters and deeper backbone provide room to learn
subtle waveform morphology from approximately 155k training sheets. This is
my performance-focused choice within the reference's single-GPU workflow.
The larger model costs more compute; an ECG advantage over Tiny or Small must
be established by validation and repeated seeds.

The native 768×1024 input preserves the complete 3×4 lead layout and rhythm
strip. This model accepts rectangular inputs without position embeddings.
It uses V1 LayerScale and has no V2 Global Response Normalization.

| Setting | Default |
|---|---|
| Outputs | 21 independent logits; same label order as the reference |
| Resolution | 768×1024 H×W throughout; RGB, bicubic, ImageNet normalization |
| Loss | BCE with square root of training negatives/positives as class weights |
| Augmentation | Brightness and contrast ±6% on training only; no spatial transformations |
| Optimizer | Fused CUDA AdamW, peak head LR 2e-4, per-block layer LR decay 0.925 (≈ Meta's 12-group 0.8), weight decay 0.02 |
| Classifier initialization | Weight and bias scale 0.001 |
| Schedule | 2 warm-up epochs, cosine decay to 1% of each group's peak LR |
| Batch | Effective 64; profile micro-batches 32/16/8/4/2/1 and select fastest of first three safe sizes |
| Evaluation batch | Fastest of first three safe forward-only sizes, up to 64 images, with optimizer and EMA resident |
| Host image batches | uint8, pinned memory; 2.25 MiB per native image instead of 9 MiB float32 |
| Preprocessing | Batched float conversion, augmentation, and normalization on GPU |
| Decoder workers | Benchmark 4/6/8 workers; prefetch 4, persistent workers, one CPU thread per worker |
| Regularization | Drop-path 0.2, gradient norm clip 1.0 |
| EMA | Decay ceiling 0.9998 with warm-up; update after successful optimizer steps |
| Training length | Maximum 20 epochs; patience 5, eligible from epoch 10 |
| Selection | Highest validation pathology-only Macro AP across epochs and raw/EMA weights |
| Thresholds | Per-label maximum F1 on validation only; frozen before test |
| Precision | BF16 when supported; FP16 with dynamic loss scaling otherwise |
| Compilation | Compare eager and in-place torch.compile; keep compiled only for ≥2% projected epoch improvement |

Square-root weighting balances rarer findings without the reference's full
inverse-prevalence weighting. Mild photometric changes provide regularization
without changing lead positions or cutting traces. Epoch/row-specific random
draws make augmentation reproducible across worker counts and restarts.
Both raw and EMA validation scores are logged. The test set selects neither
the weight source nor the epoch or thresholds. These settings are a reasoned
starting point; they have not undergone a hyperparameter search.

## Runtime and storage

The `g6e.2xlarge` L40S is the target. That instance
has 8 vCPUs, 64 GiB RAM, and 450 GB instance storage; see the
[AWS specifications](https://aws.amazon.com/ec2/instance-types/g6e/).
Its GPU is advertised as 48 GB; the memory gate uses CUDA's actual reported
capacity, rather than treating that figure as 48 GiB. The probe includes model,
AdamW state, gradients, GPU preprocessing, and EMA and requires 10% allocation
headroom. Both training and inference sizes are checked. Other machines can
use measured smaller settings, but are not this notebook's intended profile.

The input pipeline transfers uint8 images, then applies ImageNet normalization
and stateless brightness/contrast changes on the GPU. This cuts image-transfer
bytes and host image-batch storage by four, while leaving 768×1024 sheet geometry
intact. PNG decoding remains on the CPU; only the seed-derived two-number
augmentation factors travel alongside training images. A /dev/shm check covers
the training and validation prefetch queues. An explicit `CS_NUM_WORKERS`
disables automatic worker selection.

Fused AdamW, BF16, TF32, channels-last, and compiler benchmarking target the
L40S. The main CPU thread and each decoder worker use one PyTorch thread to
avoid oversubscription. These choices follow the
[PyTorch performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide).

The benchmark compares eager and `max-autotune-no-cudagraphs` compilation on
disposable copies using the actual loaders, preprocessing, accumulation, EMA,
and fused optimizer. It records training and evaluation throughput and estimates
**two validation passes** per epoch for raw and EMA. Compilation is retained
only when it meets the memory gate and improves projected epoch time by at
least 2%; failures and slower results retain eager execution. Warm-up and
compilation startup are additional to the steady-state time estimate.
Copies are compiled in place so parameter identities and checkpoint keys stay
ordinary timm keys. EMA is constructed before compilation.

Read the batch, worker, and execution comparison before starting the long
training cell. QA renders one full sheet per label. The setup requires a
CUDA PyTorch 2.1+ environment; the install cell
pins timm 1.0.28 without replacing PyTorch or CUDA.

The same dataset archive is used:

```text
s3://ecg-thesis-share/generated_images/clean_ecg_signal_images_kit_v1.tar
```

Expect approximately 178 GiB for the archive and 182 GiB for extracted PNGs,
plus checkpoints and scratch margin. The notebook detects and mounts EC2
instance storage and reads local images during epochs. Its free-space gate
checks archive and extraction volumes separately when they differ. For a
smaller instance store, mount a sufficiently large archive EBS volume first
and set `CS_ARCHIVE_DIR` to a directory on it. No archive EBS disk is formatted
by the notebook. An EBS-only runtime can use a mounted 600 GiB volume.

The archive is deleted after verified extraction by default. Keep an archive
on persistent EBS with `CS_KEEP_LOCAL_TAR=1` if it should survive NVMe wipes.
The SHA-256 sidecar is checked when available; strict mode requires it.
Without it, staging checks archive byte count and extracted manifests.
The complete corpus is not copied into a raw RGB cache: it would exceed the
450 GB NVMe once the PNG dataset is included. OS page cache uses available RAM;
model downloads, checkpoints, and the Inductor compiler cache use NVMe scratch.

## Runs, resume, and exports

Default run name: `convnext_v1_base_kitv1_clean_g6e2xlarge_seed42`. Artifacts use a new
prefix, separate from the reference model:

```text
<scratch>/runs/convnext_v1_base_kitv1_clean_v1/<CS_RUN_NAME>/
s3://ecg-thesis-share/runs/convnext_v1_base_kitv1/<CS_RUN_NAME>/
```

Useful overrides, set before running the configuration cell:

```bash
export CS_SEED=42
# export CS_RUN_NAME=convnext_v1_base_kitv1_clean_g6e2xlarge_seed43
# export CS_ARCHIVE_DIR=/mnt/archive-volume/kitv1
# export CS_KEEP_LOCAL_TAR=1
# export CS_NUM_WORKERS=6             # fixed worker count; bypasses tuning
# export CS_PREFETCH_FACTOR=4
# export CS_AUTO_TUNE_WORKERS=0
# export CS_TORCH_COMPILE=0           # skip eager/compiled comparison
# export CS_FUSED_ADAMW=0
# export CS_POS_WEIGHT_MODE=sqrt       # full | sqrt | none
# export CS_REQUIRE_TAR_SHA256=1
# export CS_DETERMINISTIC=1
# export CS_S3_UPLOAD=0               # intentional local-only run
```

`CS_SEED` also changes the default run name. Use a new run name for recipe
changes; training identity is fingerprinted. Hardware-derived micro-batch,
evaluation batch, workers, prefetch, fused optimizer, compilation decision,
and AMP dtype are recorded as runtime metadata.
Seeded default mode favors speed and is not bitwise deterministic.
`CS_DETERMINISTIC=1` disables compilation, fused AdamW, TF32, and cuDNN autotuning.
GPU float photometric augmentation changes the preprocessing identity relative
to the original CPU/PIL recipe; its existing run names/checkpoints are not
compatible with this revision. The new default name separates those runs.

`last.pt` resumes at the latest complete epoch, including raw and EMA weights,
EMA update count, optimizer, scheduler, scaler, history, and RNG state.
After interruption, restart the kernel and rerun the notebook with the same
run name; setup restores the last complete epoch. Repeating just the long
training cell in a partially advanced kernel does not restore that state.
`last_intra.pt` supplies mid-epoch visibility only; it never resumes training.
`best.pt` retains full training state plus the validation-selected weight
source. S3 restoration works after loss of local scratch. Required best and
epoch-boundary uploads raise on failure. IAM needs dataset `GetObject` and
run-prefix `GetObject`/`PutObject` access.

After training completes, final evaluation loads the recorded raw or EMA
weights, tunes thresholds on validation, and runs test inference. It writes
`inference.pt`: only selected model weights, label order, preprocessing,
thresholds, best epoch, and fingerprint. It can be loaded with
`torch.load(..., weights_only=True)`. Resume checkpoints remain separate.
The remaining outputs include prediction CSVs, per-label and per-source
metrics, combination reports, learning curves, and `report.md`/`result.json`.
`input_pipeline_benchmark.json` records batch and worker probes;
`benchmark.json` records both execution modes and the selected hardware settings.

## Regeneration and local verification

```bash
python3 make_notebook.py
python3 validate_notebook.py
```

The CPU validator compiles every Python cell and checks real V1 Base native
resolution inference/backward and optimizer grouping. It also exercises
augmentation with persistent workers, validation selection, accumulation,
simulated interruption and resume, EMA state, skipped AMP updates, and the
selected inference export. The revision also checks uint8 normalization and
augmentation parity with tensor transforms, worker tuning, compiler selection
and fallback, and ordinary checkpoint keys after an in-place compiled forward
(using the CPU eager backend for that API check).
Sizing/OOM fallback and full benchmark execution are also checked with a small
CPU model and simulated CUDA memory readings; those are logic checks, not
GPU memory or throughput measurements.
It does not access AWS or mount/format disks.
GPU memory/throughput, S3 staging, and full-corpus training require the target
runtime and are not validated by this script.

Primary model references: [timm model card](https://huggingface.co/timm/convnext_base.fb_in22k_ft_in1k_384)
and [Meta's original ConvNeXt training recipes](https://github.com/facebookresearch/ConvNeXt/blob/main/TRAINING.md).
