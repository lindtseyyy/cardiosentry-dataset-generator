# ConvNeXt-V2 Base training on kit-v1 clean ECG sheets — g6e.2xlarge

Open **CardioSentry_ConvNeXtV2_Base_KitV1_Clean_EC2.ipynb** on a CUDA-ready EC2
Jupyter kernel and run the cells in order. The notebook is self-contained;
the companion Python files are not needed on the training machine.

This notebook is built from `../convnext_v1_base_kitv1_clean_v1/`. It keeps that
notebook's dataset staging, patient audits, 21-label order, source reports,
g6e.2xlarge input-pipeline tuning, compiler benchmark, and checkpoint resume.
The model, the recipe, the GRN implementation, and the collapse guard are new.
No training results or L40S timings are claimed yet; the local checks run on
CPU only.

## Model choice

Use `convnextv2_base.fcmae_ft_in22k_in1k_384`: ConvNeXt-V2 Base with FCMAE
self-supervised pretraining, then supervised ImageNet-22K, then ImageNet-1K at
384 px. It has about 88 M parameters, depths 3/3/27/3, GRN in every block,
and no LayerScale.

- **Tiny** has already been trained on kit-v1 (`../convnextv2_tiny_kitv1_clean_v1/`).
  Base is the next step up in capacity for ~155k sheets with 21 overlapping labels.
- **Large** has ~2.2× Base's FLOPs. At 768×1024 on one L40S it would roughly
  double an already long run and force smaller micro-batches.
- Of the four timm V2 Base checkpoints, the 22K-intermediate 384 px one is the
  strongest, and it was trained at the resolution closest to this notebook's input.

The weights are licensed CC-BY-NC 4.0 (non-commercial).

## Why the recipe is more conservative than V1 Base

An earlier 4-label V2 Base run collapsed at native 768×1024
(`prototype-results/convnext_base_compilations/convnextv2_base@nat768`).
That run used a flat LR of 1e-4, no layer decay, and the IN-1K weights.
In epoch 2, validation macro AUROC fell from 0.93 to 0.50, every label was
predicted positive, and training loss stayed flat at about 1.03 until early
stopping. V1 Base on the identical recipe trained normally (best validation
AUPRC 0.962). This notebook therefore adds:

| Setting | V2 Base (this notebook) | V1 Base notebook |
|---|---|---|
| Peak head LR | **1e-4** (`CS_BASE_LR`) | 2e-4 |
| Warm-up | **3 epochs** | 2 epochs |
| Weight decay | **0.05** (ConvNeXt-V2 default) | 0.02 |
| Layer decay | 0.925 per block (stem ≈ 5.6e-6) | 0.925 per block |
| Classifier init | 0.001 (matters more without LayerScale) | 0.001 |
| Collapse guard | **yes** | no |
| GRN | **mixed-precision replacement** | n/a |

The rest is unchanged: 768×1024 native input, sqrt-pos-weight BCE, drop-path
0.2, ±6% brightness/contrast on the GPU, effective batch 64, EMA 0.9998 with
raw/EMA selection, 20 epochs max with patience 5 from epoch 10, validation-only
F1 thresholds, BF16, fused AdamW, and the eager-vs-compiled benchmark.

For reference, Meta's own V2 Base fine-tune from FCMAE on ImageNet-1K uses
group layer decay 0.6, weight decay 0.05, drop-path 0.1, and 100 epochs with
20 warm-up epochs at batch 1024. Those settings suit MAE weights and ImageNet
augmentation; here the starting point is already supervised, and the
target domain is far from ImageNet.

## Collapse guard

`CONSTANT_PREDICTOR_LOSS` is computed in section 4. It is the weighted BCE of
the best constant prediction given training prevalence and `pos_weight`;
a model that ignores the image cannot go lower.

- **Mid-epoch:** starting after one full epoch of updates, the guard checks
  every 100 micro-batches. It stops training after 5 consecutive windows with
  rolling training BCE at ≥ 97% of the constant-predictor loss.
- **Epoch end:** it stops training if the raw weights' validation macro AUROC
  falls below 0.60.

Both checks raise `TrainingCollapseError` and write `collapse_report.json`
(also uploaded to S3) **before** `last.pt`/`best.pt` are replaced, so the
last healthy epoch is preserved. Recovery: start a new `CS_RUN_NAME` with a
lower LR, e.g. `CS_BASE_LR=5e-5`. Rerunning the same run name resumes the
healthy epoch at the same LR and may collapse again. `CS_COLLAPSE_GUARD=0`
disables both checks.

## Mixed-precision GRN

timm's `GlobalResponseNorm` calls `x.norm()`, which autocasts to FP32. The
result then promotes the full `x * x_n` product and residual to FP32, making
FP32 copies of the largest activations in the network (the 4×-expanded MLP
tensor, 192×256×512 per image at stage 1). fc2 immediately casts them back
to BF16. This is a likely reason the earlier V2 Base run trained 2.5× slower than V1 Base
on an L4 (3.6 vs 9.1 img/s).

`MixedPrecisionGRN` reuses the same `weight`/`bias` parameters, so checkpoint
keys match stock timm. It computes only the B×1×1×C statistic in FP32 and
applies it in the activation dtype. It matches timm exactly in FP32
(outputs and gradients; checked by the validator) and differs in BF16 only by
the rounding fc2 applies anyway. `CS_MIXED_PRECISION_GRN=0` restores timm's
module. The setting is part of the run fingerprint. `inference.pt` loads into
a stock `timm.create_model("convnextv2_base", num_classes=21)`.

## Runtime and storage

Same as the V1 Base notebook: the target is `g6e.2xlarge` (8 vCPU, 64 GiB RAM,
L40S, 450 GB NVMe). The dataset is the same archive:

```text
s3://ecg-thesis-share/generated_images/clean_ecg_signal_images_kit_v1.tar
```

It is ~178 GiB as a TAR and ~182 GiB extracted onto the NVMe instance store.
The micro-batch probe (32/16/8/…), evaluation-batch probe, decoder-worker
tuning (4/6/8), and eager-vs-`torch.compile` benchmark all run before
training and report the projected epoch time. Read that estimate before
starting the long cell. V2 Base costs more per image than V1 Base, and the
compiled mode is more likely to win here because inductor fuses GRN.

## Runs, resume, and exports

Default run name: `convnextv2_base_kitv1_clean_g6e2xlarge_seed42`.

```text
<scratch>/runs/convnextv2_base_kitv1_clean_v1/<CS_RUN_NAME>/
s3://ecg-thesis-share/runs/convnextv2_base_kitv1/<CS_RUN_NAME>/
```

Useful overrides, set before starting the kernel:

```bash
export CS_SEED=42
# export CS_RUN_NAME=convnextv2_base_kitv1_clean_g6e2xlarge_seed43
# export CS_BASE_LR=5e-5               # new run name required (fingerprinted)
# export CS_COLLAPSE_GUARD=0
# export CS_MIXED_PRECISION_GRN=0      # stock timm GRN (fingerprinted)
# export CS_ARCHIVE_DIR=/mnt/archive-volume/kitv1
# export CS_KEEP_LOCAL_TAR=1
# export CS_NUM_WORKERS=6
# export CS_TORCH_COMPILE=0
# export CS_POS_WEIGHT_MODE=sqrt       # full | sqrt | none
# export CS_DETERMINISTIC=1
# export CS_S3_UPLOAD=0
```

Resume, `best.pt`/`last.pt`/`last_intra.pt` semantics, S3 sync, IAM needs, and
the final outputs (`inference.pt`, prediction CSVs, per-label/per-source
metrics, `report.md`, `result.json`) are identical to the V1 Base notebook's
README. `training_history.csv` also records raw and EMA validation macro
AUROC and the constant-predictor loss for every epoch.

## Regeneration and local verification

```bash
python3 make_notebook.py
../../.venv/bin/python validate_notebook.py   # needs torch, timm 1.0.28, sklearn
```

The CPU validator compiles every code cell and runs these checks:

- Real V2 Base: 36 `MixedPrecisionGRN` modules; keys and native 768×1024
  outputs match stock timm; backward pass; optimizer groups (GRN params
  without weight decay).
- `MixedPrecisionGRN` against timm: equal in FP32 (outputs and gradients),
  and it stays BF16 under autocast.
- The constant-predictor loss matches a brute-force optimum.
- Both collapse guards trip, and they do so before any checkpoint is written.
- The inherited checks: data pipeline, worker tuning, compile selection,
  OOM probe fallback, raw/EMA selection, resume after interruption, and
  inference export.

It does not measure GPU memory or throughput, touch S3, or train on the corpus.

References: [timm model card](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k_384),
[ConvNeXt-V2 paper](https://arxiv.org/abs/2301.00808),
[ConvNeXt-V2 training recipes](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/TRAINING.md).
