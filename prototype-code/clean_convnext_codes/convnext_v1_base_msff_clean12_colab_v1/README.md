# ConvNeXt-V1 Base + multi-scale feature fusion on HEEDB clean12 — Google Colab (A100)

Open **CardioSentry_ConvNeXtV1_Base_MSFF_Clean12_Colab.ipynb** in Google Colab on
an A100 runtime and choose Runtime → Run all. The notebook is self-contained:
upload only the `.ipynb`. The Python files here build and check it locally.

- **Dataset:** the same as every clean12 run, read from S3 read-only:
  `s3://cardiosentry/dataset/clean_ecg_signal_images_clean12.tar` and its `.sha256`.
  The patient split must reproduce SHA-256 `4eba80c7…`, so the results use the
  same patients as the V1 Base and V2 Base clean12 runs.
- **Outputs:** Google Drive, `MyDrive/CardioSentry/new_runs/<CS_RUN_NAME>/`.
  The default run name is `convnext_v1_base_<fusion>_clean12_colab_a100_seed42`.

## Where it comes from

| part | source |
|---|---|
| dataset audit, 12 labels, patient split, V1 Base recipe | `../convnext_v1_base_clean12_v1/`. LR 2e-4, warm-up 2, WD 0.02, layer decay 0.925, drop-path 0.2, head init 0.001, EMA 0.9998, 20 epochs. |
| Colab/Drive plumbing, streamed S3 extraction, collapse guard, rare-label sampler, patient-bootstrap CIs | `../convnextv2_base_clean12_colab_v1/`. The generator started as a copy of that one. The V2 model, GRN fix and V2 recipe were swapped back to V1. |
| Residual FPN design | the colleague's `../multi_scale_feature_fusion/CardioSentry_ConvNeXtV1_Base_RFPN_KitV1_Clean_EC2.ipynb` |
| FPN, PAN, BiFPN, the head switch, and the LR-profile fix | new, in `fusion_necks.py` |

The colleague's notebook is **ConvNeXt-V1 Base** (`convnext_base.fb_in22k_ft_in1k_384`),
not V2. It trains on kit-v1 on EC2 and implements only the Residual FPN
(`CS_HYBRID=rfpn|none`).

## Choosing the fusion: `CS_FUSION`

All necks read ConvNeXt stages 2–4. These are strides 8/16/32, with
96×128, 48×64 and 24×32 maps on a 768×1024 sheet, and 256/512/1024 channels
projected to 256 each. All necks use GroupNorm, because micro-batches are small.
Each output level is average-pooled in FP32, and the three vectors are
concatenated.

| `CS_FUSION` | neck | head | new params | compute at 768×1024 |
|---|---|---|---|---|
| `rfpn` (default) | top-down FPN | **residual**: ConvNeXt logits + zero-init Linear(LayerNorm(pooled FPN)) | 2.24 M | +4.5% |
| `fpn` | top-down FPN | **replace**: LayerNorm + Linear on the pooled levels, timm head removed | 2.24 M | +4.5% |
| `pan` | FPN + bottom-up path (PANet) | replace | 4.60 M | +6.4% |
| `bifpn` | 2× BiFPN, fast normalized fusion, depthwise-separable nodes (EfficientDet), 3 levels | replace | 1.02 M | +1.7% |
| `none` | — | plain V1 Base (in-notebook baseline) | 0 | 240.6 GMAC |

- **Residual head with PAN or BiFPN:** `CS_FUSION_HEAD=residual` gives a residual PAN or BiFPN (run names `rpan`, `rbifpn`).
- **Width and depth:** `CS_FUSION_CHANNELS` (default 256) and `CS_BIFPN_REPEATS` (default 2).
- **Run names:** the fusion settings are in the run name and the config fingerprint, so each variant gets its own Drive folder. A resume can never mix two variants.
- **Residual vs replace:**
  - *Residual:* at step 0 the model is exactly the plain V1 Base (checked), so the branch can only add what the data supports.
  - *Replace:* the classifier starts from scratch on newly initialized necks. The collapse guard stays on for that reason.
- **Learning rates:**
  - The backbone keeps **the plain model's per-block LR profile exactly**. In replace mode the timm head is removed, so timm's layer-decay grouping would otherwise give every backbone block a 1/0.925 higher LR; `fusion_param_groups` shifts it back.
  - The neck and the new classifier train at the full head LR. They get weight decay on conv/linear weights only.
- **Initialization:** neck conv weights use timm's ConvNeXt init (trunc-normal 0.02). The colleague's Residual FPN used PyTorch's default init, which has a similar scale.

**Default is `rfpn`**: it starts exactly as the plain V1 Base, so it cannot begin worse than the baseline. In the colleague's kit-v1 runs (one seed) PAN 0.6635, FPN 0.6613 and BiFPN 0.6567 all beat the baseline 0.6476 and are within noise of each other. Set `CS_FUSION=pan|bifpn|fpn` for the others. Add `none` only if you
want the plain baseline with Colab timings. Every run name differs, so the runs
can't collide on Drive. Compare variants with the bootstrap CIs in
`test_bootstrap_ci.csv`, not with point estimates: 3AVB has only 154 test
positives.

## Other differences from the V2 Colab notebook

- **Rare-label sampling is off by default** (`CS_RARE_SAMPLING=0`). A fusion run
  then differs from the plain V1 Base clean12 run only in the neck. Set
  `CS_RARE_SAMPLING=1` for the V2 Colab setting, which uses a new run name.
- No GRN replacement (V1 has no GRN). The collapse guard is kept.
- **`inference.pt`:**
  - **What it records:** `fusion_neck`, `fusion_head`, `fusion_channels`, `bifpn_repeats` and `fusion_stage_indices`.
  - **Keys:** the timm keys sit under `backbone.`, plus `neck.*`, `fuse_norm.*` and `fuse_fc.*`.
  - **Rebuilding:** create the timm model, then wrap it with `ConvNeXtMultiScaleFusion` from `fusion_necks.py`.
  - **The app:** `cardiosentry-application` has no `backend/arch/` module for these models yet. Serving a fusion `inference.pt` there needs one added.

The Colab setup is unchanged: Secrets `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
for the read-only IAM user (`colab_s3_read_policy.json`), Drive mounted by the
config cell, streamed ~104.5 GiB extraction (~113 GiB free disk; each new
runtime re-downloads, about USD 10 of S3 egress), resume from `last.pt` on
Drive, and runtime release at the end. See `../convnextv2_base_clean12_colab_v1/README.md`.

## Runtime

No A100 timing has been measured yet. The necks add 1.7–6.4% compute over
plain V1 Base, and the stride-8 neck maps add some activation memory. The probe
cell picks the micro-batch against 90% of real VRAM, and section 8 prints the
measured epoch estimate before training starts. On an L40S, plain V1 Base on
kit-v1 took ~3,650 s per epoch (compiled). Expect very roughly 30–40 min per
clean12 epoch on an A100, or 10–13 h for 20 epochs. Early stopping may cut that
short.

## Overrides

Set them in a cell before the config cell, e.g. `import os; os.environ["CS_FUSION"] = "bifpn"`.

```text
CS_FUSION=rfpn                 none | fpn | pan | bifpn | rfpn
CS_FUSION_HEAD                 replace | residual (default: residual for rfpn, replace otherwise)
CS_FUSION_CHANNELS=256         CS_BIFPN_REPEATS=2
CS_RUN_NAME                    default convnext_v1_base_<fusion>_clean12_colab_a100_seed<CS_SEED>
CS_SEED=42                     training seed; does not change the split
CS_BASE_LR=2e-4                CS_RARE_SAMPLING=0   CS_BOOTSTRAP_ROUNDS=1000
CS_RELEASE_RUNTIME=1           CS_DRIVE_RUNS_DIR    CS_SPLIT_FILE
CS_TORCH_COMPILE=1  CS_NUM_WORKERS  CS_POS_WEIGHT_MODE=sqrt|full|none  CS_DETERMINISTIC=0
```

## Regeneration and local verification

```bash
python3 make_notebook.py                         # inlines fusion_necks.py
../../../.venv/bin/python validate_notebook.py   # torch, timm 1.0.28, scikit-learn, boto3, psutil
```

All 16 CPU checks pass, verified 2026-10-02:

- **Inherited from the V2 Colab notebook:** the real clean12 split reproduces
  `4eba80c7…`. Also covered: the data pipeline, Drive persistence, streamed
  extraction against a fake S3, compile selection, OOM probe fallback, raw/EMA
  selection, both collapse guards, the sampler, and the bootstrap.
- **Real V1 Base, every variant:** all of `rfpn`, `fpn`, `pan`, `bifpn` and
  `bifpn+residual` were checked for:
  - stock timm keys under `backbone.`;
  - a backbone LR profile identical to plain V1 Base, parameter by parameter;
  - full-LR fusion groups;
  - backward reaching every parameter;
  - residual output exactly equal to the plain model at step 0.

  These checks run at 384×512.
- **The fusion cell:** the notebook's copy is identical to `fusion_necks.py`.
- **The training loop:** the epoch loop, resume and inference export all run
  through a tiny ConvNeXt with a BiFPN neck.
- **BF16 and odd sizes:** the necks run under BF16 autocast and on odd input sizes.

The validator also tries one native 768×1024 forward. It skips it when less
than 4 GiB of RAM is free (`CS_CHECK_NATIVE=1` forces it), and it was skipped
on this 11 GiB machine. Run separately, native 768×1024 forwards of the
`pan`/`bifpn`/`fpn` variants on real V1 Base passed. `torch.compile(fullgraph=True)`
traced every variant without graph breaks.

**Not exercised locally:** the A100 itself, real S3 and Drive I/O, and
full-corpus training.
