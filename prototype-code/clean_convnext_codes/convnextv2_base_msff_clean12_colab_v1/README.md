# ConvNeXt-V2 Base + multi-scale feature fusion on HEEDB clean12 — Google Colab (A100)

Open **CardioSentry_ConvNeXtV2_Base_MSFF_Clean12_Colab.ipynb** in Google Colab on
an A100 runtime and choose Runtime → Run all. The notebook is self-contained:
upload only the `.ipynb`. The Python files here build and check it locally.

- **Dataset:** the same as every clean12 run, read from S3 read-only:
  `s3://cardiosentry/dataset/clean_ecg_signal_images_clean12.tar` and its `.sha256`.
  The patient split must reproduce SHA-256 `4eba80c7…`.
- **Outputs:** Google Drive, `MyDrive/CardioSentry/new_runs/<CS_RUN_NAME>/`.
  The default run name is `convnextv2_base_rfpn_clean12_colab_a100_seed42`.

## Where it comes from

| part | source |
|---|---|
| everything except the neck | `../convnextv2_base_clean12_colab_v1/`, the plain V2 Base Colab notebook, unchanged. That covers the split, the V2 recipe, the mixed-precision GRN, the collapse guard, repeat-factor sampling, the bootstrap CIs and the Colab/Drive plumbing. |
| fusion necks | `fusion_necks.py`, the same file as `../convnext_v1_base_msff_clean12_colab_v1/` |

So:
- **vs the plain V2 Base Colab run:** a fusion run here differs **only in the neck**.
- **vs the V1 fusion notebook with the same `CS_FUSION`:** the fusion run differs in the backbone, together with each notebook's own recipe (V1: LR 2e-4, WD 0.02, uniform epochs; V2: LR 1e-4, WD 0.05, 3-epoch warm-up, rare-label sampling on).

## Choosing the fusion: `CS_FUSION`

All necks read ConvNeXt stages 2–4. These are strides 8/16/32, with 256/512/1024
channels projected to 256 each, and GroupNorm. Each output level is
average-pooled in FP32, and the three vectors are concatenated.

| `CS_FUSION` | neck | head | new params | compute at 768×1024 |
|---|---|---|---|---|
| `rfpn` (default) | top-down FPN | **residual**: V2 logits + zero-init Linear(LayerNorm(pooled FPN)); step 0 = plain V2 | 2.24 M | +4.5% |
| `fpn` | top-down FPN | **replace**: LayerNorm + Linear, timm head removed | 2.24 M | +4.5% |
| `pan` | FPN + bottom-up path (PANet) | replace | 4.60 M | +6.4% |
| `bifpn` | 2× BiFPN, fast normalized fusion | replace | 1.02 M | +1.7% |
| `none` | — | plain V2 Base (same as the plain Colab run) | 0 | 240.6 GMAC (convs/matmuls) |

- **More variants:**
  - `CS_FUSION_HEAD=residual` with `pan` or `bifpn` gives a residual PAN or BiFPN (run names `rpan` / `rbifpn`).
  - `CS_FUSION_CHANNELS` and `CS_BIFPN_REPEATS` change the neck width and depth.
- **Separate runs:** each choice is in the run name and the fingerprint.
- **GRN:** the GRN swap covers exactly the 36 backbone blocks. The neck has no GRN.
- **Learning rates:** the backbone keeps the plain V2 per-block LR profile exactly. The neck and the new head train at the full head LR (1e-4).
- **Replace heads:** the new head uses the same tiny init as the V2 head (0.001). These heads start from a random neck, which is exactly the case the collapse guard is there for.
- **`inference.pt`:** records `fusion_neck`, `fusion_head`, `fusion_channels`, `bifpn_repeats` and `fusion_stage_indices`. The keys are the timm keys under `backbone.`, plus `neck.*`, `fuse_norm.*` and `fuse_fc.*`.
- **The app:** `cardiosentry-application` has no `backend/arch/` module for these models yet.

## Colab setup, runtime, overrides

The Colab setup is the same as `../convnextv2_base_clean12_colab_v1/README.md`:
- **AWS:** read-only IAM user (`colab_s3_read_policy.json`) with keys in Colab Secrets.
- **Dataset:** streamed ~104.5 GiB extraction, needing ~113 GiB free. Every new runtime re-downloads it, about USD 10 of S3 egress.
- **Drive:** resume from `last.pt` on Drive.
- **End of run:** runtime release.

**Speed:** no A100 timing yet. Plain V2 Base was about 10% slower than V1 Base
on an L40S on kit-v1, and the necks add 1.7–6.4%. Expect very roughly 35–45 min
per epoch on an A100, i.e. 6–15 h depending on early stopping. Section 8 prints
the measured estimate before training starts.

```text
CS_FUSION=rfpn                 none | fpn | pan | bifpn | rfpn
CS_FUSION_HEAD                 replace | residual (default: residual for rfpn, replace otherwise)
CS_FUSION_CHANNELS=256         CS_BIFPN_REPEATS=2
CS_RUN_NAME                    default convnextv2_base_<fusion>_clean12_colab_a100_seed<CS_SEED>
CS_SEED=42  CS_BASE_LR=1e-4    CS_RARE_SAMPLING=1   CS_RARE_SAMPLING_THRESHOLD=0.10
CS_BOOTSTRAP_ROUNDS=1000       CS_RELEASE_RUNTIME=1  CS_DRIVE_RUNS_DIR  CS_SPLIT_FILE
CS_TORCH_COMPILE=1  CS_NUM_WORKERS  CS_POS_WEIGHT_MODE=sqrt|full|none  CS_DETERMINISTIC=0
```

## Regeneration and local verification

```bash
python3 make_notebook.py                         # inlines fusion_necks.py
../../.venv/bin/python validate_notebook.py      # torch, timm 1.0.28, scikit-learn, boto3, psutil
```

All 17 CPU checks pass, verified 2026-10-02.

- **Inherited from the plain V2 Colab notebook:** all 16 of its checks, including
  the real clean12 split reproducing `4eba80c7…` and MixedPrecisionGRN
  equivalence.
- **The fusion checks,** on real V2 Base at 384×512 for `rfpn`, `fpn`, `pan`,
  `bifpn` and `bifpn+residual`:
  - stock keys and outputs for the plain model;
  - exactly 36 MixedPrecisionGRN modules inside every wrapper;
  - a backbone LR profile identical to plain V2, parameter by parameter;
  - full-LR fusion groups;
  - backward reaching every parameter;
  - residual output exactly equal to plain V2 at step 0.
- **New check:** BF16 autocast on a GRN-swapped tiny V2, plus odd input sizes.
- **The training loop:** the epoch loop, resume and export run through a tiny
  ConvNeXt-V2 with a BiFPN neck.

The validator's native 768×1024 forward was skipped on this 11 GiB machine, for
lack of free RAM. Run separately, a native 768×1024 forward of V2 Base + `rfpn`
passed, and `torch.compile(fullgraph=True)` traced every neck on a V2 backbone
without graph breaks.

**Not exercised locally:** the A100, real S3 and Drive I/O, and full-corpus training.
