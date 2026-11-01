# ConvNeXt-V1 Base + multi-scale feature fusion on HEEDB clean12 — EC2 `g6e.2xlarge`

Open **CardioSentry_ConvNeXtV1_Base_MSFF_Clean12_EC2.ipynb** on a CUDA-ready EC2
Jupyter kernel (AWS Deep Learning AMI) and run the cells in order. The notebook
is self-contained: upload only the `.ipynb`. The Python files here build and
check it locally.

This is the EC2 twin of `../convnext_v1_base_msff_clean12_colab_v1/`, the Colab
A100 notebook. Everything that affects the model is identical, so EC2 and Colab
results compare directly:
- **model:** V1 Base + fusion neck, `CS_FUSION=rfpn` by default (`fpn`, `pan`, `bifpn`, `none`);
- **recipe:** LR 2e-4, warm-up 2, WD 0.02, layer decay 0.925, 20 epochs, rare sampling off;
- **extras:** the collapse guard and patient-bootstrap CIs;
- **data:** the same dataset and patient split (`4eba80c7…`).

The platform code is from `../convnext_v1_base_clean12_v1/`, the plain V1 Base
EC2 notebook.

## What differs from the Colab notebook

| | Colab notebook | this notebook |
|---|---|---|
| hardware profile | `colab-a100` | `g6e.2xlarge` (L40S 48 GB, 8 vCPUs, 64 GiB RAM) |
| AWS access | read-only IAM user keys in Colab Secrets | the instance's IAM role (`clean12_training_policy.json`) |
| dataset staging | streamed S3 → `tar`, archive never stored | download to NVMe, SHA-256 check, extract per site, archive deleted |
| scratch | Colab disk | NVMe instance store, mounted at `/mnt/nvme` (EBS fallback) |
| run outputs | `MyDrive/CardioSentry/new_runs/<run>/` | `s3://cardiosentry/runs/<run>/` (+ `s3_upload_manifest.json`) |
| mid-epoch snapshot | off | `last_intra.pt` every 15 min for Spot visibility; never used for resume |
| shared split | read only | read; published once if absent (conditional write, never overwritten) |
| decoder workers probed | 4/6/8/10 | 4/6/8 |
| default run name | `convnext_v1_base_rfpn_clean12_colab_a100_seed42` | `convnext_v1_base_rfpn_clean12_g6e2xlarge_seed42` |
| end of run | flushes Drive, releases the runtime | nothing; **stop the instance yourself** |

The dataset is extracted to `/mnt/nvme/dataset` with the same extraction markers
as the plain V1 Base EC2 notebook. So on an instance whose NVMe still holds that
run's dataset, section 3 skips the ~104 GiB download. Run directories are
separate: `/mnt/nvme/runs/convnext_v1_base_msff_clean12_ec2_v1/<run>/`.

## Running it

1. Attach `clean12_training_policy.json` to the instance role. It is the same
   policy as the plain V1 Base notebook's (see that README):
   ```bash
   aws iam put-role-policy --role-name CardioSentryEC2Role \
     --policy-name CardioSentryClean12Training \
     --policy-document file://clean12_training_policy.json
   ```
   For a role in another AWS account, use `cross_account_*.json`. That role
   cannot write `dataset/`, so before the first run either upload
   `../convnext_v1_base_clean12_v1/splits/` to `s3://cardiosentry/dataset/splits/`
   from the bucket's account, or set `CS_SPLIT_PUBLISH=0`. Otherwise section 4b
   computes the split, then fails with AccessDenied when it tries to publish it.
   If the bucket is not in `us-east-1`, set `AWS_REGION`.
2. Choose the variant, then run all cells. Set the variables before starting
   the kernel, or in a cell above the configuration cell:
   ```python
   import os
   os.environ["CS_FUSION"] = "rfpn"     # rfpn | fpn | pan | bifpn | none
   ```
   Each variant has its own run name and S3 prefix, so variants never collide.
3. Read the sample grid and the section 8 benchmark (epoch estimate) before the
   long training cell.
4. **Resume after an interruption** (Spot or a kernel restart): re-run all cells
   with the same `CS_RUN_NAME`. On a fresh machine the cells first re-stage the
   dataset, then pull `last.pt` and the other run files from S3, then continue
   from the last finished epoch.

The other overrides are listed in section 0 of the notebook. They are those of
the Colab notebook plus the EC2 ones:
- `CS_S3_UPLOAD`, `CS_S3_RUNS_PREFIX`, `CS_INTRA_EPOCH_SYNC_MINUTES`;
- `CS_SPLIT_PUBLISH`;
- `CS_KEEP_LOCAL_TAR`, `CS_ARCHIVE_DIR`, `CS_NVME_MOUNT`, `CS_NVME_DEVICE`.

## Runtime

No L40S timing has been measured for the fusion variants yet. On the L40S,
plain V1 Base on kit-v1 took ~3,650 s per compiled epoch. clean12 has ~58% of
those training images, and the necks add 1.7–6.4% compute. That suggests
roughly 35–40 min per epoch, or 12–14 h for 20 epochs, before early stopping.
Section 8 prints the measured estimate before training starts.

## Files

| file | purpose |
|---|---|
| `CardioSentry_ConvNeXtV1_Base_MSFF_Clean12_EC2.ipynb` | the notebook (generated) |
| `make_notebook.py` | generator; inlines `fusion_necks.py` into section 6 |
| `fusion_necks.py` | FPN / PAN / BiFPN / residual head. It is identical to the copies in the other three MSFF folders; change them together. |
| `validate_notebook.py` | CPU checks |
| `clean12_training_policy.json`, `cross_account_*.json` | IAM, copied from `../convnext_v1_base_clean12_v1/` |

The `inference.pt` format is the Colab notebook's. The timm keys sit under
`backbone.`, plus `neck.*`, `fuse_norm.*` and `fuse_fc.*`, with the fusion
settings recorded. `cardiosentry-application` has no loader for it yet.

## Regeneration and local verification

```bash
python3 make_notebook.py
../../../.venv/bin/python validate_notebook.py   # torch, timm 1.0.28, scikit-learn, boto3, psutil
```

**Verified 2026-10-02:**
- **Real data split:** the split from the real clean12 manifest reproduces SHA-256 `4eba80c7…`.
- **EC2 platform checks** (new here; S3 is a fake in-memory client):
  - S3 upload with size check, background upload and join;
  - restoring only missing run files on a fresh machine;
  - the upload manifest;
  - a 403 is refused, never read as a fresh run;
  - archive download with sidecar check, reuse of a verified local copy, and deletion of a bad-hash download;
  - every extraction class through the notebook's `tar` command.
- **Inherited from the Colab validator** (training code is the same as the Colab notebook's):
  - data pipeline and BF16 fusion;
  - compile selection;
  - batch probes;
  - raw/EMA selection;
  - epoch loop, resume and inference export;
  - collapse guards;
  - sampler and bootstrap.

- **Real V1 Base** (`check_base_model`, `check_bf16_fusion`): 87.6 M parameters, stock timm keys, and every fusion variant (rfpn/fpn 2.24 M, pan 4.60 M, bifpn 1.02 M new); one native 768×1024 PAN forward. It was run on its own under a 3 GiB memory cap, because the full validator is OOM-killed on an 11 GiB machine with a browser open.

**Not exercised locally:** the L40S, NVMe mounting, real S3 I/O, and full training.
