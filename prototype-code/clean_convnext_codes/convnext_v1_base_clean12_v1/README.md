# ConvNeXt-V1 Base training on the HEEDB clean12 ECG sheets — g6e.2xlarge

Open **CardioSentry_ConvNeXtV1_Base_Clean12_EC2.ipynb** on a CUDA-ready EC2
Jupyter kernel and run the cells in order. The notebook is self-contained:
upload only the `.ipynb` to the training machine. The companion Python files
here build and check it locally.

This is adapted from `../convnext_v1_base_kitv1_clean_v1/`. The model, recipe,
input pipeline tuning, compiler benchmark, checkpoint resume, S3 sync, metrics,
and reports are the reference's, unchanged. What changed:

| | kit-v1 reference | this notebook |
|---|---|---|
| Archive | `s3://ecg-thesis-share/generated_images/clean_ecg_signal_images_kit_v1.tar` | `s3://cardiosentry/dataset/clean_ecg_signal_images_clean12.tar` (+ `.sha256`, required by default) |
| Runs | `s3://ecg-thesis-share/runs/convnext_v1_base_kitv1/<run>/` | `s3://cardiosentry/runs/<run>/` |
| Labels | 21 (Normal ECG + 20) | 12: NORM + SB, STACH, LBBB, LQT, AF, AFL, 3AVB, 2AVB, 1AVB, STEMI, STE |
| Image layout | `images/{train,validation,test}/` | `images/{I0001,I0006}/WFDB/...` (unsplit) |
| Split | shipped in the archive | built at training time, by patient (section 4b), then frozen once at `s3://cardiosentry/dataset/splits/` and reused |
| Selection metric | pathology-only Macro AP over 20 labels | pathology-only Macro AP over the 11 non-NORM labels |
| Per-source reports | site | site (I0001 / I0006) |

The images come from `../clean12_ecg_signal_images_ec2/` (1024×768 PNG, kit
3×4 + Lead II rhythm strip, 25 mm/s, 10 mm/mV, no printed ID header). The
records are the 113,358 physician-verified 500 Hz HEEDB records selected in
`dataset-files/HEEDB/clean12_selection_2026-10-01/`.

## Patient-level split (section 4b)

The archive has no split, so the notebook builds one before anything trains:

- **Unit: patient.** Every `patient_key` (HEEDB `BDSPPatientID`) goes to exactly
  one split, with all its ECGs. 12,193 of the 96,574 patients have several
  records (up to 26), so a record-level split would leak. Records without a
  patient ID were already excluded by the selection, no key appears at both
  sites, and there are no duplicate signals (sha256) in the manifest.
- **Method:** multi-label iterative stratification (Sechidis et al., 2011) over
  patient groups, 80/10/10. Visit order and ties use `sha256(seed:patient_key)`,
  so the split depends only on `record_manifest.csv` and `CS_SPLIT_SEED`
  (default 20261002). It does not depend on row order, library versions, or
  the training `CS_SEED`, so every seed and model gets the same split.
- **Computed from all 113,358 records**, not only the rendered ones, so a render
  failure never moves a patient.
- **Checks (the cell raises):**
  - no patient is in two splits (checked three ways);
  - every split is within 1 point of its record share;
  - every label is within 1.5 points of its target share in every split;
  - every label has positives in every split.
- **Frozen:**
  - `patient_split.csv` and `split_audit.json` (method, seed, SHA-256, per-split support) are written to the run directory and uploaded to S3.
  - The split's SHA-256 is part of the training identity, so a run cannot resume on a different split.
  - A run directory that already holds a different split is refused; on a fresh machine it is first restored from S3 and then compared.
  - To reuse a split elsewhere, set `CS_SPLIT_FILE` to a local path or `s3://` URI of a `patient_split.csv`.

### Frozen split, saved once and reused (no second TAR)

The split is saved once as small files next to the dataset. The image archive
is not touched:

```text
s3://cardiosentry/dataset/splits/clean12_patient_split_v1.csv           patient_key, split, site, records   (the authority, 2.3 MB)
s3://cardiosentry/dataset/splits/clean12_patient_split_v1.csv.sha256    sha256sum line for the CSV
s3://cardiosentry/dataset/splits/clean12_patient_split_v1_audit.json    version, seed, split SHA-256, record_manifest SHA-256, per-split support
s3://cardiosentry/dataset/splits/clean12_patient_split_v1_records.csv   record_path, patient_key, site, split   (9.7 MB, for other notebooks)
```

Section 4b picks its split in this order:

1. **`CS_SPLIT_FILE`**, if set.
2. **The shared copy** at `CS_SPLIT_S3_URI`, which defaults to the `.csv` above. Before using it, the notebook checks:
   - the CSV against its `.sha256`;
   - the split SHA-256 against the audit file;
   - that the version and seed match;
   - that it was built for this exact `record_manifest.csv`.
   
   It loads the shared copy even if the split code changes later.
3. **Compute it**, then publish the four files. The patient CSV is published last with a conditional write (`IfNoneMatch="*"`), so a published split is never overwritten. If two runs race, the second stops and asks for a rerun, and the rerun loads the published copy.

The first training run publishes the split automatically. Every later run uses
it, whatever the model or `CS_SEED`.

A copy is also in this folder, `splits/`, built by the same code. Its patient
CSV is byte-identical to what the notebook publishes: sha256 `17f7ab6f…`,
split SHA-256 `4eba80c7…`. To freeze the split in S3 before the first run:

```bash
aws s3 cp splits/ s3://cardiosentry/dataset/splits/ --recursive
```

Opt out with `CS_SPLIT_S3_URI=` (compute each time, publish nothing), or read the
shared copy but never publish with `CS_SPLIT_PUBLISH=0`.

Result on the real manifest (checked locally, SHA-256 `4eba80c71bba27e8…`):

| split | records | patients | 3AVB | 2AVB | STEMI | other labels |
|---|---:|---:|---:|---:|---:|---|
| train | 90,734 | 77,334 | 1,230 | 2,923 | 7,297 | 12,000–13,662 |
| validation | 11,301 | 9,638 | 153 | 365 | 912 | 1,500–1,709 |
| test | 11,323 | 9,602 | 154 | 365 | 912 | 1,500–1,707 |

The worst share error is 0.04 points for records and 0.05 points for any label.
Sites come out at 80/10/10 as well (I0001 59,652 / 7,302 / 7,402; I0006
31,082 / 3,999 / 3,921).

## Runtime and storage

Target: `g6e.2xlarge` (L40S, 8 vCPUs, 64 GiB RAM, 450 GB NVMe instance store),
as in the reference. The archive is ~104 GiB and extracts to ~104 GiB; the
space check wants ~225 GiB free (archive + images + 15 GiB margin) and the
archive is deleted after a verified extraction (`CS_KEEP_LOCAL_TAR=1` keeps it).
With ~90.7k training sheets an epoch has about 58% of the kit-v1 reference's
training images. The benchmark cell (section 8) prints the measured epoch time
and the 20-epoch maximum before the long training cell starts.

## IAM for the training instance role

Add this inline policy to the role on the training instance:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadClean12ImageArchive",
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::cardiosentry/dataset/clean_ecg_signal_images_clean12.tar*"
    },
    {
      "Sid": "ReadWriteFrozenSplits",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::cardiosentry/dataset/splits/*"
    },
    {
      "Sid": "ReadWriteRuns",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"],
      "Resource": "arn:aws:s3:::cardiosentry/runs/*"
    },
    {
      "Sid": "ListDatasetAndRuns",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::cardiosentry",
      "Condition": {"StringLike": {"s3:prefix": ["dataset/*", "runs/*"]}}
    }
  ]
}
```

```bash
aws iam put-role-policy --role-name CardioSentryEC2Role \
  --policy-name CardioSentryClean12Training \
  --policy-document file://clean12_training_policy.json
```

- **No delete permission.** A frozen split and a run's checkpoints can be added
  but not removed by the instance.
- **`s3:ListBucket` is required.** Without it a missing object reads as 403
  instead of 404, and the notebook stops rather than assume a fresh run or a
  missing split.
- **`AbortMultipartUpload`** lets boto3 clean up when a large checkpoint upload
  fails.
- **No KMS permission is needed:** the bucket uses SSE-S3.

The same JSON is in `clean12_training_policy.json`.

### Training from another AWS account

Cross-account access needs both sides:
- **Bucket owner:** add the statements in `cross_account_bucket_policy.json` to the `cardiosentry` bucket policy. Merge them with any existing policy and replace `PARTNER_ACCOUNT_ID` and `PARTNER_ROLE_NAME` with the other account's values.
- **Other account:** attach `cross_account_role_policy.json` to its training role.

What the other account can do:
- read everything under `dataset/`;
- read and write under `runs/`;
- nothing else, and no delete.

It cannot write `dataset/splits/`. So upload `splits/` from the owner account before its first run; its notebook then loads the frozen split instead of publishing one. Keep the bucket's Object Ownership on **Bucket owner enforced**, so the owner account owns and can read the run files the other account uploads.

## Runs and overrides

Default run name: `convnext_v1_base_clean12_g6e2xlarge_seed42`.

```text
<scratch>/runs/convnext_v1_base_clean12_v1/<CS_RUN_NAME>/
s3://cardiosentry/runs/<CS_RUN_NAME>/
```

Set these before running the configuration cell:

```bash
export CS_SEED=42                     # training seed; does not change the split
# export CS_SPLIT_SEED=20261002       # changes the split (use a new run name)
# export CS_SPLIT_FILE=s3://cardiosentry/runs/<run>/patient_split.csv
# export CS_SPLIT_S3_URI=            # no shared frozen split (compute every time)
# export CS_SPLIT_PUBLISH=0          # use the shared split, never publish one
# export CS_S3_RUNS_PREFIX=runs      # runs land at s3://cardiosentry/<prefix>/<CS_RUN_NAME>/
# export CS_RUN_NAME=convnext_v1_base_clean12_g6e2xlarge_seed43
# export CS_ARCHIVE_DIR=/mnt/archive-volume/clean12
# export CS_KEEP_LOCAL_TAR=1
# export CS_NUM_WORKERS=6
# export CS_TORCH_COMPILE=0
# export CS_POS_WEIGHT_MODE=sqrt      # full | sqrt | none
# export CS_REQUIRE_TAR_SHA256=0      # allow the size + manifest fallback
# export CS_DETERMINISTIC=1
# export CS_S3_UPLOAD=0               # intentional local-only run
```

Resume, the inference export, and the output files work as in the reference
README. New run files: `patient_split.csv` (+ `.sha256`), `record_split.csv`, and
`split_audit.json`.

## Regeneration and local verification

```bash
python3 make_notebook.py
python3 validate_notebook.py   # needs timm 1.0.28, torch, scikit-learn (repo .venv works)
```

The validator runs the reference's CPU checks against this notebook with 12
labels:
- real V1 Base inference and backward at 768×1024;
- the uint8 pipeline and augmentation parity;
- compiler and batch selection;
- epoch loop, resume and inference export.

It adds a split check:
- synthetic data: patient-disjoint, independent of row order, changes with the seed;
- the real clean12 manifest, when present: share tolerances.

Separately, sections 4–4c were run end to end on the real `manifest.csv` with
placeholder images and two simulated render failures:
- the split was identical on a rerun;
- a different split was refused in the same run directory;
- `CS_SPLIT_FILE` reuse reproduced the same SHA-256.

The shared split was tested against an in-memory fake S3:
- the first run published the four files;
- the second run loaded and verified them, published nothing, and got the same split;
- a tampered CSV, a seed mismatch and a concurrent publish were each refused.

GPU memory/throughput, S3 staging, and full-corpus training are not exercised
locally.
