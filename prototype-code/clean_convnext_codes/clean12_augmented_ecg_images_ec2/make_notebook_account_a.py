#!/usr/bin/env python3
"""Build Clean12_Augmented_ECG_Images_EC2_AccountA.ipynb.

Single-account version of Clean12_Augmented_ECG_Images_EC2.ipynb (make_notebook.py):
the EC2 instance runs in Account A, the account that owns the clean12 signal
bucket, and the augmented archive is uploaded back into that bucket. There is
no cross-account grant and no second account.

Every cell is make_notebook.py's (and through it the clean12 render notebook's);
only the account wiring differs: the top cell's setup, the upload prefix and
the Account 1/Account 2 wording.

    python3 make_notebook_account_a.py
"""
from __future__ import annotations

import json
from pathlib import Path

import make_notebook as ec2

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "Clean12_Augmented_ECG_Images_EC2_AccountA.ipynb"
BUCKET = "cardiosentry-fcmae-758821198776-us-east-1-an"
UPLOAD_PREFIX = f"s3://{BUCKET}/CardioSentry/dataset/clean12_augmented/"


def between(text: str, start: str, end: str) -> str:
    """text from the one occurrence of start up to (not including) end."""
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"marker not unique: {start!r} / {end!r}")
    i = text.index(start)
    return text[i:text.index(end, i)]


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"{text.count(old)} occurrences of {old!r}")
    return text.replace(old, new)


INTRO = f"""# Augmented clean12 ECG images on EC2, all in Account A: one paper/photo sheet per record

The single-account version of `Clean12_Augmented_ECG_Images_EC2.ipynb`. It reads the **clean12 signal dataset**, `ECG_500Hz_clean12.tar.gz`: 113,358 native-500 Hz WFDB records in Account A's S3. It writes **one augmented ECG sheet (1024x768 PNG) per record**, so the dataset ends with 113,358 images for 113,358 records. Each image path mirrors its record path (`images/<site>/WFDB/.../<record>.png`), the same layout as the clean12 sheets, so clean and augmented sheets match on `image_path`.

The output tar and its `.sha256` are uploaded to `{UPLOAD_PREFIX}` in the same bucket (`S3_UPLOAD_PREFIX`). After the upload is verified, **the instance terminates itself**. That happens only if every pipeline cell before it finished and no cell in this kernel raised an error (§11).

The images are identical to the two-account notebook's: same kit settings, augmentation, balanced plan and seed (all copied from it when this notebook is built).

__PER_RECORD__
## Setup: data, EC2 and output in Account A

Account A owns `{BUCKET}` (the account id is part of the bucket's name), and the instance runs in Account A too. There is no cross-account grant and no bucket-policy change.

**Files to copy to the instance**, all three from this notebook's folder in the repo, into one folder:
- this notebook;
- `ecg_image_v1_code.tar.gz`;
- `ecg_image_v1_code.tar.gz.sha256`.

`ecg_image_v1_code.tar.gz` is the ECG-image-kit itself: the program that draws the ECG from the signals, plus the wrinkled-paper photographs. If you would rather not copy it, put both kit files next to the signals (`S3_KIT_SOURCE`); §4 downloads them if no local copy is found.

**Instance role.** Create the EC2 role (with its instance profile) `cardiosentry-augment-ec2` with this policy, then launch the instance with it:
```json
{{
  "Version": "2012-10-17",
  "Statement": [
    {{"Sid": "ReadClean12", "Effect": "Allow", "Action": "s3:GetObject",
     "Resource": "arn:aws:s3:::{BUCKET}/CardioSentry/dataset/clean12/*"}},
    {{"Sid": "WriteAugmented", "Effect": "Allow",
     "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:AbortMultipartUpload"],
     "Resource": "arn:aws:s3:::{BUCKET}/CardioSentry/dataset/clean12_augmented/*"}},
    {{"Sid": "TerminateItself", "Effect": "Allow", "Action": "ec2:TerminateInstances",
     "Resource": "arn:aws:ec2:us-east-1:758821198776:instance/*",
     "Condition": {{"ArnEquals": {{"ec2:InstanceProfile": "arn:aws:iam::758821198776:instance-profile/cardiosentry-augment-ec2"}}}}}}
  ]
}}
```
Notes:
- Within one account the role policy is enough; the bucket policy needs no change unless it has an explicit `Deny`.
- If the bucket's default encryption is **SSE-KMS with a customer-managed key**, also allow `kms:Decrypt` and `kms:GenerateDataKey` on that key. The default SSE-S3 encryption needs nothing more.
- The condition limits termination to instances launched with this profile.

**Instance.**
- Type: **c8a.4xlarge** in **us-east-1**, the same region as the data.
- Storage: a **250 GB gp3** volume at 1,000 MB/s.
- Run Jupyter inside `tmux` (or with `nohup`), from the folder holding the three files.
- Run All.

Every stage can be restarted. A rerun reuses:
- the verified archive;
- the extracted tree;
- every sheet already rendered, as recorded in `WORK_DIR/generation_progress.csv`.
"""


def config() -> str:
    text = ec2.CONFIG.replace("__KIT_SETTINGS__\n", ec2.clean12_kit_settings())
    text = replace_once(text, "# Account 1 (read-only, cross-account): the clean12 signal dataset.",
                        "# Account A (owns the bucket and runs this instance): the clean12 signal dataset.")
    return replace_once(text, '# Account 2: where the augmented archive is uploaded.\nS3_UPLOAD_PREFIX = "s3://cardiosentry-heedb/dataset/"',
                        f'# Account A, same bucket: where the augmented archive is uploaded.\nS3_UPLOAD_PREFIX = "{UPLOAD_PREFIX}"')


DOWNLOAD_MD = """## 3. Download, verify and extract the signals

This is the clean12 render notebook's download cell, unchanged:
- it reads the `.sha256` sidecar, downloads the archive and hashes all of it;
- it extracts safely, rejecting links and any path outside the destination;
- a completion marker lets a rerun skip the work.

The first S3 call is where a missing permission shows up. `AccessDenied` here means the instance role lacks the `ReadClean12` statement (top cell), or the bucket's KMS key does not let the role decrypt."""


def build() -> dict:
    md, code = ec2.md, ec2.code
    per_record = between(ec2.INTRO, "**Per record:**", "## Setup:")
    workers = (ec2.WORKERS.replace("__CLEAN_WORKER__", ec2.embedded(ec2.clean12_worker_source()))
               .replace("__AUGMENT_WORKER__", ec2.embedded((HERE / "augment_worker.py").read_text())))
    preflight_md = replace_once(ec2.PREFLIGHT_MD, "**upload:** Account 2 S3", "**upload:** S3")
    preflight = replace_once(ec2.PREFLIGHT, '"S3 upload permission (Account 2)"', '"S3 upload permission"')
    upload_md = replace_once(ec2.UPLOAD_MD, "## 10. Package and upload to Account 2 S3", "## 10. Package and upload to S3")
    notes = replace_once(ec2.NOTES, "rebuild this notebook with `python3 make_notebook.py`",
                         "rebuild this notebook with `python3 make_notebook_account_a.py`")
    cells = [
        md(INTRO.replace("__PER_RECORD__\n", per_record)), code(config()),
        md(ec2.INSTALL_MD), code(ec2.clean12_cell("%pip install") + ec2.INSTALL_EXTRA),
        md(DOWNLOAD_MD), code(ec2.clean12_cell("def split_s3_uri(uri):") + ec2.DOWNLOAD_EXTRA),
        md(ec2.KIT_MD), code(ec2.clean12_cell("KIT_BUNDLE = None") + ec2.KIT_EXTRA),
        md(ec2.RECORDS_MD), code(ec2.RECORDS),
        md(ec2.WORKERS_MD), code(workers),
        md(ec2.PLAN_MD), code(ec2.PLAN),
        md(preflight_md), code(preflight),
        md(ec2.BENCH_MD), code(ec2.BENCH),
        md(ec2.GENERATE_MD), code(ec2.GENERATE),
        md(upload_md), code(ec2.UPLOAD.replace("__UPLOADER__", ec2.uploader_class())),
        md(ec2.TERMINATE_MD), code(ec2.TERMINATE),
        md(notes),
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    NOTEBOOK.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print("wrote", NOTEBOOK)
