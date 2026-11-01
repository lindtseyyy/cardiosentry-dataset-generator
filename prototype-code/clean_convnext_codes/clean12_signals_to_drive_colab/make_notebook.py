#!/usr/bin/env python3
"""Build Copy_Clean12_Signals_To_Drive_Colab.ipynb.

One-off Colab notebook: copies one archive from Account A's S3 into Google
Drive with its .sha256, streaming S3 ranged reads straight into a resumable
Drive upload (nothing stored on the Colab disk), and checks Drive's
sha256Checksum against the S3 sidecar. COPY_WHAT picks the archive:
  * clean12_augmented: s3://cardiosentry-fcmae-758821198776-us-east-1-an/CardioSentry/dataset/clean12_augmented/
    clean_ecg_signal_images_clean12_augmented.tar (~160 GiB, made by
    ../clean12_augmented_ecg_images_ec2/Clean12_Augmented_ECG_Images_EC2_AccountA.ipynb) -> MyDrive/CardioSentry/new_dataset
  * clean12_signals: .../clean12/ECG_500Hz_clean12.tar.gz (~5.5 GB) -> MyDrive/CardioSentry/datasets
The Drive and AWS helpers are taken from ../clean12_augmented_ecg_images_colab/make_notebook.py.

    python3 make_notebook.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "Copy_Clean12_Signals_To_Drive_Colab.ipynb"
_spec = importlib.util.spec_from_file_location(
    "colab_augmented_builder", HERE.parent / "clean12_augmented_ecg_images_colab" / "make_notebook.py")
colab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(colab)

INTRO = """# Copy an archive from S3 to Google Drive

Copies one archive from Account A's S3 bucket into Google Drive. `COPY_WHAT` in §1 picks which:

| `COPY_WHAT` | S3 source (`s3://cardiosentry-fcmae-758821198776-us-east-1-an/CardioSentry/dataset/…`) | Size | Drive destination |
|---|---|---|---|
| `clean12_augmented` (default) | `clean12_augmented/clean_ecg_signal_images_clean12_augmented.tar`, the output of `Clean12_Augmented_ECG_Images_EC2_AccountA.ipynb` | ~160 GiB | `MyDrive/CardioSentry/new_dataset/`, where the clean12 + augmented training notebook reads it |
| `clean12_signals` | `clean12/ECG_500Hz_clean12.tar.gz`, the 113,358 native-500 Hz WFDB records | ~5.5 GB | `MyDrive/CardioSentry/datasets/` |

Next to the archive it writes `<archive>.sha256` (the S3 sidecar, unchanged) and `<archive>.source.json` (S3 URI, size, ETag, SHA-256).
- **No local copy.** S3 byte ranges go straight into a resumable Drive upload, so nothing is stored on the Colab disk.
- **Verified.** Drive's own SHA-256 of the stored file must equal the S3 sidecar. If it does not, the copy is deleted.
- **Cost.** One S3 download of the archive out of AWS, at S3's internet data-transfer rate (about USD 0.09/GB: ~USD 15 for the augmented archive, ~USD 0.50 for the signals). Writing to Drive is free within your storage quota.
- **Safe to rerun.** If the verified archive is already on Drive, nothing is copied again. A copy cut off by a runtime disconnect starts again from the beginning.

**Before you run**
1. The destination folder must exist (or set `DRIVE_DEST_DIR` / `DRIVE_DEST_FOLDER_ID` in §1), and the Drive account needs the archive's size + 1 GiB free.
2. Colab **Secrets** (key icon, notebook access on): `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` of an IAM user in Account A with `s3:GetObject` on `CardioSentry/dataset/clean12_augmented/*` (and on `CardioSentry/dataset/clean12/*` for the signals).
3. A CPU runtime is enough. Runtime > Run all, and sign in at both Google prompts (Drive mount, then Drive API) with the same account. Keep the tab open until §3 prints `Done`."""

CONFIG = r"""# 1. Configuration
import os
from pathlib import Path

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Which archive to copy (table in the top cell).
COPY_WHAT = "clean12_augmented"
_S3_DATASET = "s3://cardiosentry-fcmae-758821198776-us-east-1-an/CardioSentry/dataset/"
S3_SOURCE, DRIVE_DEST_DIR = {
    "clean12_augmented": (_S3_DATASET + "clean12_augmented/clean_ecg_signal_images_clean12_augmented.tar",
                          "/content/drive/MyDrive/CardioSentry/new_dataset"),
    "clean12_signals": (_S3_DATASET + "clean12/ECG_500Hz_clean12.tar.gz",
                        "/content/drive/MyDrive/CardioSentry/datasets"),
}[COPY_WHAT]
S3_SHA256_SOURCE = S3_SOURCE + ".sha256"
ARCHIVE_NAME = S3_SOURCE.rsplit("/", 1)[1]

DRIVE_DEST_FOLDER_ID = ""        # optional: the folder's id from its URL; skips the path lookup
DRIVE_API = "https://www.googleapis.com"
DRIVE_CHUNK_BYTES = 64 * 2**20   # resumable-upload chunk; a multiple of 256 KiB
assert DRIVE_CHUNK_BYTES % (256 * 1024) == 0
print("Copy", S3_SOURCE, "->", f"{DRIVE_DEST_DIR}/{ARCHIVE_NAME}")
"""

INSTALL = """%pip install --quiet boto3 google-auth requests
import hashlib
import json
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import boto3
"""

ACCESS_MD = """## 2. Google Drive and AWS access

Mounts Drive and signs in to the Drive API (same account), resolves `DRIVE_DEST_DIR` to its Drive id, and copies the AWS read key from Colab Secrets. These are the same helpers the augmentation notebook uses."""

ACCESS_TOP = r"""
if IN_COLAB and not Path("/content/drive/MyDrive").is_dir():
    from google.colab import drive
    drive.mount("/content/drive")
drive_http = drive_authorized_session()
_about = drive_request("GET", "/drive/v3/about", params={"fields": "user(emailAddress)"}).json()
dest_folder_id, _how = resolve_drive_folder(DRIVE_DEST_DIR, DRIVE_DEST_FOLDER_ID)
_folder = drive_meta(dest_folder_id)
if not _folder or _folder["mimeType"] != FOLDER_MIME:
    raise RuntimeError(f"Drive id {dest_folder_id} ({_how}) is not a folder this account can see")
if not _folder.get("capabilities", {}).get("canAddChildren", False):
    raise RuntimeError(f"Account {_about['user']['emailAddress']} cannot add files to Drive folder {_folder['name']!r}")
_free = drive_free_bytes()
print(f"Google Drive: {_about['user']['emailAddress']} | destination {DRIVE_DEST_DIR} = folder {_folder['name']!r} "
      f"({dest_folder_id}, via {_how}) | free storage " + ("unlimited" if _free is None else f"{_free / 2**30:,.1f} GiB"))

"""

COPY_MD = """## 3. Copy S3 -> Drive and verify

1. Reads the S3 `.sha256` sidecar and the archive's size and ETag.
2. Uploads the archive to Drive in `DRIVE_CHUNK_BYTES` pieces, each read from S3 by byte range. Every read is pinned to the ETag, so a changed S3 object stops the copy. A dropped connection resumes from the bytes Drive reports it holds.
3. Checks Drive's size and `sha256Checksum` against the sidecar. If Drive has not reported the checksum after 10 minutes, the file is read back from Drive and hashed instead.
4. Writes the `.sha256` sidecar (byte-for-byte the S3 one, so `sha256sum -c` works) and `.source.json` next to it."""

COPY = r"""def split_s3_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected s3://bucket/key, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


s3 = boto3.Session().client("s3")
src_bucket, src_key = split_s3_uri(S3_SOURCE)
sha_bucket, sha_key = split_s3_uri(S3_SHA256_SOURCE)
sidecar_body = s3.get_object(Bucket=sha_bucket, Key=sha_key)["Body"].read()
_fields = sidecar_body.decode("utf-8").strip().split()
if len(_fields) != 2 or _fields[1].lstrip("*") != ARCHIVE_NAME:
    raise ValueError(f"S3 sidecar must name {ARCHIVE_NAME}: {sidecar_body[:200]!r}")
expected_sha256 = _fields[0].lower()
if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
    raise ValueError("S3 sidecar does not contain a valid SHA-256")
_head = s3.head_object(Bucket=src_bucket, Key=src_key)
size, etag = int(_head["ContentLength"]), _head["ETag"]
source_identity = {"uri": S3_SOURCE, "sha256_uri": S3_SHA256_SOURCE, "sha256": expected_sha256,
                   "bytes": size, "etag": etag}
print(f"S3: {size / 2**30:.2f} GiB, ETag {etag}, SHA-256 {expected_sha256}")


def drive_sha256(meta):
    # Drive's sha256Checksum (waits up to 10 min for it), else hash the file read back from Drive.
    for _ in range(20):
        meta = drive_meta(meta["id"]) or meta
        if meta.get("sha256Checksum"):
            return meta["sha256Checksum"], "Drive's sha256Checksum"
        time.sleep(30)
    digest = hashlib.sha256()
    for offset in range(0, size, DRIVE_CHUNK_BYTES):
        digest.update(drive_download(meta["id"], offset, min(DRIVE_CHUNK_BYTES, size - offset)))
    return digest.hexdigest(), "hash of the file read back from Drive"


existing = drive_children(dest_folder_id, ARCHIVE_NAME)
if len(existing) > 1:
    raise RuntimeError(f"{len(existing)} files named {ARCHIVE_NAME} in {DRIVE_DEST_DIR}; keep at most one")
if existing:
    if int(existing[0].get("size", -1)) != size or drive_sha256(existing[0])[0] != expected_sha256:
        raise RuntimeError(f"{DRIVE_DEST_DIR}/{ARCHIVE_NAME} exists but differs from S3. Move or delete it, "
                           "then rerun this cell.")
    print(f"Already on Drive and verified: {DRIVE_DEST_DIR}/{ARCHIVE_NAME}")
else:
    if _free is not None and _free < size + 2**30:
        raise RuntimeError(f"{_free / 2**30:.1f} GiB free on Drive; need {(size + 2**30) / 2**30:.1f} GiB")

    def read_s3(offset, n):
        return s3.get_object(Bucket=src_bucket, Key=src_key, IfMatch=etag,
                             Range=f"bytes={offset}-{offset + n - 1}")["Body"].read()

    t0, _reported = time.monotonic(), [-1]
    def on_commit(committed):
        step = int(10 * committed / size)
        if step > _reported[0]:
            _reported[0] = step
            elapsed = time.monotonic() - t0
            print(f"  {committed / 2**30:.2f} / {size / 2**30:.2f} GiB | {elapsed / 60:.1f} min"
                  + (f" | {committed / 2**20 / elapsed:.0f} MiB/s" if committed and elapsed else ""), flush=True)

    meta = drive_resumable_upload(drive_resumable_start(dest_folder_id, ARCHIVE_NAME, size), size, read_s3,
                                  on_commit=on_commit)
    got, how = drive_sha256(meta)
    if int(meta.get("size", size)) != size or got != expected_sha256:
        drive_delete(meta["id"])
        raise RuntimeError(f"Copy on Drive differs from S3 (sha256 {got} via {how}); it was deleted. Rerun this cell.")
    print(f"Copied and verified in {(time.monotonic() - t0) / 60:.1f} min ({how} = S3 sidecar)")

drive_put_bytes(dest_folder_id, ARCHIVE_NAME + ".sha256", sidecar_body)
drive_put_bytes(dest_folder_id, ARCHIVE_NAME + ".source.json", (json.dumps(
    {**source_identity, "copied_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n").encode())
print(f"Done: {DRIVE_DEST_DIR}/{ARCHIVE_NAME} (+ .sha256, .source.json)")
"""


def build() -> dict:
    md, code = colab.ec2.md, colab.ec2.code
    helpers = colab.between(colab.ACCESS, "import time\nimport requests",
                            'if IN_COLAB and not Path("/content/drive/MyDrive").is_dir():')
    aws = colab.between(colab.ACCESS, "def colab_secret(name):", 'STAGES_DONE.add("access")').replace(
        "AWS read key (Account 1 signals)", "AWS read key")
    cells = [
        md(INTRO),
        code(CONFIG),
        code(INSTALL),
        md(ACCESS_MD), code(helpers.rstrip() + "\n\n" + ACCESS_TOP + aws),
        md(COPY_MD), code(COPY),
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    NOTEBOOK.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print("wrote", NOTEBOOK)
