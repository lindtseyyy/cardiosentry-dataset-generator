#!/usr/bin/env python3
"""Build Clean12_Augmented_ECG_Images_Colab.ipynb.

Colab version of ../clean12_augmented_ecg_images_ec2/: the same records, kit
settings, stage-4 augmentation and balanced plan, but the archive is written to
Google Drive (new_dataset/clean_ecg_signal_images_clean12_augmented.tar +
.sha256, where the clean12+augmented training notebook reads it) instead of S3.
Everything the two notebooks share is taken from the EC2 builder (and through
it from the clean12 render notebook) at build time: augment_worker.py, the
render worker, the kit settings, the install, download, kit, records and plan
cells.

What differs, because a Colab runtime has ~190 GiB of free disk and sessions
that end:
  * images are rendered in shards (~3.6 GiB); each shard becomes a tar fragment
    uploaded to a Drive .parts/ folder, verified, and deleted locally;
  * finished shards are recorded on Drive, so Run All in a new runtime resumes
    at the first missing shard;
  * after the last shard, one resumable Drive upload streams the fragments back
    from Drive into the single tar (resumable across sessions too); its SHA-256
    is carried across sessions in a saved OpenSSL SHA-256 state and checked
    against Drive's own sha256Checksum;
  * the signals are still read from S3 (Account 1) with a key from Colab
    Secrets; the runtime is released at the end.

    python3 make_notebook.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "Clean12_Augmented_ECG_Images_Colab.ipynb"
EC2_DIR = HERE.parent / "clean12_augmented_ecg_images_ec2"

_spec = importlib.util.spec_from_file_location("ec2_augmented_builder", EC2_DIR / "make_notebook.py")
ec2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ec2)


def between(text: str, start: str, end: str) -> str:
    """text from the one occurrence of start up to (not including) end."""
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"marker not unique: {start!r} / {end!r}")
    i = text.index(start)
    return text[i:text.index(end, i)]


INTRO = """# Augmented clean12 ECG images on Google Colab, saved to Google Drive

The Colab version of `Clean12_Augmented_ECG_Images_EC2.ipynb`. It reads the **clean12 signal dataset**, `ECG_500Hz_clean12.tar.gz`: 113,358 native-500 Hz WFDB records in Account 1's S3. It writes **one augmented ECG sheet (1024x768 PNG) per record**: 113,358 images for 113,358 records. Each image path mirrors its record path (`images/<site>/WFDB/.../<record>.png`), the same layout as the clean12 sheets, so clean and augmented sheets match on `image_path`.

The result goes to **Google Drive**, next to the clean archive, where the clean12 + augmented training notebook reads it:
```text
<DRIVE_DATASET_DIR>/clean_ecg_signal_images_clean12_augmented.tar          ~160 GiB, one tar
<DRIVE_DATASET_DIR>/clean_ecg_signal_images_clean12_augmented.tar.sha256   its SHA-256 (sha256sum format)
```
`DRIVE_DATASET_DIR` defaults to `MyDrive/CardioSentry/new_dataset`. The archive has the same top folder, member paths and metadata files as the EC2 notebook's.

__PER_RECORD__
## How it fits on Colab

| Problem on Colab | What this notebook does |
|---|---|
| ~190 GiB free disk; the PNGs alone are ~160 GiB | Records are rendered in **shards** of ~2,500 (~3.6 GiB). A complete shard is packed into a tar fragment, uploaded to `<archive>.parts/` on Drive and verified, then deleted locally. At most about 4 shards' worth of files are on disk. |
| Sessions end (12-24 h limit, disconnects) | Each finished shard is recorded on Drive. **Run all** in a new runtime downloads the signals again and continues at the first shard that is not on Drive. A lost session costs only the shards that were still rendering. |
| Drive files cannot be appended to | After the last shard, **one resumable Drive upload** streams the fragments back from Drive, followed by the metadata files and the tar end marker, into the single archive. That join resumes across sessions too. Its whole-file SHA-256 is computed while the shards are made, carried from session to session, and checked against Drive's own `sha256Checksum`. |
| CPU count varies by runtime | §1 detects the usable vCPUs, RAM and disk; one render worker per vCPU, capped by RAM. |

**The GPU is not used.** The ECG-image-kit and the augmentation run on the CPU only, so what matters is the number of vCPUs, which §1 prints. On an L4 runtime the GPU sits idle; if a CPU-only runtime offers as many vCPUs, it does the same work for fewer compute units.

**Cost.** Writing to and reading from Drive is free within your storage quota. The only AWS charge is the signal download, ~5.5 GB of S3 egress from Account 1 per new runtime (on the order of USD 0.50 each).

## Before you run (once)

**1. Drive space.** The account you sign in with at the Drive prompts needs free storage for the fragments plus the final archive: about **320 GiB** during the join. With less (but at least ~165 GiB) the join deletes each fragment as soon as it has been copied, see §11. The preflight (§9) checks the quota. Files this notebook creates count against that account, even inside a folder shared by another account.

**2. Kit bundle.** Two files from `prototype-code/dataset_generation_code/` in the repo: `ecg_image_v1_code.tar.gz` and `ecg_image_v1_code.tar.gz.sha256`. It is the ECG-image-kit (code plus the wrinkled-paper photographs), 12 MB. Use one of these:
- **Recommended:** put both files in Drive at `MyDrive/CardioSentry/codes/clean12_augmented_ecg_images_colab/` (`KIT_DRIVE_DIR`). §4 copies them in every new runtime.
- Upload both to `/content` with the Files pane (gone when the runtime is recycled).
- Put both next to the signals in Account 1 (`S3_KIT_SOURCE`); §4 downloads them if no other copy is found.

**3. AWS read key** in Colab **Secrets** (key icon; turn on notebook access for each): `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optionally `AWS_DEFAULT_REGION` (default `us-east-1`). The key needs only:
```json
{"Effect": "Allow", "Action": "s3:GetObject",
 "Resource": "arn:aws:s3:::cardiosentry-fcmae-758821198776-us-east-1-an/CardioSentry/dataset/clean12/*"}
```
An Account 1 key with that policy needs nothing else. A key from another account also needs the same statement in Account 1's bucket policy, with that user as `Principal`.

**4. Runtime.** Runtime > Change runtime type: pick the runtime with the **most vCPUs** you can get (High-RAM if offered). Then **Runtime > Run all**. Sign in at the two Google prompts (Drive mount, then Drive API) **with the same account**, and keep the tab open. A session that ends is fine: reconnect and Run all again.

**Run state on Drive:** `<DRIVE_DATASET_DIR>/clean_ecg_signal_images_clean12_augmented.parts/`. It holds:
- `run_config.json`;
- per shard: `shard_NNNNN_of_NNNNN.tarpart`, `.manifest.csv` and `.json`;
- during the join: `tail.bin` and `join.json`;
- at the end: `compose.json`.

Fragments and the join files are deleted once the archive is verified; the small state files stay.
"""

CONFIG = r"""# 1. Configuration and runtime hardware: edit before running any later cell.
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import shutil
import subprocess
from pathlib import Path

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Account 1 (read-only, cross-account): the clean12 signal dataset.
S3_SIGNAL_PREFIX = "s3://cardiosentry-fcmae-758821198776-us-east-1-an/CardioSentry/dataset/clean12/"
S3_SOURCE = S3_SIGNAL_PREFIX + "ECG_500Hz_clean12.tar.gz"
S3_SHA256_SOURCE = S3_SIGNAL_PREFIX + "ECG_500Hz_clean12.tar.gz.sha256"
# Fallback only: the kit bundle normally comes from /content or KIT_DRIVE_DIR.
S3_KIT_SOURCE = S3_SIGNAL_PREFIX + "ecg_image_v1_code.tar.gz"
S3_KIT_SHA256_SOURCE = S3_SIGNAL_PREFIX + "ecg_image_v1_code.tar.gz.sha256"
KIT_DRIVE_DIR = "/content/drive/MyDrive/CardioSentry/codes/clean12_augmented_ecg_images_colab"

# Google Drive: the folder that receives <basename>.tar (+ .sha256) and the run state
# <basename>.parts/. new_dataset/ is where the clean12 + augmented training notebook
# looks. DRIVE_DATASET_FOLDER_ID (the id in the folder's URL) overrides the path lookup.
DRIVE_DATASET_DIR = "/content/drive/MyDrive/CardioSentry/new_dataset"
DRIVE_DATASET_FOLDER_ID = ""
ARCHIVE_BASENAME = "clean_ecg_signal_images_clean12_augmented"
DRIVE_API = "https://www.googleapis.com"
DRIVE_CHUNK_BYTES = 64 * 2**20   # resumable-upload chunk; a multiple of 256 KiB

WORK_DIR = (Path("/content") if Path("/content").is_dir() else Path.home()) / "ecg_clean12_augment"
LOCAL_ARCHIVE = WORK_DIR / "input" / "ECG_500Hz_clean12.tar.gz"
INPUT_DIR = WORK_DIR / "input" / "ECG_500Hz_clean12"
OUTPUT_DIR = WORK_DIR / ARCHIVE_BASENAME   # metadata files; the archive's top folder
SHARDS_DIR = WORK_DIR / "shards"           # images of the shards not yet on Drive

EXPECTED_RECORDS = 113_358
CONDITIONS = ["SB", "STACH", "LBBB", "LQT", "AF", "AFL",
              "3AVB", "2AVB", "1AVB", "STEMI", "STE", "NORM"]
# Kit/render settings: copied from Clean12_ECG_Signal_Images_EC2.ipynb at build
# time, so every grey master is the master its clean12 sheet was made from.
__KIT_SETTINGS__
# The kit's lead-label glyphs depend on the matplotlib version; 3.11.1 reproduces
# the clean12 sheets pixel for pixel. §2 installs it, §9 checks it.
MATPLOTLIB_VERSION = "3.11.1"
AUGMENT_SEED = 20261023          # plan shuffles + per-record effect draws
EST_IMAGE_MIB = 1.5              # measured 1.13-1.49 MiB per augmented sheet

# Shards: ~2,500 records (~3.6 GiB) each. Fixed for a run: the layout is locked on Drive (§8).
SHARD_RECORDS = 2_500
SHARD_LOOKAHEAD = 2              # shards rendered ahead of the one uploading

# Release the Colab runtime (stop using compute units) once the archive is verified
# on Drive (§12), and only if every pipeline cell completed and no cell raised.
RELEASE_RUNTIME_WHEN_DONE = True
RELEASE_GRACE_SECONDS = 60       # interrupt the kernel within this window to cancel


def detect_cpus():
    # vCPUs this process may use: CPU affinity, capped by a cgroup CPU quota.
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    for quota_file, period_file in (("/sys/fs/cgroup/cpu.max", None),
                                    ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        try:
            if period_file is None:
                quota, period = Path(quota_file).read_text().split()[:2]
            else:
                quota, period = Path(quota_file).read_text().strip(), Path(period_file).read_text().strip()
        except (OSError, ValueError):
            continue
        if quota not in ("max", "-1"):
            cpus = min(cpus, max(1, int(quota) // int(period)))
        break
    return cpus


def detect_ram_gib():
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value = Path(path).read_text().strip()
        except OSError:
            continue
        if value.isdigit():
            total = min(total, int(value))
    return total / 2**30


def describe_cpu():
    model, cores, socket = "unknown", set(), None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, value = (part.strip() for part in line.partition(":"))
            if key == "model name":
                model = value
            elif key == "physical id":
                socket = value
            elif key == "core id":
                cores.add((socket, value))
    except OSError:
        pass
    return model, len(cores) or None


USABLE_CPUS = detect_cpus()
RAM_GIB = detect_ram_gib()
CPU_MODEL, PHYSICAL_CORES = describe_cpu()
GPU_NAME = None
if shutil.which("nvidia-smi"):
    _gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True)
    if _gpu.returncode == 0 and _gpu.stdout.strip():
        GPU_NAME = _gpu.stdout.strip().splitlines()[0]

# One render worker per usable vCPU, capped by RAM (~0.7 GiB per worker + 3 GiB).
# A worker runs one kit process per batch, then augments that batch's masters.
WORKERS = max(1, min(USABLE_CPUS, int((RAM_GIB - 3) / 0.7)))
BATCH_SIZE = 16                  # small batches keep every worker busy at shard ends
MAX_PENDING_BATCHES = 3 * WORKERS   # queued work keeps workers busy while a shard uploads
PROGRESS_EVERY = 500
RETRY_PASSES = 1                 # a failed record is retried once before its shard fails
_shm = Path("/dev/shm")
KIT_SCRATCH_DIR = (_shm / "ecg_clean12_augment_scratch"
                   if os.access(_shm, os.W_OK) and shutil.disk_usage(_shm).free >= WORKERS * 256 * 2**20
                   else WORK_DIR / "kit_scratch")
S3_TRANSFER = dict(max_concurrency=32, multipart_chunksize=64 * 2**20, use_threads=True)

# Safety record for §12. Re-running THIS cell starts a fresh record: both reset.
STAGES_DONE = set()              # each pipeline cell adds its name when it completes
CELL_ERRORS = []                 # every cell that raised in this kernel since this cell
try:
    _ipython = get_ipython()
except NameError:                # plain Python (no IPython): only STAGES_DONE applies
    _ipython = None
if _ipython is not None:
    for _callback in list(_ipython.events.callbacks["post_run_cell"]):
        if getattr(_callback, "__name__", "") == "_record_cell_error":
            _ipython.events.unregister("post_run_cell", _callback)

    def _record_cell_error(result):
        if not result.success:
            error = result.error_in_exec or result.error_before_exec
            CELL_ERRORS.append(f"{type(error).__name__}: {error}"[:300])

    _ipython.events.register("post_run_cell", _record_cell_error)

assert WORKERS >= 1 and BATCH_SIZE >= 1 and MAX_PENDING_BATCHES >= WORKERS and SHARD_LOOKAHEAD >= 0
assert DRIVE_CHUNK_BYTES % (256 * 1024) == 0
WORK_DIR.mkdir(parents=True, exist_ok=True)
SHARDS_DIR.mkdir(parents=True, exist_ok=True)
_disk = shutil.disk_usage(WORK_DIR)
print(f"Runtime: {'Colab' if IN_COLAB else 'not Colab'} | CPU {CPU_MODEL}")
print(f"  usable vCPUs {USABLE_CPUS}" + (f" ({PHYSICAL_CORES} physical cores)" if PHYSICAL_CORES else "")
      + f" | RAM {RAM_GIB:.1f} GiB | disk {_disk.free / 2**30:.0f} GiB free of {_disk.total / 2**30:.0f} GiB"
      + f" | GPU {GPU_NAME or 'none'}" + (" (not used)" if GPU_NAME else ""))
print(f"  -> {WORKERS} render workers, batches of {BATCH_SIZE}; kit scratch {KIT_SCRATCH_DIR}")
print("Working directory:", WORK_DIR)
print("Source:", S3_SOURCE)
print("Archive:", f"{DRIVE_DATASET_DIR}/{ARCHIVE_BASENAME}.tar", "| release runtime when done:", RELEASE_RUNTIME_WHEN_DONE)
"""

INSTALL_MD = """## 2. Install and verify runtime dependencies

The clean12 render notebook's install cell, unchanged, after one Colab addition: matplotlib is pinned to `MATPLOTLIB_VERSION`, because Colab ships an older matplotlib and the kit's lead-label glyphs change between versions. The kit runs as a subprocess of this kernel's Python, so the pinned version is the one it uses even if the kernel had already imported another. `google-auth` and `requests` (for the Drive API) are already on Colab."""

INSTALL_PIN = """%pip install --quiet "matplotlib=={MATPLOTLIB_VERSION}" google-auth requests
"""

ACCESS_MD = """## 3. Google Drive and AWS access

- **Drive:** mounts Drive (prompt 1), then signs in to the Drive API (prompt 2, **same account**). The archive and the run state are written through the API, which is authoritative and resumable; the mount is used only to read the kit and folder ids.
- **Destination folder:** resolves `DRIVE_DATASET_DIR` to its Drive id. The id comes from the mount's `user.drive.id` attribute, or else by walking the path with the API and following shortcuts. `DRIVE_DATASET_FOLDER_ID` overrides both.
- **Account check:** prints the signed-in account and its free storage.
- **AWS:** copies the read key from Colab Secrets and prints its identity, so a wrong key shows up here rather than hours later."""

ACCESS = r"""import time
import requests

FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
DRIVE_FILE_FIELDS = "id,name,mimeType,size,md5Checksum,sha256Checksum,shortcutDetails,capabilities/canAddChildren"


class DriveTransientError(Exception):
    pass


class DriveSessionExpired(Exception):
    pass


def drive_authorized_session():
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    if IN_COLAB:
        from google.colab import auth as colab_auth
        colab_auth.authenticate_user()
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive"])
    return AuthorizedSession(credentials)


def drive_request(method, path, ok=(200,), tries=8, **kwargs):
    # One Drive API call; retries rate limits, server errors and dropped connections.
    url = path if path.startswith("http") else DRIVE_API + path
    error = None
    for attempt in range(tries):
        try:
            response = drive_http.request(method, url, timeout=(30, 600), **kwargs)
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code in ok:
                return response
            retryable = response.status_code in (408, 429, 500, 502, 503, 504) or (
                response.status_code == 403 and "ateLimitExceeded" in response.text)
            if not retryable:
                raise RuntimeError(f"Drive API {method} {url.split('?')[0]} -> {response.status_code}: "
                                   f"{response.text[:500]}")
            error = f"{response.status_code}: {response.text[:200]}"
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"Drive API {method} {url.split('?')[0]} failed {tries} times; last: {error}")


def drive_children(folder_id, name=None):
    query = f"'{folder_id}' in parents and trashed=false"
    if name is not None:
        query += " and name='" + name.replace("\\", "\\\\").replace("'", "\\'") + "'"
    files, token = [], None
    while True:
        params = {"q": query, "fields": f"nextPageToken,files({DRIVE_FILE_FIELDS})", "pageSize": 1000,
                  "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"}
        if token:
            params["pageToken"] = token
        page = drive_request("GET", "/drive/v3/files", params=params).json()
        files += page.get("files", [])
        token = page.get("nextPageToken")
        if not token:
            return files


def drive_meta(file_id):
    response = drive_request("GET", f"/drive/v3/files/{file_id}", ok=(200, 404),
                             params={"fields": DRIVE_FILE_FIELDS, "supportsAllDrives": "true"})
    return response.json() if response.status_code == 200 else None


def drive_download(file_id, start=None, length=None):
    headers = {"Range": f"bytes={start}-{start + length - 1}"} if start is not None else {}
    data = drive_request("GET", f"/drive/v3/files/{file_id}", ok=(200, 206), headers=headers,
                         params={"alt": "media", "supportsAllDrives": "true"}).content
    if length is not None and len(data) != length:
        raise DriveTransientError(f"ranged download returned {len(data)} of {length} bytes")
    return data


def drive_delete(file_id):
    drive_request("DELETE", f"/drive/v3/files/{file_id}", ok=(200, 204, 404),
                  params={"supportsAllDrives": "true"})


def drive_folder(parent_id, name):
    # The one folder called name in parent_id, created if missing.
    hits = [f for f in drive_children(parent_id, name) if f["mimeType"] == FOLDER_MIME]
    if len(hits) > 1:
        raise RuntimeError(f"{len(hits)} folders named {name!r} in Drive folder {parent_id}; keep one")
    if hits:
        return hits[0]["id"]
    return drive_request("POST", "/drive/v3/files", params={"supportsAllDrives": "true", "fields": "id"},
                         json={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]}).json()["id"]


def drive_free_bytes():
    quota = drive_request("GET", "/drive/v3/about", params={"fields": "storageQuota"}).json()["storageQuota"]
    return None if "limit" not in quota else int(quota["limit"]) - int(quota["usage"])


def drive_resumable_start(parent_id, name, size=None):
    # A resumable upload session (valid for a week); size=None leaves the length open.
    headers = {"X-Upload-Content-Type": "application/octet-stream"}
    if size is not None:
        headers["X-Upload-Content-Length"] = str(size)
    response = drive_request("POST", "/upload/drive/v3/files", headers=headers,
                             params={"uploadType": "resumable", "supportsAllDrives": "true",
                                     "fields": DRIVE_FILE_FIELDS},
                             json={"name": name, "parents": [parent_id]})
    return response.headers["Location"]


def _committed(response):
    # Bytes Drive holds after a 308: "Range: bytes=0-N" means N + 1; no header means 0.
    value = response.headers.get("Range")
    return int(value.rsplit("-", 1)[1]) + 1 if value else 0


def drive_resumable_status(uri, total, declared=True):
    # (bytes Drive holds, file metadata if the upload is already complete).
    for attempt in range(12):
        try:
            response = drive_http.put(uri, data=b"", timeout=(30, 120),
                                      headers={"Content-Range": f"bytes */{total if declared else '*'}"})
        except requests.RequestException:
            time.sleep(min(60, 2 ** attempt))
            continue
        if response.status_code in (200, 201):
            return total, response.json()
        if response.status_code == 308:
            return _committed(response), None
        if response.status_code in (404, 410):
            raise DriveSessionExpired(f"upload session expired ({response.status_code})")
        if response.status_code not in (408, 429, 500, 502, 503, 504):
            raise RuntimeError(f"upload status -> {response.status_code}: {response.text[:300]}")
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError("upload status query kept failing")


def drive_resumable_upload(uri, total, read, declared=True, on_commit=None):
    # Send bytes [committed, total) in chunks; read(offset, n) supplies them. Returns file metadata.
    committed, meta = drive_resumable_status(uri, total, declared)
    failures = 0
    while meta is None:
        if on_commit:
            on_commit(committed)
        start = committed
        data = read(start, min(DRIVE_CHUNK_BYTES, total - start))
        end = start + len(data)
        size_field = total if declared or end == total else "*"
        try:
            response = drive_http.put(uri, data=data, timeout=(30, 900),
                                      headers={"Content-Range": f"bytes {start}-{end - 1}/{size_field}"})
            if response.status_code in (200, 201):
                committed, meta = total, response.json()
            elif response.status_code == 308:
                committed, failures = _committed(response), 0
            elif response.status_code in (404, 410):
                raise DriveSessionExpired(f"upload session expired ({response.status_code})")
            elif response.status_code in (408, 429, 500, 502, 503, 504):
                raise DriveTransientError(f"{response.status_code}: {response.text[:200]}")
            else:
                raise RuntimeError(f"upload chunk -> {response.status_code}: {response.text[:300]}")
        except (requests.RequestException, DriveTransientError) as exc:
            failures += 1
            if failures > 12:
                raise RuntimeError(f"upload kept failing at byte {start:,}: {exc}") from exc
            time.sleep(min(60, 2 ** failures))
            committed, meta = drive_resumable_status(uri, total, declared)
    if on_commit:
        on_commit(total)
    return meta


def check_drive_copy(meta, size, sha256_hex, md5_hex=None):
    # Drive's own checksums must equal the local ones.
    if int(meta.get("size", -1)) != size:
        raise RuntimeError(f"Drive file {meta.get('name')} is {meta.get('size')} bytes, expected {size:,}")
    if meta.get("sha256Checksum"):
        if meta["sha256Checksum"] != sha256_hex:
            raise RuntimeError(f"Drive file {meta['name']} sha256 {meta['sha256Checksum']} != local {sha256_hex}")
    elif md5_hex and meta.get("md5Checksum"):
        if meta["md5Checksum"] != md5_hex:
            raise RuntimeError(f"Drive file {meta['name']} md5 {meta['md5Checksum']} != local {md5_hex}")
    else:
        raise RuntimeError(f"Drive returned no checksum for {meta.get('name')}")


def drive_put_bytes(parent_id, name, body):
    # Replace any same-named files in parent_id with one file holding body; verified.
    for stale in drive_children(parent_id, name):
        drive_delete(stale["id"])
    uri = drive_resumable_start(parent_id, name, len(body))
    meta = drive_resumable_upload(uri, len(body), lambda offset, n: body[offset:offset + n])
    check_drive_copy(meta, len(body), hashlib.sha256(body).hexdigest(), hashlib.md5(body).hexdigest())
    return meta


def drive_get_bytes(parent_id, name):
    hits = drive_children(parent_id, name)
    if len(hits) > 1:
        raise RuntimeError(f"{len(hits)} files named {name!r} in Drive folder {parent_id}")
    return drive_download(hits[0]["id"]) if hits else None


def resolve_drive_folder(mounted_path, folder_id=""):
    # Drive id of a mounted folder: explicit id, the mount's xattr, or an API walk from My Drive.
    if folder_id:
        return folder_id, "DRIVE_DATASET_FOLDER_ID"
    try:
        return os.getxattr(mounted_path, "user.drive.id").decode().strip(), "mount attribute"
    except (OSError, AttributeError):
        pass
    parts = Path(mounted_path).relative_to("/content/drive/MyDrive").parts
    parent = "root"
    for part in parts:
        hits = [f for f in drive_children(parent, part) if f["mimeType"] in (FOLDER_MIME, SHORTCUT_MIME)]
        if len(hits) != 1:
            raise RuntimeError(f"{len(hits)} folders named {part!r} on the way to {mounted_path}; create the "
                               "folder (or set DRIVE_DATASET_FOLDER_ID to its id from the folder's URL)")
        parent = hits[0]["shortcutDetails"]["targetId"] if hits[0]["mimeType"] == SHORTCUT_MIME else hits[0]["id"]
    return parent, "Drive API path walk"


if IN_COLAB and not Path("/content/drive/MyDrive").is_dir():
    from google.colab import drive
    drive.mount("/content/drive")
drive_http = drive_authorized_session()
_about = drive_request("GET", "/drive/v3/about", params={"fields": "user(emailAddress)"}).json()
dataset_folder_id, _how = resolve_drive_folder(DRIVE_DATASET_DIR, DRIVE_DATASET_FOLDER_ID)
_folder = drive_meta(dataset_folder_id)
if not _folder or _folder["mimeType"] != FOLDER_MIME:
    raise RuntimeError(f"Drive id {dataset_folder_id} ({_how}) is not a folder this account can see")
if not _folder.get("capabilities", {}).get("canAddChildren", False):
    raise RuntimeError(f"Account {_about['user']['emailAddress']} cannot add files to Drive folder {_folder['name']!r}")
_free = drive_free_bytes()
print(f"Google Drive: {_about['user']['emailAddress']} | destination {DRIVE_DATASET_DIR} = folder "
      f"{_folder['name']!r} ({dataset_folder_id}, via {_how}) | free storage "
      + ("unlimited" if _free is None else f"{_free / 2**30:,.0f} GiB"))


def colab_secret(name):
    # An environment variable wins; otherwise the Colab secret of that name.
    if os.environ.get(name):
        return os.environ[name].strip()
    if not IN_COLAB:
        return None
    try:
        from google.colab import userdata
        value = userdata.get(name)
    except Exception:  # secret absent, or notebook access not granted
        return None
    return value.strip() if value else None


for _name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
    _value = colab_secret(_name)
    if _value:
        os.environ[_name] = _value
os.environ["AWS_DEFAULT_REGION"] = colab_secret("AWS_DEFAULT_REGION") or "us-east-1"
if boto3.Session().get_credentials() is None:
    raise RuntimeError("No AWS credentials. In Colab open Secrets (key icon), add AWS_ACCESS_KEY_ID and "
                       "AWS_SECRET_ACCESS_KEY, turn on notebook access for both, and rerun this cell.")
try:
    _arn = boto3.Session().client("sts").get_caller_identity()["Arn"]
except Exception as exc:
    raise RuntimeError(f"The AWS key was rejected: {type(exc).__name__}: {exc}") from exc
print(f"AWS read key (Account 1 signals): {_arn}")
STAGES_DONE.add("access")
"""

DOWNLOAD_MD = """## 3b. Download, verify and extract the signals (Account 1)

The clean12 render notebook's download cell, unchanged:
- it reads the `.sha256` sidecar, downloads the archive (5.5 GiB) and hashes all of it;
- it extracts safely, rejecting links and any path outside the destination;
- a completion marker lets a rerun in the same runtime skip the work.

A new runtime downloads it again (Colab's disk is wiped). `AccessDenied` here means the read key lacks `s3:GetObject` on the signals (top cell)."""

KIT_MD = """## 4. Kit bundle: locate, verify, extract, patch gate

Colab addition first: if the two kit files are not in `/content`, they are copied from `KIT_DRIVE_DIR`. Then the clean12 render notebook's kit cell, unchanged: it falls back to `S3_KIT_SOURCE` if there is still no local copy, the bundle must match its `.sha256` sidecar, and rendering is refused unless every kit patch marker and the fixed font are present. The wrinkled-paper photographs come from the same bundle."""

KIT_LOCATE = r"""_kit_files = ("ecg_image_v1_code.tar.gz", "ecg_image_v1_code.tar.gz.sha256")
if KIT_DRIVE_DIR and not all((Path.cwd() / name).is_file() for name in _kit_files):
    if all((Path(KIT_DRIVE_DIR) / name).is_file() for name in _kit_files):
        for name in _kit_files:
            shutil.copy2(Path(KIT_DRIVE_DIR) / name, Path.cwd() / name)
        print("Kit bundle copied from", KIT_DRIVE_DIR)
    else:
        print(f"Kit bundle not in {Path.cwd()} or {KIT_DRIVE_DIR}; trying the other locations")

"""

PLAN_MD = """## 7. Balanced plan, run configuration and balance audit

The EC2 notebook's plan cell. `AW.balanced_plan` assigns every record one level of each balanced factor; it is deterministic for a given `AUGMENT_SEED`, so every session reproduces it exactly. It writes into the metadata folder, which goes into the final archive:
- `augmentation_plan.csv`;
- `balance_by_condition.csv` (count of each condition × factor × level);
- `record_manifest.csv` (the clean12 labels, for joins);
- `config.json`.

It stops unless every factor is balanced to ±1 overall and inside every `(site, label set)` stratum. It also defines the pool runner used by the preflight."""

SHARDS_MD = """## 8. Shards and resume state (Drive)

- **Shard k** holds `records[k::N_SHARDS]`, every N-th record in manifest order. Each shard is a representative slice, and the assignment is the same in every session.
- **Layout lock:** `run_config.json` in `<basename>.parts/` records the config hash, the record list and the shard layout. A session with different settings is refused, so one archive never mixes settings.
- **Finished shards:** a shard counts as finished only when its `shard_…json` marker exists. The marker is written last, after Drive's checksum of the fragment and its manifest match the local ones. Finished shards must be 0, 1, 2, … with no gaps, because the archive hash runs through them in order.
- **Fragments:** tar members only, with no end-of-archive blocks, so the fragments concatenate into one tar.
- **Running SHA-256:** Python's `hashlib` objects cannot be saved, so the archive-wide hash uses OpenSSL's `SHA256_*` functions on a raw `SHA256_CTX` buffer. That buffer's bytes go into every marker, and the next session continues from them. The cell first checks this hash against `hashlib`, with a save and restore in the middle.

Shard folders left over from an earlier kernel in this runtime are reused (`progress.csv`), except those already on Drive, which are deleted."""

SHARDS = r"""import base64
import ctypes
import ctypes.util
import io
import re
from collections import deque


class ResumableSHA256:
    # SHA-256 whose state survives the session: OpenSSL SHA256_Init/Update/Final on a
    # raw SHA256_CTX (112 bytes; the buffer has slack), saved as base64.
    CTX_BYTES = 256
    _lib = None

    @classmethod
    def _crypto(cls):
        if cls._lib is None:
            import _hashlib  # noqa: F401  (loads the libcrypto Python itself uses)
            for name in filter(None, (ctypes.util.find_library("crypto"), "libcrypto.so.3", "libcrypto.so")):
                try:
                    lib = ctypes.CDLL(name)
                    lib.SHA256_Init, lib.SHA256_Update, lib.SHA256_Final
                except (OSError, AttributeError):
                    continue
                break
            else:
                raise RuntimeError("OpenSSL libcrypto with SHA256_Init/Update/Final not found")
            lib.SHA256_Init.argtypes = [ctypes.c_void_p]
            lib.SHA256_Update.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
            lib.SHA256_Final.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
            for fn in (lib.SHA256_Init, lib.SHA256_Update, lib.SHA256_Final):
                fn.restype = ctypes.c_int
            cls._lib = lib
        return cls._lib

    def __init__(self, state=None):
        lib = self._crypto()
        self._ctx = ctypes.create_string_buffer(self.CTX_BYTES)
        if state is None:
            if lib.SHA256_Init(self._ctx) != 1:
                raise RuntimeError("SHA256_Init failed")
        else:
            raw = base64.b64decode(state)
            if len(raw) != self.CTX_BYTES:
                raise ValueError("saved SHA-256 state has the wrong size")
            ctypes.memmove(self._ctx, raw, self.CTX_BYTES)

    def update(self, data):
        if not isinstance(data, bytes):
            data = bytes(data)
        if self._lib.SHA256_Update(self._ctx, data, len(data)) != 1:
            raise RuntimeError("SHA256_Update failed")

    def state(self):
        return base64.b64encode(self._ctx.raw).decode()

    def hexdigest(self):
        ctx = ctypes.create_string_buffer(self._ctx.raw, self.CTX_BYTES)   # Final destroys its context
        out = ctypes.create_string_buffer(32)
        if self._lib.SHA256_Final(out, ctx) != 1:
            raise RuntimeError("SHA256_Final failed")
        return out.raw.hex()


_probe = os.urandom(3 * 2**20 + 77)
_check = ResumableSHA256()
_check.update(_probe[:1_000_003])
_check = ResumableSHA256(_check.state())
_check.update(_probe[1_000_003:])
if _check.hexdigest() != hashlib.sha256(_probe).hexdigest():
    raise RuntimeError("Resumable SHA-256 disagrees with hashlib; the archive hash cannot be carried across sessions")
del _probe, _check


def tar_header(arcname, size, mtime):
    # One regular-file member header, as tarfile.TarFile.addfile writes it (PAX format).
    info = tarfile.TarInfo(arcname)
    info.size, info.mtime, info.mode, info.type = size, int(mtime), 0o644, tarfile.REGTYPE
    return info.tobuf(tarfile.PAX_FORMAT, "utf-8", "surrogateescape")


archive_name = f"{ARCHIVE_BASENAME}.tar"
ARCHIVE_TOPDIR = OUTPUT_DIR.name
parts_folder_id = drive_folder(dataset_folder_id, ARCHIVE_BASENAME + ".parts")

N_SHARDS = math.ceil(len(records) / SHARD_RECORDS)
shards = [records[k::N_SHARDS] for k in range(N_SHARDS)]
def shard_name(k):
    return f"shard_{k:05d}_of_{N_SHARDS:05d}"
def shard_dir(k):
    return SHARDS_DIR / shard_name(k)

shard_layout = {
    "archive": archive_name, "archive_topdir": ARCHIVE_TOPDIR, "config_sha256": config_hash,
    "records": len(records),
    "records_sha256": hashlib.sha256("\n".join(r["record_path"] for r in records).encode()).hexdigest(),
    "n_shards": N_SHARDS, "shard_records": SHARD_RECORDS,
    "assignment": "shard k = records[k::n_shards], manifest order", "matplotlib": MATPLOTLIB_VERSION,
}
_remote_layout = drive_get_bytes(parts_folder_id, "run_config.json")
if _remote_layout is None:
    drive_put_bytes(parts_folder_id, "run_config.json", (json.dumps(shard_layout, indent=2) + "\n").encode())
elif json.loads(_remote_layout) != shard_layout:
    _old = json.loads(_remote_layout)
    _diff = {k: (_old.get(k), v) for k, v in shard_layout.items() if _old.get(k) != v}
    raise RuntimeError(f"{ARCHIVE_BASENAME}.parts/run_config.json on Drive holds a run with other settings "
                       f"{_diff} (Drive vs now). Restore those settings, or use a new ARCHIVE_BASENAME.")

_marker_name = re.compile(r"^shard_(\d{5})_of_" + f"{N_SHARDS:05d}" + r"\.json$")


def load_markers():
    # Finished shards from Drive, in order; refuses gaps, duplicates and foreign settings.
    found = {}
    for item in drive_children(parts_folder_id):
        match = _marker_name.match(item["name"])
        if match:
            k = int(match.group(1))
            if k in found:
                raise RuntimeError(f"two {item['name']} files in {ARCHIVE_BASENAME}.parts/; delete one")
            found[k] = json.loads(drive_download(item["id"]))
    order = sorted(found)
    if order != list(range(len(order))):
        raise RuntimeError(f"Finished shards on Drive are not 0..n-1 without gaps: {order[:20]}")
    for k, marker in found.items():
        if (marker["shard"] != k or marker["config_sha256"] != config_hash
                or marker["records"] != len(shards[k])):
            raise RuntimeError(f"{shard_name(k)}.json does not belong to this run")
    return [found[k] for k in order]


def shard_progress(k):
    # Records of shard k already rendered in this runtime: success rows whose image exists.
    done, path = {}, shard_dir(k) / "progress.csv"
    if path.is_file():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if None in row or None in row.values():          # torn line from a killed kernel
                    continue
                if row["status"] == "success" and (shard_dir(k) / row["image_path"]).is_file():
                    done[row["record_path"]] = row
                elif row["status"] != "success":
                    done.pop(row["record_path"], None)
    return done


def write_fragment(k, rows, running):
    # Pack shard k's sheets into a local tar fragment; hash it as it is written.
    path = shard_dir(k) / "fragment.tarpart"
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("wb") as out:
        def emit(data):
            out.write(data)
            sha.update(data)
            md5.update(data)
            running.update(data)

        for row in rows:
            source = shard_dir(k) / row["image_path"]
            data = source.read_bytes()
            emit(tar_header(f"{ARCHIVE_TOPDIR}/{row['image_path']}", len(data), source.stat().st_mtime))
            emit(data)
            if len(data) % 512:
                emit(b"\0" * (512 - len(data) % 512))
    return path, sha.hexdigest(), md5.hexdigest()


def upload_fragment(k, rows, render_seconds):
    # Upload shard k's fragment to Drive and verify it, then write its manifest and marker.
    if len(markers) != k:
        raise RuntimeError(f"{shard_name(k)} is out of order: {len(markers)} shards are finished")
    prev = markers[-1] if markers else None
    running = ResumableSHA256(prev["running_sha256_state"]) if prev else ResumableSHA256()
    path, sha_hex, md5_hex = write_fragment(k, rows, running)
    size = path.stat().st_size
    name = shard_name(k) + ".tarpart"
    for stale in drive_children(parts_folder_id, name):   # left by a session that ended before its marker
        drive_delete(stale["id"])
    t0 = time.monotonic()
    with path.open("rb") as source:
        def read(offset, n):
            source.seek(offset)
            return source.read(n)
        meta = drive_resumable_upload(drive_resumable_start(parts_folder_id, name, size), size, read)
    check_drive_copy(meta, size, sha_hex, md5_hex)
    seconds = time.monotonic() - t0

    manifest = io.StringIO()
    writer = csv.DictWriter(manifest, fieldnames=AW.MANIFEST_FIELDS, restval="", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    manifest_body = manifest.getvalue().encode("utf-8")
    drive_put_bytes(parts_folder_id, shard_name(k) + ".manifest.csv", manifest_body)
    marker = {
        "shard": k, "n_shards": N_SHARDS, "config_sha256": config_hash, "records": len(rows),
        "file_id": meta["id"], "fragment_bytes": size, "fragment_sha256": sha_hex, "fragment_md5": md5_hex,
        "image_bytes": sum(int(row["image_bytes"]) for row in rows),
        "manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
        "running_bytes": (prev["running_bytes"] if prev else 0) + size,
        "running_sha256_state": running.state(),
        "render_seconds": round(render_seconds, 1), "upload_seconds": round(seconds, 1),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    drive_put_bytes(parts_folder_id, shard_name(k) + ".json", (json.dumps(marker, indent=2) + "\n").encode())
    markers.append(marker)
    return marker


markers = load_markers()
next_shard = len(markers)
for _k in range(next_shard):
    shutil.rmtree(shard_dir(_k), ignore_errors=True)   # already on Drive
print(f"{N_SHARDS} shards of ~{len(shards[0]):,} records (~{len(shards[0]) * EST_IMAGE_MIB / 1024:.1f} GiB each); "
      f"{next_shard} already finished in {DRIVE_DATASET_DIR}/{ARCHIVE_BASENAME}.parts/")
if next_shard < N_SHARDS:
    _local = {k: n for k in range(next_shard, N_SHARDS) if shard_dir(k).is_dir() and (n := len(shard_progress(k)))}
    if _local:
        print("Rendered earlier in this runtime (reused):", {shard_name(k): n for k, n in _local.items()})
STAGES_DONE.add("shards")
"""

PREFLIGHT_MD = """## 9. Preflight

Takes about a minute. The run continues only if all of these pass:
- **Drive:** a small file and a two-chunk resumable upload are written to `<basename>.parts/`, checked against Drive's checksums and status replies, then deleted;
- **Drive storage:** enough free space for the fragments still to come (a warning if there is not also room for the final archive next to them, see §11);
- **matplotlib** in the kit's Python is `MATPLOTLIB_VERSION`;
- **disk** for `SHARD_LOOKAHEAD + 2` shards and **RAM** for the workers;
- **one real record:** rendered by the kit in a spawned worker, augmented, and checked to be 1024x768. The sheet goes to a scratch folder, which is deleted afterwards."""

PREFLIGHT = r"""pf_failures, pf_warnings = [], []
def pf_check(name, fn):
    try:
        print(f"  ok    {name}: {fn()}", flush=True)
    except Exception as exc:
        pf_failures.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}", flush=True)

if GPU_NAME:
    pf_warnings.append(f"GPU {GPU_NAME} is attached but unused; only the {USABLE_CPUS} vCPUs do the work")

def _pf_drive():
    small = f"preflight {datetime.now(timezone.utc).isoformat()}\n".encode()
    meta = drive_put_bytes(parts_folder_id, ".preflight_small.txt", small)
    try:
        assert drive_download(meta["id"]) == small, "small file did not read back"
    finally:
        drive_delete(meta["id"])
    body = os.urandom(256 * 1024 + 4321)           # one full 256 KiB chunk + a short final chunk
    uri = drive_resumable_start(parts_folder_id, ".preflight_resumable.bin", len(body))
    response = drive_http.put(uri, data=body[:256 * 1024], timeout=(30, 120),
                              headers={"Content-Range": f"bytes 0-{256 * 1024 - 1}/{len(body)}"})
    assert response.status_code == 308, f"first chunk -> {response.status_code}"
    assert drive_resumable_status(uri, len(body)) == (256 * 1024, None), "status after the first chunk"
    meta = drive_resumable_upload(uri, len(body), lambda offset, n: body[offset:offset + n])
    try:
        check_drive_copy(meta, len(body), hashlib.sha256(body).hexdigest(), hashlib.md5(body).hexdigest())
        assert drive_download(meta["id"], 1000, 5000) == body[1000:6000], "ranged read differs"
    finally:
        drive_delete(meta["id"])
    return f"write, resumable upload, status, ranged read, delete OK in {ARCHIVE_BASENAME}.parts/"
pf_check("Google Drive", _pf_drive)

def _pf_quota():
    free = drive_free_bytes()
    if free is None:
        return "unlimited"
    fragments = (N_SHARDS - next_shard) * len(shards[0]) * EST_IMAGE_MIB * 2**20
    archive = len(records) * EST_IMAGE_MIB * 2**20
    assert free >= fragments + 2**30, (f"{free / 2**30:.1f} GiB free on Drive, need {(fragments + 2**30) / 2**30:.1f} "
                                       "GiB for the fragments still to come (+1 GiB margin)")
    if free < fragments + archive + 2**30:
        pf_warnings.append(f"Drive has {free / 2**30:.0f} GiB free: not enough to keep all fragments while the "
                           f"{archive / 2**30:.0f} GiB archive is written, so §11 will delete each fragment once copied")
    return f"{free / 2**30:,.0f} GiB free; ~{fragments / 2**30:.0f} GiB of fragments to come, archive ~{archive / 2**30:.0f} GiB"
pf_check("Drive storage", _pf_quota)

def _pf_matplotlib():
    probe = subprocess.run([sys.executable, "-c", "import matplotlib; print(matplotlib.__version__)"],
                           capture_output=True, text=True, timeout=300)
    version = probe.stdout.strip()
    assert version == MATPLOTLIB_VERSION, (f"kit Python has matplotlib {version or probe.stderr.strip()[-300:]}, "
                                           f"need {MATPLOTLIB_VERSION}; rerun §2")
    return f"matplotlib {version} in {sys.executable}"
pf_check("kit matplotlib", _pf_matplotlib)

def _pf_disk():
    need = (SHARD_LOOKAHEAD + 2) * len(shards[0]) * EST_IMAGE_MIB * 2**20 + 5 * 2**30
    free = shutil.disk_usage(SHARDS_DIR).free
    assert free >= need, f"{free / 2**30:.0f} GiB free, need {need / 2**30:.0f} GiB (lower SHARD_LOOKAHEAD)"
    return f"{free / 2**30:.0f} GiB free, {need / 2**30:.0f} GiB needed"
pf_check("disk space", _pf_disk)

def _pf_ram():
    need = WORKERS * 0.7 + 3   # kit ~0.25 GiB, then augment ~0.6 GiB peak per worker; + parent
    assert RAM_GIB >= need, f"{RAM_GIB:.0f} GiB RAM < {need:.0f} GiB for {WORKERS} workers; lower WORKERS"
    return f"{RAM_GIB:.0f} GiB RAM for {WORKERS} workers"
pf_check("RAM", _pf_ram)

def _pf_one_record():
    probe_dir = WORK_DIR / "preflight"
    shutil.rmtree(probe_dir, ignore_errors=True)
    rows = []
    seconds = run_pool(records[:1], {**WORKER_CONFIG, "output_dir": str(probe_dir)}, rows.append, "preflight")
    row = rows[0]
    assert row["status"] == "success", row["error"]
    with Image.open(probe_dir / row["image_path"]) as image:
        assert image.size == (1024, 768) and image.mode == "RGB", (image.size, image.mode)
    shutil.rmtree(probe_dir)
    return (f"{row['record_path']} -> {row['grid_palette']}, tilt {row['tilt_deg']:+.2f}, creases "
            f"{row['fold_creases']}, {int(row['image_bytes']) / 2**20:.2f} MiB in {seconds:.1f} s incl. spawn + kit start")
pf_check("one record end to end (kit + augmentation, spawned worker)", _pf_one_record)

for warning in pf_warnings:
    print("  WARN ", warning)
if pf_failures:
    raise RuntimeError("PREFLIGHT FAIL; fix and rerun this cell:\n- " + "\n- ".join(pf_failures))
print("PREFLIGHT PASS")
STAGES_DONE.add("preflight")
"""

GENERATE_MD = """## 10. Render, augment and upload the shards

One pool of `WORKERS` spawned processes works through the unfinished shards in order, in batches of `BATCH_SIZE` (one kit call per batch). It renders up to `SHARD_LOOKAHEAD` shards ahead.

When the oldest open shard is complete:
1. its effect counts are checked against the plan;
2. its tar fragment is written locally and uploaded to Drive, Drive's checksum is compared with the local one, then the manifest and marker are written;
3. its local files are deleted.

The queued batches keep the workers busy while a shard uploads.

A record that fails is retried `RETRY_PASSES` times. If it still fails, the cell raises with `failures_<shard>.csv`, and that shard is not uploaded.

Progress lines give the measured rate and the time left. If the session ends, reconnect and **Run all**: the run continues at the first unfinished shard."""

GENERATE = r"""def shard_job(k, r):
    # The worker writes SHARDS_DIR/<shard>/images/...; rows are logged with the plain image_path.
    return {"record_path": r["record_path"], "image_path": f"{shard_name(k)}/{r['image_path']}",
            "plan": plan[r["record_path"]]}


class OpenShard:
    def __init__(self, k):
        self.k, self.started = k, time.monotonic()
        folder = shard_dir(k)
        folder.mkdir(parents=True, exist_ok=True)
        for stray in folder.rglob("*.part"):     # temp file of a killed worker
            stray.unlink()
        self.done, self.failed, self.tries = shard_progress(k), {}, Counter()
        self.todo = [r for r in shards[k] if r["record_path"] not in self.done]
        self.open = len(self.todo)
        self.log = (folder / "progress.csv").open("a", newline="")
        self.writer = csv.DictWriter(self.log, fieldnames=AW.MANIFEST_FIELDS, restval="", extrasaction="ignore")
        if self.log.tell() == 0:
            self.writer.writeheader()


def finish_shard(st):
    k = st.k
    st.log.close()
    if st.failed:
        path = WORK_DIR / f"failures_{shard_name(k)}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=AW.MANIFEST_FIELDS, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(st.failed.values())
        raise RuntimeError(f"{len(st.failed)} records of {shard_name(k)} failed after {RETRY_PASSES} retry pass(es); "
                           f"see {path}. This shard was not uploaded.")
    rows = [st.done[r["record_path"]] for r in shards[k]]
    missing = [row["image_path"] for row in rows if not (shard_dir(k) / row["image_path"]).is_file()]
    if missing:
        raise RuntimeError(f"{shard_name(k)}: {len(missing)} sheets missing on disk, e.g. {missing[:3]}")
    for factor in AW.PLAN_FACTORS:
        rendered_counts = Counter(str(row[factor]) for row in rows)
        planned_counts = Counter(str(plan[r["record_path"]][factor]) for r in shards[k])
        if rendered_counts != planned_counts:
            raise RuntimeError(f"{shard_name(k)} {factor}: rendered {dict(rendered_counts)} != planned {dict(planned_counts)}")
    marker = upload_fragment(k, rows, time.monotonic() - st.started)
    shutil.rmtree(shard_dir(k))
    print(f"{shard_name(k)} finished: {marker['records']:,} sheets, {marker['fragment_bytes'] / 2**30:.2f} GiB "
          f"uploaded in {marker['upload_seconds'] / 60:.1f} min; {len(markers)}/{N_SHARDS} shards on Drive", flush=True)


def log_rows(st, rows, retry=True):
    # Log one finished batch; a failed record is queued again until RETRY_PASSES is used up.
    global rendered
    for row in rows:
        path = row["record_path"]
        row["image_path"] = record_of[path]["image_path"]
        st.writer.writerow(row)
        if row["status"] == "success":
            st.done[path] = row
            st.open -= 1
            rendered += 1
        elif retry:
            st.tries[path] += 1
            if st.tries[path] <= RETRY_PASSES:
                queue.appendleft((st.k, [record_of[path]]))
            else:
                st.failed[path] = row
                st.open -= 1
    st.log.flush()


if next_shard < N_SHARDS:
    record_of = {r["record_path"]: r for r in records}
    states, queue, pending = {}, deque(), {}
    next_open = next_upload = next_shard
    rendered, last_report, run_t0 = 0, 0, time.monotonic()
    print(f"Rendering shards {next_shard}..{N_SHARDS - 1} with {WORKERS} workers", flush=True)
    pool = ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("spawn"), initializer=AW.init_worker,
                               initargs=({**WORKER_CONFIG, "output_dir": str(SHARDS_DIR)},))
    try:
        while next_upload < N_SHARDS:
            while next_open < N_SHARDS and next_open <= next_upload + SHARD_LOOKAHEAD:
                need = 2 * len(shards[next_open]) * EST_IMAGE_MIB * 2**20 + 5 * 2**30
                if shutil.disk_usage(SHARDS_DIR).free < need and next_open > next_upload:
                    break                      # wait for an upload to free space
                st = states[next_open] = OpenShard(next_open)
                for i in range(0, len(st.todo), BATCH_SIZE):
                    queue.append((next_open, st.todo[i:i + BATCH_SIZE]))
                next_open += 1
            while queue and len(pending) < MAX_PENDING_BATCHES:
                k, batch = queue.popleft()
                pending[pool.submit(AW.process_batch, [shard_job(k, r) for r in batch])] = (k, batch)
            if states[next_upload].open == 0:
                finish_shard(states.pop(next_upload))
                next_upload += 1
                continue
            if not pending:
                raise RuntimeError(f"Scheduler stalled: {shard_name(next_upload)} is incomplete with nothing queued")
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                k, batch = pending.pop(future)
                try:
                    rows = future.result()
                except Exception as exc:   # task raised outside its per-record handling
                    rows = [{"record_path": r["record_path"], "status": "failed",
                             "error": f"task failure: {type(exc).__name__}: {exc}"[:1000]} for r in batch]
                log_rows(states[k], rows)
            if rendered - last_report >= PROGRESS_EVERY:
                last_report = rendered
                rate = rendered / (time.monotonic() - run_t0)
                left = (sum(len(shards[k]) for k in range(next_upload, N_SHARDS))
                        - sum(len(s.done) for s in states.values()))
                print(f"  {rendered:,} sheets this session | {rate:.2f} sheets/s | {left:,} left in "
                      f"{N_SHARDS - next_upload} shards | ETA {left / rate / 3600:.1f} h", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for future, (k, _batch) in pending.items():   # batches that finished while stopping
            if k in states and future.done() and not future.cancelled() and future.exception() is None:
                log_rows(states[k], future.result(), retry=False)
        for st in states.values():
            st.log.close()
    print(f"All {N_SHARDS} shards rendered and uploaded; this session: {rendered:,} sheets in "
          f"{(time.monotonic() - run_t0) / 3600:.2f} h", flush=True)
else:
    print(f"All {N_SHARDS} shards are already on Drive.")
if len(markers) != N_SHARDS:
    raise RuntimeError(f"{len(markers)}/{N_SHARDS} shards finished")
STAGES_DONE.add("generate")
"""

COMPOSE_MD = """## 11. Join the shards into the archive on Drive and verify

1. **Manifest check.** The shard manifests are read back from Drive and joined into `manifest.csv`. The cell checks that every record has exactly one successful sheet (113,358) and that the effect counts equal the plan.
2. **Tail.** The metadata files become tar members, followed by the end-of-archive blocks: `manifest.csv`, `augmentation_plan.csv`, `balance_by_condition.csv`, `effect_counts.json`, `record_manifest.csv`, `config.json`, `summary.json`, `failures.csv`, plus `shards.json`. The tail is saved as `tail.bin` and the upload session as `join.json`, so a new session continues the same join.
3. **Join.** One resumable upload creates `<basename>.tar` in `DRIVE_DATASET_DIR`. The bytes come from the fragments, read back from Drive by byte range, then the tail. After a disconnect, Drive reports how many bytes it holds, and the upload continues from there, in this session or the next. An upload session stays valid for a week.
4. **Verify.** Drive's size and `sha256Checksum` of the archive must equal the local values. The SHA-256 is the running hash after the last shard, continued over the tail. It is written to `<basename>.tar.sha256` and read back.

**Drive space.** If Drive has room for the archive on top of the fragments, the fragments stay until the archive is verified. Otherwise each fragment is deleted as soon as the upload has passed it; the length is then left open, so Drive does not reserve the full size up front. In that mode the join cannot be restarted from scratch, so finish it within the week the upload session lasts.

If the archive is already complete and verified (`compose.json`), the cell does nothing. If a file with the archive's name already sits in the folder and was not made by this run (for example a copy of the EC2 archive), the cell refuses to overwrite it."""

COMPOSE = r"""COMPOSE_VERIFIED = False
markers = load_markers()
if len(markers) != N_SHARDS:
    raise RuntimeError(f"{len(markers)}/{N_SHARDS} shards are finished; run §10 first")
sidecar_name = archive_name + ".sha256"

_record = drive_get_bytes(parts_folder_id, "compose.json")
_existing = drive_children(dataset_folder_id, archive_name)
if _record is not None:
    _record = json.loads(_record)
    _meta = drive_meta(_record["file_id"])
    if _meta is None or int(_meta["size"]) != _record["bytes"] or (
            _meta.get("sha256Checksum") and _meta["sha256Checksum"] != _record["archive_sha256"]):
        raise RuntimeError(f"{archive_name} on Drive no longer matches compose.json. Delete the archive and "
                           f"{ARCHIVE_BASENAME}.parts/ to start over.")
    archive_sha256 = _record["archive_sha256"]
    _sidecar_body = f"{archive_sha256}  {archive_name}\n".encode()
    if drive_get_bytes(dataset_folder_id, sidecar_name) != _sidecar_body:
        drive_put_bytes(dataset_folder_id, sidecar_name, _sidecar_body)
    COMPOSE_VERIFIED = True
    print(f"Already complete and verified: {DRIVE_DATASET_DIR}/{archive_name} ({_record['bytes'] / 2**30:.2f} GiB), "
          f"SHA-256 {archive_sha256}")

if not COMPOSE_VERIFIED:
    join = drive_get_bytes(parts_folder_id, "join.json")
    join = json.loads(join) if join is not None else None
    if _existing:
        _owned = False                         # the archive this run's join already completed?
        if join is not None:
            try:
                _done = drive_resumable_status(join["session_uri"], join["total_bytes"], join["keep_fragments"])[1]
                _owned = _done is not None and {f["id"] for f in _existing} == {_done["id"]}
            except DriveSessionExpired:
                pass
        if not _owned:
            raise RuntimeError(f"{archive_name} already exists in {DRIVE_DATASET_DIR} and was not made by this run. "
                               "Move or delete it, or use a new ARCHIVE_BASENAME.")
    for k, marker in enumerate(markers):
        if marker["fragment_bytes"] % 512:
            raise RuntimeError(f"{shard_name(k)} fragment is not a whole number of tar blocks")

    if join is None:
        manifest_rows = []
        for k, marker in enumerate(markers):
            body = drive_get_bytes(parts_folder_id, shard_name(k) + ".manifest.csv")
            if body is None or hashlib.sha256(body).hexdigest() != marker["manifest_sha256"]:
                raise RuntimeError(f"{shard_name(k)}.manifest.csv is missing or differs from its marker")
            rows = list(csv.DictReader(io.StringIO(body.decode("utf-8"))))
            if [row["record_path"] for row in rows] != [r["record_path"] for r in shards[k]]:
                raise RuntimeError(f"{shard_name(k)}.manifest.csv does not list that shard's records")
            manifest_rows.extend(rows)
        by_path = {row["record_path"]: row for row in manifest_rows}
        problems = []
        if len(manifest_rows) != EXPECTED_RECORDS or len(by_path) != EXPECTED_RECORDS:
            problems.append(f"{len(manifest_rows):,} rows / {len(by_path):,} records; expected {EXPECTED_RECORDS:,}")
        if set(by_path) != {r["record_path"] for r in records}:
            problems.append("manifest records differ from the clean12 records")
        if any(row["status"] != "success" for row in manifest_rows):
            problems.append("some manifest rows are not 'success'")
        if any(by_path[r["record_path"]]["image_path"] != r["image_path"] for r in records if r["record_path"] in by_path):
            problems.append("some image paths differ from the record paths")
        effect_counts = {}
        for factor in AW.PLAN_FACTORS:
            rendered_counts = Counter(str(row[factor]) for row in manifest_rows)
            planned_counts = Counter(str(plan[r["record_path"]][factor]) for r in records)
            if rendered_counts != planned_counts:
                problems.append(f"{factor}: rendered {dict(rendered_counts)} != planned {dict(planned_counts)}")
            effect_counts[factor] = dict(sorted(rendered_counts.items()))
        if problems:
            raise RuntimeError("Shards do not add up to the dataset:\n- " + "\n- ".join(problems))

        manifest_rows = [by_path[r["record_path"]] for r in records]
        with (OUTPUT_DIR / "manifest.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=AW.MANIFEST_FIELDS, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(manifest_rows)
        with (OUTPUT_DIR / "failures.csv").open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=AW.MANIFEST_FIELDS).writeheader()
        (OUTPUT_DIR / "effect_counts.json").write_text(json.dumps(effect_counts, indent=2) + "\n")
        (OUTPUT_DIR / "shards.json").write_text(json.dumps({**shard_layout, "shards": [
            {key: m[key] for key in ("shard", "records", "fragment_bytes", "fragment_sha256", "image_bytes",
                                     "render_seconds", "upload_seconds", "completed_utc")} for m in markers]},
            indent=2) + "\n")
        summary = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "source": S3_SOURCE, "source_sha256": expected_sha256,
            "records": len(records), "images": len(manifest_rows), "failed": 0,
            "image_bytes": sum(int(row["image_bytes"]) for row in manifest_rows),
            "render_seconds": round(sum(m["render_seconds"] for m in markers), 1),
            "config_sha256": config_hash, "built_on": "Google Colab, sharded, saved to Google Drive",
            "shards": N_SHARDS, "effect_counts": effect_counts,
        }
        (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        tail = io.BytesIO()
        for name in ("manifest.csv", "augmentation_plan.csv", "balance_by_condition.csv", "effect_counts.json",
                     "record_manifest.csv", "config.json", "summary.json", "failures.csv", "shards.json"):
            data = (OUTPUT_DIR / name).read_bytes()
            tail.write(tar_header(f"{ARCHIVE_TOPDIR}/{name}", len(data), (OUTPUT_DIR / name).stat().st_mtime))
            tail.write(data)
            tail.write(b"\0" * (-len(data) % 512))
        tail.write(b"\0" * (2 * 512))                           # end-of-archive blocks
        _length = markers[-1]["running_bytes"] + tail.tell()
        tail.write(b"\0" * (-_length % tarfile.RECORDSIZE))     # pad to a whole record, as tarfile does
        tail = tail.getvalue()
        _running = ResumableSHA256(markers[-1]["running_sha256_state"])
        _running.update(tail)
        _tail_meta = drive_put_bytes(parts_folder_id, "tail.bin", tail)
        total_bytes = markers[-1]["running_bytes"] + len(tail)
        _free = drive_free_bytes()
        keep_fragments = _free is None or _free >= total_bytes + 2**30
        join = {"tail_file_id": _tail_meta["id"], "tail_sha256": hashlib.sha256(tail).hexdigest(),
                "total_bytes": total_bytes, "archive_sha256": _running.hexdigest(), "keep_fragments": keep_fragments,
                "session_uri": drive_resumable_start(dataset_folder_id, archive_name,
                                                     total_bytes if keep_fragments else None),
                "started_utc": datetime.now(timezone.utc).isoformat()}
        drive_put_bytes(parts_folder_id, "join.json", (json.dumps(join, indent=2) + "\n").encode())
    else:
        tail = drive_download(join["tail_file_id"])
        if hashlib.sha256(tail).hexdigest() != join["tail_sha256"]:
            raise RuntimeError("tail.bin on Drive differs from join.json")
        print(f"Continuing the join started {join['started_utc']} (upload sessions last a week)")

    total_bytes, archive_sha256, keep_fragments = join["total_bytes"], join["archive_sha256"], join["keep_fragments"]
    segments, _offset = [], 0                 # (start, end, fragment file id or None for the tail)
    for marker in markers:
        segments.append((_offset, _offset + marker["fragment_bytes"], marker["file_id"]))
        _offset += marker["fragment_bytes"]
    segments.append((_offset, _offset + len(tail), None))
    if segments[-1][1] != total_bytes:
        raise RuntimeError("fragment sizes + tail do not add up to the archive size in join.json")

    def read_archive(offset, n):
        out = bytearray()
        for start, end, file_id in segments:
            if end <= offset or start >= offset + n:
                continue
            lo, hi = max(start, offset), min(end, offset + n)
            out += tail[lo - start:hi - start] if file_id is None else drive_download(file_id, lo - start, hi - lo)
        return bytes(out)

    deleted, _reported, t0 = set(), [-1], time.monotonic()
    def on_commit(committed):
        if not keep_fragments:
            for start, end, file_id in segments:
                if file_id and end <= committed and file_id not in deleted:
                    drive_delete(file_id)
                    deleted.add(file_id)
        step = int(20 * committed / total_bytes)
        if step > _reported[0]:
            _reported[0] = step
            print(f"  join {committed / 2**30:,.1f} / {total_bytes / 2**30:,.1f} GiB "
                  f"({(time.monotonic() - t0) / 60:.1f} min)", flush=True)

    print(f"Joining {N_SHARDS} fragments + {len(tail) / 2**20:.1f} MiB tail -> {DRIVE_DATASET_DIR}/{archive_name} "
          f"({total_bytes / 2**30:.2f} GiB; fragments {'kept until verified' if keep_fragments else 'deleted as copied'})",
          flush=True)
    try:
        meta = drive_resumable_upload(join["session_uri"], total_bytes, read_archive, keep_fragments, on_commit)
    except DriveSessionExpired:
        missing = [m["file_id"] for m in markers if drive_meta(m["file_id"]) is None]
        if missing:
            raise RuntimeError(f"The join's upload session expired after {len(missing)} fragments were deleted; "
                               f"delete {ARCHIVE_BASENAME}.parts/ on Drive and start over.") from None
        join["session_uri"] = drive_resumable_start(dataset_folder_id, archive_name,
                                                    total_bytes if keep_fragments else None)
        join["started_utc"] = datetime.now(timezone.utc).isoformat()
        drive_put_bytes(parts_folder_id, "join.json", (json.dumps(join, indent=2) + "\n").encode())
        print("The upload session had expired; started a new one", flush=True)
        meta = drive_resumable_upload(join["session_uri"], total_bytes, read_archive, keep_fragments, on_commit)
    for _ in range(20):                      # Drive can fill in sha256Checksum a little after the upload
        meta = drive_meta(meta["id"]) or meta
        if meta.get("sha256Checksum"):
            break
        time.sleep(30)
    if int(meta["size"]) != total_bytes or (meta.get("sha256Checksum") and meta["sha256Checksum"] != archive_sha256):
        drive_delete(meta["id"])
        raise RuntimeError(f"The archive on Drive ({meta['size']} bytes, sha256 {meta.get('sha256Checksum')}) differs "
                           f"from the local {total_bytes:,} bytes / {archive_sha256}; it was deleted.")
    if not meta.get("sha256Checksum"):
        print("  WARN  Drive reported no sha256Checksum after 10 min; only the size and each fragment's checksum "
              "are verified", flush=True)

    _sidecar_body = f"{archive_sha256}  {archive_name}\n".encode()
    (WORK_DIR / sidecar_name).write_bytes(_sidecar_body)
    drive_put_bytes(dataset_folder_id, sidecar_name, _sidecar_body)
    drive_put_bytes(parts_folder_id, "compose.json", (json.dumps({
        "file_id": meta["id"], "archive_sha256": archive_sha256, "bytes": total_bytes,
        "completed_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n").encode())
    for marker in markers:
        if marker["file_id"] not in deleted:
            drive_delete(marker["file_id"])
    for name in ("tail.bin", "join.json"):
        for item in drive_children(parts_folder_id, name):
            drive_delete(item["id"])
    COMPOSE_VERIFIED = True
    print(f"Archive written and verified in {(time.monotonic() - t0) / 60:.1f} min: {DRIVE_DATASET_DIR}/{archive_name}")
    print(f"SHA-256 {archive_sha256}" + (" (equals Drive's sha256Checksum)" if meta.get("sha256Checksum") else "")
          + f" -> {sidecar_name}")
STAGES_DONE.add("compose")
"""

RELEASE_MD = """## 12. Release the Colab runtime

The runtime is released (`google.colab.runtime.unassign()`), so it stops using compute units, only when **all** of these hold:
- `RELEASE_RUNTIME_WHEN_DONE` is on;
- §11 verified the archive on Drive in this kernel;
- every pipeline cell (§2-§11) completed in this kernel since §1 was last run;
- **no cell raised an error** in that time.

The cell waits `RELEASE_GRACE_SECONDS` first; interrupt the kernel to cancel. Releasing deletes the runtime's disk, but everything is on Drive at that point."""

RELEASE = r"""REQUIRED_STAGES = ["install", "access", "download", "kit", "records", "workers", "plan", "shards",
                   "preflight", "generate", "compose"]
missing_stages = [s for s in REQUIRED_STAGES if s not in STAGES_DONE]
blockers = []
if not RELEASE_RUNTIME_WHEN_DONE:
    blockers.append("RELEASE_RUNTIME_WHEN_DONE is False")
if not globals().get("COMPOSE_VERIFIED"):
    blockers.append("the archive was not verified in this kernel")
if missing_stages:
    blockers.append(f"these cells did not complete since §1: {missing_stages}")
if CELL_ERRORS:
    blockers.append(f"{len(CELL_ERRORS)} cell(s) raised an error since §1, e.g. {CELL_ERRORS[0]}")
if not IN_COLAB:
    blockers.append("not running on Colab")
if blockers:
    print("NOT releasing the runtime:\n- " + "\n- ".join(blockers))
else:
    print(f"All {len(REQUIRED_STAGES)} pipeline cells completed without errors; archive verified: "
          f"{DRIVE_DATASET_DIR}/{archive_name}")
    print(f"Releasing this Colab runtime in {RELEASE_GRACE_SECONDS} s; interrupt the kernel to cancel.", flush=True)
    time.sleep(RELEASE_GRACE_SECONDS)
    from google.colab import runtime
    runtime.unassign()
"""

NOTES = """## Notes

**Colab disk at peak** (on top of what Colab itself uses):

| Item | Size |
|---|---|
| Signal archive + extracted records | ~19 GiB |
| Open shards (`SHARD_LOOKAHEAD + 1` = 3 × ~3.6 GiB) + one fragment being uploaded | ~15 GiB |
| **Total** | **~34 GiB** |

**Drive storage at peak:** ~160 GiB of fragments plus the ~160 GiB archive during the join. With less free space the join deletes fragments as it goes, and the peak stays near ~165 GiB. Drive also limits uploads to about 750 GB per account per day; this run uploads about 340 GB in total (fragments, then the archive).

**Time.** Measured on a 4-core laptop: about 4 s per record per core (kit about 2.6 s, augmentation and PNG about 1.4 s). Colab vCPUs are usually hyperthreads, so expect a bit more per vCPU. The total is roughly 113,358 × 4-6 s ÷ vCPUs; for example, about 10-16 h on 12 vCPUs. The progress lines in §10 give the measured rate and ETA. The join in §11 reads and writes ~160 GiB within Google; expect tens of minutes to a few hours, depending on Drive throughput.

**Sessions.** One run can span several sessions. Each new runtime:
- signs in again;
- downloads and extracts the signals again (~5 min);
- reruns the preflight;
- continues at the first unfinished shard, or continues the join.

A lost session costs at most the `SHARD_LOOKAHEAD + 1` shards that were rendering.

**Output equals the EC2 notebook's archive:**
- same file name and top folder `clean_ecg_signal_images_clean12_augmented/`, member paths, metadata files and plan; the same PNG for each record;
- the `.sha256` sidecar is in `sha256sum` format.

Differences:
- tar member order (shard by shard), no directory entries (extractors create folders), and the extra `shards.json`;
- `summary.json` has `render_seconds` instead of `generate_cell_seconds`;
- PNG bytes can differ from an EC2 render only if the Python packages differ (matplotlib is pinned).

The clean12 + augmented training notebook reads it from `new_dataset/` as it is.

**Using the output.**
- `manifest.csv` has one row per record: `record_path`, `image_path` and every effect value.
- Join to `record_manifest.csv` on `record_path` for labels and `patient_key`.
- Check a downloaded copy with `sha256sum -c clean_ecg_signal_images_clean12_augmented.tar.sha256`.

**Starting over.** Delete `<DRIVE_DATASET_DIR>/clean_ecg_signal_images_clean12_augmented.parts/` (and the archive, if any) on Drive, or set a new `ARCHIVE_BASENAME`. Empty the Drive trash if you deleted them in the web interface, because trashed files still count against the quota.

**Changing the design.**
1. Edit `../clean12_augmented_ecg_images_ec2/augment_worker.py` (`AUG_PALETTES`, `INK_BINS`, `TILT_BINS`, `PLAN_FACTORS`).
2. Rebuild both notebooks with `python3 make_notebook.py` in each folder.
3. Use a new `ARCHIVE_BASENAME`: the layout lock refuses to mix settings."""


def build() -> dict:
    md, code = ec2.md, ec2.code
    per_record = between(ec2.INTRO, "**Per record:**", "## Setup:")
    workers = (ec2.WORKERS.replace("__CLEAN_WORKER__", ec2.embedded(ec2.clean12_worker_source()))
               .replace("__AUGMENT_WORKER__", ec2.embedded((EC2_DIR / "augment_worker.py").read_text())))
    plan = between(ec2.PLAN, "plan = AW.balanced_plan", 'progress_path = WORK_DIR / "generation_progress.csv"')
    plan += 'STAGES_DONE.add("plan")\n'
    cells = [
        md(INTRO.replace("__PER_RECORD__", per_record.rstrip() + "\n")),
        code(CONFIG.replace("__KIT_SETTINGS__\n", ec2.clean12_kit_settings())),
        md(INSTALL_MD), code(INSTALL_PIN + ec2.clean12_cell("%pip install") + ec2.INSTALL_EXTRA),
        md(ACCESS_MD), code(ACCESS),
        md(DOWNLOAD_MD), code(ec2.clean12_cell("def split_s3_uri(uri):") + '\nSTAGES_DONE.add("download")\n'),
        md(KIT_MD), code(KIT_LOCATE + ec2.clean12_cell("KIT_BUNDLE = None") + ec2.KIT_EXTRA),
        md(ec2.RECORDS_MD), code(ec2.RECORDS),
        md(ec2.WORKERS_MD), code(workers),
        md(PLAN_MD), code(plan),
        md(SHARDS_MD), code(SHARDS),
        md(PREFLIGHT_MD), code(PREFLIGHT),
        md(GENERATE_MD), code(GENERATE),
        md(COMPOSE_MD), code(COMPOSE),
        md(RELEASE_MD), code(RELEASE),
        md(NOTES),
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    NOTEBOOK.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print("wrote", NOTEBOOK)
