#!/usr/bin/env python3
"""Build Clean12_Augmented_ECG_Images_EC2.ipynb.

The notebook is standalone on EC2. Proven pieces of the clean12 render
notebook (../clean12_ecg_signal_images_ec2/Clean12_ECG_Signal_Images_EC2.ipynb)
are copied from it at build time, so both notebooks render ECGs identically:
its kit settings, dependency install, signal download/extract, kit bundle
patch gate, render worker (ecg_clean_worker) and streaming tar->S3 uploader.
augment_worker.py is embedded verbatim and written to WORK_DIR at run time.

    python3 make_notebook.py
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "Clean12_Augmented_ECG_Images_EC2.ipynb"
CLEAN12_NOTEBOOK = HERE.parent / "clean12_ecg_signal_images_ec2" / "Clean12_ECG_Signal_Images_EC2.ipynb"
CLEAN12_CELLS = ["".join(c["source"]) for c in json.loads(CLEAN12_NOTEBOOK.read_text())["cells"]]


def clean12_cell(marker: str) -> str:
    """The one clean12 code cell containing marker."""
    hits = [s for s in CLEAN12_CELLS if marker in s]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} clean12 cells contain {marker!r}")
    return hits[0].rstrip() + "\n"


def clean12_kit_settings() -> str:
    """Kit/render constants from the clean12 config cell, so masters match its sheets."""
    namespace: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(clean12_cell("KIT_RENDER_FLAGS = "), namespace)  # assignments + prints only
    names = ["EXPECTED_FS", "EXPECTED_SAMPLES", "MAX_DURATION_S", "KIT_SEED", "KIT_FIXED_FONT",
             "KIT_RENDER_FLAGS", "KIT_ENV", "PNG_COMPRESS_LEVEL"]
    return "\n".join(f"{name} = {namespace[name]!r}" for name in names) + "\n"


def clean12_worker_source() -> str:
    source = clean12_cell("WORKER_SOURCE = r'''")
    start = source.index("r'''") + 4
    return source[start:source.index("'''", start)]


def uploader_class() -> str:
    source = clean12_cell("class TarUploadStream")
    return source[source.index("STREAM_PART_BYTES ="):source.index("upload_prefix = urlparse(S3_UPLOAD_PREFIX)")].rstrip() + "\n"


def embedded(source: str) -> str:
    if "'''" in source:
        raise ValueError("embedded module must not contain '''")
    return source


INTRO = """# Augmented clean12 ECG images on EC2: one paper/photo sheet per record

The notebook reads the **clean12 signal dataset**, `ECG_500Hz_clean12.tar.gz`: 113,358 native-500 Hz WFDB records in Account 1. It writes **one augmented ECG sheet (1024x768 PNG) per record**, so the dataset ends with 113,358 images for 113,358 records. Each image path mirrors its record path (`images/<site>/WFDB/.../<record>.png`), the same layout as the clean12 sheets, so clean and augmented sheets match on `image_path`.

The output tar and its `.sha256` are uploaded to `s3://cardiosentry-heedb/dataset/` in Account 2 (`S3_UPLOAD_PREFIX`). After the upload is verified, **the instance terminates itself**. That happens only if every pipeline cell before it finished and no cell in this kernel raised an error (§11).

**Per record:**
1. The record is rendered with the **ECG-image-kit**, using the same settings and the same code as `Clean12_ECG_Signal_Images_EC2.ipynb` (both copied from that notebook when this one is built). This gives the 2200x1700 grey sheet the clean12 images were made from.
2. That sheet goes through the image pipeline's **stage-4 paper/photo augmentation**: `stage4_augment.py`, the code behind the `augmented_<grid>_tilt<deg>.png` previews in `correct_prediction_samples`. Its wrinkled-paper texture and fold creases are **stronger** than stage 4 (table below).
3. JPEG q85 round trip, then the clean view's own 1024x768 fit.

The copy of stage 4 in `augment_worker.py` reproduces `stage4_augment.augment` pixel for pixel when run with the stage-4 settings. Only the wrinkle and crease strengths were raised:

| Setting | Stage 4 | This notebook |
|---|---|---|
| wrinkle texture strength | 0.30–0.62 | 0.65–1.00 |
| wrinkle shading limits | ×0.86–1.10 | ×0.76–1.14 |
| crease amplitude | 0.06–0.13 | 0.16–0.26 |
| crease shading limits | ×0.88–1.14 | ×0.76–1.22 |
| crease width (fraction of the sheet's long side) | 0.006–0.020 | 0.004–0.012 (sharper) |
| paper bowing | 0.02–0.06 | 0.03–0.08 |

**Equal representation.** These factors are assigned by plan, not by chance. Grid red is skipped because it is the clean view's grid.

| Factor | Levels | Share each |
|---|---|---|
| grid colour | orange, pink, grey | 1/3 |
| ink strength | 0.72–0.82 (faded), 0.82–0.92, 0.92–1.02 (fresh) | 1/3 |
| tilt | six 1° bins across −3°…+3° | 1/6 |
| perspective | off / on | 1/2 |
| fold creases | off / on | 1/2 |
| edge shadow | off / on | 1/2 |
| blur | off / on | 1/2 |

Every sheet gets wrinkled-paper texture (at the stronger setting above), uneven lighting, colour tint, brightness, contrast, paper-level normalisation and sensor noise, each at a random strength within its range.

How the plan is assigned:
- Levels are dealt round-robin inside each `(site, label set)` stratum, ignoring the labels themselves.
- Each factor uses its own seeded shuffle, so factors are assigned independently of each other.
- Result: level counts are within ±1 overall and within every label combination. §7 writes the audit and stops if it fails.

## Setup: data in Account 1, EC2 and output in Account 2

**Files to copy to the instance**, all three from this notebook's folder in the repo, into one folder:
- this notebook;
- `ecg_image_v1_code.tar.gz`;
- `ecg_image_v1_code.tar.gz.sha256`.

`ecg_image_v1_code.tar.gz` is the ECG-image-kit itself: the program that draws the ECG from the signals, plus the wrinkled-paper photographs. It is code, not data, and does not need to go to S3. The clean12 render notebook needs it in exactly the same way. If you would rather not copy it, put both kit files next to the signals in Account 1 (`S3_KIT_SOURCE`); §4 downloads them if no local copy is found.

**Account 1** (owns `cardiosentry-fcmae-758821198776-us-east-1-an`). Add this statement to the **bucket policy**: S3 console > bucket > Permissions > Bucket policy.
```json
{
  "Sid": "Account2Ec2ReadsClean12",
  "Effect": "Allow",
  "Principal": {"AWS": "arn:aws:iam::<ACCOUNT_2_ID>:role/cardiosentry-augment-ec2"},
  "Action": "s3:GetObject",
  "Resource": [
    "arn:aws:s3:::cardiosentry-fcmae-758821198776-us-east-1-an/CardioSentry/dataset/clean12/ECG_500Hz_clean12.tar.gz",
    "arn:aws:s3:::cardiosentry-fcmae-758821198776-us-east-1-an/CardioSentry/dataset/clean12/ECG_500Hz_clean12.tar.gz.sha256",
    "arn:aws:s3:::cardiosentry-fcmae-758821198776-us-east-1-an/CardioSentry/dataset/clean12/ecg_image_v1_code.tar.gz*"
  ]
}
```
Notes on Account 1:
- If the bucket's default encryption is **SSE-KMS with a customer-managed key**, the key policy in Account 1 must also allow `kms:Decrypt` for that role. The default SSE-S3 encryption needs nothing more.
- Block Public Access does not block this policy, because the principal is a specific role.

**Account 2.** Create the EC2 role (with its instance profile) `cardiosentry-augment-ec2` with this policy, then launch the instance with it:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "ReadClean12FromAccount1", "Effect": "Allow", "Action": "s3:GetObject",
     "Resource": "arn:aws:s3:::cardiosentry-fcmae-758821198776-us-east-1-an/CardioSentry/dataset/clean12/*"},
    {"Sid": "WriteOutputInAccount2", "Effect": "Allow",
     "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:AbortMultipartUpload"],
     "Resource": "arn:aws:s3:::cardiosentry-heedb/dataset/*"},
    {"Sid": "TerminateItself", "Effect": "Allow", "Action": "ec2:TerminateInstances",
     "Resource": "arn:aws:ec2:us-east-1:<ACCOUNT_2_ID>:instance/*",
     "Condition": {"ArnEquals": {"ec2:InstanceProfile": "arn:aws:iam::<ACCOUNT_2_ID>:instance-profile/cardiosentry-augment-ec2"}}}
  ]
}
```
A cross-account read needs **both** sides to allow it: the Account 1 bucket policy and this Account 2 role policy. The condition limits termination to instances launched with this profile.

**Instance.**
- Type: **c8a.4xlarge** in **us-east-1**, the same region as the data.
- Storage: a **250 GB gp3** volume at 1,000 MB/s.
- Run Jupyter inside `tmux` (or with `nohup`), from the folder holding the three files.
- Run All. The output goes to `s3://cardiosentry-heedb/dataset/` (Account 2), set in §1.

Every stage can be restarted. A rerun reuses:
- the verified archive;
- the extracted tree;
- every sheet already rendered, as recorded in `WORK_DIR/generation_progress.csv`.
"""

CONFIG = """# 1. Configuration: edit before running any later cell.
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

from pathlib import Path

# Account 1 (read-only, cross-account): the clean12 signal dataset. Same object as
# https://cardiosentry-fcmae-758821198776-us-east-1-an.s3.us-east-1.amazonaws.com/CardioSentry/dataset/clean12/ECG_500Hz_clean12.tar.gz
S3_SIGNAL_PREFIX = "s3://cardiosentry-fcmae-758821198776-us-east-1-an/CardioSentry/dataset/clean12/"
S3_SOURCE = S3_SIGNAL_PREFIX + "ECG_500Hz_clean12.tar.gz"
S3_SHA256_SOURCE = S3_SIGNAL_PREFIX + "ECG_500Hz_clean12.tar.gz.sha256"
# Fallback only: the kit bundle normally sits next to this notebook on the instance.
S3_KIT_SOURCE = S3_SIGNAL_PREFIX + "ecg_image_v1_code.tar.gz"
S3_KIT_SHA256_SOURCE = S3_SIGNAL_PREFIX + "ecg_image_v1_code.tar.gz.sha256"

# Account 2: where the augmented archive is uploaded.
S3_UPLOAD_PREFIX = "s3://cardiosentry-heedb/dataset/"
ARCHIVE_FORMAT = "tar"           # PNG is already compressed
ARCHIVE_BASENAME = "clean_ecg_signal_images_clean12_augmented"   # -> <prefix><basename>.tar

WORK_DIR = Path.home() / "ecg_clean12_augment"
LOCAL_ARCHIVE = WORK_DIR / "input" / "ECG_500Hz_clean12.tar.gz"
INPUT_DIR = WORK_DIR / "input" / "ECG_500Hz_clean12"
OUTPUT_DIR = WORK_DIR / "clean_ecg_signal_images_clean12_augmented"

EXPECTED_RECORDS = 113_358
CONDITIONS = ["SB", "STACH", "LBBB", "LQT", "AF", "AFL",
              "3AVB", "2AVB", "1AVB", "STEMI", "STE", "NORM"]
# Kit/render settings: copied from Clean12_ECG_Signal_Images_EC2.ipynb at build
# time, so every grey master is the master its clean12 sheet was made from.
__KIT_SETTINGS__
AUGMENT_SEED = 20261023          # plan shuffles + per-record effect draws
EST_IMAGE_MIB = 1.5              # measured 1.13-1.49 MiB per augmented sheet

# One worker per vCPU (c8a: 1 vCPU = 1 core). A worker runs one kit process for
# a whole batch (amortises the kit's 1-3 s import), then augments its masters.
WORKERS = max(1, os.cpu_count() or 1)
BATCH_SIZE = 48
MAX_PENDING_BATCHES = 2 * WORKERS
PROGRESS_EVERY = 1000
RETRY_PASSES = 1                 # failed records are retried once before giving up
KIT_SCRATCH_DIR = (Path("/dev/shm") / "ecg_clean12_augment_scratch"
                   if os.access("/dev/shm", os.W_OK) else WORK_DIR / "kit_scratch")
S3_TRANSFER = dict(max_concurrency=32, multipart_chunksize=64 * 2**20, use_threads=True)

# Terminate this instance once the archive is uploaded and verified (§11), and
# only if every pipeline cell completed and no cell raised in this kernel.
AUTO_TERMINATE = True
TERMINATE_GRACE_SECONDS = 60     # interrupt the kernel within this window to cancel

# Safety record for §11. Re-running THIS cell starts a fresh run: both reset.
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

assert WORKERS >= 1 and BATCH_SIZE >= 1 and MAX_PENDING_BATCHES >= WORKERS
WORK_DIR.mkdir(parents=True, exist_ok=True)
print("Working directory:", WORK_DIR, "| workers:", WORKERS, "| batch:", BATCH_SIZE)
print("Source:", S3_SOURCE)
print("Upload:", S3_UPLOAD_PREFIX + ARCHIVE_BASENAME + "." + ARCHIVE_FORMAT, "| auto-terminate:", AUTO_TERMINATE)
"""

INSTALL_MD = """## 2. Install and verify runtime dependencies

This is the clean12 render notebook's install cell, unchanged: the kit's batch driver runs on this kernel's Python, and it requires `wfdb>=4.3`. `pandas` is also used by the plan audit."""

INSTALL_EXTRA = """
import subprocess
import urllib.request
from collections import defaultdict
from pathlib import PurePosixPath

import pandas as pd
from botocore.exceptions import ClientError
STAGES_DONE.add("install")
"""

DOWNLOAD_MD = """## 3. Download, verify and extract the signals (Account 1)

This is the clean12 render notebook's download cell, unchanged:
- it reads the `.sha256` sidecar, downloads the archive across accounts and hashes all of it;
- it extracts safely, rejecting links and any path outside the destination;
- a completion marker lets a rerun skip the work.

The first S3 call is where a missing cross-account grant shows up. `AccessDenied` here means the Account 1 bucket policy or the Account 2 role policy (top cell) is missing a statement."""

DOWNLOAD_EXTRA = """
def ec2_identity():
    \"\"\"This instance's id, region and type from IMDSv2 (used by §8 and §11).\"\"\"
    token = urllib.request.urlopen(urllib.request.Request(
        "http://169.254.169.254/latest/api/token", method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"}), timeout=3).read().decode()
    document = urllib.request.urlopen(urllib.request.Request(
        "http://169.254.169.254/latest/dynamic/instance-identity/document",
        headers={"X-aws-ec2-metadata-token": token}), timeout=3).read().decode()
    doc = json.loads(document)
    return {"instance_id": doc["instanceId"], "region": doc["region"], "instance_type": doc["instanceType"]}

STAGES_DONE.add("download")
"""

KIT_MD = """## 4. Kit bundle: verify, extract, patch gate

This is the clean12 render notebook's kit cell, unchanged. It finds `ecg_image_v1_code.tar.gz` next to this notebook, or falls back to `S3_KIT_SOURCE`. The bundle must match its `.sha256` sidecar, and rendering is refused unless every kit patch marker and the fixed font are present. The wrinkled-paper photographs come from the same bundle."""

KIT_EXTRA = """
TEXTURE_DIR = Path(KIT_DIR) / "CreasesWrinkles" / "wrinkles-dataset"
_textures = sorted(p.name for p in TEXTURE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
if len(_textures) != 19:
    raise RuntimeError(f"Expected the kit's 19 wrinkled-paper photographs, found {len(_textures)}")
print(f"{len(_textures)} wrinkled-paper photographs in {TEXTURE_DIR}")
STAGES_DONE.add("kit")
"""

RECORDS_MD = """## 5. Discover and validate the 113,358 records

Records come from `manifest.csv`, never from globbing. Each header must be 12 leads × 5,000 samples at 500 Hz, and the label totals must equal `_checks/final_counts.csv`. These are the same checks the clean12 render notebook makes."""

RECORDS = """with (INPUT_DIR / "manifest.csv").open(newline="", encoding="utf-8") as handle:
    source_rows = list(csv.DictReader(handle))
if len(source_rows) != EXPECTED_RECORDS or len({r["record_path"] for r in source_rows}) != EXPECTED_RECORDS:
    raise ValueError(f"manifest.csv lists {len(source_rows):,} rows / "
                     f"{len({r['record_path'] for r in source_rows}):,} unique; expected {EXPECTED_RECORDS:,}")

site_counts, label_totals, patients, records = Counter(), Counter(), set(), []
for row in source_rows:
    rel = PurePosixPath(row["record_path"])
    if (rel.is_absolute() or ".." in rel.parts or rel.suffix != ".hea"
            or rel.parts[0] != row["site"] or rel.parts[1] != "WFDB"):
        raise ValueError(f"Unexpected record_path: {row['record_path']!r}")
    header = INPUT_DIR.joinpath(*rel.parts)
    if not header.is_file():
        raise FileNotFoundError(f"Manifest record missing from extracted tree: {rel}")
    first = header.read_text(encoding="utf-8", errors="replace").split("\\n", 1)[0].split()
    if (len(first) < 4 or first[1] != "12"
            or float(first[2].split("/")[0]) != EXPECTED_FS or int(first[3]) != EXPECTED_SAMPLES):
        raise ValueError(f"{rel}: header {' '.join(first[:4])!r} is not 12 x {EXPECTED_SAMPLES} @ {EXPECTED_FS} Hz")
    site_counts[row["site"]] += 1
    patients.add(row["patient_key"])
    for condition in CONDITIONS:
        label_totals[condition] += int(row[condition])
    records.append({"record_path": row["record_path"], "site": row["site"], "labels": row["labels"],
                    "image_path": (PurePosixPath("images") / rel.with_suffix(".png")).as_posix()})

with (INPUT_DIR / "_checks" / "final_counts.csv").open(newline="", encoding="utf-8") as handle:
    verified_totals = {row["condition"]: int(row["total"]) for row in csv.DictReader(handle)}
if verified_totals != dict(label_totals):
    raise ValueError(f"Label totals {dict(label_totals)} differ from _checks/final_counts.csv {verified_totals}")
record_by_path = {row["record_path"]: row for row in source_rows}
print(f"{len(records):,} records from {len(patients):,} patients | per site: {dict(sorted(site_counts.items()))}")
print("Records per condition:", dict(label_totals))
STAGES_DONE.add("records")
"""

WORKERS_MD = """## 6. Worker modules

Two modules are written into `WORK_DIR`. The spawned pool processes import them, so `WORK_DIR` is put on `PYTHONPATH`.
- `ecg_clean_worker.py` is the clean12 render worker, verbatim: header parsing, millivolt loading, canonical `.dat` staging, the kit call, and the 1024x768 fit.
- `augment_worker.py` contains the stage-4 copy, the balanced planner, and the pool task. One task makes one kit call for a batch of records, then augments each grey sheet and writes it out."""

WORKERS = """CLEAN_WORKER_SOURCE = r'''__CLEAN_WORKER__'''

AUGMENT_WORKER_SOURCE = r'''__AUGMENT_WORKER__'''

for _name, _source in (("ecg_clean_worker.py", CLEAN_WORKER_SOURCE), ("augment_worker.py", AUGMENT_WORKER_SOURCE)):
    _path = WORK_DIR / _name
    compile(_source, str(_path), "exec")
    if not _path.is_file() or _path.read_text() != _source:
        _path.write_text(_source)
WORKER_SHA256 = hashlib.sha256((CLEAN_WORKER_SOURCE + AUGMENT_WORKER_SOURCE).encode()).hexdigest()
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
if str(WORK_DIR) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(WORK_DIR), os.environ.get("PYTHONPATH")]))
importlib.invalidate_caches()
import ecg_clean_worker
import augment_worker as AW
ecg_clean_worker = importlib.reload(ecg_clean_worker)
AW = importlib.reload(AW)

render_config = {
    "kit_dir": KIT_DIR, "kit_python": sys.executable, "kit_seed": KIT_SEED,
    "kit_fixed_font": KIT_FIXED_FONT, "kit_render_flags": KIT_RENDER_FLAGS, "kit_env": KIT_ENV,
    "kit_scratch_dir": str(KIT_SCRATCH_DIR), "max_duration_s": MAX_DURATION_S,
    "png_compress_level": PNG_COMPRESS_LEVEL,
}
WORKER_CONFIG = {**render_config, "seed": AUGMENT_SEED, "texture_dir": str(TEXTURE_DIR),
                 "input_dir": str(INPUT_DIR), "output_dir": str(OUTPUT_DIR)}
KIT_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
print("Workers written; SHA-256", WORKER_SHA256[:16], "| palettes", AW.AUG_PALETTES)
STAGES_DONE.add("workers")
"""

PLAN_MD = """## 7. Balanced plan, run configuration and balance audit

`AW.balanced_plan` assigns every record one level of each balanced factor. It is deterministic for a given `AUGMENT_SEED`, so a rerun reproduces it exactly. This cell writes into the output folder, which is packaged with the images:
- `augmentation_plan.csv`;
- `balance_by_condition.csv` (count of each condition × factor × level);
- `record_manifest.csv` (the clean12 labels, for joins);
- `config.json`.

It stops unless every factor is balanced to ±1 overall and inside every `(site, label set)` stratum. A config lock refuses to mix images made with different settings in one folder. It also defines the pool runner used by §8, §8b and §9, and loads the progress log so that already-rendered sheets are not rendered again."""

PLAN = """plan = AW.balanced_plan(records, AUGMENT_SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "images").mkdir(exist_ok=True)

run_config = {
    "dataset": "HEEDB clean12, augmented paper/photo sheet (one per record)",
    "source_uri": S3_SOURCE, "source_sha256": expected_sha256, "source_etag": source_identity["etag"],
    "record_count": len(records), "seed": AUGMENT_SEED, "worker_sha256": WORKER_SHA256,
    "kit_bundle_sha256": KIT_BUNDLE_SHA256, "render": {k: v for k, v in render_config.items()
                                                       if k not in ("kit_dir", "kit_python", "kit_scratch_dir")},
    "augmentation": "kit grey master (clean12 settings) -> stage4_augment.augment with the balanced "
                    "plan -> JPEG q85 4:2:0 round trip -> clean12 to_final (1024x768)",
    "balanced_factors": {k: list(v) for k, v in AW.PLAN_FACTORS.items()},
    "ink_bins": AW.INK_BINS, "tilt_bins_deg": AW.TILT_BINS,
    "grid_palettes": {k: AW.GRID_PALETTES[k] for k in AW.AUG_PALETTES},
    "always_on": {"wrinkle_texture_strength": AW.TEXTURE_STRENGTH, "wrinkle_clip": AW.TEXTURE_CLIP,
                  "light_ramp": [0.10, 0.22], "corner_falloff": [0.06, 0.18], "tint_sigma": 0.012,
                  "brightness": [0.92, 1.06], "contrast": [0.94, 1.08], "paper_level": AW.PAPER_LEVEL_TARGET,
                  "sensor_noise_sigma": [1.5, 4.5], "jpeg_quality": AW.JPEG_QUALITY},
    "when_on": {"perspective_mag": AW.KEYSTONE_MAG, "crease_amplitude": AW.CREASE_AMP,
                "crease_clip": AW.CREASE_CLIP, "crease_width": AW.CREASE_WIDTH, "paper_bow": AW.BOW_AMP,
                "edge_shadow_depth": [0.10, 0.26], "blur_radius_px": AW.DEFOCUS_RADIUS},
}
config_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
lock_file = OUTPUT_DIR / ".augment_config_sha256"
if lock_file.is_file() and lock_file.read_text().strip() != config_hash:
    raise RuntimeError("OUTPUT_DIR holds images made with different settings. Use a fresh OUTPUT_DIR "
                       "(and ARCHIVE_BASENAME) or restore the previous settings.")
lock_file.write_text(config_hash + "\\n")
(OUTPUT_DIR / "config.json").write_text(json.dumps(run_config, indent=2) + "\\n")
shutil.copy2(INPUT_DIR / "manifest.csv", OUTPUT_DIR / "record_manifest.csv")

factors = list(AW.PLAN_FACTORS)
with (OUTPUT_DIR / "augmentation_plan.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["record_path", "image_path", "site", "labels", *factors])
    writer.writeheader()
    for r in records:
        writer.writerow({"record_path": r["record_path"], "image_path": r["image_path"],
                         "site": r["site"], "labels": r["labels"], **plan[r["record_path"]]})

audit_failures = []
for factor, levels in AW.PLAN_FACTORS.items():
    counts = Counter(plan[r["record_path"]][factor] for r in records)
    if set(counts) != set(levels) or max(counts.values()) - min(counts.values()) > 1:
        audit_failures.append(f"{factor}: overall counts {dict(counts)}")
strata = defaultdict(list)
for r in records:
    strata[(r["site"], r["labels"])].append(r["record_path"])
for key, paths in strata.items():
    for factor, levels in AW.PLAN_FACTORS.items():
        counts = Counter(plan[p][factor] for p in paths)
        if max(counts.get(l, 0) for l in levels) - min(counts.get(l, 0) for l in levels) > 1:
            audit_failures.append(f"{factor} in stratum {key}: {dict(counts)}")
if audit_failures:
    raise RuntimeError("Plan is not balanced:\\n- " + "\\n- ".join(audit_failures[:20]))

audit_rows, worst = [], {}
for condition in CONDITIONS:
    paths = [r["record_path"] for r in records if int(record_by_path[r["record_path"]][condition]) == 1]
    if not paths:
        continue
    for factor, levels in AW.PLAN_FACTORS.items():
        counts = Counter(plan[p][factor] for p in paths)
        for level in levels:
            audit_rows.append({"condition": condition, "records": len(paths), "factor": factor, "level": level,
                               "count": counts.get(level, 0), "share": round(counts.get(level, 0) / len(paths), 4)})
        spread = max(counts.get(l, 0) for l in levels) - min(counts.get(l, 0) for l in levels)
        worst[factor] = max(worst.get(factor, 0.0), spread / (len(paths) / len(levels)))
pd.DataFrame(audit_rows).to_csv(OUTPUT_DIR / "balance_by_condition.csv", index=False)
print(f"Plan: {len(records):,} records, balanced to +-1 overall and in all {len(strata)} (site, label set) strata.")
print("Worst per-condition spread (max-min)/expected:", {f: f"{100 * v:.1f}%" for f, v in worst.items()})
display(pd.DataFrame({f: pd.Series(Counter(str(plan[r["record_path"]][f]) for r in records)) for f in factors})
        .T.fillna("").astype(str))

def make_job(r):
    return {"record_path": r["record_path"], "image_path": r["image_path"], "plan": plan[r["record_path"]]}

def run_pool(subset, config, on_row, label):
    \"\"\"Render subset in BATCH_SIZE tasks on WORKERS spawned processes; on_row(row) per record.\"\"\"
    batches = iter([subset[i:i + BATCH_SIZE] for i in range(0, len(subset), BATCH_SIZE)])
    finished_n, last_report, t0 = 0, 0, time.monotonic()
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("spawn"),
                             initializer=AW.init_worker, initargs=(config,)) as pool:
        pending = {}

        def submit_next():
            batch = next(batches, None)
            if batch is not None:
                pending[pool.submit(AW.process_batch, [make_job(r) for r in batch])] = batch

        for _ in range(MAX_PENDING_BATCHES):
            submit_next()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                batch = pending.pop(future)
                try:
                    rows = future.result()
                except Exception as exc:   # task raised outside its per-record handling
                    rows = [{"record_path": r["record_path"], "image_path": r["image_path"], "status": "failed",
                             "error": f"task failure: {type(exc).__name__}: {exc}"[:1000]} for r in batch]
                for row in rows:
                    on_row(row)
                finished_n += len(rows)
                submit_next()
            if finished_n - last_report >= PROGRESS_EVERY or finished_n == len(subset):
                rate = finished_n / max(time.monotonic() - t0, 1e-9)
                print(f"{label}: {finished_n:,}/{len(subset):,} | {rate:.2f} records/s | "
                      f"ETA {(len(subset) - finished_n) / rate / 3600:.1f} h", flush=True)
                last_report = finished_n
    return time.monotonic() - t0

progress_path = WORK_DIR / "generation_progress.csv"
done, failed = {}, {}
if progress_path.is_file():
    with progress_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "success" and (OUTPUT_DIR / row["image_path"]).is_file():
                done[row["record_path"]] = row
                failed.pop(row["record_path"], None)
            elif row["status"] != "success":
                done.pop(row["record_path"], None)
                failed[row["record_path"]] = row

def render_and_log(subset, label):
    \"\"\"run_pool into OUTPUT_DIR, appending every row to the progress log.\"\"\"
    with progress_path.open("a", newline="") as progress_file:
        writer = csv.DictWriter(progress_file, fieldnames=AW.MANIFEST_FIELDS, restval="")
        if progress_file.tell() == 0:
            writer.writeheader()

        def on_row(row):
            writer.writerow(row)
            progress_file.flush()
            if row["status"] == "success":
                done[row["record_path"]] = row
                failed.pop(row["record_path"], None)
            else:
                failed[row["record_path"]] = row

        return run_pool(subset, WORKER_CONFIG, on_row, label)

print(f"Progress log: {len(done):,} sheets already rendered")
STAGES_DONE.add("plan")
"""

PREFLIGHT_MD = """## 8. Preflight (run before the long job)

Takes about a minute. The run continues only if all of these pass:
- **instance:** identity from IMDS, with a warning if this is not a c8a.4xlarge;
- **auto-terminate:** an `ec2:TerminateInstances` **dry run** (only when `AUTO_TERMINATE`);
- **upload:** Account 2 S3 write, read and delete permission at `S3_UPLOAD_PREFIX`;
- **disk** and **RAM** headroom;
- **one real record:** rendered by the kit in a spawned worker, augmented, and checked to be 1024x768 with its planned levels. The sheet goes to a scratch folder, which is deleted afterwards."""

PREFLIGHT = """pf_failures, pf_warnings = [], []
def pf_check(name, fn):
    try:
        print(f"  ok    {name}: {fn()}", flush=True)
    except Exception as exc:
        pf_failures.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}", flush=True)

instance = None
def _pf_instance():
    global instance
    instance = ec2_identity()
    if instance["instance_type"] != "c8a.4xlarge":
        pf_warnings.append(f"instance type is {instance['instance_type']}, not c8a.4xlarge (timings assume it)")
    return instance
pf_check("EC2 instance identity (IMDS)", _pf_instance)

def _pf_terminate():
    if not AUTO_TERMINATE:
        return "AUTO_TERMINATE is off (skipped)"
    assert instance is not None, "no instance identity"
    ec2 = aws_session.client("ec2", region_name=instance["region"])
    try:
        ec2.terminate_instances(InstanceIds=[instance["instance_id"]], DryRun=True)
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "DryRunOperation":
            return f"ec2:TerminateInstances allowed on {instance['instance_id']} (dry run)"
        raise RuntimeError(f"{exc.response['Error']['Code']}: give the instance role ec2:TerminateInstances "
                           "(top cell), or set AUTO_TERMINATE = False") from exc
    raise RuntimeError("dry run unexpectedly succeeded")
pf_check("auto-terminate permission", _pf_terminate)

def _pf_upload():
    up = urlparse(S3_UPLOAD_PREFIX)
    key = "/".join(p for p in (up.path.strip("/"), ".preflight_write_probe.txt") if p)
    body = f"preflight {datetime.now(timezone.utc).isoformat()}\\n".encode()
    s3.put_object(Bucket=up.netloc, Key=key, Body=body)
    try:
        assert int(s3.head_object(Bucket=up.netloc, Key=key)["ContentLength"]) == len(body)
    finally:
        s3.delete_object(Bucket=up.netloc, Key=key)
    return f"write+read+delete OK at s3://{up.netloc}/{key}"
pf_check("S3 upload permission (Account 2)", _pf_upload)

def _pf_disk():
    remaining = len(records) - len(done)
    need = remaining * EST_IMAGE_MIB * 2**20 + 5 * 2**30
    free = shutil.disk_usage(OUTPUT_DIR).free
    assert free >= need, f"{free / 2**30:.0f} GiB free, need {need / 2**30:.0f} GiB for {remaining:,} sheets"
    return f"{free / 2**30:.0f} GiB free, {need / 2**30:.0f} GiB needed for {remaining:,} sheets"
pf_check("disk space", _pf_disk)

def _pf_ram():
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    need = WORKERS * 0.7 + 3   # kit ~0.25 GiB, then augment ~0.6 GiB peak per worker; + parent
    assert total >= need, f"{total:.0f} GiB RAM < {need:.0f} GiB for {WORKERS} workers; lower WORKERS"
    return f"{total:.0f} GiB RAM for {WORKERS} workers"
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
    raise RuntimeError("PREFLIGHT FAIL; fix and rerun this cell:\\n- " + "\\n- ".join(pf_failures))
print("PREFLIGHT PASS")
STAGES_DONE.add("preflight")
"""

BENCH_MD = """## 8b. Throughput benchmark and time estimate (about 5 min)

Renders one full wave, `WORKERS × BATCH_SIZE` records, into the real output folder and logs them, so nothing is wasted. It then estimates the time for the rest. Measured on a 4-core laptop: about 4 s per record per core (kit about 2.6 s, augmentation and PNG about 1.4 s). On the c8a.4xlarge's 16 cores, expect roughly 7–9 h for 113,358 records."""

BENCH = """bench_subset = [r for r in records if r["record_path"] not in done][: WORKERS * BATCH_SIZE]
if bench_subset:
    bench_seconds = render_and_log(bench_subset, "benchmark")
    bench_rows = [done[r["record_path"]] for r in bench_subset if r["record_path"] in done]
    if len(bench_rows) != len(bench_subset):
        raise RuntimeError(f"{len(bench_subset) - len(bench_rows)} benchmark records failed: "
                           f"{[failed[r['record_path']]['error'] for r in bench_subset if r['record_path'] in failed][:3]}")
    rate = len(bench_rows) / bench_seconds
    mean_mib = np.mean([int(row["image_bytes"]) for row in bench_rows]) / 2**20
    remaining = len(records) - len(done)
    print(f"{rate:.2f} records/s with {WORKERS} workers ({WORKERS / rate:.2f} s per record per worker, "
          f"incl. pool start); mean PNG {mean_mib:.2f} MiB")
    print(f"Estimated time for the remaining {remaining:,} records: {remaining / rate / 3600:.1f} h; "
          f"output ~{len(records) * mean_mib / 1024:.0f} GiB in total")
else:
    print("Nothing left to benchmark: every record is already rendered.")
STAGES_DONE.add("benchmark")
"""

GENERATE_MD = """## 9. Generate one augmented sheet per record

Records missing from the progress log are rendered. Failed records are retried `RETRY_PASSES` times. The cell raises unless:
- every record has its sheet (113,358 images for 113,358 records);
- the rendered effect counts equal the plan.

If it raises, nothing is packaged or uploaded, and the instance is not terminated."""

GENERATE = """for stray in (OUTPUT_DIR / "images").rglob("*.part"):   # temp file of a killed worker
    stray.unlink()
started = time.monotonic()
for attempt in range(1 + RETRY_PASSES):
    todo = [r for r in records if r["record_path"] not in done]
    if not todo:
        break
    print(f"{'Retry pass ' + str(attempt) if attempt else 'Render'}: {len(todo):,} records", flush=True)
    render_and_log(todo, f"retry {attempt}" if attempt else "render")

still_failed = [failed.get(r["record_path"], {"record_path": r["record_path"], "error": "not rendered"})
                for r in records if r["record_path"] not in done]
with (OUTPUT_DIR / "failures.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=AW.MANIFEST_FIELDS, restval="")
    writer.writeheader()
    writer.writerows(still_failed)
if still_failed:
    raise RuntimeError(f"{len(still_failed):,} records failed after {RETRY_PASSES} retry pass(es); see "
                       f"{OUTPUT_DIR / 'failures.csv'}. Not packaged, not uploaded, instance left running.")

manifest_rows = [done[r["record_path"]] for r in records]
on_disk = sum(1 for _ in (OUTPUT_DIR / "images").rglob("*.png"))
missing = [row["image_path"] for row in manifest_rows if not (OUTPUT_DIR / row["image_path"]).is_file()]
if missing or on_disk != EXPECTED_RECORDS:
    raise RuntimeError(f"{len(missing)} sheets missing; {on_disk:,} PNGs on disk, expected {EXPECTED_RECORDS:,}")
effect_counts = {}
for factor in AW.PLAN_FACTORS:
    rendered_counts = Counter(str(row[factor]) for row in manifest_rows)
    planned_counts = Counter(str(plan[r["record_path"]][factor]) for r in records)
    if rendered_counts != planned_counts:
        raise RuntimeError(f"{factor}: rendered {dict(rendered_counts)} != planned {dict(planned_counts)}")
    effect_counts[factor] = dict(sorted(rendered_counts.items()))
(OUTPUT_DIR / "effect_counts.json").write_text(json.dumps(effect_counts, indent=2) + "\\n")
with (OUTPUT_DIR / "manifest.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=AW.MANIFEST_FIELDS, restval="")
    writer.writeheader()
    writer.writerows(manifest_rows)
summary = {
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "source": S3_SOURCE, "source_sha256": expected_sha256,
    "records": len(records), "images": on_disk, "failed": 0,
    "image_bytes": sum(int(row["image_bytes"]) for row in manifest_rows),
    "generate_cell_seconds": round(time.monotonic() - started, 1), "config_sha256": config_hash,
    "effect_counts": effect_counts,
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\\n")
print(json.dumps({k: v for k, v in summary.items() if k != "effect_counts"}, indent=2))
print("Effect counts:", effect_counts)
STAGES_DONE.add("generate")
"""

UPLOAD_MD = """## 10. Package and upload to Account 2 S3 (one streaming pass)

The output folder is streamed as a tar straight into an S3 multipart upload. It contains:
- `images/`;
- `manifest.csv` (every effect value for each sheet);
- `augmentation_plan.csv`, `balance_by_condition.csv`, `effect_counts.json`;
- `record_manifest.csv`, `config.json`, `summary.json`;
- `failures.csv` (header only).

No local archive is written. The stream is hashed as it is sent and checked against S3's composite checksum; a mismatch deletes the upload. The `.sha256` sidecar is uploaded last and read back. The uploader is the clean12 render notebook's."""

UPLOAD = """import base64
import io
import threading
from collections import deque
from botocore.config import Config as BotoConfig

__UPLOADER__
UPLOAD_VERIFIED = False
upload_prefix = urlparse(S3_UPLOAD_PREFIX)
if upload_prefix.scheme != "s3" or not upload_prefix.netloc:
    raise ValueError("S3_UPLOAD_PREFIX must be s3://bucket/ or s3://bucket/prefix/")
archive_name = f"{ARCHIVE_BASENAME}.{ARCHIVE_FORMAT}"
upload_bucket = upload_prefix.netloc
upload_key = "/".join(part for part in (upload_prefix.path.strip("/"), archive_name) if part)
uploaded_uri = f"s3://{upload_bucket}/{upload_key}"
image_prefix = OUTPUT_DIR.name + "/images/"
packed = Counter()

def _member_filter(info):
    name = info.name[len(OUTPUT_DIR.name) + 1:]
    if info.name.endswith(".part") or name.startswith("."):   # temp files, config lock
        packed["left_out"] += 1
        return None
    if info.isfile() and info.name.startswith(image_prefix) and info.name.endswith(".png"):
        packed["images"] += 1
    return info

def _pack(stream):
    try:
        with tarfile.open(fileobj=stream, mode="w", copybufsize=8 * 2**20) as archive:
            archive.add(OUTPUT_DIR, arcname=OUTPUT_DIR.name, recursive=True, filter=_member_filter)
        stream.finish()
    except BaseException as exc:
        stream.finish(exc)

upload_client = aws_session.client("s3", config=BotoConfig(
    retries={"max_attempts": 10, "mode": "standard"}, max_pool_connections=64))
stream_transfer = TransferConfig(multipart_threshold=STREAM_PART_BYTES, multipart_chunksize=STREAM_PART_BYTES,
                                 max_concurrency=32, use_threads=True, preferred_transfer_client="classic")
stream_transfer.max_in_memory_upload_chunks = 32  # ~2 GiB of parts in flight

stream = TarUploadStream(STREAM_PART_BYTES, STREAM_BUFFER_BYTES)
packer = threading.Thread(target=_pack, args=(stream,), name="tar-packer", daemon=True)
print(f"Streaming {OUTPUT_DIR.name} -> {uploaded_uri} ...", flush=True)
t0 = time.monotonic()
packer.start()
try:
    upload_client.upload_fileobj(stream, upload_bucket, upload_key,
                                 ExtraArgs={"ChecksumAlgorithm": "SHA256"}, Config=stream_transfer)
except BaseException:
    stream.abort()
    packer.join()
    raise
packer.join()
elapsed = time.monotonic() - t0
archive_sha256 = stream.digest.hexdigest()

remote = upload_client.head_object(Bucket=upload_bucket, Key=upload_key, ChecksumMode="ENABLED")
problems = []
if int(remote["ContentLength"]) != stream.bytes_written:
    problems.append(f"S3 size {int(remote['ContentLength']):,} != streamed {stream.bytes_written:,} bytes")
if remote.get("ChecksumSHA256") != stream.s3_composite_sha256():
    problems.append(f"S3 ChecksumSHA256 {remote.get('ChecksumSHA256')!r} != local {stream.s3_composite_sha256()!r}")
if packed["images"] != EXPECTED_RECORDS:
    problems.append(f"archive has {packed['images']:,} images; expected {EXPECTED_RECORDS:,}")
if problems:
    upload_client.delete_object(Bucket=upload_bucket, Key=upload_key)
    raise RuntimeError("Uploaded archive failed verification and was deleted:\\n- " + "\\n- ".join(problems))

sidecar_body = f"{archive_sha256}  {archive_name}\\n".encode()
(WORK_DIR / (archive_name + ".sha256")).write_bytes(sidecar_body)
upload_client.put_object(Bucket=upload_bucket, Key=upload_key + ".sha256", Body=sidecar_body)
if upload_client.get_object(Bucket=upload_bucket, Key=upload_key + ".sha256")["Body"].read() != sidecar_body:
    raise RuntimeError("SHA-256 sidecar did not read back intact")
UPLOAD_VERIFIED = True
print(f"Uploaded {packed['images']:,} images; {stream.bytes_written / 2**30:.2f} GiB in "
      f"{len(stream.part_digests):,} parts, {elapsed / 60:.1f} min -> {uploaded_uri}")
print(f"S3 composite checksum verified: {remote['ChecksumSHA256']}")
print(f"SHA-256 {archive_sha256} -> {uploaded_uri}.sha256")
STAGES_DONE.add("upload")
"""

TERMINATE_MD = """## 11. Auto-terminate the instance

The instance terminates through the EC2 API only when **all** of these hold:
- `AUTO_TERMINATE` is on;
- §10 verified the upload;
- every pipeline cell (§2–§10, including the preflight and the benchmark) completed in this kernel since §1 was last run;
- **no cell raised an error** in that time. The error log in §1 records every failing cell, including cells re-run by hand after a fix. After any error, restart the kernel and Run All, or terminate the instance yourself.

Before terminating, the cell waits `TERMINATE_GRACE_SECONDS`; interrupting the kernel during that wait cancels. If the API call fails, the cell falls back to `sudo shutdown -h now`. That terminates the instance if its shutdown behaviour is *terminate*; otherwise it stops it.

Termination deletes the root EBS volume (the default *delete on termination*). The archive is already in S3 and verified at that point. Extra volumes you attached yourself may survive."""

TERMINATE = """REQUIRED_STAGES = ["install", "download", "kit", "records", "workers", "plan", "preflight",
                   "benchmark", "generate", "upload"]
missing_stages = [s for s in REQUIRED_STAGES if s not in STAGES_DONE]
blockers = []
if not AUTO_TERMINATE:
    blockers.append("AUTO_TERMINATE is False")
if not globals().get("UPLOAD_VERIFIED"):
    blockers.append("the upload was not verified in this kernel")
if missing_stages:
    blockers.append(f"these cells did not complete since §1: {missing_stages}")
if CELL_ERRORS:
    blockers.append(f"{len(CELL_ERRORS)} cell(s) raised an error since §1, e.g. {CELL_ERRORS[0]}")
if blockers:
    print("NOT terminating; the instance keeps running:\\n- " + "\\n- ".join(blockers))
else:
    identity = globals().get("instance") or ec2_identity()
    print(f"All {len(REQUIRED_STAGES)} pipeline cells completed without errors; upload verified: {uploaded_uri}")
    print(f"Terminating {identity['instance_id']} ({identity['instance_type']}, {identity['region']}) in "
          f"{TERMINATE_GRACE_SECONDS} s; interrupt the kernel to cancel.", flush=True)
    time.sleep(TERMINATE_GRACE_SECONDS)
    try:
        response = aws_session.client("ec2", region_name=identity["region"]).terminate_instances(
            InstanceIds=[identity["instance_id"]])
        print("Termination requested; instance state:",
              response["TerminatingInstances"][0]["CurrentState"]["Name"], flush=True)
    except Exception as exc:
        print(f"EC2 API termination failed ({type(exc).__name__}: {exc}); falling back to OS shutdown", flush=True)
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
"""

NOTES = """## Notes

**Storage at peak:**

| Item | Size |
|---|---|
| Signal archive | 5.5 GiB |
| Extracted records | 13.5 GiB |
| Augmented PNGs (113,358 × ~1.45 MiB) | ~160 GiB |
| OS + packages | ~10 GiB |
| **Total** | **~190 GiB** |

**250 GB gp3** leaves headroom. Set the volume to 1,000 MB/s, so that §10 reads the 160 GiB in minutes rather than the ~25 min it takes at the 125 MB/s default.

**Memory.** Kit processes (~0.25 GiB) and augmentation (~0.6 GiB peak) never overlap within one worker, so 16 workers need about 10 GiB of the 32 GiB.

**matplotlib version.** The kit draws lead labels with matplotlib's default font, and the glyph pixels differ between matplotlib versions. With matplotlib 3.11 (installed by §2's unpinned `pip`), a re-render matched the clean12 sample sheets pixel for pixel; with 3.8.4, only the label text differed (about 4,600 pixels per sheet). Traces and grid are unaffected.

**Time on c8a.4xlarge (estimates; §8b measures).**

| Stage | Time |
|---|---|
| Download + extract | ~5 min |
| Render + augment | ~7–9 h |
| Upload | ~20–40 min |

**Using the output.**
- `manifest.csv` has one row per record: `record_path`, `image_path` and every effect value.
- Join to `record_manifest.csv` on `record_path` for labels and `patient_key`.
- There is no train/val/test split. Split by `patient_key`, using the same split as the clean sheets.
- Check a downloaded copy with `sha256sum -c clean_ecg_signal_images_clean12_augmented.tar.sha256`.

**Changing the design.** Edit `augment_worker.py` (`AUG_PALETTES`, `INK_BINS`, `TILT_BINS`, `PLAN_FACTORS`) and rebuild this notebook with `python3 make_notebook.py`. Use a new `OUTPUT_DIR` and `ARCHIVE_BASENAME`, because the config lock refuses to mix settings."""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def build() -> dict:
    cells = [
        md(INTRO), code(CONFIG.replace("__KIT_SETTINGS__\n", clean12_kit_settings())),
        md(INSTALL_MD), code(clean12_cell("%pip install") + INSTALL_EXTRA),
        md(DOWNLOAD_MD), code(clean12_cell("def split_s3_uri(uri):") + DOWNLOAD_EXTRA),
        md(KIT_MD), code(clean12_cell("KIT_BUNDLE = None") + KIT_EXTRA),
        md(RECORDS_MD), code(RECORDS),
        md(WORKERS_MD), code(WORKERS.replace("__CLEAN_WORKER__", embedded(clean12_worker_source()))
                             .replace("__AUGMENT_WORKER__", embedded((HERE / "augment_worker.py").read_text()))),
        md(PLAN_MD), code(PLAN),
        md(PREFLIGHT_MD), code(PREFLIGHT),
        md(BENCH_MD), code(BENCH),
        md(GENERATE_MD), code(GENERATE),
        md(UPLOAD_MD), code(UPLOAD.replace("__UPLOADER__", uploader_class())),
        md(TERMINATE_MD), code(TERMINATE),
        md(NOTES),
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    NOTEBOOK.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print("wrote", NOTEBOOK)
