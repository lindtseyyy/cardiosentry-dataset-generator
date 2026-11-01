"""Build the kit-v1 -> clean12 cross-evaluation Colab notebook (evaluation only, no training)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "CrossEval_KitV1_on_Clean12_Colab.ipynb"
# Library code lives in two importable files (tested by validate_notebook.py);
# the notebook inlines them verbatim so only the .ipynb has to be uploaded.
TAR_RANGES_SOURCE = (HERE / "tar_ranges.py").read_text()
CROSS_EVAL_LIB_SOURCE = (HERE / "cross_eval_lib.py").read_text()

# Copied verbatim from the clean12 training notebook (section 5, image pipeline);
# validate_notebook.py checks every line still appears there, in this order.
TRAINING_PREPROCESSING_SOURCE = r'''INPUT_SIZE = (768, 1024)                 # (height, width)
ImageFile.LOAD_TRUNCATED_IMAGES = False
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_COLOR_MODE = "RGB"
RESIZE_INTERPOLATION = InterpolationMode.BICUBIC
RESIZE_ANTIALIAS = True

image_transform = transforms.Compose([
    transforms.Resize(
        INPUT_SIZE,
        interpolation=RESIZE_INTERPOLATION,
        antialias=RESIZE_ANTIALIAS,
    ),
    transforms.PILToTensor(),  # keep uint8: 2.25 MiB per native sheet, not 9 MiB
])
GPU_MEAN = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
GPU_STD = torch.tensor(IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
GPU_LUMINANCE = torch.tensor((0.2989, 0.5870, 0.1140), device=DEVICE).view(1, 3, 1, 1)
'''
TRAINING_PREPARE_BATCH_SOURCE = r'''def prepare_image_batch(images, augmentation_factors=None):
    if images.dtype != torch.uint8:
        raise TypeError("The input pipeline requires uint8 RGB batches; normalization happens here once.")
    images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
    images = images.float().mul_(1.0 / 255.0)
    if augmentation_factors is not None:
        factors = augmentation_factors.to(DEVICE, non_blocking=True)
        brightness = factors[:, 0].view(-1, 1, 1, 1)
        contrast = factors[:, 1].view(-1, 1, 1, 1)
        images.mul_(brightness).clamp_(0.0, 1.0)
        gray_mean = (images.mean(dim=(2, 3), keepdim=True) * GPU_LUMINANCE).sum(dim=1, keepdim=True)
        images.sub_(gray_mean).mul_(contrast).add_(gray_mean).clamp_(0.0, 1.0)
    return images.sub_(GPU_MEAN).div_(GPU_STD)
'''


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(True)}


cells = [
    md(r"""# CardioSentry: kit-v1 baseline evaluated on the clean12 validation split (Google Colab, L4)

**Evaluation only; nothing is trained.** This notebook asks a single question:
how well does the old **kit-v1** ConvNeXt-V1 Base model (21 noisy labels) rank
the **clean12** labels, compared with the clean12 model? It does this on the
clean12 validation split, using exactly the clean12 training image pipeline.

1. **Load both checkpoints.** It loads `best.pt` of the kit-v1 baseline run and
   of the clean12 run from S3. Each uses the weight source (EMA or raw) that its
   training selected on validation, as recorded in the checkpoint.
2. **Rebuild the clean12 validation split.** It uses the clean12 run's frozen
   patient split, whose SHA-256 must match the run's config.
3. **Exclude kit-v1 training patients.** Every validation record whose
   `patient_key` appears in the kit-v1 **TRAIN** split (from kit-v1's
   `record_manifest.csv`) is dropped, and the excluded count is printed.
4. **Fetch only the images it needs.** The clean12 archive is a plain 104 GiB
   TAR in S3. Instead of downloading it, the notebook indexes the member
   headers with small byte-range reads, then downloads only the remaining
   validation PNGs (about 6 GB).
5. **Run both models on the same tensors.** Each batch is decoded, resized and
   normalized once with the clean12 training code (copied verbatim), then fed
   to both models.
6. **Map the outputs.** kit-v1 outputs map to clean12 labels: Atrial
   Fibrillation→AF, Atrial Flutter→AFL, Strict STEMI→STEMI, ST
   elevation→STE, LBBB→LBBB, First-degree AV block→1AVB, 2AVB→2AVB,
   Third-degree / complete AV block→3AVB, QT/QTc prolongation→LQT, Sinus
   bradycardia→SB, Sinus tachycardia→STACH, Normal ECG→NORM. The other 9
   kit-v1 outputs are unused.
7. **Report AP.** It reports per-label **average precision** for both models
   and the **macro over the 11 pathologies** (NORM excluded). Only summary
   numbers are saved, to **Google Drive**:
   `MyDrive/CardioSentry/new_runs/cross_eval_kitv1_on_clean12/`
   gets `cross_eval_kitv1_on_clean12.csv` plus a provenance `.json`. No images
   and no per-record predictions are saved. Nothing is written to S3; the AWS
   keys are only used for reading.

**Built-in check.** The clean12 model is re-run here and compared with that
run's own saved `validation_predictions.csv`. If the per-label APs disagree by
more than 0.005, the cell stops, because the image pipeline would not be
reproducing training.

**Reading the result.**
- **AP needs no thresholds.** The kit-v1 thresholds are not used.
- **Validation is not fully independent for the clean12 model.** This split
  chose the clean12 run's best epoch, so its clean12 AP is slightly
  optimistic. The test split never selected anything; it is reported in
  `metrics.csv` of each run.
"""),
    md(r"""## 0. Before you run

**Runtime.** Runtime → Change runtime type → **L4 GPU** or **T4 GPU**. On a T4
the models run in float16 instead of bfloat16, and the reproduction check
still guards the result. High-RAM is not needed. Then Runtime → Run all.

**AWS keys.** Open the key icon (**Secrets**), add `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` (plus `AWS_DEFAULT_REGION=us-east-1` if you like), and
turn on notebook access for each. The IAM user needs the read-only
`colab_s3_cross_eval_policy.json` from this folder. It allows reading:
- the two image archives;
- the run files used: `config_snapshot.json` and `best.pt` from both runs, plus
  `patient_split.csv` and `validation_predictions.csv` from the clean12 run.

**Google Drive.** The config cell mounts Drive (one consent prompt). The
results go to `MyDrive/CardioSentry/new_runs/cross_eval_kitv1_on_clean12/`.

**What it transfers and how long it takes.**

| Step | Data | Time (L4, typical) |
|---|---|---|
| Index the clean12 TAR (113k headers) | ~0.5 GB of headers, ~115k small GETs | 2–5 min |
| Index the kit-v1 TAR (to find `record_manifest.csv`) | ~0.8 GB, ~195k small GETs | 3–8 min |
| Validation PNGs that remain after the exclusion | ~6 GB | 2–6 min |
| Two `best.pt` checkpoints | 2.8 GB | 1–3 min |
| Inference, both models, one pass | — | 5–10 min |

That is about 10 GB of S3 egress (roughly USD 1); the results on Drive are a
few KB. Skip the kit-v1 indexing by
pointing `CS_KITV1_RECORD_MANIFEST` at a local copy of kit-v1's
`record_manifest.csv`, for example uploaded to Colab files or Drive. Its
SHA-256 must equal the one in the kit-v1 run's config (`b3da84a9…`), the same
file as `ECG_500Hz_patient_splits/record_manifest.csv`.

**Overrides** (set before the config cell with `os.environ[...] = ...`):

| Variable | Default |
|---|---|
| `CS_CLEAN12_RUN_NAME` | `convnext_v1_base_clean12_g6e2xlarge_seed42` |
| `CS_KITV1_RUN_PREFIX` | `runs/convnext_v1_base_kitv1/convnext_v1_base_kitv1_clean_g6e2xlarge_seed42` |
| `CS_EXCLUDE_KITV1_SPLITS` | `train` (comma list; `train,validation` is stricter) |
| `CS_DRIVE_OUTPUT_DIR` | `/content/drive/MyDrive/CardioSentry/new_runs/cross_eval_kitv1_on_clean12` |
| `CS_OVERWRITE_OUTPUT` | `0` (an existing result on Drive is never replaced) |
| `CS_EVAL_BATCH_SIZE` | `16` |

The S3 keys in the IAM policy name the two default runs. Other runs need the
policy edited.
"""),
    code(r"""# 1. Install what Colab lacks (Colab's PyTorch/CUDA build is kept).
%pip install -q "timm==1.0.28" "boto3>=1.35.16" "scikit-learn>=1.4" "pandas>=2.1" "pillow>=10" "tqdm>=4.66"
"""),
    code(r"""# 2. Configuration, AWS credentials, S3 access, and device.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gc, hashlib, io, json, platform, threading, time, warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
import sklearn
import timm
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

def env_flag(name, default):
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off", ""}

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def load_aws_credentials():
    '''Copy AWS keys from Colab Secrets into the environment; return where they came from.'''
    source = "environment / boto3 default chain"
    if IN_COLAB:
        from google.colab import userdata
        found = []
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_DEFAULT_REGION"):
            if os.environ.get(name):
                continue
            try:
                value = userdata.get(name)
            except Exception:  # secret absent, or notebook access not granted
                value = None
            if value:
                os.environ[name] = value.strip()
                found.append(name)
        if found:
            source = "Colab Secrets (" + ", ".join(found) + ")"
    if boto3.session.Session().get_credentials() is None:
        raise RuntimeError(
            "No AWS credentials found. In Colab open the key icon (Secrets), add "
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY for an IAM user with "
            "the read-only colab_s3_cross_eval_policy.json, enable notebook access, and re-run."
        )
    return source

AWS_CREDENTIAL_SOURCE = load_aws_credentials()
S3_BUCKET = os.environ.get("CS_S3_BUCKET", "cardiosentry").strip()
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

# Runs and archives.
S3_RUNS_PREFIX = os.environ.get("CS_S3_RUNS_PREFIX", "runs").strip("/")
CLEAN12_RUN_NAME = os.environ.get("CS_CLEAN12_RUN_NAME", "convnext_v1_base_clean12_g6e2xlarge_seed42").strip()
CLEAN12_RUN_PREFIX = f"{S3_RUNS_PREFIX}/{CLEAN12_RUN_NAME}"
KITV1_RUN_PREFIX = os.environ.get(
    "CS_KITV1_RUN_PREFIX",
    "runs/convnext_v1_base_kitv1/convnext_v1_base_kitv1_clean_g6e2xlarge_seed42").strip("/")
CLEAN12_TAR_KEY = os.environ.get("CS_CLEAN12_TAR_KEY", "dataset/clean_ecg_signal_images_clean12.tar")
KITV1_TAR_KEY = os.environ.get("CS_KITV1_TAR_KEY", "dataset/clean_ecg_signal_images_kit_v1.tar")
KIT_TOPDIR = "clean_ecg_signal_images_clean12"           # top folder inside the clean12 archive
KITV1_RECORD_MANIFEST_OVERRIDE = os.environ.get("CS_KITV1_RECORD_MANIFEST", "").strip()
EXCLUDE_KITV1_SPLITS = tuple(
    part.strip() for part in os.environ.get("CS_EXCLUDE_KITV1_SPLITS", "train").split(",") if part.strip())
if not EXCLUDE_KITV1_SPLITS or not set(EXCLUDE_KITV1_SPLITS) <= {"train", "validation", "test"}:
    raise ValueError("CS_EXCLUDE_KITV1_SPLITS must list train/validation/test")

# Output: summary numbers only, to Google Drive (S3 is only read).
DRIVE_MOUNT_POINT = Path("/content/drive")
if IN_COLAB and not (DRIVE_MOUNT_POINT / "MyDrive").is_dir():
    from google.colab import drive
    drive.mount(str(DRIVE_MOUNT_POINT))
DRIVE_OUTPUT_DIR = Path(os.environ.get(
    "CS_DRIVE_OUTPUT_DIR",
    str(DRIVE_MOUNT_POINT / "MyDrive" / "CardioSentry" / "new_runs" / "cross_eval_kitv1_on_clean12"))).expanduser()
if not DRIVE_OUTPUT_DIR.parent.is_dir():
    raise RuntimeError(f"{DRIVE_OUTPUT_DIR.parent} is not available. On Colab this cell mounts Google Drive; "
                       "elsewhere set CS_DRIVE_OUTPUT_DIR to a folder whose parent exists.")
OUTPUT_CSV_NAME = "cross_eval_kitv1_on_clean12.csv"
OUTPUT_JSON_NAME = "cross_eval_kitv1_on_clean12.json"
OUTPUT_CSV_PATH = DRIVE_OUTPUT_DIR / OUTPUT_CSV_NAME
OUTPUT_JSON_PATH = DRIVE_OUTPUT_DIR / OUTPUT_JSON_NAME
OVERWRITE_OUTPUT = env_flag("CS_OVERWRITE_OUTPUT", "0")
AP_TOLERANCE = float(os.environ.get("CS_AP_TOLERANCE", "0.005"))

# Throughput.
INDEX_SEGMENTS = int(os.environ.get("CS_INDEX_SEGMENTS", "1024"))
INDEX_WORKERS = int(os.environ.get("CS_INDEX_WORKERS", "256"))
FETCH_WORKERS = int(os.environ.get("CS_FETCH_WORKERS", "32"))
EVAL_BATCH_SIZE = int(os.environ.get("CS_EVAL_BATCH_SIZE", "16"))
NUM_WORKERS = int(os.environ.get("CS_NUM_WORKERS", str(min(8, os.cpu_count() or 2))))

SCRATCH_ROOT = Path(os.environ.get(
    "CS_SCRATCH_ROOT", "/content/cs_cross_eval" if IN_COLAB else str(Path.home() / "cs_cross_eval"))).resolve()
INDEX_DIR, IMAGE_ROOT = SCRATCH_ROOT / "index", SCRATCH_ROOT / "images"
CHECKPOINT_DIR, OUTPUT_DIR = SCRATCH_ROOT / "checkpoints", SCRATCH_ROOT / "output"
for path in (INDEX_DIR, IMAGE_ROOT, CHECKPOINT_DIR, OUTPUT_DIR):
    path.mkdir(parents=True, exist_ok=True)

S3 = boto3.client("s3", region_name=AWS_REGION, config=BotoConfig(
    max_pool_connections=max(INDEX_WORKERS, FETCH_WORKERS) + 8,
    retries={"max_attempts": 10, "mode": "adaptive"},
))

class S3RangeReader:
    '''Byte-range reads of one S3 object version (every request pinned to the first ETag).'''
    def __init__(self, client, bucket, key):
        head = client.head_object(Bucket=bucket, Key=key)
        self.client, self.bucket, self.key = client, bucket, key
        self.size = int(head["ContentLength"])
        self.etag = head["ETag"]
        self.requests = 0
        self.bytes_read = 0
        self._lock = threading.Lock()

    def read(self, offset, length):
        end = min(offset + length, self.size) - 1
        if offset < 0 or end < offset:
            return b""
        delay = 1.0
        for attempt in range(8):
            try:
                response = self.client.get_object(Bucket=self.bucket, Key=self.key,
                                                  Range=f"bytes={offset}-{end}", IfMatch=self.etag)
                data = response["Body"].read()
                if len(data) != end - offset + 1:
                    raise IOError(f"short read at byte {offset}: {len(data)} of {end - offset + 1}")
                with self._lock:
                    self.requests += 1
                    self.bytes_read += len(data)
                return data
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in ("PreconditionFailed", "412"):
                    raise RuntimeError(f"s3://{self.bucket}/{self.key} changed while being read") from exc
                if code in ("AccessDenied", "403", "NoSuchKey", "404") or attempt == 7:
                    raise
            except (BotoCoreError, OSError):
                if attempt == 7:
                    raise
            time.sleep(delay)
            delay = min(delay * 2, 20.0)

def s3_get_bytes(key):
    return S3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()

def s3_get_json(key):
    return json.loads(s3_get_bytes(key))

def s3_download(key, destination):
    '''Download once; a complete local copy (same size as S3) is reused.'''
    destination = Path(destination)
    size = int(S3.head_object(Bucket=S3_BUCKET, Key=key)["ContentLength"])
    if destination.is_file() and destination.stat().st_size == size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    S3.download_file(S3_BUCKET, key, str(partial), Config=TransferConfig(
        multipart_chunksize=64 * 2**20, max_concurrency=16, use_threads=True))
    if partial.stat().st_size != size:
        raise IOError(f"s3://{S3_BUCKET}/{key}: downloaded {partial.stat().st_size} of {size} bytes")
    os.replace(partial, destination)
    return destination

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

if not torch.cuda.is_available() and not env_flag("CS_ALLOW_CPU", "0"):
    raise RuntimeError("No GPU. Runtime -> Change runtime type -> L4 GPU (CS_ALLOW_CPU=1 runs on CPU, very slowly).")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
# Native bfloat16 needs Ampere or newer (L4/A100). On a T4 (sm_75) is_bf16_supported() is True only
# through slow emulation, so use float16 there; the reproduction check in section 11 still applies.
AMP_DTYPE = (torch.bfloat16 if USE_AMP and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def autocast_context():
    return torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP)

print("AWS credentials:  ", AWS_CREDENTIAL_SOURCE)
print("clean12 run:      ", f"s3://{S3_BUCKET}/{CLEAN12_RUN_PREFIX}/")
print("kit-v1 run:       ", f"s3://{S3_BUCKET}/{KITV1_RUN_PREFIX}/")
print("exclude patients in kit-v1 split(s):", ", ".join(EXCLUDE_KITV1_SPLITS))
print("output (Drive):   ", OUTPUT_CSV_PATH, "(overwrite)" if OVERWRITE_OUTPUT else "(never overwrites)")
print("device:           ", torch.cuda.get_device_name(0) if USE_AMP else "cpu",
      "| AMP", str(AMP_DTYPE).replace("torch.", "") if USE_AMP else "off")
print("versions:          torch", torch.__version__, "| timm", timm.__version__, "| python", platform.python_version())
"""),
    md(r"""## 3. Library code

These two cells are verbatim copies of `tar_ranges.py` (random access into a
plain TAR in S3) and `cross_eval_lib.py` (manifest join, exclusion, AP table).
Both are tested locally by `validate_notebook.py`.
"""),
    code("# 3a. tar_ranges.py (verbatim)\n" + TAR_RANGES_SOURCE),
    code("# 3b. cross_eval_lib.py (verbatim)\n" + CROSS_EVAL_LIB_SOURCE),
    md(r"""## 4. Run identities

Both runs' `config_snapshot.json` are read. The checks:
- **Pipeline.** The kit-v1 run must declare the same backbone and image
  pipeline as clean12: size, color mode, resize and normalization.
- **Archive.** The clean12 archive's `.sha256` sidecar must equal the archive
  hash the clean12 run trained on.
- **Output.** If an earlier result exists on Drive and overwriting is off, the
  notebook stops here, before any large transfer.
"""),
    code(r"""# 4. Load and cross-check the two runs' identities.
clean12_config = s3_get_json(f"{CLEAN12_RUN_PREFIX}/config_snapshot.json")
kitv1_config = s3_get_json(f"{KITV1_RUN_PREFIX}/config_snapshot.json")
CLEAN12_IDENTITY, KITV1_IDENTITY = clean12_config["identity"], kitv1_config["identity"]

if list(CLEAN12_IDENTITY["label_order"]) != CLEAN12_LABELS:
    raise ValueError(f"clean12 run label order {CLEAN12_IDENTITY['label_order']} != {CLEAN12_LABELS}")
KITV1_LABEL_ORDER = list(KITV1_IDENTITY["label_order"])
KITV1_COLUMNS = kitv1_columns_for_clean12(KITV1_LABEL_ORDER)   # raises if a mapped name is missing
print(f"kit-v1 outputs: {len(KITV1_LABEL_ORDER)}; mapped to clean12:")
for label, column in zip(CLEAN12_LABELS, KITV1_COLUMNS):
    print(f"  {KITV1_LABEL_ORDER[column]:<34} -> {label}")

PIPELINE_KEYS = ("model", "input_size_hw", "input_color_mode", "resize_interpolation", "resize_antialias",
                 "resize_preserves_aspect_ratio", "tensor_conversion", "normalization_mean", "normalization_std")
pipeline_mismatch = {key: (CLEAN12_IDENTITY.get(key), KITV1_IDENTITY.get(key))
                     for key in PIPELINE_KEYS if CLEAN12_IDENTITY.get(key) != KITV1_IDENTITY.get(key)}
if pipeline_mismatch:
    message = f"kit-v1 and clean12 runs declare different image pipelines/backbones: {pipeline_mismatch}"
    if not env_flag("CS_ALLOW_PIPELINE_MISMATCH", "0"):
        raise RuntimeError(message + " (CS_ALLOW_PIPELINE_MISMATCH=1 to continue anyway)")
    warnings.warn(message)
print("image pipeline and backbone: identical in both runs" if not pipeline_mismatch else "pipeline mismatch allowed")

EXPECTED = {
    "clean12_archive_sha256": CLEAN12_IDENTITY["dataset_archive_sha256"],
    "clean12_record_manifest_sha256": CLEAN12_IDENTITY["dataset_metadata_sha256"],
    "clean12_render_manifest_sha256": CLEAN12_IDENTITY["dataset_manifest_sha256"],
    "clean12_split_sha256": CLEAN12_IDENTITY["split"]["sha256"],
    "clean12_validation_records": int(clean12_config["runtime"]["split_record_counts"]["validation"]),
    "clean12_images": int(sum(clean12_config["runtime"]["split_image_counts"].values())),
    "kitv1_record_manifest_sha256": KITV1_IDENTITY["dataset_metadata_sha256"],
}
sidecar_sha = s3_get_bytes(CLEAN12_TAR_KEY + ".sha256").decode().split()[0].lower()
if sidecar_sha != EXPECTED["clean12_archive_sha256"]:
    raise RuntimeError(f"{CLEAN12_TAR_KEY}.sha256 is {sidecar_sha}, but the clean12 run trained on "
                       f"{EXPECTED['clean12_archive_sha256']}")
print("clean12 archive sidecar matches the run:", sidecar_sha[:16] + "...")
for key, value in EXPECTED.items():
    print(f"  {key}: {value}")

# Fail before the big transfers if a result already exists on Drive.
existing = [path for path in (OUTPUT_CSV_PATH, OUTPUT_JSON_PATH) if path.exists()]
if existing and not OVERWRITE_OUTPUT:
    raise FileExistsError(f"{existing[0]} already exists; set CS_OVERWRITE_OUTPUT=1 to replace it")
print("an earlier result exists and will be replaced (CS_OVERWRITE_OUTPUT=1)" if existing
      else "no earlier result on Drive: OK")
"""),
    md(r"""## 5. Index the archives and read their manifests

The header chain of each plain TAR is followed with small byte-range reads,
in parallel segments (see `tar_ranges.py`). Every request is pinned to the
archive's ETag. The index proves that the headers form one gap-free chain, so
a false header match inside image data cannot slip through. It is cached
under the scratch folder, so a re-run of this cell in the same runtime is
instant.

Checks:
- **clean12 manifests.** `manifest.csv` (render inventory) and
  `record_manifest.csv` (labels, patient keys) must hash to the values recorded
  by the clean12 run.
- **PNG count.** The number of PNGs must equal the run's image count.
- **kit-v1 manifest.** kit-v1's `record_manifest.csv` must hash to the value
  recorded by the kit-v1 run.
"""),
    code(r"""# 5. Archive indexes and manifests.
def cached_tar_index(reader, name):
    cache = INDEX_DIR / f"{name}.json"
    if cache.is_file():
        stored = json.loads(cache.read_text())
        if stored.get("etag") == reader.etag and stored.get("size") == reader.size:
            print(f"{name}: cached index ({len(stored['members']):,} members)")
            return stored["members"]
    print(f"{name}: indexing s3://{reader.bucket}/{reader.key} ({reader.size / 2**30:.1f} GiB) ...")
    members = build_tar_index(reader, segments=INDEX_SEGMENTS, workers=INDEX_WORKERS)
    print(f"{name}: {reader.requests:,} range requests, {reader.bytes_read / 2**20:,.0f} MiB read")
    temporary = cache.with_suffix(".partial")
    temporary.write_text(json.dumps({"etag": reader.etag, "size": reader.size, "members": members}))
    os.replace(temporary, cache)
    return members

clean12_reader = S3RangeReader(S3, S3_BUCKET, CLEAN12_TAR_KEY)
clean12_index = cached_tar_index(clean12_reader, "clean12")
clean12_png = sum(1 for name, entry in clean12_index.items()
                  if entry["type"] in ("0", "7") and name.startswith(f"{KIT_TOPDIR}/images/") and name.endswith(".png"))
if clean12_png != EXPECTED["clean12_images"]:
    raise RuntimeError(f"clean12 archive holds {clean12_png:,} PNGs; the run saw {EXPECTED['clean12_images']:,}")
print(f"clean12 archive: {clean12_png:,} PNGs (matches the run)")

def read_verified_member(reader, index, name, expected_sha):
    if name not in index:
        raise FileNotFoundError(f"{name} is not in s3://{reader.bucket}/{reader.key}")
    data = read_member(reader, index, name)
    if sha256_bytes(data) != expected_sha:
        raise RuntimeError(f"{name}: sha256 {sha256_bytes(data)} != expected {expected_sha}")
    return data

clean12_manifest_bytes = read_verified_member(
    clean12_reader, clean12_index, f"{KIT_TOPDIR}/manifest.csv", EXPECTED["clean12_render_manifest_sha256"])
clean12_record_bytes = read_verified_member(
    clean12_reader, clean12_index, f"{KIT_TOPDIR}/record_manifest.csv", EXPECTED["clean12_record_manifest_sha256"])
print("clean12 manifest.csv and record_manifest.csv: sha256 match the run")

if KITV1_RECORD_MANIFEST_OVERRIDE:
    kitv1_record_bytes = Path(KITV1_RECORD_MANIFEST_OVERRIDE).read_bytes()
    if sha256_bytes(kitv1_record_bytes) != EXPECTED["kitv1_record_manifest_sha256"]:
        raise RuntimeError(f"{KITV1_RECORD_MANIFEST_OVERRIDE}: sha256 {sha256_bytes(kitv1_record_bytes)} "
                           f"!= kit-v1 run's {EXPECTED['kitv1_record_manifest_sha256']}")
    KITV1_MANIFEST_SOURCE = f"local file {KITV1_RECORD_MANIFEST_OVERRIDE}"
else:
    kitv1_reader = S3RangeReader(S3, S3_BUCKET, KITV1_TAR_KEY)
    kitv1_index = cached_tar_index(kitv1_reader, "kitv1")
    candidates = sorted(name for name in kitv1_index
                        if name == "record_manifest.csv" or name.endswith("/record_manifest.csv"))
    kitv1_record_bytes, KITV1_MANIFEST_SOURCE = None, None
    for name in candidates:
        data = read_member(kitv1_reader, kitv1_index, name)
        if sha256_bytes(data) == EXPECTED["kitv1_record_manifest_sha256"]:
            kitv1_record_bytes, KITV1_MANIFEST_SOURCE = data, f"s3://{S3_BUCKET}/{KITV1_TAR_KEY}::{name}"
            break
    if kitv1_record_bytes is None:
        raise RuntimeError(f"No record_manifest.csv in the kit-v1 archive matches the kit-v1 run "
                           f"({EXPECTED['kitv1_record_manifest_sha256']}); candidates: {candidates}")
print("kit-v1 record_manifest.csv: sha256 matches the kit-v1 run ->", KITV1_MANIFEST_SOURCE)
"""),
    md(r"""## 6. Clean12 validation split, minus kit-v1 training patients

The image/label join is the clean12 training notebook's. The patient split is
the clean12 run's own `patient_split.csv`, and its canonical SHA-256 must
equal the run's. Validation records whose `patient_key` is in kit-v1's TRAIN
split are excluded, and the count is printed.

For information only, the cell also reports:
- the overlap with kit-v1's validation and test splits;
- the site-qualified count. Patient keys never collide across the two
  hospitals, so it should equal the plain count.
"""),
    code(r"""# 6. Validation records to evaluate.
manifest_df = pd.read_csv(io.BytesIO(clean12_manifest_bytes), low_memory=False)
record_df = pd.read_csv(io.BytesIO(clean12_record_bytes), low_memory=False, keep_default_na=False,
                        dtype={"record_path": str, "site": str, "patient_key": str, "labels": str})
clean12_frame = build_clean12_frame(manifest_df, record_df)

split_df = pd.read_csv(io.BytesIO(s3_get_bytes(f"{CLEAN12_RUN_PREFIX}/patient_split.csv")),
                       dtype={"patient_key": str, "split": str}, keep_default_na=False)
patient_split = split_df.set_index("patient_key")["split"]
if split_sha256(patient_split) != EXPECTED["clean12_split_sha256"]:
    raise RuntimeError(f"patient_split.csv canonical sha256 {split_sha256(patient_split)} != the run's "
                       f"{EXPECTED['clean12_split_sha256']}")
clean12_frame = assign_split(clean12_frame, patient_split)
validation_df = clean12_frame[clean12_frame["split"].eq("validation")].reset_index(drop=True)
if len(validation_df) != EXPECTED["clean12_validation_records"]:
    raise RuntimeError(f"{len(validation_df):,} validation records; the run used {EXPECTED['clean12_validation_records']:,}")

kitv1_records = pd.read_csv(io.BytesIO(kitv1_record_bytes), dtype=str, keep_default_na=False, low_memory=False)
eval_df, EXCLUSION = exclude_kitv1_patients(validation_df, kitv1_records, EXCLUDE_KITV1_SPLITS)

print(f"EXCLUDED {EXCLUSION['excluded_records']:,} of {EXCLUSION['validation_records']:,} clean12 validation "
      f"records ({EXCLUSION['excluded_patients']:,} patients) because their patient_key is in the kit-v1 "
      f"{'/'.join(EXCLUDE_KITV1_SPLITS).upper()} split.")
print(f"Evaluating {EXCLUSION['kept_records']:,} records from {EXCLUSION['kept_patients']:,} patients.")
print("for information: validation records whose patient is in kit-v1 "
      + ", ".join(f"{split} {EXCLUSION[f'records_with_patient_in_kitv1_{split}']:,}" for split in ("train", "validation", "test")))
if EXCLUSION.get("excluded_records_site_qualified") not in (None, EXCLUSION["excluded_records"]):
    warnings.warn(f"site-qualified exclusion count {EXCLUSION['excluded_records_site_qualified']:,} differs "
                  f"from the patient_key count {EXCLUSION['excluded_records']:,}")
support = pd.DataFrame({"positives": eval_df[CLEAN12_LABELS].sum(),
                        "of_validation": validation_df[CLEAN12_LABELS].sum()})
print(support.T.to_string())
"""),
    md(r"""## 7. Fetch only the validation PNGs that remain

Each PNG is one byte range of the archive. Every file is checked for:
- the size in the TAR header;
- the size in the render manifest (`image_bytes`);
- the PNG signature.

Files already on disk with the right size are reused.
"""),
    code(r"""# 7. Download the needed images (byte ranges of the clean12 archive).
member_names = [f"{KIT_TOPDIR}/{path}" for path in eval_df["image_path"]]
missing = [name for name in member_names if name not in clean12_index]
if missing:
    raise FileNotFoundError(f"{len(missing):,} images are not in the archive, e.g. {missing[:3]}")
expected_bytes = (pd.to_numeric(eval_df["image_bytes"]).astype(int).tolist()
                  if "image_bytes" in eval_df.columns else [None] * len(eval_df))
eval_df["local_path"] = [str(IMAGE_ROOT / path) for path in eval_df["image_path"]]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

def fetch_image(row):
    name, local, manifest_size = member_names[row], Path(eval_df.at[row, "local_path"]), expected_bytes[row]
    entry = clean12_index[name]
    if manifest_size is not None and entry["size"] != manifest_size:
        raise RuntimeError(f"{name}: archive size {entry['size']} != manifest image_bytes {manifest_size}")
    if local.is_file() and local.stat().st_size == entry["size"]:
        return 0
    data = read_member(clean12_reader, clean12_index, name)
    if not data.startswith(PNG_SIGNATURE):
        raise RuntimeError(f"{name} is not a PNG")
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_suffix(".partial")
    temporary.write_bytes(data)
    os.replace(temporary, local)
    return len(data)

started = time.time()
with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
    fetched = list(tqdm(pool.map(fetch_image, range(len(eval_df))), total=len(eval_df), desc="images"))
seconds = max(time.time() - started, 1e-6)
print(f"{len(eval_df):,} images ready ({sum(1 for n in fetched if n):,} downloaded, "
      f"{sum(fetched) / 2**30:.2f} GiB at {sum(fetched) / 2**20 / seconds:.0f} MiB/s)")
"""),
    md(r"""## 8. Image pipeline: the clean12 training code, verbatim

`INPUT_SIZE`, the transform, the normalization constants and
`prepare_image_batch` are copied character for character from the clean12
training notebook. `validate_notebook.py` re-checks this. Images are decoded
the same way as the training dataset: PIL, `.convert("RGB")`, the transform,
then uint8 batches. They are normalized on the GPU in FP32. Evaluation uses
no augmentation.
"""),
    code("# 8. Clean12 training image pipeline (verbatim) and the evaluation loader.\n"
         + TRAINING_PREPROCESSING_SOURCE + "\n" + TRAINING_PREPARE_BATCH_SOURCE + r"""
# The run's declared pipeline must be the code above.
assert list(INPUT_SIZE) == list(CLEAN12_IDENTITY["input_size_hw"]), CLEAN12_IDENTITY["input_size_hw"]
assert list(IMAGENET_MEAN) == list(CLEAN12_IDENTITY["normalization_mean"])
assert list(IMAGENET_STD) == list(CLEAN12_IDENTITY["normalization_std"])
assert IMAGE_COLOR_MODE == CLEAN12_IDENTITY["input_color_mode"]
assert RESIZE_INTERPOLATION.value == CLEAN12_IDENTITY["resize_interpolation"]
assert RESIZE_ANTIALIAS == CLEAN12_IDENTITY["resize_antialias"]

class CrossEvalImageDataset(Dataset):
    '''Same decode path as the training ECGImageDataset.__getitem__; returns (uint8 image, row).'''
    def __init__(self, paths):
        self.paths = [str(path) for path in paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert(IMAGE_COLOR_MODE)
            tensor = image_transform(image)
        return tensor, index

eval_dataset = CrossEvalImageDataset(eval_df["local_path"])
loader_options = {"batch_size": EVAL_BATCH_SIZE, "shuffle": False, "drop_last": False,
                  "num_workers": NUM_WORKERS, "pin_memory": DEVICE.type == "cuda"}
if NUM_WORKERS > 0:
    loader_options.update(prefetch_factor=4, persistent_workers=False)
eval_loader = DataLoader(eval_dataset, **loader_options)
sample, _ = eval_dataset[0]
print("decoded sample:", tuple(sample.shape), sample.dtype, "| batches:", len(eval_loader),
      "| batch", EVAL_BATCH_SIZE, "| workers", NUM_WORKERS)
"""),
    md(r"""## 9. Load both checkpoints

Each run's `best.pt` is downloaded from S3. Its state dict is the one
training selected on validation: `model_ema` if `selected_weight_source` is
`ema`, `model` if it is `raw`. That state dict is loaded strictly into a
fresh timm ConvNeXt with the run's number of outputs. The checkpoint must
carry the same config fingerprint as the run's `config_snapshot.json`.

`best.pt` also holds optimizer and RNG state, so it is loaded with
`weights_only=False`. These are the project's own training checkpoints.
"""),
    code(r"""# 9. Models.
def load_selected_model(run_prefix, tag, run_config):
    local = s3_download(f"{run_prefix}/best.pt", CHECKPOINT_DIR / tag / "best.pt")
    checkpoint = torch.load(local, map_location="cpu", weights_only=False)
    state, info = select_checkpoint_weights(checkpoint)
    del checkpoint
    gc.collect()
    identity = run_config["identity"]
    if info["config_fingerprint"] and info["config_fingerprint"] != run_config["fingerprint"]:
        raise RuntimeError(f"{tag}: best.pt fingerprint {info['config_fingerprint']} != "
                           f"config_snapshot {run_config['fingerprint']}")
    if info["label_order"] and list(info["label_order"]) != list(identity["label_order"]):
        raise RuntimeError(f"{tag}: best.pt label order differs from config_snapshot.json")
    model = timm.create_model(identity["model"], pretrained=False, num_classes=len(identity["label_order"]))
    model.load_state_dict(state, strict=True)
    del state
    model = model.to(DEVICE, memory_format=torch.channels_last).eval()
    info["checkpoint_bytes"] = local.stat().st_size
    print(f"{tag}: {identity['model']}, {len(identity['label_order'])} outputs, epoch {info['epoch']}, "
          f"selected weights = {info['selected_weight_source']} "
          f"(validation pathology macro AP at training: {info['validation_pathology_only_macro_average_precision']})")
    return model, info

kitv1_model, KITV1_CHECKPOINT = load_selected_model(KITV1_RUN_PREFIX, "kitv1", kitv1_config)
clean12_model, CLEAN12_CHECKPOINT = load_selected_model(CLEAN12_RUN_PREFIX, "clean12", clean12_config)
"""),
    code(r"""# 10. One pass over the images; both models see the identical normalized batch.
@torch.inference_mode()
def predict_both(loader):
    kitv1_parts, clean12_parts, rows = [], [], []
    for pixels, row_index in tqdm(loader, desc="inference"):
        images = prepare_image_batch(pixels)
        with autocast_context():
            kitv1_logits = kitv1_model(images)
            clean12_logits = clean12_model(images)
        kitv1_parts.append(torch.sigmoid(kitv1_logits.float()).cpu().numpy())
        clean12_parts.append(torch.sigmoid(clean12_logits.float()).cpu().numpy())
        rows.append(row_index.numpy())
    rows = np.concatenate(rows)
    if not np.array_equal(rows, np.arange(len(loader.dataset))):
        raise RuntimeError("loader returned rows out of order")
    return np.concatenate(kitv1_parts), np.concatenate(clean12_parts)

started = time.time()
kitv1_probability_all, clean12_probability = predict_both(eval_loader)
INFERENCE_SECONDS = time.time() - started
kitv1_probability = kitv1_probability_all[:, KITV1_COLUMNS]      # clean12 label order
print(f"inference: {len(eval_df):,} images x 2 models in {INFERENCE_SECONDS / 60:.1f} min")
del kitv1_probability_all
"""),
    md(r"""## 11. Average precision, and a check against the clean12 run

AP is computed per label on the filtered validation records:
- the kit-v1 column is the mapped kit-v1 output;
- the clean12 column is the clean12 model's own output.

The macro averages the 11 pathologies (NORM excluded).

**The check:** the clean12 run's saved `validation_predictions.csv` is
restricted to the same records. Its per-label AP must agree with this
notebook's clean12 AP within `CS_AP_TOLERANCE` (0.005). This proves that this
notebook feeds the models the same images and preprocessing as training. The
small remaining difference comes from eager versus compiled kernels and the
GPU type.
"""),
    code(r"""# 11. Summary table and the reproduction check.
targets = eval_df[CLEAN12_LABELS].to_numpy(dtype=np.int8)
summary = ap_summary(targets, kitv1_probability, clean12_probability)

saved = pd.read_csv(io.BytesIO(s3_get_bytes(f"{CLEAN12_RUN_PREFIX}/validation_predictions.csv")))
if "primary_metric" in saved.columns:
    saved = saved[saved["primary_metric"].eq(1)]
saved = saved.drop_duplicates("record_id").set_index("record_id")
absent = set(eval_df["record_path"]) - set(saved.index)
if absent:
    raise RuntimeError(f"{len(absent):,} evaluated records are missing from validation_predictions.csv")
saved = saved.loc[eval_df["record_path"]]
saved_targets = saved[[f"true_{label}" for label in CLEAN12_LABELS]].to_numpy(dtype=np.int8)
if not np.array_equal(saved_targets, targets):
    raise RuntimeError("validation_predictions.csv targets differ from record_manifest.csv labels")
saved_probability = saved[[f"prob_{label}" for label in CLEAN12_LABELS]].to_numpy(dtype=np.float64)
saved_summary = ap_summary(targets, kitv1_probability, saved_probability)
ap_gap = (summary.set_index("label")["clean12_ap"] - saved_summary.set_index("label")["clean12_ap"]).abs()
probability_gap = np.abs(saved_probability - clean12_probability)
VERIFICATION = {
    "max_abs_ap_gap_vs_saved_validation_predictions": float(ap_gap.max()),
    "per_label_ap_gap": {label: float(value) for label, value in ap_gap.items()},
    "probability_abs_gap_median": float(np.median(probability_gap)),
    "probability_abs_gap_p99": float(np.quantile(probability_gap, 0.99)),
    "probability_abs_gap_max": float(probability_gap.max()),
    "tolerance": AP_TOLERANCE,
}
print(f"reproduction check: max |AP(this notebook) - AP(saved run predictions)| = {ap_gap.max():.5f} "
      f"(tolerance {AP_TOLERANCE}); probability |diff| median {VERIFICATION['probability_abs_gap_median']:.2e}, "
      f"p99 {VERIFICATION['probability_abs_gap_p99']:.2e}")
if ap_gap.max() > AP_TOLERANCE:
    raise RuntimeError("This notebook does not reproduce the clean12 run's own validation scores; "
                       "the image pipeline differs from training. Nothing was written to Drive.")

display_table = summary.copy()
for column in ("prevalence", "kitv1_ap", "clean12_ap", "ap_difference"):
    display_table[column] = display_table[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
display_table["positives"] = display_table["positives"].map(lambda value: "" if pd.isna(value) else f"{int(value):,}")
print(f"\nAverage precision on {EXCLUSION['kept_records']:,} clean12 validation records "
      f"(kit-v1 {'/'.join(EXCLUDE_KITV1_SPLITS)} patients excluded):\n")
print(display_table.to_string(index=False))
"""),
    md(r"""## 12. Save the summary to Google Drive

Two files are written to `MyDrive/CardioSentry/new_runs/cross_eval_kitv1_on_clean12/`,
and both hold summary numbers only:
- **`cross_eval_kitv1_on_clean12.csv`.** One row per clean12 label plus
  `MACRO_11_PATHOLOGIES`. Columns: `label`, `kitv1_output`, `records`,
  `positives`, `prevalence`, `kitv1_ap`, `clean12_ap`, and `ap_difference`
  (clean12 − kit-v1).
- **`cross_eval_kitv1_on_clean12.json`.** Provenance: the exclusion counts,
  checkpoint epochs and weight sources, the dataset/split/manifest hashes, the
  reproduction check, the GPU and the package versions.

Each file is written to `*.partial`, size-checked, then renamed. Unless
`CS_OVERWRITE_OUTPUT=1`, an existing result is never replaced. Drive is
flushed at the end so the files are on Drive before the runtime goes away.
"""),
    code(r"""# 12. Write the summary to Google Drive.
csv_columns = ["label", "kitv1_output", "records", "positives", "prevalence", "kitv1_ap", "clean12_ap", "ap_difference"]
local_csv = OUTPUT_DIR / OUTPUT_CSV_NAME
summary[csv_columns].to_csv(local_csv, index=False, float_format="%.6f")

provenance = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "question": "kit-v1 baseline vs clean12 model, per-label AP on clean12 validation records",
    "clean12_run": f"s3://{S3_BUCKET}/{CLEAN12_RUN_PREFIX}/",
    "kitv1_run": f"s3://{S3_BUCKET}/{KITV1_RUN_PREFIX}/",
    "clean12_checkpoint": {k: v for k, v in CLEAN12_CHECKPOINT.items() if k != "label_order"},
    "kitv1_checkpoint": {k: v for k, v in KITV1_CHECKPOINT.items() if k != "label_order"},
    "label_mapping_kitv1_to_clean12": KITV1_TO_CLEAN12,
    "exclusion": EXCLUSION,
    "kitv1_record_manifest_source": KITV1_MANIFEST_SOURCE,
    "expected_identities": EXPECTED,
    "clean12_archive": {"key": CLEAN12_TAR_KEY, "etag": clean12_reader.etag, "bytes": clean12_reader.size},
    "macro_definition": "unweighted mean of per-label AP over the 11 pathologies (NORM excluded)",
    "reproduction_check": VERIFICATION,
    "image_pipeline": {"input_size_hw": list(INPUT_SIZE), "normalization_mean": list(IMAGENET_MEAN),
                       "normalization_std": list(IMAGENET_STD), "color_mode": IMAGE_COLOR_MODE,
                       "resize_interpolation": RESIZE_INTERPOLATION.value, "resize_antialias": RESIZE_ANTIALIAS},
    "runtime": {"device": torch.cuda.get_device_name(0) if USE_AMP else "cpu",
                "amp_dtype": str(AMP_DTYPE) if USE_AMP else None, "eval_batch_size": EVAL_BATCH_SIZE,
                "inference_seconds": round(INFERENCE_SECONDS, 1), "torch": torch.__version__,
                "timm": timm.__version__, "scikit_learn": sklearn.__version__, "python": platform.python_version()},
}
local_json = OUTPUT_DIR / OUTPUT_JSON_NAME
local_json.write_text(json.dumps(provenance, indent=2, default=str))

for local, destination in ((local_csv, OUTPUT_CSV_PATH), (local_json, OUTPUT_JSON_PATH)):
    print("saved:", save_summary_file(local.read_bytes(), destination, OVERWRITE_OUTPUT))
if IN_COLAB:
    from google.colab import drive
    drive.flush_and_unmount()
    print("Google Drive flushed.")
print(local_csv.read_text())
"""),
    code(r"""# 13. Optional: release the Colab runtime (CS_RELEASE_RUNTIME=1). The results are already on Drive.
if IN_COLAB and env_flag("CS_RELEASE_RUNTIME", "0"):
    from google.colab import runtime
    print("releasing the Colab runtime ...")
    runtime.unassign()
else:
    print("Done. The runtime is kept; Runtime -> Disconnect and delete runtime when finished.")
"""),
]

for index, cell in enumerate(cells):
    cell["id"] = f"cs-xeval-kitv1-clean12-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "provenance": []},
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1))
print(f"wrote {OUTPUT} ({len(cells)} cells)")
