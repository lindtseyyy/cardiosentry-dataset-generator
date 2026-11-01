"""Build the self-contained CardioSentry ConvNeXt-V2 Tiny V1-image EC2 notebook."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "CardioSentry_ConvNeXtV2_Tiny_V1Img_EC2.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    md(r"""# CardioSentry — ConvNeXt-V2 Tiny fine-tuning on the V1 ECG image corpus

This notebook fine-tunes one **ConvNeXt-V2 Tiny** model on the V1 clean-view
ECG image corpus (1024×768 PNG sheets rendered from 10 s 12-lead signals). It
is designed for an AWS `g5.2xlarge` (8 vCPU, 32 GiB RAM, one NVIDIA A10G with
about 22 GiB usable accelerator memory) and does not depend on Google Colab.

## Dataset: clean views only

The corpus ships exactly one `clean` view per ECG for training, validation,
and test. There is a single dataset mode, so every run with the same archive
trains and evaluates on the same data identity. The benchmark cell reports the
time estimates for this mode before committing to full training.

## Fixed training contract

| setting | choice |
|---|---|
| task | multi-label; twelve independent logits in this order: `NORM`, `AF`, `IAVB`, `LBBB`, `RBBB`, `PAC`, `PVC`, `LAFB`, `LAE`, `TInv`, `LQT`, `PRWP` |
| pretrained model | `convnextv2_tiny.fcmae_ft_in22k_in1k_384` |
| input | 768×1024 RGB, bicubic resize (a no-op on native V1 sheets) |
| loss | per-label `BCEWithLogitsLoss(pos_weight=training negatives / training positives)`; no auxiliary penalty in the baseline |
| contradiction | disabled (`CONTRADICTION_LAMBDA = 0.0`); the noisy-OR term `p(NORM) × (1 - ∏(1-p(pathology)))` is available only as a later ablation |
| optimizer | AdamW, peak head LR `1e-4`, layer-wise LR decay `0.9`, weight decay `0.05`, gradient clip `1.0` |
| schedule | 3-epoch linear warm-up, then cosine decay |
| training | at most 30 epochs; early stopping patience 6 |
| checkpoint monitor | **validation pathology-only Macro Average Precision on the primary clean subset** (conflicting-duplicate rows excluded; NORM detected but never selecting); accuracy never selects a checkpoint |
| thresholds | fixed 0.5 baseline plus per-label F1 thresholds tuned on validation only, then frozen for test |
| subgroup reports | per source and source × condition; support always shown |
| acceleration | FP16 AMP, TF32, channels-last, pinned memory, persistent workers, automatic largest-safe micro-batch probe |

The selected checkpoint was **FCMAE-pretrained on ImageNet-1K**, then
supervised-fine-tuned on ImageNet-22K and ImageNet-1K at 384 px. ConvNeXt-V2 is
fully convolutional with adaptive pooling, so the rectangular input needs no
positional-embedding interpolation.

The default CUDA configuration is seeded but performance-oriented, not bitwise
deterministic. Set `CS_DETERMINISTIC=1` before starting the kernel for stricter
repeatability. Final claims should ideally be confirmed across multiple
independently named seeds.

The original ConvNeXt-V2 weights are distributed under CC-BY-NC-4.0. Confirm
that this is compatible with the intended use before commercial deployment.

Official references: [timm checkpoint model card](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384)
and [Meta training recipes](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/TRAINING.md).
"""),
    md(r"""## 0. Data expected on EC2

The V1 image-generation build packages the clean-view corpus as one
deterministic archive plus a SHA-256 sidecar:

```text
cardiosentry_v1_10s_768x1024_clean_only.tar
cardiosentry_v1_10s_768x1024_clean_only.tar.sha256
```

The archive top level holds the three image subsets (each with `images/` and a
per-subset `metadata.csv`), the record-level `splits/` manifests, and the
corpus-wide `full_metadata.csv` (one row per image, 12 binary label columns,
`primary_metric` flags for the conflicting-duplicate audit):

```text
train/  val_clean/  test_clean/
splits/  full_metadata.csv  generation_config.json  ...
```

This notebook downloads the archive from S3 when it is not already on disk,
verifies its SHA-256 against the sidecar object, and extracts the metadata
plus all clean images. Extraction markers make re-runs incremental: each
selection class is extracted once and skipped afterwards.

Expected corpus: 24,881 / 3,115 / 3,113 train/val/test records with one clean
image each (31,109 images total). The archive is roughly 28 GiB, so provision
a **150 GiB EBS volume** (same-region EC2 keeps the download fast). By default
the local archive is deleted after verified extraction to free room; set
`KEEP_LOCAL_TAR_AFTER_EXTRACT = True` to keep it.

## Automatic S3 persistence

Uploads default to:

```text
s3://cardiosentry-fcmae-758821198776-us-east-1-an/
└── runs/convnext_v2_tiny_V1/<CS_RUN_NAME>/
    ├── best.pt
    ├── last.pt
    ├── training_history.csv
    ├── benchmark.json
    └── ... all final reports, predictions, plots, and s3_upload_manifest.json
```

The notebook performs a real S3 write/read-metadata preflight before training,
uploads recovery-critical checkpoints after every epoch, and uploads the
complete run after final evaluation. Every final object is byte-count checked,
and `s3_upload_manifest.json` records each local SHA-256. If training is
interrupted, re-running the notebook pulls the missing run files back from S3
and resumes from the newest checkpoint instead of starting over. The EC2 IAM
role needs `s3:PutObject` and `s3:GetObject` on the runs prefix plus
`s3:GetObject` on the `datasets/V1/` prefix. Export `CS_S3_UPLOAD=0` only for
an intentional local-only run.
"""),
    code(r"""# 1. Install only the Python packages the training harness needs.
# Use an AWS Deep Learning AMI or another CUDA-ready environment: this cell does
# not replace its PyTorch/CUDA build.
%pip install -q "timm>=1.0.15,<1.1" "scikit-learn>=1.4" "pandas>=2.1" \
    "pillow>=10" "matplotlib>=3.8" "seaborn>=0.13" "tqdm>=4.66" "psutil>=5.9" \
    "boto3>=1.34"
"""),
    code(r"""# 2. Configuration and EC2/GPU preflight — edit the toggles here if needed.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import contextlib, csv, hashlib, json, math, multiprocessing as mp, platform
import random, re, shutil, subprocess, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import boto3
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.optim import param_groups_layer_decay
from PIL import Image, ImageFile
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
)
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

# ----------------------------------------------------------------------------
# Single dataset mode: clean views only. The legacy two-mode flags were
# removed with the switch to the clean-only archive; fail loudly if they are
# still set so a stale environment can never silently mislabel a run.
# ----------------------------------------------------------------------------
def env_flag(name, default):
    return os.environ.get(name, default).strip().lower() not in {
        "0", "false", "no", "off", "",
    }

for _legacy_flag in ("CS_USE_AUGMENTATIONS", "CS_EVAL_STRESS"):
    if env_flag(_legacy_flag, "0"):
        raise RuntimeError(
            f"{_legacy_flag} is set, but this notebook trains on the clean-only "
            "V1 archive, which has no v01–v03 or stress views. Unset the flag "
            "and re-run."
        )
KEEP_LOCAL_TAR_AFTER_EXTRACT = env_flag("CS_KEEP_LOCAL_TAR", "0")

# Local paths. Defaults target the g5.2xlarge 450 GB NVMe instance store;
# every directory below derives from PROJECT_ROOT, so one override moves all
# of them. An explicit CARDIOSENTRY_PROJECT_ROOT always wins and skips the
# mount guard (e.g. for a non-G5 machine or an EBS-only layout).
_PROJECT_ROOT_OVERRIDE = os.environ.get("CARDIOSENTRY_PROJECT_ROOT")
if _PROJECT_ROOT_OVERRIDE:
    PROJECT_ROOT = Path(_PROJECT_ROOT_OVERRIDE).expanduser().resolve()
else:
    PROJECT_ROOT = Path("/mnt/cardiosentry").resolve()
    if not os.path.ismount(PROJECT_ROOT):
        raise RuntimeError(
            "Default project root /mnt/cardiosentry is not a mounted volume. "
            "Mount the g5.2xlarge 450 GB NVMe instance store there (see README), "
            "or export CARDIOSENTRY_PROJECT_ROOT to use another location."
        )
ARCHIVE_DIR = Path(os.environ.get(
    "CS_ARCHIVE_DIR", str(PROJECT_ROOT / "datasets" / "V1")
)).expanduser().resolve()
EXTRACT_ROOT = Path(os.environ.get(
    "CS_EXTRACT_ROOT", str(PROJECT_ROOT / "experiments" / "convnextv2_tiny_v1img_v1" / "data")
)).expanduser().resolve()
RUN_ROOT = Path(os.environ.get(
    "CS_RUN_ROOT", str(PROJECT_ROOT / "runs" / "convnextv2_tiny_v1img_v1")
)).expanduser().resolve()
HF_HOME = Path(os.environ.get(
    "HF_HOME", str(PROJECT_ROOT / "hf_cache")
)).expanduser().resolve()
os.environ["HF_HOME"] = str(HF_HOME)

RUN_NAME = os.environ.get("CS_RUN_NAME", "convnextv2_tiny_v1img_clean_seed42")
RUN_DIR = RUN_ROOT / RUN_NAME
for path in (ARCHIVE_DIR, EXTRACT_ROOT, RUN_DIR, HF_HOME):
    path.mkdir(parents=True, exist_ok=True)

# Persistent S3 destinations. Disable only for an intentional local-only run by
# exporting CS_S3_UPLOAD=0 before starting the notebook kernel.
S3_UPLOAD_ENABLED = env_flag("CS_S3_UPLOAD", "1")
S3_BUCKET = os.environ.get(
    "CS_S3_BUCKET", "cardiosentry-fcmae-758821198776-us-east-1-an"
).strip()
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
S3_DATASET_PREFIX = os.environ.get("CS_S3_DATASET_PREFIX", "datasets/V1").strip("/")
DATASET_TAR_NAME = os.environ.get(
    "CS_DATASET_TAR_NAME", "cardiosentry_v1_10s_768x1024_clean_only.tar"
).strip()
S3_RUNS_PREFIX = os.environ.get(
    "CS_S3_RUNS_PREFIX", "runs/convnext_v2_tiny_V1"
).strip("/")
S3_UPLOAD_EVERY_N_EPOCHS = int(os.environ.get("CS_S3_UPLOAD_EVERY_N_EPOCHS", "1"))
if S3_UPLOAD_EVERY_N_EPOCHS < 1:
    raise ValueError("CS_S3_UPLOAD_EVERY_N_EPOCHS must be at least 1")
S3_RUN_PREFIX = f"{S3_RUNS_PREFIX}/{RUN_NAME}"
S3_CLIENT = boto3.client("s3", region_name=AWS_REGION) if S3_UPLOAD_ENABLED else None
S3_DATASET_KEY = f"{S3_DATASET_PREFIX}/{DATASET_TAR_NAME}"
LOCAL_TAR = ARCHIVE_DIR / DATASET_TAR_NAME

def s3_uri(key):
    return f"s3://{S3_BUCKET}/{key.lstrip('/')}"

def upload_file_to_s3(local_path, key, required=False):
    '''Upload one completed local artifact, then verify the remote byte count.'''
    local_path = Path(local_path)
    if not S3_UPLOAD_ENABLED:
        return None
    if not local_path.is_file():
        message = f"Cannot upload missing local artifact: {local_path}"
        if required:
            raise FileNotFoundError(message)
        warnings.warn(message)
        return None
    key = key.lstrip("/")
    try:
        S3_CLIENT.upload_file(str(local_path), S3_BUCKET, key)
        remote = S3_CLIENT.head_object(Bucket=S3_BUCKET, Key=key)
        if int(remote["ContentLength"]) != local_path.stat().st_size:
            raise RuntimeError(
                f"S3 size verification failed for {s3_uri(key)}: "
                f"local={local_path.stat().st_size}, remote={remote['ContentLength']}"
            )
    except Exception as exc:
        message = f"S3 upload failed for {local_path} -> {s3_uri(key)}: {exc}"
        if required:
            raise RuntimeError(message) from exc
        warnings.warn(message)
        return None
    print("uploaded:", s3_uri(key))
    return s3_uri(key)

def upload_run_files(paths, required=False):
    uploaded = []
    for path in paths:
        path = Path(path)
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        relative = path.relative_to(RUN_DIR).as_posix()
        uri = upload_file_to_s3(path, f"{S3_RUN_PREFIX}/{relative}", required=required)
        if uri:
            uploaded.append(uri)
    return uploaded

def download_file_from_s3(key, local_path, required=False):
    '''Download one run artifact, then verify the local byte count.'''
    local_path = Path(local_path)
    if not S3_UPLOAD_ENABLED:
        return None
    key = key.lstrip("/")
    try:
        expected_bytes = int(S3_CLIENT.head_object(Bucket=S3_BUCKET, Key=key)["ContentLength"])
    except Exception:
        # The startup preflight already proved credentials, so a failed HEAD
        # here means the object is absent (fresh run); that is not an error.
        if required:
            raise FileNotFoundError(f"S3 object not found: {s3_uri(key)}")
        return None
    try:
        temporary = local_path.with_suffix(local_path.suffix + ".tmp")
        S3_CLIENT.download_file(S3_BUCKET, key, str(temporary))
        if int(temporary.stat().st_size) != expected_bytes:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"S3 size verification failed for {s3_uri(key)}: "
                f"expected={expected_bytes}"
            )
        os.replace(temporary, local_path)
    except Exception as exc:
        message = f"S3 download failed for {s3_uri(key)} -> {local_path}: {exc}"
        if required:
            raise RuntimeError(message) from exc
        warnings.warn(message)
        return None
    print("downloaded:", s3_uri(key), "->", local_path)
    return local_path

def restore_missing_run_files_from_s3(filenames):
    '''Fresh-machine resume: fetch run files that are absent locally.

    Never overwrites local files; returns the restored names. Missing S3
    objects are normal for a brand-new run and are silently skipped.
    '''
    if not S3_UPLOAD_ENABLED:
        return []
    restored = []
    for name in filenames:
        local_path = RUN_DIR / name
        if local_path.is_file():
            continue
        if download_file_from_s3(f"{S3_RUN_PREFIX}/{name}", local_path) is not None:
            restored.append(name)
    if restored:
        print("restored from S3:", ", ".join(restored))
    return restored

def sync_complete_run_to_s3(required=True):
    '''Upload every final run artifact and publish a SHA-256 manifest last.'''
    if not S3_UPLOAD_ENABLED:
        print("S3 upload disabled; final artifacts remain local at", RUN_DIR)
        return None
    paths = sorted(
        path for path in RUN_DIR.rglob("*")
        if path.is_file()
        and not path.name.endswith(".tmp")
        and path.name != "s3_upload_manifest.json"
    )
    if not paths:
        raise RuntimeError(f"No final artifacts found under {RUN_DIR}")
    records = []
    for path in paths:
        relative = path.relative_to(RUN_DIR).as_posix()
        key = f"{S3_RUN_PREFIX}/{relative}"
        uri = upload_file_to_s3(path, key, required=required)
        if uri:
            records.append({
                "relative_path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "s3_uri": uri,
            })
    manifest_path = RUN_DIR / "s3_upload_manifest.json"
    manifest = {
        "uploaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": S3_BUCKET,
        "run_prefix": S3_RUN_PREFIX,
        "run_uri": s3_uri(S3_RUN_PREFIX) + "/",
        "artifacts": records,
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(temporary, manifest_path)
    upload_file_to_s3(
        manifest_path,
        f"{S3_RUN_PREFIX}/{manifest_path.name}",
        required=required,
    )
    print(f"Final S3 sync verified: {len(records):,} artifacts plus upload manifest")
    print("S3 run URI:", s3_uri(S3_RUN_PREFIX) + "/")
    return manifest

# V1 corpus identity (from the dataset-generation build).
LABELS = ["NORM", "AF", "IAVB", "LBBB", "RBBB", "PAC", "PVC",
          "LAFB", "LAE", "TInv", "LQT", "PRWP"]
PATHOLOGY_LABELS = [label for label in LABELS if label != "NORM"]
# The clean-only archive carries exactly one view per ECG.
TRAIN_VARIANTS = ["clean"]
EVAL_VARIANTS = ["clean"]
SUBSETS = ("train", "val_clean", "test_clean")
EXPECTED_RECORDS_BY_SPLIT = {"train": 24881, "val": 3115, "test": 3113}

# Training hyperparameters (same recipe as the master-corpus notebook).
MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
INPUT_SIZE = (768, 1024)                 # (height, width)
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 6
WARMUP_EPOCHS = 3
BASE_LR = 1e-4
WEIGHT_DECAY = 0.05
LAYER_DECAY = 0.90
DROP_PATH_RATE = 0.20
EFFECTIVE_BATCH_SIZE = 32
# Baseline uses plain weighted BCE. The noisy-OR contradiction term saturates
# with 11 pathologies (0.5 each => P(any) ~= 0.9995) and would mostly suppress
# NORM; re-enable it only as an explicit ablation with its own CS_RUN_NAME.
CONTRADICTION_LAMBDA = 0.0
GRAD_CLIP_NORM = 1.0
SEED = 42
NUM_WORKERS = int(os.environ.get(
    "CS_NUM_WORKERS", str(min(6, max(1, (os.cpu_count() or 2) - 2)))
))
if NUM_WORKERS < 0:
    raise ValueError("CS_NUM_WORKERS must be nonnegative")
DATALOADER_START_METHOD = os.environ.get("CS_DATALOADER_START_METHOD", "fork").strip()
if NUM_WORKERS > 0 and DATALOADER_START_METHOD not in mp.get_all_start_methods():
    raise ValueError(
        f"Unsupported CS_DATALOADER_START_METHOD={DATALOADER_START_METHOD!r}; "
        f"available methods: {mp.get_all_start_methods()}"
    )
PREFETCH_FACTOR = 2
PROGRESS_UPDATE_EVERY_N_BATCHES = int(os.environ.get(
    "CS_PROGRESS_UPDATE_EVERY_N_BATCHES", "50"
))
if PROGRESS_UPDATE_EVERY_N_BATCHES < 1:
    raise ValueError("CS_PROGRESS_UPDATE_EVERY_N_BATCHES must be at least 1")
AUTO_BATCH_CANDIDATES = [32, 16, 8, 4, 2, 1]  # divisors of effective batch
VRAM_SAFETY_FRACTION = 0.90
USE_AMP = True
DETERMINISTIC = env_flag("CS_DETERMINISTIC", "0")
if DETERMINISTIC:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Benchmark size (number of timed training/validation batches after warm-up).
BENCHMARK_TRAIN_BATCHES = int(os.environ.get("CS_BENCHMARK_TRAIN_BATCHES", "30"))
BENCHMARK_VAL_BATCHES = int(os.environ.get("CS_BENCHMARK_VAL_BATCHES", "10"))
BENCHMARK_WARMUP_BATCHES = 3
COMBINATION_LOW_SUPPORT_THRESHOLD = 30

def gib(value):
    return value / 2**30

print("dataset mode:       clean only (single mode)")
print("Python:             ", sys.version.split()[0])
print("PyTorch:            ", torch.__version__)
print("timm:               ", timm.__version__)
print("CPU count:          ", os.cpu_count())
print(f"RAM:                 {gib(psutil.virtual_memory().total):.1f} GiB")
print("dataset S3 source:  ", s3_uri(S3_DATASET_KEY))
print("local archive:      ", LOCAL_TAR)
print("extract root:       ", EXTRACT_ROOT)
print("run directory:      ", RUN_DIR)
print("HF cache:           ", HF_HOME)
print("S3 uploads enabled: ", S3_UPLOAD_ENABLED)
if S3_UPLOAD_ENABLED:
    print("S3 run destination:", s3_uri(S3_RUN_PREFIX) + "/")
print(f"free extract disk:   {gib(shutil.disk_usage(EXTRACT_ROOT).free):.1f} GiB")

# Prove PutObject and HeadObject access before committing hours to training.
if S3_UPLOAD_ENABLED:
    preflight_key = f"{S3_RUN_PREFIX}/upload_preflight.json"
    preflight_body = json.dumps({
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "region": AWS_REGION,
        "purpose": "CardioSentry V1-image training artifact upload preflight",
    }, indent=2).encode()
    try:
        S3_CLIENT.put_object(Bucket=S3_BUCKET, Key=preflight_key, Body=preflight_body)
        preflight_head = S3_CLIENT.head_object(Bucket=S3_BUCKET, Key=preflight_key)
        if int(preflight_head["ContentLength"]) != len(preflight_body):
            raise RuntimeError("preflight object length differs after upload")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot write and verify {s3_uri(preflight_key)}. Attach an EC2 IAM role "
            "with s3:PutObject and s3:GetObject for this prefix, or set CS_S3_UPLOAD=0 "
            "for an intentional local-only run."
        ) from exc
    print("S3 write preflight: OK ->", s3_uri(preflight_key))

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Use a GPU-enabled AWS Deep Learning AMI/kernel and "
        "verify the NVIDIA driver before training."
    )
gpu = torch.cuda.get_device_properties(0)
print(f"GPU:                 {gpu.name}")
print(f"actual GPU VRAM:     {gib(gpu.total_memory):.1f} GiB")
print("CUDA runtime:        ", torch.version.cuda)
if os.cpu_count() and os.cpu_count() < 8:
    warnings.warn("Fewer than 8 logical CPUs are visible; image decoding may bottleneck the GPU.")
if psutil.virtual_memory().total < 28 * 2**30:
    warnings.warn("Less than 28 GiB RAM is visible; check the EC2 instance configuration.")

torch.backends.cudnn.benchmark = not DETERMINISTIC
torch.backends.cudnn.deterministic = DETERMINISTIC
torch.use_deterministic_algorithms(DETERMINISTIC)
torch.backends.cuda.matmul.allow_tf32 = not DETERMINISTIC
torch.backends.cudnn.allow_tf32 = not DETERMINISTIC
torch.set_float32_matmul_precision("highest" if DETERMINISTIC else "high")
DEVICE = torch.device("cuda:0")

def seed_global_rngs(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_global_rngs(SEED)
print("reproducibility:     ", "deterministic opt-in" if DETERMINISTIC else "seeded performance mode")
"""),
    md(r"""## 3. Download, verify, and extract the V1 archive

This cell ensures the corpus is on local disk: it downloads the archive from
S3 when the local copy is missing, verifies its SHA-256 against the sidecar,
and extracts the metadata plus all clean images. Each selection class writes
a marker recording the archive hash, so re-running extracts only what is
still missing.
"""),
    code(r"""def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()

def expected_tar_sha256() -> str:
    '''Read the SHA-256 authority: local sidecar first, else the S3 sidecar.'''
    local_sidecar = LOCAL_TAR.with_suffix(LOCAL_TAR.suffix + ".sha256")
    if local_sidecar.is_file():
        text = local_sidecar.read_text().split()[0].lower()
    else:
        dataset_client = boto3.client("s3", region_name=AWS_REGION)
        try:
            response = dataset_client.get_object(
                Bucket=S3_BUCKET, Key=S3_DATASET_KEY + ".sha256"
            )
            text = response["Body"].read().decode().split()[0].lower()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot read the dataset checksum sidecar "
                f"{s3_uri(S3_DATASET_KEY + '.sha256')}. The EC2 IAM role needs "
                f"s3:GetObject on {S3_DATASET_PREFIX}/, or place "
                f"{local_sidecar.name} beside the archive."
            ) from exc
    if len(text) != 64:
        raise ValueError("Malformed SHA-256 sidecar for the dataset archive")
    return text

def download_dataset_tar(expected_sha: str) -> None:
    if LOCAL_TAR.is_file() and sha256_file(LOCAL_TAR) == expected_sha:
        print("local archive already verified; skipping download")
        return
    if LOCAL_TAR.is_file():
        print("local archive hash differs; re-downloading from S3")
        LOCAL_TAR.unlink()
    dataset_client = boto3.client("s3", region_name=AWS_REGION)
    try:
        total = dataset_client.head_object(
            Bucket=S3_BUCKET, Key=S3_DATASET_KEY
        )["ContentLength"]
    except Exception as exc:
        raise RuntimeError(
            f"Dataset archive not found at {s3_uri(S3_DATASET_KEY)}. "
            f"Check CS_S3_BUCKET/CS_S3_DATASET_PREFIX/CS_DATASET_TAR_NAME."
        ) from exc
    free = shutil.disk_usage(ARCHIVE_DIR).free
    if free < total + 5 * 2**30:
        raise RuntimeError(
            f"Insufficient disk for the dataset download: {gib(free):.1f} GiB free, "
            f"{gib(total):.1f} GiB archive plus margin required."
        )
    progress = tqdm(total=total, unit="B", unit_scale=True, desc="downloading V1 archive")
    def callback(chunk):
        progress.update(chunk)
    temporary = LOCAL_TAR.with_suffix(LOCAL_TAR.suffix + ".tmp")
    try:
        dataset_client.download_file(
            S3_BUCKET, S3_DATASET_KEY, str(temporary), Callback=callback
        )
    finally:
        progress.close()
    print("verifying download hash ...", flush=True)
    if sha256_file(temporary) != expected_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Downloaded archive SHA-256 mismatch; partial file removed")
    os.replace(temporary, LOCAL_TAR)
    print(f"download verified: {LOCAL_TAR} ({gib(LOCAL_TAR.stat().st_size):.1f} GiB)")

EXPECTED_TAR_SHA256 = expected_tar_sha256()
print("expected archive sha256:", EXPECTED_TAR_SHA256[:16] + "...")

# Extraction selection classes. The clean-only archive holds exactly these,
# so every class is always required. Markers make re-runs incremental.
EXTRACTION_CLASSES = {
    "meta": {"patterns": None, "exclude": ["*/images/*.png"]},
    "train-clean": {"patterns": ["train/images/*__clean.png"], "exclude": []},
    "eval-clean": {"patterns": ["val_clean/images/*.png", "test_clean/images/*.png"],
                   "exclude": []},
}
REQUIRED_CLASSES = ["meta", "train-clean", "eval-clean"]

def extraction_marker(name):
    return EXTRACT_ROOT / f".v1_extract_{name}.json"

def extraction_complete(name):
    marker = extraction_marker(name)
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("sha256") == EXPECTED_TAR_SHA256

missing_classes = [name for name in REQUIRED_CLASSES if not extraction_complete(name)]
if missing_classes and not LOCAL_TAR.is_file():
    print("local archive missing but extraction is incomplete; downloading ...")
    download_dataset_tar(EXPECTED_TAR_SHA256)
elif LOCAL_TAR.is_file():
    print("hashing local archive ...", flush=True)
    actual_sha = sha256_file(LOCAL_TAR)
    if actual_sha != EXPECTED_TAR_SHA256:
        raise RuntimeError(
            f"Local archive SHA-256 mismatch: expected {EXPECTED_TAR_SHA256}, "
            f"got {actual_sha}. Delete it and re-run to re-download."
        )
    print("local archive hash: OK")
elif not missing_classes:
    print("extraction already complete; no archive needed")

for name in missing_classes:
    spec = EXTRACTION_CLASSES[name]
    command = ["tar", "-xf", str(LOCAL_TAR), "-C", str(EXTRACT_ROOT),
               "--no-same-owner", "--no-same-permissions"]
    for pattern in spec["exclude"]:
        command += ["--exclude", pattern]
    if spec["patterns"]:
        command += ["--wildcards"] + spec["patterns"]
    print(f"extracting {name} ...", flush=True)
    subprocess.run(command, check=True)
    temporary = extraction_marker(name).with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "class": name, "sha256": EXPECTED_TAR_SHA256,
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    os.replace(temporary, extraction_marker(name))

FULL_METADATA_CSV = EXTRACT_ROOT / "full_metadata.csv"
for required in [FULL_METADATA_CSV] + [
    EXTRACT_ROOT / subset / "metadata.csv"
    for subset in ("train", "val_clean", "test_clean")
]:
    if not required.is_file():
        raise FileNotFoundError(
            f"Expected {required} after extraction. If the archive layout changed, "
            "inspect the TAR listing before proceeding."
        )

# Count check: every extracted image on disk must be referenced by metadata.
with open(FULL_METADATA_CSV, newline="") as handle:
    referenced_images = {row["image_path"] for row in csv.DictReader(handle)}
on_disk_images = {
    f"{subset}/images/{path.name}"
    for subset in SUBSETS
    if (EXTRACT_ROOT / subset / "images").is_dir()
    for path in (EXTRACT_ROOT / subset / "images").glob("*.png")
}
if not on_disk_images:
    raise RuntimeError("No PNG images found under the extraction root")
if on_disk_images - referenced_images:
    raise RuntimeError(
        f"{len(on_disk_images - referenced_images):,} extracted images are not "
        "referenced by full_metadata.csv. If a different archive was extracted "
        f"to {EXTRACT_ROOT} before, delete that directory and re-run."
    )
print(f"extracted images on disk: {len(on_disk_images):,} "
      f"(classes: {', '.join(sorted(REQUIRED_CLASSES))})")

if LOCAL_TAR.is_file() and not KEEP_LOCAL_TAR_AFTER_EXTRACT:
    print(f"removing local archive to free {gib(LOCAL_TAR.stat().st_size):.1f} GiB ...")
    LOCAL_TAR.unlink()
    print("archive removed; S3 remains the source of truth for re-downloads")
print("Dataset is ready at", EXTRACT_ROOT)
"""),
    md(r"""## 4. Audit the V1 manifest and select the training rows

This cell loads `full_metadata.csv`, checks the corpus structure (expected
record counts, exactly one clean view per record, binary labels,
NORM/pathology semantics, no record or patient group crossing splits), then
builds the train/val/test frames. Rows flagged `primary_metric=0` (the
conflicting-duplicate audit) are reported separately and excluded from the
checkpoint-selection metric.
"""),
    code(r"""metadata_df = pd.read_csv(FULL_METADATA_CSV, low_memory=False)
required_columns = {
    "record_id", "dup_group_id", "patient_id", "variant_id", "variant_type",
    "primary_metric", "metric_note", "split", "subset", "image_path",
    "source_dataset", *LABELS,
}
missing_columns = sorted(required_columns - set(metadata_df.columns))
if missing_columns:
    raise ValueError(f"full_metadata.csv is missing required columns: {missing_columns}")

metadata_df["split"] = metadata_df["split"].astype(str).str.lower().str.strip()
if set(metadata_df["split"].unique()) != {"train", "val", "test"}:
    raise ValueError(f"Expected train/val/test only; got {sorted(metadata_df['split'].unique())}")
metadata_df["subset"] = metadata_df["subset"].astype(str).str.strip()
metadata_df["variant_id"] = metadata_df["variant_id"].astype(str).str.strip()

for label in LABELS:
    metadata_df[label] = pd.to_numeric(metadata_df[label], errors="raise").astype(np.int8)
    if not metadata_df[label].isin([0, 1]).all():
        raise ValueError(f"{label} contains values outside 0/1")
if (metadata_df[LABELS].sum(axis=1) == 0).any():
    raise ValueError("full_metadata.csv contains rows with no positive label")

# NORM must be exclusive with every pathology (verified in the V1 build).
pathology_sum = metadata_df[PATHOLOGY_LABELS].sum(axis=1)
normal_semantics = metadata_df["NORM"].eq(1) == pathology_sum.eq(0)
if not normal_semantics.all():
    bad = metadata_df.loc[~normal_semantics, ["record_id", *LABELS]].head()
    raise ValueError("NORM/pathology target contradiction in full_metadata.csv:\n"
                     + bad.to_string(index=False))

# Expected corpus structure: record counts and per-record variant sets.
observed_records = metadata_df.groupby("split")["record_id"].nunique().to_dict()
if observed_records != EXPECTED_RECORDS_BY_SPLIT:
    raise RuntimeError(
        f"Record counts differ from the V1 build: expected "
        f"{EXPECTED_RECORDS_BY_SPLIT}, observed {observed_records}"
    )
train_variant_sets = metadata_df.loc[
    metadata_df["split"].eq("train")
].groupby("record_id")["variant_id"].agg(set)
if not (train_variant_sets == set(TRAIN_VARIANTS)).all():
    raise RuntimeError("At least one training record does not have exactly the clean view")
eval_variant_sets = metadata_df.loc[
    ~metadata_df["split"].eq("train")
].groupby("record_id")["variant_id"].agg(set)
if not (eval_variant_sets == set(EVAL_VARIANTS)).all():
    raise RuntimeError("At least one val/test record does not have exactly the clean view")
subset_check = metadata_df.groupby(["split", "variant_id"])["subset"].agg(set)
for (split, variant), subsets in subset_check.items():
    expected = "train" if split == "train" else f"{split}_{variant}"
    if subsets != {expected}:
        raise RuntimeError(
            f"Rows ({split}, {variant}) map to subsets {subsets}, need {{{expected}}}"
        )

# Splits must be atomic at the record, duplicate-group, and patient level.
for key in ("record_id", "dup_group_id"):
    split_counts = metadata_df.groupby(key)["split"].nunique()
    if (split_counts > 1).any():
        examples = split_counts[split_counts > 1].index[:10].tolist()
        raise ValueError(f"{key} crosses dataset splits: {examples}")
nonempty_patient = metadata_df[metadata_df["patient_id"].fillna("").astype(str).str.strip() != ""]
patient_split_counts = nonempty_patient.groupby("patient_id")["split"].nunique()
if (patient_split_counts > 1).any():
    examples = patient_split_counts[patient_split_counts > 1].index[:10].tolist()
    raise ValueError(f"patient_id crosses dataset splits: {examples}")

# Labels must be identical across every variant of one record.
for label in LABELS:
    inconsistent = metadata_df.groupby("record_id")[label].nunique(dropna=False)
    if inconsistent.gt(1).any():
        examples = inconsistent[inconsistent.gt(1)].index[:10].tolist()
        raise RuntimeError(f"{label} changes across variants of records: {examples}")

# Selection: one clean view per ECG everywhere (the variant-set audits above
# already proved the archive holds nothing else).
train_df = metadata_df[metadata_df["split"].eq("train")].reset_index(drop=True)
val_df = metadata_df[metadata_df["subset"].eq("val_clean")].reset_index(drop=True)
test_df = metadata_df[metadata_df["subset"].eq("test_clean")].reset_index(drop=True)

selected_frames = {"train": train_df, "val_clean": val_df, "test_clean": test_df}
for name, frame in selected_frames.items():
    frame["local_path"] = frame["image_path"].map(
        lambda relative: str((EXTRACT_ROOT / relative).resolve())
    )
    missing_files = [p for p in frame["local_path"] if not Path(p).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files):,} {name} images are missing after extraction; "
            f"first: {missing_files[:5]}. Re-run the extraction cell."
        )
    if frame["local_path"].duplicated().any():
        raise ValueError(f"Duplicate local image paths in the {name} selection")

# Exactly the requested formula, independently per output, using train rows only.
train_positive = train_df[LABELS].sum(axis=0).astype(np.int64)
train_negative = len(train_df) - train_positive
if (train_positive == 0).any():
    raise ValueError(f"Training split has a label with zero positives: {train_positive.to_dict()}")
pos_weight_series = train_negative / train_positive
POS_WEIGHT = torch.tensor(pos_weight_series.to_numpy(), dtype=torch.float32)

DATASET_MODE = "clean-only"
print("dataset mode:", DATASET_MODE)
split_table = pd.DataFrame([
    {"selection": name, "images": len(frame),
     "records": int(frame["record_id"].nunique())}
    for name, frame in selected_frames.items()
]).set_index("selection")
display(split_table)
print("positive image counts")
display(pd.concat(
    {name: frame[LABELS].sum(axis=0).astype(int)
     for name, frame in selected_frames.items()}, axis=1
))
print("training-only pos_weight = negatives / positives")
display(pd.DataFrame({
    "positive": train_positive,
    "negative": train_negative,
    "prevalence": train_positive / len(train_df),
    "pos_weight": pos_weight_series,
}).round(4))
print("source x split (unique ECG records)")
display(pd.crosstab(
    metadata_df.drop_duplicates("record_id")["source_dataset"],
    metadata_df.drop_duplicates("record_id")["split"],
))

# Primary-metric coverage: conflicting-duplicate clean rows are excluded from
# checkpoint selection and the primary reported metrics, never silently.
for name in ("val_clean", "test_clean"):
    frame = selected_frames[name]
    primary = frame["primary_metric"].astype(int).eq(1)
    print(f"{name}: {int(primary.sum()):,} primary rows, "
          f"{int((~primary).sum()):,} conflicting-duplicate rows held out")

# Hard label-support gate: every label needs at least one positive and one
# negative row in each primary evaluation split. Without this, ranking metrics
# would be undefined for the unsupported label and nanmean would silently
# shrink e.g. "12-label Macro AP" to an 11-label average.
for name in ("val_clean", "test_clean"):
    primary_frame = selected_frames[name][
        selected_frames[name]["primary_metric"].astype(int).eq(1)
    ].reset_index(drop=True)
    support_table = pd.DataFrame({
        "positive": primary_frame[LABELS].sum(axis=0).astype(int),
        "negative": (len(primary_frame) - primary_frame[LABELS].sum(axis=0)).astype(int),
    })
    support_table["prevalence"] = support_table["positive"] / len(primary_frame)
    print(f"primary {name} label support")
    display(support_table.round(4))
    unsupported = support_table[
        (support_table["positive"] == 0) | (support_table["negative"] == 0)
    ].index.tolist()
    if unsupported:
        raise RuntimeError(
            f"Primary {name} split lacks positive or negative support for "
            f"{unsupported}; refusing to train with undefined ranking metrics."
        )

# Record-level source/label audit (unique ECGs, so variants cannot inflate it).
record_level = metadata_df.drop_duplicates("record_id")
source_label_rows = []
for (split, source), frame in record_level.groupby(["split", "source_dataset"], sort=True):
    row = {"split": split, "source": source, "records": int(len(frame))}
    row.update({f"positive_{label}": int(frame[label].sum()) for label in LABELS})
    source_label_rows.append(row)
source_label_support = pd.DataFrame(source_label_rows)
source_label_support.to_csv(RUN_DIR / "source_label_support_by_split.csv", index=False)
print("source x label support (unique ECG records)")
display(source_label_support)

METADATA_SHA256 = sha256_file(FULL_METADATA_CSV)
print("metadata rows:", f"{len(metadata_df):,}")
print("metadata sha256:", METADATA_SHA256)
print("archive sha256:", EXPECTED_TAR_SHA256)
"""),
    md(r"""## 5. Dataset and transforms

This corpus ships the clean views only, so online augmentation is deliberately
disabled: every sample is decoded to RGB, resized to 768×1024 with bicubic
interpolation, and normalized with the ImageNet mean and standard deviation.
ECG-unsafe flips and random crops are never used.
"""),
    code(r"""ImageFile.LOAD_TRUNCATED_IMAGES = False
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
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class ECGImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True).copy()
        self.targets = self.frame[LABELS].to_numpy(np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.local_path) as image:
            tensor = image_transform(image.convert(IMAGE_COLOR_MODE))
        target = torch.from_numpy(self.targets[index])
        metadata = {
            "row_index": int(index),
            "record_id": str(row.record_id),
            "source": str(row.source_dataset),
            "split": str(row.split),
            "subset": str(row.subset),
            "variant_id": str(row.variant_id),
            "primary_metric": int(row.primary_metric),
        }
        return tensor, target, metadata

train_dataset = ECGImageDataset(train_df)
val_dataset = ECGImageDataset(val_df)
test_dataset = ECGImageDataset(test_df)

# Decode a few real samples now, before the long run.
for dataset_name, dataset in (
    [("train", train_dataset), ("val_clean", val_dataset), ("test_clean", test_dataset)]
):
    image, target, metadata = dataset[0]
    print(dataset_name, tuple(image.shape), target.tolist(), metadata)
"""),
    code(r"""# Optional visual QA: one training ECG (single clean view).
import matplotlib.pyplot as plt

qa_record = train_df["record_id"].iloc[0]
qa_rows = metadata_df[
    metadata_df["record_id"].eq(qa_record) & metadata_df["split"].eq("train")
].sort_values("variant_id")
fig, axes = plt.subplots(1, len(qa_rows), figsize=(5 * len(qa_rows), 4))
axes = np.atleast_1d(axes)
for axis, (_, row) in zip(axes, qa_rows.iterrows()):
    path = EXTRACT_ROOT / row.image_path
    if path.is_file():
        axis.imshow(np.asarray(Image.open(path).convert("RGB")))
        axis.set_title(f"{row.variant_id} (selected: {row.image_path in set(train_df['image_path'])})")
    else:
        axis.text(0.5, 0.5, f"{row.variant_id}\nnot extracted", ha="center")
        axis.set_title(row.variant_id)
    axis.axis("off")
qa_labels = [label for label in LABELS if int(qa_rows.iloc[0][label]) == 1]
fig.suptitle(f"{qa_record}: {' + '.join(qa_labels)}")
plt.tight_layout()
plt.show()
"""),
    md(r"""## 6. Model and loss

Twelve independent sigmoid outputs (NORM plus the 11 pathologies) on the
FCMAE-bridge ConvNeXt-V2 Tiny checkpoint. The baseline trains on plain
weighted BCE (`CONTRADICTION_LAMBDA = 0.0`): the noisy-OR contradiction term
saturates with 11 pathologies and would mostly suppress NORM instead of
teaching exclusivity. Re-enable it only as an explicit ablation run.
"""),
    code(r"""def build_model(pretrained: bool) -> nn.Module:
    return timm.create_model(
        MODEL_NAME,
        pretrained=pretrained,
        num_classes=len(LABELS),
        drop_path_rate=DROP_PATH_RATE,
    )

model = build_model(pretrained=True)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
print(model.default_cfg)
print(f"parameters: {parameter_count / 1e6:.2f} M total; {trainable_count / 1e6:.2f} M trainable")
print("classifier outputs:", model.get_classifier().out_features)
assert model.get_classifier().out_features == len(LABELS)

class CardioSentryLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, contradiction_lambda: float):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight.clone().float())
        self.contradiction_lambda = float(contradiction_lambda)
        self.normal_index = LABELS.index("NORM")
        self.pathology_indices = [LABELS.index(label) for label in PATHOLOGY_LABELS]

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="mean"
        )
        probabilities = torch.sigmoid(logits)
        normal_probability = probabilities[:, self.normal_index:self.normal_index + 1]
        pathology_probabilities = probabilities[:, self.pathology_indices]
        any_pathology_probability = 1.0 - torch.prod(
            1.0 - pathology_probabilities, dim=1, keepdim=True
        )
        contradiction = (normal_probability * any_pathology_probability).mean()
        weighted_contradiction = self.contradiction_lambda * contradiction
        total = bce + weighted_contradiction
        return total, {
            "bce": bce.detach(),
            "contradiction": contradiction.detach(),
            "weighted_contradiction": weighted_contradiction.detach(),
        }

def build_optimizer(active_model: nn.Module):
    # Meta's published ConvNeXt-V2 Tiny fine-tuning recipe uses layer decay 0.9.
    # timm assigns the newly initialized classifier to the highest-LR group.
    parameter_groups = param_groups_layer_decay(
        active_model, weight_decay=WEIGHT_DECAY, layer_decay=LAYER_DECAY
    )
    expected = {id(parameter) for parameter in active_model.parameters() if parameter.requires_grad}
    grouped_ids = [
        id(parameter)
        for group in parameter_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    if set(grouped_ids) != expected or len(grouped_ids) != len(expected):
        raise RuntimeError(
            "Layer-wise optimizer grouping does not cover every trainable parameter exactly"
        )
    for group in parameter_groups:
        group["lr"] = BASE_LR * float(group.get("lr_scale", 1.0))
    optimizer = torch.optim.AdamW(
        parameter_groups, lr=BASE_LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY
    )
    return optimizer

criterion_cpu = CardioSentryLoss(POS_WEIGHT, CONTRADICTION_LAMBDA)
dummy_logits = torch.zeros(2, len(LABELS))
dummy_targets = torch.zeros(2, len(LABELS))
dummy_targets[0, LABELS.index("NORM")] = 1.0
dummy_targets[1, 1:] = 1.0
dummy_loss, dummy_parts = criterion_cpu(dummy_logits, dummy_targets)
# All-zero logits give p=0.5 everywhere: 0.5 * (1 - 0.5**11).
assert torch.isclose(
    dummy_parts["contradiction"], torch.tensor(0.5 * (1.0 - 0.5 ** len(PATHOLOGY_LABELS)))
)
_, pathology_only_parts = criterion_cpu(
    torch.tensor([[-20.0] + [20.0] * len(PATHOLOGY_LABELS)]),
    torch.tensor([[0.0] + [1.0] * len(PATHOLOGY_LABELS)]),
)
assert pathology_only_parts["contradiction"] < 1e-6
print("loss smoke test:", float(dummy_loss), {k: float(v) for k, v in dummy_parts.items()})
"""),
    md(r"""## 7. Automatically choose the largest safe micro-batch

The probe runs two real optimizer updates at each candidate size on the actual
GPU and keeps the largest size whose peak fits under 90% of VRAM. Gradient
accumulation recovers the effective batch of 32. Global RNGs are reseeded
afterwards so the probe cannot shift the training random stream.
"""),
    code(r"""def make_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=USE_AMP)
    except TypeError:  # compatibility with older supported PyTorch builds
        return torch.cuda.amp.GradScaler(enabled=USE_AMP)

def autocast_context():
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP)

def probe_micro_batch(candidates):
    total_vram = torch.cuda.get_device_properties(0).total_memory
    base_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    accepted = None
    rows = []
    for batch_size in candidates:
        if EFFECTIVE_BATCH_SIZE % batch_size:
            continue
        probe = probe_criterion = optimizer = scaler = inputs = targets = logits = loss = None
        try:
            torch.cuda.empty_cache()
            probe = build_model(pretrained=False)
            probe.load_state_dict(base_state, strict=True)
            probe = probe.to(DEVICE, memory_format=torch.channels_last).train()
            probe_criterion = CardioSentryLoss(POS_WEIGHT, CONTRADICTION_LAMBDA).to(DEVICE)
            optimizer = build_optimizer(probe)
            scaler = make_grad_scaler()
            inputs = torch.randn(
                batch_size, 3, *INPUT_SIZE, device=DEVICE
            ).contiguous(memory_format=torch.channels_last)
            targets = torch.randint(0, 2, (batch_size, len(LABELS)), device=DEVICE).float()

            # First update allocates AdamW state and warms kernels.
            for iteration in range(2):
                optimizer.zero_grad(set_to_none=True)
                if iteration == 1:
                    torch.cuda.reset_peak_memory_stats()
                with autocast_context():
                    logits = probe(inputs)
                    loss, _ = probe_criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            fraction = peak / total_vram
            rows.append({"micro_batch": batch_size, "peak_GiB": gib(peak), "VRAM_fraction": fraction, "status": "fits"})
            print(f"batch {batch_size:2d}: peak {gib(peak):5.1f} GiB ({fraction:.0%})")
            if fraction <= VRAM_SAFETY_FRACTION:
                accepted = batch_size
                break
            print("  rejected: insufficient safety headroom")
        except torch.cuda.OutOfMemoryError:
            rows.append({"micro_batch": batch_size, "peak_GiB": float("nan"), "VRAM_fraction": float("nan"), "status": "OOM"})
            print(f"batch {batch_size:2d}: OOM")
        finally:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            del probe, probe_criterion, optimizer, scaler, inputs, targets, logits, loss
            torch.cuda.empty_cache()
    if accepted is None:
        raise RuntimeError("Even micro-batch 1 failed the VRAM preflight")
    return accepted, pd.DataFrame(rows)

MICRO_BATCH_SIZE, batch_probe = probe_micro_batch(AUTO_BATCH_CANDIDATES)
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE
display(batch_probe)
print("selected micro-batch:", MICRO_BATCH_SIZE)
print("gradient accumulation steps:", ACCUMULATION_STEPS)
print("effective batch:", MICRO_BATCH_SIZE * ACCUMULATION_STEPS)

# The hardware-dependent probe creates random tensors and temporary models.
# Restore the intended starting RNG state before DataLoaders and real training
# so the number of probe attempts does not alter the training random stream.
seed_global_rngs(SEED)
print("global RNGs reseeded after the micro-batch probe")

def seed_data_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

loader_options = dict(
    batch_size=MICRO_BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
    prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    worker_init_fn=seed_data_worker,
    # Python 3.14 defaults to forkserver, which cannot import classes defined
    # in a Jupyter __main__. EC2 Linux fork workers retain parallel decoding.
    multiprocessing_context=DATALOADER_START_METHOD if NUM_WORKERS > 0 else None,
)
TRAIN_GENERATOR = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    train_dataset, shuffle=True, drop_last=True, generator=TRAIN_GENERATOR, **loader_options
)
val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_options)
test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_options)

usable_train_batches = len(train_loader) - (len(train_loader) % ACCUMULATION_STEPS)
if usable_train_batches <= 0:
    raise RuntimeError("Training loader is too short for the configured effective batch")
OPTIMIZER_STEPS_PER_EPOCH = usable_train_batches // ACCUMULATION_STEPS
print("loader batches (train/val/test):", len(train_loader), len(val_loader), len(test_loader))
print("DataLoader workers/start method:", NUM_WORKERS, DATALOADER_START_METHOD if NUM_WORKERS else "single-process")
print("optimizer steps per epoch:", OPTIMIZER_STEPS_PER_EPOCH)
"""),
    md(r"""## 8. Benchmark before full training

Run this cell **before** the long training loop. It times a small slice of the
real training step (forward + backward + amortized optimizer updates) and a few
validation batches on a throwaway model copy, using the actual loaders — then
projects seconds per batch, seconds per epoch, total training time, and GPU
memory headroom. The real model, optimizer, and RNG states are untouched (the
loader shuffle state is saved and restored, and the global RNGs are reseeded
afterwards).
"""),
    code(r"""def format_duration(seconds):
    seconds = float(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes):02d}m"
    if minutes >= 1:
        return f"{int(minutes)}m {int(secs):02d}s"
    return f"{secs:.1f}s"

train_benchmark_batches = min(BENCHMARK_TRAIN_BATCHES, usable_train_batches)
if train_benchmark_batches < ACCUMULATION_STEPS:
    raise RuntimeError(
        f"BENCHMARK_TRAIN_BATCHES={BENCHMARK_TRAIN_BATCHES} is too small to "
        f"cover one optimizer update ({ACCUMULATION_STEPS} accumulation steps)"
    )
val_benchmark_batches = min(BENCHMARK_VAL_BATCHES, len(val_loader))
print(f"dataset mode: {DATASET_MODE} "
      f"({len(train_dataset):,} train images, {len(val_dataset):,} val images)")
print(f"timing {train_benchmark_batches} training batches + "
      f"{val_benchmark_batches} validation batches "
      f"(+{BENCHMARK_WARMUP_BATCHES} warm-up, micro-batch {MICRO_BATCH_SIZE})")

# A throwaway copy carries the exact training compute without touching the
# real model/optimizer/scheduler/scaler objects built later.
benchmark_model = build_model(pretrained=False)
benchmark_model.load_state_dict(model.state_dict(), strict=True)
benchmark_model = benchmark_model.to(DEVICE, memory_format=torch.channels_last).train()
benchmark_criterion = CardioSentryLoss(POS_WEIGHT, CONTRADICTION_LAMBDA).to(DEVICE)
benchmark_optimizer = build_optimizer(benchmark_model)
benchmark_scaler = make_grad_scaler()

train_generator_state = TRAIN_GENERATOR.get_state()
try:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    benchmark_optimizer.zero_grad(set_to_none=True)
    train_iterator = iter(train_loader)
    # Warm-up (worker spin-up + kernel autotune) is excluded from timing.
    for _ in range(BENCHMARK_WARMUP_BATCHES):
        images, labels, _ = next(train_iterator)
        images = images.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast_context():
            logits = benchmark_model(images)
            loss, _ = benchmark_criterion(logits, labels)
        benchmark_scaler.scale(loss / ACCUMULATION_STEPS).backward()
    benchmark_optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    timed_batches = train_benchmark_batches
    started = time.perf_counter()
    for batch_index in range(timed_batches):
        images, labels, _ = next(train_iterator)
        images = images.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast_context():
            logits = benchmark_model(images)
            loss, _ = benchmark_criterion(logits, labels)
            scaled_loss = loss / ACCUMULATION_STEPS
        benchmark_scaler.scale(scaled_loss).backward()
        if (batch_index + 1) % ACCUMULATION_STEPS == 0:
            benchmark_scaler.unscale_(benchmark_optimizer)
            torch.nn.utils.clip_grad_norm_(benchmark_model.parameters(), GRAD_CLIP_NORM)
            benchmark_scaler.step(benchmark_optimizer)
            benchmark_scaler.update()
            benchmark_optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    train_elapsed = time.perf_counter() - started
    train_peak_allocated = torch.cuda.max_memory_allocated()
    train_peak_reserved = torch.cuda.max_memory_reserved()

    benchmark_model.eval()
    val_iterator = iter(val_loader)
    for _ in range(min(BENCHMARK_WARMUP_BATCHES, len(val_loader))):
        images, _, _ = next(val_iterator)
        images = images.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.inference_mode(), autocast_context():
            benchmark_model(images)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(val_benchmark_batches):
            images, _, _ = next(val_iterator)
            images = images.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
            with autocast_context():
                benchmark_model(images)
    torch.cuda.synchronize()
    val_elapsed = time.perf_counter() - started
finally:
    TRAIN_GENERATOR.set_state(train_generator_state)
    del benchmark_model, benchmark_criterion, benchmark_optimizer, benchmark_scaler
    torch.cuda.empty_cache()

seed_global_rngs(SEED)
print("global RNGs reseeded after the benchmark; train shuffle state restored")

total_vram = torch.cuda.get_device_properties(0).total_memory
sec_per_train_batch = train_elapsed / timed_batches
sec_per_val_batch = val_elapsed / val_benchmark_batches
train_sec_per_epoch = sec_per_train_batch * usable_train_batches
val_sec_per_epoch = sec_per_val_batch * len(val_loader)
epoch_total = (train_sec_per_epoch + val_sec_per_epoch) * 1.05  # checkpoint/scheduling overhead
BENCHMARK = {
    "dataset_mode": DATASET_MODE,
    "micro_batch_size": MICRO_BATCH_SIZE,
    "accumulation_steps": ACCUMULATION_STEPS,
    "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    "train_images": len(train_dataset),
    "val_images": len(val_dataset),
    "usable_train_batches": usable_train_batches,
    "val_batches": len(val_loader),
    "timed_train_batches": timed_batches,
    "timed_val_batches": val_benchmark_batches,
    "sec_per_train_batch": sec_per_train_batch,
    "sec_per_val_batch": sec_per_val_batch,
    "train_images_per_second": MICRO_BATCH_SIZE * timed_batches / train_elapsed,
    "train_sec_per_epoch": train_sec_per_epoch,
    "val_sec_per_epoch": val_sec_per_epoch,
    "estimated_sec_per_epoch": epoch_total,
    "estimated_total_max_epochs": epoch_total * MAX_EPOCHS,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "train_peak_allocated_gib": gib(train_peak_allocated),
    "train_peak_reserved_gib": gib(train_peak_reserved),
    "vram_total_gib": gib(total_vram),
    "vram_peak_fraction": train_peak_allocated / total_vram,
}
benchmark_path = RUN_DIR / "benchmark.json"
temporary = benchmark_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(BENCHMARK, indent=2, sort_keys=True))
os.replace(temporary, benchmark_path)
print()
print(f"--- benchmark ({DATASET_MODE}) ---")
print(f"time per training batch: {sec_per_train_batch:.2f} s "
      f"({BENCHMARK['train_images_per_second']:.1f} images/s)")
print(f"time per validation batch: {sec_per_val_batch:.2f} s")
print(f"training pass per epoch: {format_duration(train_sec_per_epoch)} "
      f"({usable_train_batches} batches)")
print(f"validation pass per epoch: {format_duration(val_sec_per_epoch)} "
      f"({len(val_loader)} batches)")
print(f"estimated time per epoch: {format_duration(epoch_total)} (incl. overhead)")
print(f"estimated total for {MAX_EPOCHS} epochs: "
      f"{format_duration(epoch_total * MAX_EPOCHS)} (worst case; early stopping "
      f"usually ends the run sooner)")
print(f"GPU peak allocated: {gib(train_peak_allocated):.2f} GiB "
      f"({train_peak_allocated / total_vram:.0%} of {gib(total_vram):.1f} GiB)")
print(f"GPU peak reserved: {gib(train_peak_reserved):.2f} GiB")
if train_peak_allocated / total_vram > VRAM_SAFETY_FRACTION:
    warnings.warn("Benchmark peak exceeds the 90% VRAM safety margin.")
print("benchmark saved to", benchmark_path)
if S3_UPLOAD_ENABLED:
    upload_file_to_s3(benchmark_path, f"{S3_RUN_PREFIX}/{benchmark_path.name}", required=True)
"""),
    md(r"""## 9. Metrics

Threshold-free ranking metrics (Average Precision, AUROC) carry checkpoint
selection; thresholded metrics use the fixed-0.5 baseline plus per-label F1
thresholds tuned on validation only. Undefined metrics stay missing with an
explicit reason — they are never converted to zero. Exact-combination
reporting covers the most common test patterns with support and a low-support
flag, since V1 has 209 distinct combinations.
"""),
    code(r"""def safe_ranking_metric(metric, y_true, y_probability):
    return float(metric(y_true, y_probability)) if len(np.unique(y_true)) == 2 else float("nan")

def divide(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")

def binary_metrics(y_true, y_probability, threshold):
    prediction = (y_probability >= threshold).astype(np.int8)
    positive_count = int(y_true.sum())
    negative_count = int(len(y_true) - positive_count)
    true_positive = int(((prediction == 1) & (y_true == 1)).sum())
    false_positive = int(((prediction == 1) & (y_true == 0)).sum())
    false_negative = int(((prediction == 0) & (y_true == 1)).sum())
    true_negative = int(((prediction == 0) & (y_true == 0)).sum())
    sensitivity = divide(true_positive, true_positive + false_negative)
    specificity = divide(true_negative, true_negative + false_positive)
    precision = divide(true_positive, true_positive + false_positive)
    npv = divide(true_negative, true_negative + false_negative)
    f1 = divide(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    return {
        "auroc": safe_ranking_metric(roc_auc_score, y_true, y_probability),
        "average_precision": safe_ranking_metric(average_precision_score, y_true, y_probability),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "fpr": divide(false_positive, false_positive + true_negative),
        "fnr": divide(false_negative, false_negative + true_positive),
        "accuracy": divide(true_positive + true_negative, len(y_true)),
        "balanced_accuracy": (sensitivity + specificity) / 2 if np.isfinite(sensitivity) and np.isfinite(specificity) else float("nan"),
        "threshold": float(threshold),
        "support": positive_count,
        "negative_support": negative_count,
        "ranking_metric_status": (
            "N/A — no positive samples" if positive_count == 0
            else "N/A — only one class present (all positive)" if negative_count == 0
            else "defined"
        ),
        "n": int(len(y_true)),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
    }

def tune_thresholds_on_validation(y_true, y_probability):
    thresholds = []
    for label_index in range(y_true.shape[1]):
        y_label = y_true[:, label_index]
        p_label = y_probability[:, label_index]
        if len(np.unique(y_label)) < 2:
            thresholds.append(0.5)
            continue
        precision, recall, candidates = precision_recall_curve(y_label, p_label)
        if len(candidates) == 0:
            thresholds.append(0.5)
            continue
        f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        thresholds.append(float(candidates[int(np.nanargmax(f1))]))
    return thresholds

def evaluate_predictions(y_true, y_probability, thresholds=None):
    thresholds = list(thresholds or [0.5] * len(LABELS))
    per_label = {
        label: binary_metrics(y_true[:, index], y_probability[:, index], thresholds[index])
        for index, label in enumerate(LABELS)
    }
    result = {"per_label": per_label, "thresholds": dict(zip(LABELS, thresholds))}
    metric_names = [
        "auroc", "average_precision", "sensitivity", "specificity", "precision", "npv",
        "f1", "fpr", "fnr", "accuracy", "balanced_accuracy",
    ]
    for metric_name in metric_names:
        values = np.array([per_label[label][metric_name] for label in LABELS], dtype=float)
        result[f"macro_{metric_name}"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    prediction = (y_probability >= np.asarray(thresholds)[None, :]).astype(np.int8)
    result["hamming_accuracy"] = float((prediction == y_true).mean())
    result["exact_match_accuracy"] = float((prediction == y_true).all(axis=1).mean())
    normal_prediction = prediction[:, LABELS.index("NORM")].astype(bool)
    pathology_prediction = prediction[:, [LABELS.index(label) for label in PATHOLOGY_LABELS]].astype(bool)
    result["normal_pathology_contradiction_rate"] = float(
        (normal_prediction & pathology_prediction.any(axis=1)).mean()
    )
    result["predicted_multi_pathology_rate"] = float((pathology_prediction.sum(axis=1) >= 2).mean())
    pathology_probability = y_probability[:, [LABELS.index(label) for label in PATHOLOGY_LABELS]]
    any_pathology_probability = 1.0 - np.prod(1.0 - pathology_probability, axis=1)
    result["mean_soft_normal_any_pathology_product"] = float(
        (y_probability[:, LABELS.index("NORM")] * any_pathology_probability).mean()
    )
    result["micro_average_precision"] = safe_ranking_metric(
        average_precision_score, y_true.ravel(), y_probability.ravel()
    )
    result["micro_auroc"] = safe_ranking_metric(roc_auc_score, y_true.ravel(), y_probability.ravel())
    pooled_tp = sum(per_label[label]["tp"] for label in LABELS)
    pooled_fp = sum(per_label[label]["fp"] for label in LABELS)
    pooled_fn = sum(per_label[label]["fn"] for label in LABELS)
    pooled_tn = sum(per_label[label]["tn"] for label in LABELS)
    result["micro_precision"] = divide(pooled_tp, pooled_tp + pooled_fp)
    result["micro_sensitivity"] = divide(pooled_tp, pooled_tp + pooled_fn)
    result["micro_specificity"] = divide(pooled_tn, pooled_tn + pooled_fp)
    result["micro_f1"] = divide(2 * pooled_tp, 2 * pooled_tp + pooled_fp + pooled_fn)
    pathology_average_precision = np.asarray([
        per_label[label]["average_precision"] for label in PATHOLOGY_LABELS
    ], dtype=float)
    result["pathology_only_macro_average_precision"] = (
        float(np.nanmean(pathology_average_precision))
        if np.isfinite(pathology_average_precision).any() else float("nan")
    )
    result["n_samples"] = int(len(y_true))
    result["valid_ranking_labels"] = int(
        sum(np.isfinite(per_label[label]["average_precision"]) for label in LABELS)
    )
    return result

def evaluate_by_source(y_true, y_probability, sources, thresholds):
    sources = np.asarray(sources, dtype=str)
    return {
        source: evaluate_predictions(y_true[sources == source], y_probability[sources == source], thresholds)
        for source in sorted(np.unique(sources))
    }

def combination_name(bits):
    bits = tuple(int(value) for value in bits)
    active = [label for label, value in zip(LABELS, bits) if value]
    return "NO_LABEL" if not active else " + ".join(active)

def top_combination_metrics(y_true, y_probability, thresholds, top_n=15):
    '''Exact-match accuracy on the most common ground-truth label patterns.'''
    y_true = np.asarray(y_true, dtype=np.int8)
    thresholds = np.asarray(thresholds, dtype=float)
    prediction = (np.asarray(y_probability) >= thresholds[None, :]).astype(np.int8)
    true_names = np.asarray([combination_name(row) for row in y_true], dtype=object)
    predicted_names = np.asarray([combination_name(row) for row in prediction], dtype=object)
    incorrect_labels = np.not_equal(prediction, y_true).sum(axis=1)
    unique, counts = np.unique(true_names, return_counts=True)
    order = np.argsort(-counts)[:top_n]
    rows = []
    for name, support in zip(unique[order], counts[order]):
        truth_mask = true_names == name
        prediction_mask = predicted_names == name
        correct_exact = int((truth_mask & prediction_mask).sum())
        predicted_count = int(prediction_mask.sum())
        precision = divide(correct_exact, predicted_count)
        recall = divide(correct_exact, int(support))
        f1 = 0.0 if correct_exact == 0 else divide(2 * precision * recall, precision + recall)
        rows.append({
            "combination": str(name),
            "support": int(support),
            "predicted_count": predicted_count,
            "correct_exact": correct_exact,
            "exact_match_accuracy": recall,
            "hamming_loss": float(incorrect_labels[truth_mask].sum() / (support * len(LABELS))),
            "combination_precision": precision,
            "combination_recall": recall,
            "combination_f1": f1,
            "precision_status": "defined" if predicted_count else "N/A — no predictions for combination",
            "support_flag": (
                f"LOW SUPPORT (<{COMBINATION_LOW_SUPPORT_THRESHOLD})"
                if support < COMBINATION_LOW_SUPPORT_THRESHOLD else "adequate"
            ),
        })
    normal_prediction = prediction[:, LABELS.index("NORM")].astype(bool)
    pathology_prediction = prediction[:, [LABELS.index(label) for label in PATHOLOGY_LABELS]].astype(bool)
    contradiction_mask = normal_prediction & pathology_prediction.any(axis=1)
    no_label_mask = prediction.sum(axis=1) == 0
    summary = {
        "n_samples": int(len(y_true)),
        "distinct_ground_truth_combinations": int(len(unique)),
        "exact_match_accuracy": float(np.equal(prediction, y_true).all(axis=1).mean()),
        "hamming_loss": float(np.not_equal(prediction, y_true).mean()),
        "normal_pathology_contradiction_count": int(contradiction_mask.sum()),
        "normal_pathology_contradiction_rate": float(contradiction_mask.mean()),
        "no_label_prediction_count": int(no_label_mask.sum()),
        "no_label_prediction_rate": float(no_label_mask.mean()),
    }
    return {"summary": summary, "top_combinations": rows}

def flatten_metrics(result, split, source="ALL", threshold_policy="validation_tuned"):
    rows = []
    for label in LABELS:
        rows.append({
            "split": split, "source": source, "threshold_policy": threshold_policy,
            "label": label, **result["per_label"][label],
        })
    macro = {
        "split": split, "source": source, "threshold_policy": threshold_policy,
        "label": "MACRO",
    }
    macro.update({key: value for key, value in result.items() if key.startswith("macro_")})
    macro.update({
        "n": result["n_samples"],
        "valid_ranking_labels": result["valid_ranking_labels"],
        "exact_match_accuracy": result["exact_match_accuracy"],
        "hamming_accuracy": result["hamming_accuracy"],
        "normal_pathology_contradiction_rate": result["normal_pathology_contradiction_rate"],
        "predicted_multi_pathology_rate": result["predicted_multi_pathology_rate"],
        "mean_soft_normal_any_pathology_product": result["mean_soft_normal_any_pathology_product"],
        "micro_average_precision": result["micro_average_precision"],
        "micro_auroc": result["micro_auroc"],
        "micro_f1": result["micro_f1"],
        "pathology_only_macro_average_precision": result["pathology_only_macro_average_precision"],
    })
    rows.append(macro)
    return rows
"""),
    md(r"""## 10. Train with checkpoint resume

The harness moves the model to the GPU, builds the optimizer/scheduler/scaler,
freezes the run configuration (including the dataset mode and the benchmark
estimate), and validates any existing run directory before writing anything: a
reused `CS_RUN_NAME` with a different configuration fails loudly instead of
mixing results. Missing run files are first pulled back from S3, so after an
interruption (or on a fresh machine) re-running resumes from the newest
checkpoint — `last.pt` or `best.pt`, both of which carry full training state
including optimizer, scheduler, and RNG states — instead of starting over.
Only the primary-subset validation pathology-only Macro AP selects checkpoints
and drives early stopping; the 12-label Macro AP is tracked as a secondary
metric.
"""),
    code(r"""def atomic_torch_save(payload, destination: Path):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)

def atomic_json_save(payload, destination: Path):
    def json_ready(value):
        if isinstance(value, dict):
            return {str(key): json_ready(entry) for key, entry in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_ready(entry) for entry in value]
        if isinstance(value, np.generic):
            return json_ready(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, Path):
            return str(value)
        return value
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2, allow_nan=False, default=str))
    os.replace(temporary, destination)

def batch_metadata_to_lists(batch_metadata):
    output = {}
    for key, values in batch_metadata.items():
        if torch.is_tensor(values):
            output[key] = values.detach().cpu().tolist()
        else:
            output[key] = list(values)
    return output

LOSS_COMPONENT_KEYS = ("loss", "bce", "contradiction", "weighted_contradiction")
PREDICT_METADATA_KEYS = ("row_index", "record_id", "source", "split", "subset",
                         "variant_id", "primary_metric")

@torch.inference_mode()
def predict_loader(active_model, loader, active_criterion=None):
    active_model.eval()
    probabilities, targets = [], []
    metadata = {key: [] for key in PREDICT_METADATA_KEYS}
    loss_sums = torch.zeros(len(LOSS_COMPONENT_KEYS), device=DEVICE, dtype=torch.float32)
    sample_count = 0
    for images, labels, batch_metadata in tqdm(loader, leave=False):
        images = images.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast_context():
            logits = active_model(images)
            if active_criterion is not None:
                loss, parts = active_criterion(logits, labels)
        if active_criterion is not None:
            batch_loss_values = torch.stack((
                loss.detach(), parts["bce"], parts["contradiction"],
                parts["weighted_contradiction"],
            )).float()
            loss_sums.add_(batch_loss_values, alpha=len(images))
        sample_count += len(images)
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        targets.append(labels.cpu().numpy().astype(np.int8))
        decoded = batch_metadata_to_lists(batch_metadata)
        for key in metadata:
            metadata[key].extend(decoded[key])
    averaged_losses = (loss_sums / max(sample_count, 1)).cpu().tolist()
    return {
        "y_true": np.concatenate(targets),
        "y_probability": np.concatenate(probabilities),
        "metadata": metadata,
        **dict(zip(LOSS_COMPONENT_KEYS, averaged_losses)),
    }

def primary_mask(output):
    return np.asarray(output["metadata"]["primary_metric"], dtype=int) == 1

def train_one_epoch(active_model, loader, active_criterion, optimizer, scaler, scheduler):
    active_model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_totals = torch.zeros(len(LOSS_COMPONENT_KEYS), device=DEVICE, dtype=torch.float32)
    sample_count = 0
    optimizer_updates = 0
    amp_skipped_optimizer_steps = 0
    scale_before = scaler.get_scale()
    progress = tqdm(enumerate(loader), total=usable_train_batches, leave=False)
    for batch_index, (images, labels, _) in progress:
        if batch_index >= usable_train_batches:
            break
        images = images.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast_context():
            logits = active_model(images)
            loss, parts = active_criterion(logits, labels)
            scaled_loss = loss / ACCUMULATION_STEPS
        scaler.scale(scaled_loss).backward()
        if (batch_index + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(active_model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer.zero_grad(set_to_none=True)
            # A scale decrease means non-finite gradients caused GradScaler to
            # skip optimizer.step(). Keep LambdaLR aligned with real updates.
            if scale_after >= scale_before:
                scheduler.step()
                optimizer_updates += 1
            else:
                amp_skipped_optimizer_steps += 1
            scale_before = scale_after
        batch_samples = len(images)
        batch_loss_values = torch.stack((
            loss.detach(), parts["bce"], parts["contradiction"],
            parts["weighted_contradiction"],
        )).float()
        loss_totals.add_(batch_loss_values, alpha=batch_samples)
        sample_count += batch_samples
        if (batch_index + 1) % PROGRESS_UPDATE_EVERY_N_BATCHES == 0:
            progress.set_postfix(loss=f"{(loss_totals[0] / sample_count).item():.4f}")
    averaged_losses = (loss_totals / sample_count).cpu().tolist()
    return {
        **dict(zip(LOSS_COMPONENT_KEYS, averaged_losses)),
        "optimizer_updates": optimizer_updates,
        "amp_skipped_optimizer_steps": amp_skipped_optimizer_steps,
    }

def lr_multiplier(step):
    warmup_steps = WARMUP_EPOCHS * OPTIMIZER_STEPS_PER_EPOCH
    total_steps = MAX_EPOCHS * OPTIMIZER_STEPS_PER_EPOCH
    if step < warmup_steps:
        return max((step + 1) / max(warmup_steps, 1), 1e-8)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

model = model.to(DEVICE, memory_format=torch.channels_last)
criterion = CardioSentryLoss(POS_WEIGHT, CONTRADICTION_LAMBDA).to(DEVICE)
optimizer = build_optimizer(model)
scheduler = LambdaLR(optimizer, lr_lambda=lr_multiplier)
scaler = make_grad_scaler()

lr_scales = [float(group.get("lr_scale", 1.0)) for group in optimizer.param_groups]
print(
    f"layer-wise LR groups: {len(lr_scales)}; "
    f"scale range {min(lr_scales):.6f}..{max(lr_scales):.6f}; "
    f"initial LR range {min(group['lr'] for group in optimizer.param_groups):.3e}.."
    f"{max(group['lr'] for group in optimizer.param_groups):.3e}"
)

CONFIG = {
    "run_name": RUN_NAME,
    "model": MODEL_NAME,
    "pretraining": "FCMAE ImageNet-1K -> supervised ImageNet-22K -> supervised ImageNet-1K at 384px",
    "label_order": LABELS,
    "dataset": "CardioSentry V1 ECG images (1024x768 PNG)",
    "dataset_archive": DATASET_TAR_NAME,
    "dataset_archive_sha256": EXPECTED_TAR_SHA256,
    "dataset_metadata_sha256": METADATA_SHA256,
    "dataset_mode": DATASET_MODE,
    "train_variants": TRAIN_VARIANTS,
    "input_size_hw": list(INPUT_SIZE),
    "input_color_mode": IMAGE_COLOR_MODE,
    "resize_interpolation": RESIZE_INTERPOLATION.value,
    "resize_antialias": RESIZE_ANTIALIAS,
    "resize_preserves_aspect_ratio": False,
    "tensor_conversion": "torchvision.transforms.ToTensor: uint8 [0,255] to float32 [0,1]",
    "normalization_mean": list(IMAGENET_MEAN),
    "normalization_std": list(IMAGENET_STD),
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "monitor": "validation_primary_pathology_only_macro_average_precision",
    "monitor_secondary": "validation_primary_macro_average_precision",
    "monitor_mode": "max",
    "reported_threshold_policies": ["fixed_0_5", "validation_tuned_per_label_f1"],
    "combination_low_support_threshold": COMBINATION_LOW_SUPPORT_THRESHOLD,
    "base_lr": BASE_LR,
    "optimizer": "AdamW",
    "optimizer_parameter_groups": len(optimizer.param_groups),
    "weight_decay": WEIGHT_DECAY,
    "layer_decay": LAYER_DECAY,
    "layer_lr_scale_min": min(lr_scales),
    "layer_lr_scale_max": max(lr_scales),
    "drop_path_rate": DROP_PATH_RATE,
    "warmup_epochs": WARMUP_EPOCHS,
    "schedule": "linear_warmup_then_cosine",
    "scheduler_step_policy": "advance only after a successful optimizer update",
    "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    "micro_batch_size": MICRO_BATCH_SIZE,
    "accumulation_steps": ACCUMULATION_STEPS,
    "loss": (
        "BCEWithLogitsLoss(pos_weight=train negatives/train positives)"
        + (" + contradiction" if CONTRADICTION_LAMBDA else "")
    ),
    "train_pos_weight": dict(zip(LABELS, map(float, POS_WEIGHT.tolist()))),
    "contradiction_lambda": CONTRADICTION_LAMBDA,
    "contradiction_formula": "p(NORM) * (1 - product(1 - p(pathology)))",
    "pathology_pathology_penalty": False,
    "online_augmentation": "none; bicubic resize + ImageNet normalization",
    "amp": "FP16",
    "tf32": not DETERMINISTIC,
    "deterministic_algorithms": DETERMINISTIC,
    "reproducibility_mode": "deterministic opt-in" if DETERMINISTIC else "seeded performance mode; not bitwise deterministic",
    "channels_last": True,
    "num_workers": NUM_WORKERS,
    "dataloader_start_method": DATALOADER_START_METHOD if NUM_WORKERS > 0 else "single-process",
    "progress_update_every_n_batches": PROGRESS_UPDATE_EVERY_N_BATCHES,
    "benchmark_estimated_sec_per_epoch": BENCHMARK["estimated_sec_per_epoch"] if "BENCHMARK" in globals() else None,
    "seed": SEED,
    "s3_upload_enabled": S3_UPLOAD_ENABLED,
    "s3_bucket": S3_BUCKET if S3_UPLOAD_ENABLED else None,
    "s3_run_prefix": S3_RUN_PREFIX if S3_UPLOAD_ENABLED else None,
    "s3_upload_every_n_epochs": S3_UPLOAD_EVERY_N_EPOCHS if S3_UPLOAD_ENABLED else None,
    "split_image_counts": {name: len(frame) for name, frame in selected_frames.items()},
    "split_record_counts": {name: int(frame["record_id"].nunique()) for name, frame in selected_frames.items()},
}
CONFIG_FINGERPRINT = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]
CONFIG["fingerprint"] = CONFIG_FINGERPRINT

config_snapshot_path = RUN_DIR / "config_snapshot.json"
best_path = RUN_DIR / "best.pt"
last_path = RUN_DIR / "last.pt"
history_path = RUN_DIR / "training_history.csv"
completion_path = RUN_DIR / "training_complete.json"

# Interrupted run on a fresh machine (or a wiped run directory): pull the
# missing run files back from S3 first, so the checks below validate the
# restored state and training resumes instead of restarting.
restore_missing_run_files_from_s3([
    "last.pt", "best.pt", "training_history.csv",
    "config_snapshot.json", "benchmark.json", "training_complete.json",
])

def require_matching_config_fingerprint(artifact_name, artifact_fingerprint):
    if artifact_fingerprint != CONFIG_FINGERPRINT:
        raise RuntimeError(
            f"{artifact_name} belongs to a different configuration. "
            "Choose a new CS_RUN_NAME; existing run files were not modified."
        )

# Validate every existing source of run identity before writing anything. This
# prevents a reused run directory from receiving a new config snapshot before
# an incompatible checkpoint is rejected.
if config_snapshot_path.is_file():
    try:
        existing_config = json.loads(config_snapshot_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Existing config_snapshot.json could not be read and will not be overwritten. "
            "Choose a new CS_RUN_NAME or repair the existing snapshot."
        ) from error
    stored_fingerprint = existing_config.get("fingerprint")
    snapshot_payload = {
        key: value for key, value in existing_config.items() if key != "fingerprint"
    }
    recomputed_fingerprint = hashlib.sha256(
        json.dumps(snapshot_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    if stored_fingerprint != recomputed_fingerprint:
        raise RuntimeError(
            "Existing config_snapshot.json has an invalid fingerprint and will not be overwritten. "
            "Choose a new CS_RUN_NAME or repair the existing snapshot."
        )
    require_matching_config_fingerprint("config_snapshot.json", stored_fingerprint)

# Both checkpoints carry full training state, so resume from whichever is
# newest: last.pt normally wins, while a restored best.pt covers the case
# where the final last.pt upload never landed on S3.
checkpoint = None
resume_source = None
for candidate_path in (last_path, best_path):
    if not candidate_path.is_file():
        continue
    candidate = torch.load(candidate_path, map_location=DEVICE, weights_only=False)
    require_matching_config_fingerprint(
        candidate_path.name, candidate.get("config_fingerprint")
    )
    if "optimizer" not in candidate or "scheduler" not in candidate:
        warnings.warn(
            f"{candidate_path.name} predates full-state checkpoints and cannot "
            "resume training; it will be overwritten by this run."
        )
        del candidate
        continue
    if checkpoint is None or int(candidate.get("epoch", 0)) > int(checkpoint.get("epoch", 0)):
        if checkpoint is not None:
            del checkpoint
        checkpoint, resume_source = candidate, candidate_path.name
    else:
        del candidate

start_epoch, best_pathology_only_macro_ap, bad_epochs, history = 1, -float("inf"), 0, []
if checkpoint is not None:
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    if checkpoint.get("train_generator_state") is not None:
        TRAIN_GENERATOR.set_state(checkpoint["train_generator_state"].cpu())
    if checkpoint.get("python_random_state") is not None:
        random.setstate(checkpoint["python_random_state"])
    if checkpoint.get("numpy_random_state") is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
    if checkpoint.get("torch_random_state") is not None:
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    if checkpoint.get("cuda_random_state_all") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_random_state_all"]])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_pathology_only_macro_ap = float(checkpoint["best_pathology_only_macro_average_precision"])
    bad_epochs = int(checkpoint["bad_epochs"])
    history = list(checkpoint["history"])
    print(
        f"resuming from {resume_source} after epoch {start_epoch - 1}; "
        f"best validation primary pathology-only Macro AP={best_pathology_only_macro_ap:.6f}"
    )

# Only a fully validated compatible run may create or refresh this file.
atomic_json_save(CONFIG, config_snapshot_path)
print("run configuration frozen:", config_snapshot_path)
"""),
    code(r"""# Long-running training cell. Re-run safely after interruption.
if completion_path.is_file():
    print("Training already marked complete:")
    print(completion_path.read_text())
else:
    stopped_early = False
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_started = time.time()
        torch.cuda.reset_peak_memory_stats()
        train_values = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler)
        validation_output = predict_loader(model, val_loader, criterion)
        mask = primary_mask(validation_output)
        if not mask.any():
            raise RuntimeError("Validation output has no primary-metric rows")
        validation_metrics = evaluate_predictions(
            validation_output["y_true"][mask], validation_output["y_probability"][mask],
            [0.5] * len(LABELS),
        )
        validation_all = evaluate_predictions(
            validation_output["y_true"], validation_output["y_probability"],
            [0.5] * len(LABELS),
        )
        validation_macro_average_precision = validation_metrics["macro_average_precision"]
        validation_pathology_only_macro_average_precision = validation_metrics[
            "pathology_only_macro_average_precision"
        ]
        improved = validation_pathology_only_macro_average_precision > best_pathology_only_macro_ap
        if improved:
            best_pathology_only_macro_ap = validation_pathology_only_macro_average_precision
            bad_epochs = 0
        else:
            bad_epochs += 1

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_values["loss"],
            "train_bce": train_values["bce"],
            "train_contradiction": train_values["contradiction"],
            "train_weighted_contradiction": train_values["weighted_contradiction"],
            "train_contradiction_loss_fraction": train_values["weighted_contradiction"] / max(train_values["loss"], 1e-12),
            "optimizer_updates": train_values["optimizer_updates"],
            "amp_skipped_optimizer_steps": train_values["amp_skipped_optimizer_steps"],
            "val_loss": validation_output["loss"],
            "val_bce": validation_output["bce"],
            "val_contradiction": validation_output["contradiction"],
            "val_weighted_contradiction": validation_output["weighted_contradiction"],
            "val_macro_average_precision": validation_macro_average_precision,
            "val_pathology_only_macro_average_precision": validation_pathology_only_macro_average_precision,
            "val_macro_average_precision_all_clean": validation_all["macro_average_precision"],
            **{f"val_ap_{label}": validation_metrics["per_label"][label]["average_precision"]
               for label in LABELS},
            "val_macro_auroc": validation_metrics["macro_auroc"],
            "val_micro_average_precision": validation_metrics["micro_average_precision"],
            "val_normal_pathology_contradiction_rate_fixed_0_5": validation_metrics["normal_pathology_contradiction_rate"],
            "val_primary_rows": int(mask.sum()),
            "val_heldout_rows": int((~mask).sum()),
            "learning_rate_head": optimizer.param_groups[-1]["lr"],
            "epoch_seconds": time.time() - epoch_started,
            "peak_allocated_gib": gib(torch.cuda.max_memory_allocated()),
            "improved": improved,
            "bad_epochs": bad_epochs,
        }
        history.append(epoch_row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        # Full training state shared by both checkpoints, so either one can
        # resume training after an interruption.
        full_checkpoint_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "train_generator_state": TRAIN_GENERATOR.get_state(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state_all": torch.cuda.get_rng_state_all(),
            "best_pathology_only_macro_average_precision": best_pathology_only_macro_ap,
            "bad_epochs": bad_epochs,
            "history": history,
            "config": CONFIG,
            "config_fingerprint": CONFIG_FINGERPRINT,
        }
        if improved:
            atomic_torch_save({
                **full_checkpoint_state,
                "validation_macro_average_precision": validation_macro_average_precision,
                "validation_pathology_only_macro_average_precision": validation_pathology_only_macro_average_precision,
                "validation_average_precision_per_condition": {
                    label: validation_metrics["per_label"][label]["average_precision"]
                    for label in LABELS
                },
            }, best_path)
        atomic_torch_save(dict(full_checkpoint_state), last_path)

        print(
            f"epoch {epoch:2d}/{MAX_EPOCHS} "
            f"train_loss={train_values['loss']:.4f} "
            f"val_primary_mAP={validation_macro_average_precision:.4f} "
            f"val_all_clean_mAP={validation_all['macro_average_precision']:.4f} "
            f"val_path_mAP={validation_pathology_only_macro_average_precision:.4f} "
            f"bad={bad_epochs} "
            f"({epoch_row['epoch_seconds'] / 60:.1f} min)",
            flush=True,
        )
        if epoch % S3_UPLOAD_EVERY_N_EPOCHS == 0:
            upload_run_files(
                [last_path, history_path, config_snapshot_path]
                + ([best_path] if best_path.is_file() else []),
                required=False,
            )
        if bad_epochs >= EARLY_STOPPING_PATIENCE:
            stopped_early = True
            print(f"early stopping after epoch {epoch} (patience {EARLY_STOPPING_PATIENCE})")
            break

    upload_run_files(
        [last_path, best_path, history_path, config_snapshot_path],
        required=False,
    )
    completion = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "stopped_early": stopped_early,
        "epochs_run": len(history),
        "best_pathology_only_macro_average_precision": best_pathology_only_macro_ap,
        "config_fingerprint": CONFIG_FINGERPRINT,
    }
    atomic_json_save(completion, completion_path)
    if S3_UPLOAD_ENABLED:
        upload_file_to_s3(
            completion_path, f"{S3_RUN_PREFIX}/{completion_path.name}", required=False
        )
    print("Training finished:", json.dumps(completion, indent=2))
"""),
    md(r"""## 11. Final validation, test, and per-source evaluation

The best validation checkpoint is fixed first; the test split is evaluated
exactly once afterwards. Thresholds are tuned per label on the primary
validation rows only, then frozen for test. Primary metrics exclude the
conflicting-duplicate rows; the held-out rows are still predicted and saved
so nothing is evaluated silently.
"""),
    code(r"""if not best_path.is_file():
    raise FileNotFoundError(
        "No best.pt checkpoint found. Run the training cell first; this cell "
        "must never evaluate an unfinished run."
    )
best_checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
require_matching_config_fingerprint(
    "best.pt", best_checkpoint.get("config_fingerprint")
)
model.load_state_dict(best_checkpoint["model"])
model.eval()
print(f"evaluating best checkpoint from epoch {best_checkpoint['epoch']} "
      f"(val primary pathology-only Macro AP="
      f"{best_checkpoint['validation_pathology_only_macro_average_precision']:.6f}; "
      f"12-label Macro AP={best_checkpoint['validation_macro_average_precision']:.6f})")

validation_output = predict_loader(model, val_loader, criterion)
test_output = predict_loader(model, test_loader, criterion)

val_primary = primary_mask(validation_output)
test_primary = primary_mask(test_output)
fixed_thresholds = [0.5] * len(LABELS)
tuned_thresholds = tune_thresholds_on_validation(
    validation_output["y_true"][val_primary],
    validation_output["y_probability"][val_primary],
)
print("validation-tuned thresholds:",
      {label: round(value, 4) for label, value in zip(LABELS, tuned_thresholds)})

evaluations = {}
for policy, policy_thresholds in (
    ("fixed_0_5", fixed_thresholds), ("validation_tuned", tuned_thresholds)
):
    val_primary_result = evaluate_predictions(
        validation_output["y_true"][val_primary],
        validation_output["y_probability"][val_primary], policy_thresholds,
    )
    val_all_result = evaluate_predictions(
        validation_output["y_true"], validation_output["y_probability"],
        policy_thresholds,
    )
    test_primary_result = evaluate_predictions(
        test_output["y_true"][test_primary],
        test_output["y_probability"][test_primary], policy_thresholds,
    )
    test_all_result = evaluate_predictions(
        test_output["y_true"], test_output["y_probability"], policy_thresholds,
    )
    evaluations[policy] = {
        "validation_primary": val_primary_result,
        "validation_all_clean": val_all_result,
        "test_primary": test_primary_result,
        "test_all_clean": test_all_result,
        "validation_primary_per_source": evaluate_by_source(
            validation_output["y_true"][val_primary],
            validation_output["y_probability"][val_primary],
            np.asarray(validation_output["metadata"]["source"])[val_primary],
            policy_thresholds,
        ),
        "test_primary_per_source": evaluate_by_source(
            test_output["y_true"][test_primary],
            test_output["y_probability"][test_primary],
            np.asarray(test_output["metadata"]["source"])[test_primary],
            policy_thresholds,
        ),
        "test_primary_combinations": top_combination_metrics(
            test_output["y_true"][test_primary],
            test_output["y_probability"][test_primary], policy_thresholds,
        ),
    }

# Tuned-threshold aliases keep the human-readable tables concise. Checkpoint
# selection remains threshold-free and is unaffected by this policy.
validation_metrics = evaluations["validation_tuned"]["validation_primary"]
test_metrics = evaluations["validation_tuned"]["test_primary"]
test_per_source = evaluations["validation_tuned"]["test_primary_per_source"]

def prediction_frame(output):
    frame = pd.DataFrame(output["metadata"])
    for label_index, label in enumerate(LABELS):
        frame[f"true_{label}"] = output["y_true"][:, label_index]
        frame[f"prob_{label}"] = output["y_probability"][:, label_index]
        frame[f"pred_0_5_{label}"] = (
            output["y_probability"][:, label_index] >= fixed_thresholds[label_index]
        ).astype(np.int8)
        frame[f"pred_validation_tuned_{label}"] = (
            output["y_probability"][:, label_index] >= tuned_thresholds[label_index]
        ).astype(np.int8)
    return frame

prediction_frame(validation_output).to_csv(RUN_DIR / "validation_predictions.csv", index=False)
prediction_frame(test_output).to_csv(RUN_DIR / "test_predictions.csv", index=False)
atomic_json_save(dict(zip(LABELS, tuned_thresholds)), RUN_DIR / "validation_tuned_thresholds.json")

metric_rows = []
source_rows = []
for policy, policy_results in evaluations.items():
    metric_rows.extend(flatten_metrics(policy_results["validation_primary"], "val_primary", threshold_policy=policy))
    metric_rows.extend(flatten_metrics(policy_results["validation_all_clean"], "val_all_clean", threshold_policy=policy))
    metric_rows.extend(flatten_metrics(policy_results["test_primary"], "test_primary", threshold_policy=policy))
    metric_rows.extend(flatten_metrics(policy_results["test_all_clean"], "test_all_clean", threshold_policy=policy))
    for split, source_key in (("val_primary", "validation_primary_per_source"),
                              ("test_primary", "test_primary_per_source")):
        for source, result in policy_results[source_key].items():
            source_rows.extend(flatten_metrics(result, split, source, threshold_policy=policy))
pd.DataFrame(metric_rows).to_csv(RUN_DIR / "metrics.csv", index=False)
pd.DataFrame(source_rows).to_csv(RUN_DIR / "per_source_metrics.csv", index=False)

source_condition_columns = [
    "split", "source", "threshold_policy", "label", "n", "support", "negative_support",
    "ranking_metric_status", "average_precision", "auroc", "sensitivity", "specificity",
    "precision", "f1",
]
source_condition_metrics = pd.DataFrame(source_rows)
source_condition_metrics = source_condition_metrics[
    source_condition_metrics["label"].isin(LABELS)
][source_condition_columns]
source_condition_metrics.to_csv(RUN_DIR / "source_condition_metrics.csv", index=False)

combination_rows = []
combination_summary_rows = []
for policy, policy_results in evaluations.items():
    combo = policy_results["test_primary_combinations"]
    combination_summary_rows.append({
        "split": "test_primary", "threshold_policy": policy, **combo["summary"],
    })
    combination_rows.extend({
        "split": "test_primary", "threshold_policy": policy, **row,
    } for row in combo["top_combinations"])
pd.DataFrame(combination_rows).to_csv(RUN_DIR / "top_combination_metrics.csv", index=False)
pd.DataFrame(combination_summary_rows).to_csv(RUN_DIR / "combination_summary.csv", index=False)

# Deployment-cost context, matching the older training notebooks.
def benchmark_inference(active_model, batch_size, repetitions=30, warmup=10):
    sample = torch.randn(batch_size, 3, *INPUT_SIZE, device=DEVICE).contiguous(memory_format=torch.channels_last)
    with torch.inference_mode(), autocast_context():
        for _ in range(warmup):
            active_model(sample)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repetitions):
            active_model(sample)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "batch_size": batch_size,
        "latency_ms_per_batch": 1000 * elapsed / repetitions,
        "latency_ms_per_image": 1000 * elapsed / (repetitions * batch_size),
        "throughput_images_per_second": repetitions * batch_size / elapsed,
    }

inference_benchmark = {
    "batch_1": benchmark_inference(model, 1),
    "selected_micro_batch": benchmark_inference(model, MICRO_BATCH_SIZE),
}
model_size_mib = best_path.stat().st_size / 2**20

RESULT = {
    "config": CONFIG,
    "best_epoch": int(best_checkpoint["epoch"]),
    "selection_metric": "validation_primary_pathology_only_macro_average_precision",
    "best_checkpoint_validation_pathology_only_macro_average_precision": float(
        best_checkpoint["validation_pathology_only_macro_average_precision"]
    ),
    "best_checkpoint_validation_macro_average_precision": float(
        best_checkpoint["validation_macro_average_precision"]
    ),
    "threshold_selection": "per-label maximum F1 on primary validation rows only",
    "validation_tuned_thresholds": dict(zip(LABELS, map(float, tuned_thresholds))),
    "evaluations": evaluations,
    "best_checkpoint_loss_components": {
        "validation": {
            key: float(validation_output[key])
            for key in ("loss", "bce", "contradiction", "weighted_contradiction")
        },
        "test": {
            key: float(test_output[key])
            for key in ("loss", "bce", "contradiction", "weighted_contradiction")
        },
    },
    "model": {
        "parameters_total": int(parameter_count),
        "parameters_trainable": int(trainable_count),
        "best_checkpoint_mib": model_size_mib,
        "inference_benchmark": inference_benchmark,
    },
    "environment": {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "timm": timm.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu.name,
        "gpu_vram_gib": gib(gpu.total_memory),
    },
}
atomic_json_save(RESULT, RUN_DIR / "result.json")
print("final evaluation artifacts written to", RUN_DIR)
"""),
    code(r"""# Human-readable report and tables.
per_label_test = pd.DataFrame(test_metrics["per_label"]).T
per_label_columns = [
    "average_precision", "auroc", "sensitivity", "specificity", "precision", "npv",
    "f1", "fpr", "fnr", "accuracy", "balanced_accuracy", "threshold",
    "support", "tp", "fp", "fn", "tn",
]
display(per_label_test[per_label_columns].round(4))

per_source_macro = pd.DataFrame({source: {
    "n": result["n_samples"],
    "valid_ranking_labels": result["valid_ranking_labels"],
    "macro_average_precision": result["macro_average_precision"],
    "macro_auroc": result["macro_auroc"],
    "macro_sensitivity": result["macro_sensitivity"],
    "macro_specificity": result["macro_specificity"],
    "macro_precision": result["macro_precision"],
    "macro_f1": result["macro_f1"],
    "exact_match_accuracy": result["exact_match_accuracy"],
    "normal_pathology_contradiction_rate": result["normal_pathology_contradiction_rate"],
} for source, result in test_per_source.items()}).T
display(per_source_macro.round(4))

per_source_per_label = pd.DataFrame(source_rows)
per_source_per_label = per_source_per_label[
    (per_source_per_label.split == "test_primary")
    & (per_source_per_label.threshold_policy == "validation_tuned")
    & (per_source_per_label.label.isin(LABELS))
]
display(per_source_per_label[[
    "source", "label", "n", "support", "negative_support", "ranking_metric_status",
    "average_precision", "auroc", "sensitivity",
    "specificity", "precision", "npv", "f1", "threshold", "tp", "fp", "fn", "tn",
]].sort_values(["source", "label"]).round(4))

test_combination_result = evaluations["validation_tuned"]["test_primary_combinations"]
test_combination_table = pd.DataFrame(test_combination_result["top_combinations"])
display(test_combination_table[[
    "combination", "support", "support_flag", "predicted_count", "correct_exact",
    "exact_match_accuracy", "hamming_loss",
    "combination_precision", "combination_recall", "combination_f1", "precision_status",
]].round(4))

def value(number):
    return "NA" if number is None or not np.isfinite(number) else f"{number:.4f}"

def source_metric_value(row, metric):
    metric_value = row[metric]
    if metric_value is not None and np.isfinite(metric_value):
        return f"{metric_value:.4f}"
    if metric in {"average_precision", "auroc"}:
        return row["ranking_metric_status"]
    if metric == "sensitivity" and row["support"] == 0:
        return "N/A — no positive samples"
    if metric == "specificity" and row["negative_support"] == 0:
        return "N/A — no negative samples"
    return "N/A — undefined denominator"

test_combination_summary = test_combination_result["summary"]

report_lines = [
    "# CardioSentry ConvNeXt-V2 Tiny V1-image report",
    "",
    f"- Dataset mode: {DATASET_MODE}",
    f"- Best epoch: {best_checkpoint['epoch']}",
    "- Selection rule: maximum primary-subset validation pathology-only Macro Average Precision (never accuracy; 12-label Macro AP tracked as secondary)",
    f"- Validation primary Macro Average Precision: {validation_metrics['macro_average_precision']:.4f}",
    f"- Validation primary pathology-only Macro Average Precision: {validation_metrics['pathology_only_macro_average_precision']:.4f}",
    f"- Test primary Macro Average Precision: {test_metrics['macro_average_precision']:.4f}",
    f"- Test primary pathology-only Macro Average Precision: {test_metrics['pathology_only_macro_average_precision']:.4f}",
    f"- Test primary Macro F1 (fixed 0.5): {evaluations['fixed_0_5']['test_primary']['macro_f1']:.4f}",
    f"- Test primary macro AUROC: {test_metrics['macro_auroc']:.4f}",
    f"- Test primary Macro F1 (validation-tuned thresholds): {test_metrics['macro_f1']:.4f}",
    f"- Test primary exact-match accuracy: {test_metrics['exact_match_accuracy']:.4f}",
    f"- Test primary Hamming accuracy: {test_metrics['hamming_accuracy']:.4f}",
    f"- Test primary distinct ground-truth combinations: {test_combination_summary['distinct_ground_truth_combinations']}",
    f"- Test NORMAL/pathology contradiction rate (validation-tuned): {test_metrics['normal_pathology_contradiction_rate']:.4f}",
    f"- Test NORMAL/pathology contradiction rate (fixed 0.5): {evaluations['fixed_0_5']['test_primary']['normal_pathology_contradiction_rate']:.4f}",
    f"- Test contradiction count (validation-tuned): {test_combination_summary['normal_pathology_contradiction_count']}",
    f"- Test NO_LABEL count (validation-tuned): {test_combination_summary['no_label_prediction_count']}",
    f"- Test predicted multi-pathology rate: {test_metrics['predicted_multi_pathology_rate']:.4f}",
    "",
    "## Configuration",
    "",
] + [f"- {key}: `{entry}`" for key, entry in CONFIG.items()] + [
    "", "## Test primary metrics by label", "",
    "| label | Average Precision | AUROC | sensitivity | specificity | precision | NPV | F1 | threshold | support |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for label in LABELS:
    row = test_metrics["per_label"][label]
    report_lines.append(
        f"| {label} | {value(row['average_precision'])} | {value(row['auroc'])} | {value(row['sensitivity'])} | "
        f"{value(row['specificity'])} | {value(row['precision'])} | {value(row['npv'])} | "
        f"{value(row['f1'])} | {row['threshold']:.4f} | {row['support']} |"
    )
report_lines += [
    "", "## Test primary metrics by source", "",
    "A source macro excludes labels that have only one target class within that source; read `valid labels` with the score.",
    "",
    "| source | n | valid labels | exact-match accuracy | Macro AP | macro AUROC | macro sensitivity | macro specificity | macro F1 | contradiction rate |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for source, row in test_per_source.items():
    report_lines.append(
        f"| {source} | {row['n_samples']} | {row['valid_ranking_labels']} | "
        f"{value(row['exact_match_accuracy'])} | {value(row['macro_average_precision'])} | "
        f"{value(row['macro_auroc'])} | {value(row['macro_sensitivity'])} | "
        f"{value(row['macro_specificity'])} | {value(row['macro_f1'])} | "
        f"{value(row['normal_pathology_contradiction_rate'])} |"
    )
report_lines += [
    "", "## Test primary source x condition metrics", "",
    "Undefined ranking metrics are never converted to zero.",
    "",
    "| source | label | support | AP | AUROC | sensitivity | specificity | precision | F1 |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for _, row in per_source_per_label.sort_values(["source", "label"]).iterrows():
    report_lines.append(
        f"| {row['source']} | {row['label']} | {int(row['support'])} | "
        f"{source_metric_value(row, 'average_precision')} | {source_metric_value(row, 'auroc')} | "
        f"{source_metric_value(row, 'sensitivity')} | {source_metric_value(row, 'specificity')} | "
        f"{source_metric_value(row, 'precision')} | {source_metric_value(row, 'f1')} |"
    )
report_lines += [
    "", "## Test primary top exact-combination performance", "",
    f"Top {len(test_combination_result['top_combinations'])} ground-truth patterns by support. "
    f"Rows below {COMBINATION_LOW_SUPPORT_THRESHOLD} samples are explicitly flagged.",
    "",
    "| ground-truth combination | support | support flag | predicted count | correct exact | exact-match accuracy | Hamming loss | precision | recall | F1 |",
    "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in test_combination_result["top_combinations"]:
    report_lines.append(
        f"| {row['combination']} | {row['support']} | {row['support_flag']} | "
        f"{row['predicted_count']} | {row['correct_exact']} | {value(row['exact_match_accuracy'])} | "
        f"{value(row['hamming_loss'])} | {value(row['combination_precision'])} | "
        f"{value(row['combination_recall'])} | {value(row['combination_f1'])} |"
    )
(RUN_DIR / "report.md").write_text("\n".join(report_lines) + "\n")
print((RUN_DIR / "report.md").read_text())
"""),
    code(r"""# 12. Learning curves and per-label precision-recall curves.
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
history_frame = pd.read_csv(history_path)
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes = axes.ravel()
axes[0].plot(history_frame.epoch, history_frame.train_loss, marker="o", label="train")
axes[0].plot(history_frame.epoch, history_frame.val_loss, marker="o", label="validation")
axes[0].set_title("loss"); axes[0].legend()
axes[1].plot(history_frame.epoch, history_frame.val_macro_average_precision, marker="o", label="primary, all labels")
axes[1].plot(history_frame.epoch, history_frame.val_pathology_only_macro_average_precision, marker="o", label="primary, pathology only")
axes[1].axvline(best_checkpoint["epoch"], color="tab:red", linestyle="--", label="best")
axes[1].set_title("validation Macro Average Precision"); axes[1].legend()
for label in LABELS:
    axes[2].plot(history_frame.epoch, history_frame[f"val_ap_{label}"], marker="o", label=label)
axes[2].set_title("validation AP by condition"); axes[2].legend(ncol=3, fontsize="small")
axes[3].plot(history_frame.epoch, history_frame.val_macro_auroc, marker="o")
axes[3].set_title("validation macro AUROC")
for axis in axes:
    axis.set_xlabel("epoch")
plt.tight_layout()
fig.savefig(RUN_DIR / "learning_curves.png", dpi=180, bbox_inches="tight")
plt.show()

fig, axis = plt.subplots(figsize=(8, 6))
for label_index, label in enumerate(LABELS):
    precision, recall, _ = precision_recall_curve(
        test_output["y_true"][test_primary][:, label_index],
        test_output["y_probability"][test_primary][:, label_index],
    )
    ap = test_metrics["per_label"][label]["average_precision"]
    axis.plot(recall, precision, label=f"{label} AP={ap:.3f}")
axis.set(xlabel="recall", ylabel="precision", title="Test primary precision-recall curves")
axis.legend(ncol=2, fontsize="small")
plt.tight_layout()
fig.savefig(RUN_DIR / "test_precision_recall_curves.png", dpi=180, bbox_inches="tight")
plt.show()

final_s3_manifest = sync_complete_run_to_s3(required=True)

print("Saved artifacts:")
for path in sorted(RUN_DIR.iterdir()):
    print(f"  {path.name:38s} {path.stat().st_size / 2**20:9.2f} MiB")
"""),
    md(r"""## Persistent outputs

```text
<CS_RUN_ROOT>/<CS_RUN_NAME>/
├── best.pt                              best primary-validation Macro-AP checkpoint
├── last.pt                              interruption/resume checkpoint
├── config_snapshot.json                 full training and data identity
├── benchmark.json                       mode-aware time and VRAM estimates
├── training_history.csv                 losses, AP, contradiction, optimizer/AMP-step counts by epoch
├── training_complete.json               terminal state
├── source_label_support_by_split.csv    unique-ECG source/label audit
├── validation_tuned_thresholds.json     validation-only F1 thresholds
├── validation_predictions.csv           labels + twelve probabilities
├── test_predictions.csv                 labels + twelve probabilities
├── metrics.csv                          fixed-0.5 + tuned aggregate/per-label metrics
├── per_source_metrics.csv               both threshold policies by source
├── source_condition_metrics.csv         source × label metrics with undefined-status reason
├── top_combination_metrics.csv          exact-match performance on common patterns
├── combination_summary.csv              overall exact/invalid-pattern summaries
├── result.json                          complete machine-readable report
├── report.md                            human-readable settings and results
├── learning_curves.png
├── test_precision_recall_curves.png
└── s3_upload_manifest.json              S3 URIs, sizes, and local SHA-256 hashes
```

Only the primary-subset validation pathology-only Macro Average Precision
controls checkpointing and early stopping; the 12-label Macro AP is tracked as
a secondary metric. Accuracy is reported for context but never participates in
selection. The test split is evaluated only after the best validation
checkpoint has been fixed. Held-out conflicting-duplicate rows are predicted
and saved, but excluded from the primary metrics.

With the default configuration, the complete directory above is also persisted
to:

```text
s3://cardiosentry-fcmae-758821198776-us-east-1-an/runs/convnext_v2_tiny_V1/<CS_RUN_NAME>/
```
"""),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1))
print(f"wrote {OUTPUT} ({len(cells)} cells)")

