"""Build the self-contained CardioSentry ConvNeXt-V2 Tiny EC2 notebook."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "CardioSentry_ConvNeXtV2_Tiny_Master_EC2.ipynb"


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
    md(r"""# CardioSentry — ConvNeXt-V2 Tiny fine-tuning on the v3 master image corpus

This notebook fine-tunes one **ConvNeXt-V2 Tiny** model on the full synthetic
image corpus generated from the harmonized v3 release. It is designed for an
AWS `g5.2xlarge` (8 vCPU, 32 GiB RAM, one NVIDIA A10G with about 22 GiB usable
accelerator memory) and does not depend on Google Colab.

## Fixed training contract

| setting | choice |
|---|---|
| task | multi-label; four independent logits in this order: `NORMAL`, `STEMI`, `AF`, `LVH` |
| pretrained model | `convnextv2_tiny.fcmae_ft_in22k_in1k_384` |
| input | 768×1024 RGB, close to the source sheet aspect ratio; bicubic resize |
| loss | per-label `BCEWithLogitsLoss(pos_weight=training negatives / training positives)` + NORMAL/pathology contradiction penalty |
| contradiction | `0.20 × p(NORMAL) × (1 - ∏(1-p(pathology)))`; pathology means STEMI/AF/LVH |
| optimizer | AdamW, peak head LR `1e-4`, layer-wise LR decay `0.9`, weight decay `0.05`, gradient clip `1.0` |
| schedule | 3-epoch linear warm-up, then cosine decay |
| training | at most 30 epochs; early stopping patience 6 |
| checkpoint monitor | **validation Macro Average Precision (Macro AP)**; accuracy never selects a checkpoint |
| thresholds | fixed 0.5 baseline plus per-label F1 thresholds tuned on validation only, then frozen for test |
| subgroup reports | per source, source × condition, and exact disease combination; support always shown |
| acceleration | FP16 AMP, TF32, channels-last, pinned memory, persistent workers, automatic largest-safe micro-batch probe |

The selected checkpoint was **FCMAE-pretrained on ImageNet-1K**, then
supervised-fine-tuned on ImageNet-22K and ImageNet-1K at 384 px. Among the
official Tiny ImageNet variants, it is the strongest initialization and its
384-px fine-tuning is a better bridge to high-resolution ECG sheets than the
224-px variants. ConvNeXt-V2 remains fully convolutional and uses adaptive
pooling, so the rectangular input needs no positional-embedding interpolation.

The default CUDA configuration is seeded but performance-oriented, not bitwise
deterministic. Set `CS_DETERMINISTIC=1` before starting the kernel for stricter
repeatability; expect lower throughput and a possible error if a CUDA operation
has no deterministic implementation. Final claims should ideally be confirmed
across multiple independently named seeds.

The original ConvNeXt-V2 weights are distributed under CC-BY-NC-4.0. Confirm
that this is compatible with the intended use before commercial deployment.

Official references: [timm checkpoint model card](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384)
and [Meta training recipes](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/TRAINING.md).
"""),
    md(r"""## 0. Data expected on EC2

The image-generation notebook was checked. Its final, storage-limited packaging
path creates exactly these archives and one SHA-256 file beside every archive:

```text
images_train_00.tar ... images_train_05.tar
images_val.tar
images_test.tar
<each archive>.sha256
```

It also wrote the label manifest separately to
`runs/synthetic_images_v1/index.csv`. If that file is unavailable, this notebook
can reconstruct the training-critical manifest from the exact v3
`harmonized_ecg_metadata.csv`. Place either `index.csv` or the metadata CSV beside
the archives (or set its environment variable below).

The image names are anonymized (`ECG000001_a0.jpg`, and so on), so reconstruction
must reproduce the generator's deterministic ordinal mapping; it cannot merely
join the filename to `global_record_id`. The reconstruction cell therefore pins
the metadata to the SHA-256 of the exact v3 CSV used by the generator, checks the
known record/image/class/source counts, checks every record's expected variants,
and writes a verification report under the experiment directory.

Transfer or mount the archive folder on EC2 and edit only the path block below.
When the ConvNeXt V1 notebook already prepared the corpus on the same volume,
run the dedicated Cell 1a to reuse its extraction and reconstructed index.
If archives and extracted images share one EBS volume, provision at least 300
GiB; 350 GiB leaves more comfortable room for the roughly 120 GiB archives,
extracted corpus, caches, checkpoints, and predictions.

## Automatic S3 persistence

Uploads default to:

```text
s3://cardiosentry-fcmae-758821198776-us-east-1-an/
├── CardioSentry/experiments/convnextv2_tiny_master_v1/data/
│   ├── reconstructed_index.csv
│   └── index_reconstruction_report.json
└── CardioSentry/runs/convnextv2_tiny_master_v1/<CS_RUN_NAME>/
    ├── best.pt
    ├── last.pt
    ├── training_history.csv
    └── ... all final reports, predictions, plots, and s3_upload_manifest.json
```

The notebook performs a real S3 write/read-metadata preflight before training,
uploads recovery-critical checkpoints after every epoch, retries them when
training ends, and uploads the complete run after final evaluation. Every final
object is byte-count checked, and `s3_upload_manifest.json` records each local
SHA-256. The EC2 IAM role needs `s3:PutObject` and `s3:GetObject` on these
prefixes. Export `CS_S3_UPLOAD=0` only for an intentional local-only run.
"""),
    code(r"""# 1. Install only the Python packages the training harness needs.
# Use an AWS Deep Learning AMI or another CUDA-ready environment: this cell does
# not replace its PyTorch/CUDA build.
%pip install -q "timm>=1.0.15,<1.1" "scikit-learn>=1.4" "pandas>=2.1" \
    "pillow>=10" "matplotlib>=3.8" "seaborn>=0.13" "tqdm>=4.66" "psutil>=5.9" \
    "boto3>=1.34"
"""),
    md(r"""## 1a. Reuse the dataset prepared by the ConvNeXt V1 notebook

Run the next cell before configuration when this notebook uses the same EC2
volume as `CardioSentry_ConvNeXt_Tiny_Master_EC2.ipynb`. It points V2 at V1's
already verified extraction markers and reconstructed index, while assigning a
separate V2 output directory locally and in S3. The normal checksum/extraction
cell remains safe to run: it verifies the archives, then reports `skip
extracted` for all eight archives instead of unpacking them again.

Set `CS_REUSE_V1_PREPARED_DATA=0` before starting the kernel to disable this
fast path. For non-default V1 locations, set `CS_V1_EXTRACT_ROOT`,
`CS_V1_ARCHIVE_DIR`, or `CS_V1_INDEX_CSV` before running the cell. Set
`CS_V2_RUN_NAME` to create a differently named V2 run.
"""),
    code(r"""# 1a. Optional fast path: reuse V1 extraction/index, never V1 results.
import os
from pathlib import Path

def environment_flag(name, default="1"):
    return os.environ.get(name, default).strip().lower() not in {
        "0", "false", "no", "off", ""
    }

if environment_flag("CS_REUSE_V1_PREPARED_DATA", "1"):
    project_root_hint = Path(os.environ.get(
        "CARDIOSENTRY_PROJECT_ROOT", str(Path.home() / "CardioSentry")
    )).expanduser().resolve()

    explicit_extract_root = os.environ.get("CS_V1_EXTRACT_ROOT") or os.environ.get(
        "CS_EXTRACT_ROOT"
    )
    if explicit_extract_root:
        v1_extract_root = Path(explicit_extract_root).expanduser().resolve()
        checked_extract_roots = [v1_extract_root]
    else:
        storage_root_hint = Path(os.environ.get(
            "CS_STORAGE_ROOT", "/mnt/cardiosentry"
        )).expanduser().resolve()
        checked_extract_roots = []
        for candidate in (
            storage_root_hint / "master-dataset" / "extracted",
            Path("/mnt/cardiosentry/master-dataset/extracted"),
            project_root_hint / "master-dataset" / "extracted",
        ):
            candidate = candidate.expanduser().resolve()
            if candidate not in checked_extract_roots:
                checked_extract_roots.append(candidate)
        prepared_roots = [
            candidate for candidate in checked_extract_roots
            if (candidate / "reconstructed_index.csv").is_file()
            and all((candidate / "images" / split).is_dir() for split in ("train", "val", "test"))
        ]
        if not prepared_roots:
            raise FileNotFoundError(
                "Could not find the V1 prepared dataset. Checked:\n  "
                + "\n  ".join(map(str, checked_extract_roots))
                + "\nSet CS_V1_EXTRACT_ROOT to the V1 extraction directory, "
                "or set CS_REUSE_V1_PREPARED_DATA=0 for a fresh setup."
            )
        v1_extract_root = prepared_roots[0]

    explicit_index = os.environ.get("CS_V1_INDEX_CSV") or os.environ.get(
        "CS_RECONSTRUCTED_INDEX_CSV"
    )
    v1_index_csv = Path(
        explicit_index or v1_extract_root / "reconstructed_index.csv"
    ).expanduser().resolve()

    explicit_archive_dir = os.environ.get("CS_V1_ARCHIVE_DIR") or os.environ.get(
        "CS_ARCHIVE_DIR"
    )
    v1_archive_dir = Path(
        explicit_archive_dir or v1_extract_root.parent
    ).expanduser().resolve()

    expected_archives = [f"images_train_{index:02d}.tar" for index in range(6)] + [
        "images_val.tar", "images_test.tar"
    ]
    missing_markers = []
    for archive_name in expected_archives:
        split = (
            "train" if archive_name.startswith("images_train_")
            else archive_name.removeprefix("images_").removesuffix(".tar")
        )
        marker = v1_extract_root / "images" / split / f".{archive_name}.extracted.json"
        if not marker.is_file():
            missing_markers.append(marker)
    if not v1_index_csv.is_file() or missing_markers:
        problems = []
        if not v1_index_csv.is_file():
            problems.append(f"missing reconstructed index: {v1_index_csv}")
        problems.extend(f"missing extraction marker: {path}" for path in missing_markers)
        raise FileNotFoundError(
            "V1 prepared-data validation failed:\n  " + "\n  ".join(problems)
        )

    if v1_archive_dir.name == "master-dataset":
        shared_storage_root = v1_archive_dir.parent
    else:
        shared_storage_root = Path(os.environ.get(
            "CS_STORAGE_ROOT", str(v1_archive_dir.parent)
        )).expanduser().resolve()

    v2_run_root = Path(os.environ.get(
        "CS_V2_RUN_ROOT",
        str(shared_storage_root / "runs" / "convnextv2_tiny_master_v1"),
    )).expanduser().resolve()
    v2_run_name = os.environ.get(
        "CS_V2_RUN_NAME", "convnextv2_tiny_master_seed42"
    ).strip()
    if not v2_run_name or Path(v2_run_name).name != v2_run_name:
        raise ValueError("CS_V2_RUN_NAME must be one non-empty folder name")

    v1_run_root = (shared_storage_root / "runs" / "convnext_v1_tiny_master").resolve()
    if v2_run_root == v1_run_root:
        raise RuntimeError("The V2 run root must be different from the V1 run root")

    os.environ["CS_ARCHIVE_DIR"] = str(v1_archive_dir)
    os.environ["CS_EXTRACT_ROOT"] = str(v1_extract_root)
    os.environ["CS_INDEX_CSV"] = str(v1_index_csv)
    os.environ["CS_RECONSTRUCTED_INDEX_CSV"] = str(v1_index_csv)
    metadata_csv = v1_archive_dir / "harmonized_ecg_metadata.csv"
    if metadata_csv.is_file():
        os.environ["CS_MASTER_METADATA_CSV"] = str(metadata_csv)

    # Never inherit V1 result destinations from the shell environment.
    os.environ["CS_RUN_ROOT"] = str(v2_run_root)
    os.environ["CS_RUN_NAME"] = v2_run_name
    os.environ["CS_S3_RUNS_PREFIX"] = os.environ.get(
        "CS_V2_S3_RUNS_PREFIX", "CardioSentry/runs/convnextv2_tiny_master_v1"
    ).strip("/")
    os.environ["CS_S3_EXPERIMENT_PREFIX"] = os.environ.get(
        "CS_V2_S3_EXPERIMENT_PREFIX",
        "CardioSentry/experiments/convnextv2_tiny_master_v1/data",
    ).strip("/")

    print("V1 prepared dataset reuse: enabled")
    print("  archive directory:  ", v1_archive_dir)
    print("  extracted images:   ", v1_extract_root / "images")
    print("  reconstructed index:", v1_index_csv)
    print("  V2 local run:       ", v2_run_root / v2_run_name)
    print(
        "  V2 S3 run:          s3://"
        + os.environ.get("CS_S3_BUCKET", "cardiosentry-fcmae-758821198776-us-east-1-an")
        + "/" + os.environ["CS_S3_RUNS_PREFIX"] + "/" + v2_run_name + "/"
    )
else:
    print("V1 prepared dataset reuse: disabled; Cell 2 defaults/overrides will be used")
"""),
    code(r"""# 2. Configuration and EC2/GPU preflight — edit paths here if needed.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import contextlib, csv, hashlib, json, math, multiprocessing as mp, platform, random, re, shutil
import subprocess, sys, time, warnings
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

# Environment variables make the notebook portable without editing it.
PROJECT_ROOT = Path(os.environ.get(
    "CARDIOSENTRY_PROJECT_ROOT", str(Path.home() / "CardioSentry")
)).expanduser().resolve()
ARCHIVE_DIR = Path(os.environ.get(
    "CS_ARCHIVE_DIR", str(PROJECT_ROOT / "datasets" / "synthetic_images_v1")
)).expanduser().resolve()
PROVIDED_INDEX_CSV = Path(os.environ.get(
    "CS_INDEX_CSV", str(ARCHIVE_DIR / "index.csv")
)).expanduser().resolve()
MASTER_METADATA_CSV = Path(os.environ.get(
    "CS_MASTER_METADATA_CSV", str(ARCHIVE_DIR / "harmonized_ecg_metadata.csv")
)).expanduser().resolve()
EXTRACT_ROOT = Path(os.environ.get(
    "CS_EXTRACT_ROOT", str(PROJECT_ROOT / "experiments" / "convnextv2_tiny_master_v1" / "data")
)).expanduser().resolve()
RECONSTRUCTED_INDEX_CSV = Path(os.environ.get(
    "CS_RECONSTRUCTED_INDEX_CSV", str(EXTRACT_ROOT / "reconstructed_index.csv")
)).expanduser().resolve()
RECONSTRUCTION_REPORT = EXTRACT_ROOT / "index_reconstruction_report.json"
RUN_ROOT = Path(os.environ.get(
    "CS_RUN_ROOT", str(PROJECT_ROOT / "runs" / "convnextv2_tiny_master_v1")
)).expanduser().resolve()
HF_HOME = Path(os.environ.get(
    "HF_HOME", str(PROJECT_ROOT / "hf_cache")
)).expanduser().resolve()
os.environ["HF_HOME"] = str(HF_HOME)

RUN_NAME = os.environ.get("CS_RUN_NAME", "convnextv2_tiny_master_seed42")
RUN_DIR = RUN_ROOT / RUN_NAME
for path in (EXTRACT_ROOT, RUN_DIR, HF_HOME):
    path.mkdir(parents=True, exist_ok=True)

def env_flag(name, default="1"):
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off", ""}

# Persistent S3 destinations. Disable only for an intentional local-only run by
# exporting CS_S3_UPLOAD=0 before starting the notebook kernel.
S3_UPLOAD_ENABLED = env_flag("CS_S3_UPLOAD", "1")
S3_BUCKET = os.environ.get(
    "CS_S3_BUCKET", "cardiosentry-fcmae-758821198776-us-east-1-an"
).strip()
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
S3_RUNS_PREFIX = os.environ.get(
    "CS_S3_RUNS_PREFIX", "CardioSentry/runs/convnextv2_tiny_master_v1"
).strip("/")
S3_EXPERIMENT_PREFIX = os.environ.get(
    "CS_S3_EXPERIMENT_PREFIX", "CardioSentry/experiments/convnextv2_tiny_master_v1/data"
).strip("/")
S3_UPLOAD_EVERY_N_EPOCHS = int(os.environ.get("CS_S3_UPLOAD_EVERY_N_EPOCHS", "1"))
if S3_UPLOAD_EVERY_N_EPOCHS < 1:
    raise ValueError("CS_S3_UPLOAD_EVERY_N_EPOCHS must be at least 1")
S3_RUN_PREFIX = f"{S3_RUNS_PREFIX}/{RUN_NAME}"
S3_CLIENT = boto3.client("s3", region_name=AWS_REGION) if S3_UPLOAD_ENABLED else None

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

LABELS = ["NORMAL", "STEMI", "AF", "LVH"]
PATHOLOGY_LABELS = ["STEMI", "AF", "LVH"]
VALID_COMBINATIONS = [
    ("NORMAL", (1, 0, 0, 0)),
    ("LVH", (0, 0, 0, 1)),
    ("AF", (0, 0, 1, 0)),
    ("STEMI", (0, 1, 0, 0)),
    ("AF + LVH", (0, 0, 1, 1)),
    ("STEMI + LVH", (0, 1, 0, 1)),
    ("STEMI + AF", (0, 1, 1, 0)),
    ("STEMI + AF + LVH", (0, 1, 1, 1)),
]
VALID_COMBINATION_LOOKUP = {bits: name for name, bits in VALID_COMBINATIONS}
COMBINATION_LOW_SUPPORT_THRESHOLD = 30
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
CONTRADICTION_LAMBDA = 0.20
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

EXPECTED_ARCHIVES = [f"images_train_{i:02d}.tar" for i in range(6)] + [
    "images_val.tar", "images_test.tar"
]

def gib(value):
    return value / 2**30

print("Python:            ", sys.version.split()[0])
print("PyTorch:           ", torch.__version__)
print("timm:              ", timm.__version__)
print("CPU count:         ", os.cpu_count())
print(f"RAM:                {gib(psutil.virtual_memory().total):.1f} GiB")
print("archive directory: ", ARCHIVE_DIR)
print("provided index CSV:", PROVIDED_INDEX_CSV)
print("master metadata:   ", MASTER_METADATA_CSV)
print("reconstructed index:", RECONSTRUCTED_INDEX_CSV)
print("extract root:      ", EXTRACT_ROOT)
print("run directory:     ", RUN_DIR)
print("HF cache:          ", HF_HOME)
print("S3 uploads enabled:", S3_UPLOAD_ENABLED)
if S3_UPLOAD_ENABLED:
    print("S3 run destination:", s3_uri(S3_RUN_PREFIX) + "/")
    print("S3 experiment data:", s3_uri(S3_EXPERIMENT_PREFIX) + "/")
print(f"free extract disk:  {gib(shutil.disk_usage(EXTRACT_ROOT).free):.1f} GiB")

# Prove PutObject and HeadObject access before committing hours to training.
if S3_UPLOAD_ENABLED:
    preflight_key = f"{S3_RUN_PREFIX}/upload_preflight.json"
    preflight_body = json.dumps({
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "region": AWS_REGION,
        "purpose": "CardioSentry training artifact upload preflight",
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
print(f"GPU:                {gpu.name}")
print(f"actual GPU VRAM:    {gib(gpu.total_memory):.1f} GiB")
print("CUDA runtime:       ", torch.version.cuda)
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
print("reproducibility:    ", "deterministic opt-in" if DETERMINISTIC else "seeded performance mode")
"""),
    md(r"""## 3. Verify checksums and extract the eight archives

This cell verifies every archive before extraction. The six train shards are
extracted into one `images/train/` directory; the validation and test archives
go into `images/val/` and `images/test/`. Completion markers make re-runs safe.
An interrupted archive is simply extracted again on the next run.
"""),
    code(r"""def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()

missing = []
for name in EXPECTED_ARCHIVES:
    archive = ARCHIVE_DIR / name
    checksum = ARCHIVE_DIR / f"{name}.sha256"
    if not archive.is_file():
        missing.append(str(archive))
    if not checksum.is_file():
        missing.append(str(checksum))
if not PROVIDED_INDEX_CSV.is_file() and not MASTER_METADATA_CSV.is_file():
    missing.append(
        "provide either the original index CSV at " + str(PROVIDED_INDEX_CSV)
        + " or the exact v3 metadata CSV at " + str(MASTER_METADATA_CSV)
    )
if missing:
    raise FileNotFoundError("Missing required inputs:\n  " + "\n  ".join(missing))

verified = {}
for name in EXPECTED_ARCHIVES:
    archive = ARCHIVE_DIR / name
    checksum = ARCHIVE_DIR / f"{name}.sha256"
    expected = checksum.read_text().split()[0].lower()
    if len(expected) != 64:
        raise ValueError(f"Malformed checksum in {checksum}")
    print(f"hashing {name} ({gib(archive.stat().st_size):.1f} GiB) ...", flush=True)
    actual = sha256_file(archive)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
    verified[name] = actual
    print("  OK", actual[:16] + "...")

images_root = EXTRACT_ROOT / "images"
remaining_bytes = 0
for name in EXPECTED_ARCHIVES:
    split = "train" if name.startswith("images_train_") else name.removeprefix("images_").removesuffix(".tar")
    marker = images_root / split / f".{name}.extracted.json"
    if not marker.is_file():
        remaining_bytes += (ARCHIVE_DIR / name).stat().st_size
required_free = int(remaining_bytes * 1.10 + 20 * 2**30)
available_free = shutil.disk_usage(EXTRACT_ROOT).free
if available_free < required_free:
    raise RuntimeError(
        f"Insufficient extraction space: {gib(available_free):.1f} GiB free, "
        f"approximately {gib(required_free):.1f} GiB required for remaining archives plus margin."
    )

for name in EXPECTED_ARCHIVES:
    split = "train" if name.startswith("images_train_") else name.removeprefix("images_").removesuffix(".tar")
    destination = images_root / split
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / f".{name}.extracted.json"
    marker_payload = {
        "archive": name,
        "sha256": verified[name],
        "bytes": (ARCHIVE_DIR / name).stat().st_size,
    }
    if marker.is_file() and json.loads(marker.read_text()) == marker_payload:
        print("skip extracted:", name)
        continue
    print(f"extracting {name} -> {destination} ...", flush=True)
    subprocess.run([
        "tar", "-xf", str(ARCHIVE_DIR / name), "-C", str(destination),
        "--no-same-owner", "--no-same-permissions",
    ], check=True)
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(json.dumps(marker_payload, indent=2))
    os.replace(temporary, marker)

print("All archives are checksum-verified and extracted.")
"""),
    md(r"""## 4. Select or reconstruct the image index

If the original `index.csv` exists, it remains the preferred input. Otherwise,
the notebook reconstructs its training-critical contents from the exact v3
metadata and the extracted image inventory.

This is not a filename-to-global-ID guess. Stage 2 of the image generator
assigned anonymized names in accepted metadata-row order using
`ECG{:06d}`. The generator log proves that all 37,225 accepted records staged,
so reproducing its validation filter and row order reproduces that mapping.

The fallback refuses to proceed unless all of these independent checks pass:

1. the metadata SHA-256 is the exact one used by the generator;
2. all archive SHA-256 checks have already passed;
3. metadata record, class, source, and split totals match the generator log;
4. the extracted inventory contains the exact 128,608 images;
5. every anonymous record appears in its metadata split with variants 0–3 for
   train and variant 0 only for validation/test;
6. every metadata record maps exactly once and labels satisfy NORMAL semantics.

The reconstructed CSV and a machine-readable proof report are written under
`experiments/convnextv2_tiny_master_v1/data`, leaving the archived dataset
directory read-only.
"""),
    code(r"""# Exact provenance recorded by cardiosentry_generate_images.ipynb.
EXPECTED_METADATA_SHA256 = "b2434ab7da801f00d338cbab628cb7dae42d76d28bb8c74221537766d9423ac0"
EXPECTED_METADATA_ROWS = 37_329
EXPECTED_RECORDS_BY_SPLIT = {"train": 30_461, "val": 3_434, "test": 3_330}
EXPECTED_IMAGES_BY_SPLIT = {"train": 121_844, "val": 3_434, "test": 3_330}
EXPECTED_SOURCE_RECORDS = {"CHAPMAN": 7_716, "CUHK": 17_951, "PTBXL": 10_141, "STEMI": 1_417}
EXPECTED_CLASS_IMAGES = {
    "AF": 28_758, "AF+LVH": 8_406, "LVH": 34_077, "NORMAL": 43_538,
    "STEMI": 11_691, "STEMI+AF": 587, "STEMI+AF+LVH": 157, "STEMI+LVH": 1_394,
}
EXPECTED_LABEL_IMAGES_BY_SPLIT = {
    "train": {"NORMAL": 40_932, "STEMI": 13_080, "AF": 36_032, "LVH": 42_020},
    "val": {"NORMAL": 1_280, "STEMI": 388, "AF": 968, "LVH": 1_043},
    "test": {"NORMAL": 1_326, "STEMI": 361, "AF": 908, "LVH": 971},
}
EXPECTED_SIDECARS = 128_607  # one annotation sidecar was absent in the packaged corpus
GENERATOR_LABEL_ORDER = ["STEMI", "AF", "LVH", "NORMAL"]
INDEX_FIELDS = [
    "image_path", "record", "record_id", "original_id", "cls",
    *GENERATOR_LABEL_ORDER, "split", "source", "render_k", "grid_palette",
    "rotation_deg", "has_annotation", "patient_id", "age", "sex", "note",
]

def require_exact_dict(name, observed, expected):
    observed = {str(k): int(v) for k, v in observed.items()}
    expected = {str(k): int(v) for k, v in expected.items()}
    if observed != expected:
        raise RuntimeError(f"{name} mismatch. Expected {expected}, observed {observed}")

def clean_generator_age(value):
    try:
        age = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(age) or age <= 0:
        return ""
    if age >= 300:
        return "90"
    return str(int(round(age)))

def clean_generator_sex(value, source):
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "unknown", "none"}:
        return ""
    source_maps = {
        "PTBXL": {"0": "M", "1": "F"},
        "STEMI": {"1": "M", "0": "F"},
        "CHAPMAN": {"male": "M", "female": "F"},
    }
    table = source_maps.get(str(source).upper(), {})
    if raw in table:
        return table[raw]
    if raw.lower() in table:
        return table[raw.lower()]
    if raw.lower().startswith("m"):
        return "M"
    if raw.lower().startswith("f"):
        return "F"
    return ""

def build_metadata_record_map(metadata_path):
    metadata_sha = sha256_file(metadata_path)
    if metadata_sha != EXPECTED_METADATA_SHA256:
        raise RuntimeError(
            "Cannot safely reconstruct labels: harmonized_ecg_metadata.csv has SHA-256 "
            f"{metadata_sha}, but the generator used {EXPECTED_METADATA_SHA256}. "
            "Because ECG numbers are assigned by CSV row order, do not bypass this check."
        )

    metadata = pd.read_csv(metadata_path, low_memory=False)
    required = {
        "global_record_id", "source_dataset", "original_record_id", "split_group_id",
        "cardiosentry_label", "validation_status", "split", "age", "sex",
        *GENERATOR_LABEL_ORDER,
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Master metadata is missing required columns: {missing}")
    if len(metadata) != EXPECTED_METADATA_ROWS:
        raise RuntimeError(
            f"Master metadata row count mismatch: expected {EXPECTED_METADATA_ROWS:,}, "
            f"observed {len(metadata):,}"
        )

    for label in GENERATOR_LABEL_ORDER:
        numeric = pd.to_numeric(metadata[label], errors="coerce")
        if numeric.isna().any() or not numeric.isin([0, 1]).all():
            raise ValueError(f"Master metadata column {label} is not completely binary")
        metadata[label] = numeric.astype(np.int8)

    # Exact stage-1 logic used for this build: accept VALID/VALID_WITH_WARNINGS
    # (or blank status), then discard rows with no positive CardioSentry label.
    status = metadata["validation_status"].fillna("").astype(str).str.strip()
    accepted_status = status.eq("") | status.isin({"VALID", "VALID_WITH_WARNINGS"})
    has_label = metadata[GENERATOR_LABEL_ORDER].sum(axis=1).gt(0)
    accepted = metadata.loc[accepted_status & has_label].copy().reset_index(drop=True)
    accepted["split"] = accepted["split"].astype(str).str.lower().str.strip()

    if accepted["global_record_id"].astype(str).duplicated().any():
        raise RuntimeError("Accepted metadata contains duplicate global_record_id values")
    if accepted["split_group_id"].fillna("").astype(str).str.strip().eq("").any():
        raise RuntimeError("Accepted metadata contains an empty split_group_id")

    # stage2_transcode.py enumerated the stage-1 rows starting at one.
    record_map = pd.DataFrame({
        "record": [f"ECG{i:06d}" for i in range(1, len(accepted) + 1)],
        "record_id": accepted["global_record_id"].astype(str).str.strip(),
        "original_id": accepted["original_record_id"].fillna("").astype(str).str.strip(),
        "cls": accepted["cardiosentry_label"].astype(str).str.strip(),
        "split": accepted["split"],
        "source": accepted["source_dataset"].astype(str).str.strip(),
        "patient_id": accepted["split_group_id"].astype(str).str.strip(),
        "age": [clean_generator_age(v) for v in accepted["age"]],
        "sex": [clean_generator_sex(v, s) for v, s in zip(accepted["sex"], accepted["source_dataset"])],
        "note": status.loc[accepted_status & has_label].reset_index(drop=True),
    })
    for label in GENERATOR_LABEL_ORDER:
        record_map[label] = accepted[label].to_numpy(np.int8)

    require_exact_dict(
        "metadata records by split",
        record_map.groupby("split").size().to_dict(),
        EXPECTED_RECORDS_BY_SPLIT,
    )
    require_exact_dict(
        "metadata records by source",
        record_map.groupby("source").size().to_dict(),
        EXPECTED_SOURCE_RECORDS,
    )
    pathology = record_map[["STEMI", "AF", "LVH"]].sum(axis=1)
    if not (record_map["NORMAL"].eq(1) == pathology.eq(0)).all():
        raise RuntimeError("Master metadata violates NORMAL/pathology target semantics")
    return record_map, metadata_sha

IMAGE_NAME_PATTERN = re.compile(r"^(ECG\d{6})_a(\d+)\.jpg$", re.IGNORECASE)

def inventory_extracted_images():
    rows = []
    for split in ("train", "val", "test"):
        split_dir = images_root / split
        for path in sorted(split_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".jpg":
                continue
            match = IMAGE_NAME_PATTERN.fullmatch(path.name)
            if match is None:
                raise RuntimeError(f"Unexpected image filename; cannot map safely: {path}")
            rows.append({
                "image_path": str(path.resolve()),
                "basename": path.name,
                "record": match.group(1).upper(),
                "render_k": int(match.group(2)),
                "physical_split": split,
                "has_annotation": int(path.with_suffix(".json").is_file()),
                "bytes": int(path.stat().st_size),
            })
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        raise RuntimeError("No extracted JPG images were found")
    if inventory["basename"].duplicated().any():
        dup = inventory.loc[inventory["basename"].duplicated(False), "basename"].head().tolist()
        raise RuntimeError(f"Duplicate image basenames across extracted splits: {dup}")
    if inventory[["physical_split", "record", "render_k"]].duplicated().any():
        raise RuntimeError("Duplicate (split, record, render_k) image entries found")
    require_exact_dict(
        "extracted images by split",
        inventory.groupby("physical_split").size().to_dict(),
        EXPECTED_IMAGES_BY_SPLIT,
    )

    variant_summary = inventory.groupby(["physical_split", "record"])["render_k"].agg(
        count="size", unique="nunique", minimum="min", maximum="max"
    ).reset_index()
    expected_n = variant_summary["physical_split"].map({"train": 4, "val": 1, "test": 1})
    expected_max = variant_summary["physical_split"].map({"train": 3, "val": 0, "test": 0})
    valid_variants = (
        variant_summary["count"].eq(expected_n)
        & variant_summary["unique"].eq(expected_n)
        & variant_summary["minimum"].eq(0)
        & variant_summary["maximum"].eq(expected_max)
    )
    if not valid_variants.all():
        raise RuntimeError(
            "Unexpected per-record variant layout; examples:\n"
            + variant_summary.loc[~valid_variants].head().to_string(index=False)
        )
    require_exact_dict(
        "extracted records by split",
        variant_summary.groupby("physical_split").size().to_dict(),
        EXPECTED_RECORDS_BY_SPLIT,
    )
    return inventory

def reconstruct_index(record_map, destination):
    inventory = inventory_extracted_images()
    physical_records = set(inventory["record"])
    metadata_records = set(record_map["record"])
    if physical_records != metadata_records:
        raise RuntimeError(
            "Image/metadata anonymous record sets differ: "
            f"missing images for {len(metadata_records - physical_records)} records; "
            f"unknown image records {len(physical_records - metadata_records)}"
        )

    merged = inventory.merge(record_map, on="record", how="left", validate="many_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise RuntimeError("At least one extracted image could not be joined to metadata")
    if not merged["physical_split"].eq(merged["split"]).all():
        bad = merged.loc[
            ~merged["physical_split"].eq(merged["split"]),
            ["basename", "record", "physical_split", "split"],
        ].head()
        raise RuntimeError("Image folder split disagrees with metadata:\n" + bad.to_string(index=False))

    require_exact_dict(
        "reconstructed images by class",
        merged.groupby("cls").size().to_dict(),
        EXPECTED_CLASS_IMAGES,
    )
    for split, expected in EXPECTED_LABEL_IMAGES_BY_SPLIT.items():
        observed = merged.loc[merged["split"].eq(split), LABELS].sum(axis=0).astype(int).to_dict()
        require_exact_dict(f"{split} positive images by label", observed, expected)

    merged["grid_palette"] = ""       # not needed for classification; sidecar geometry is untouched
    merged["rotation_deg"] = ""        # not needed for classification; sidecar geometry is untouched
    index = merged[INDEX_FIELDS].sort_values(["cls", "record", "render_k"]).reset_index(drop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    index.to_csv(temporary, index=False)
    os.replace(temporary, destination)

    sidecars = int(inventory["has_annotation"].sum())
    if sidecars != EXPECTED_SIDECARS:
        warnings.warn(
            f"Expected {EXPECTED_SIDECARS:,} annotation sidecars but found {sidecars:,}. "
            "Classification can proceed because sidecars are not model inputs."
        )
    inventory_lines = (
        inventory.sort_values(["physical_split", "basename"])
        .apply(lambda r: f"{r.physical_split}/{r.basename}\t{int(r.bytes)}\n", axis=1)
        .str.cat()
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "exact_v3_metadata_row_order_to_ECG%06d_plus_extracted_inventory",
        "metadata_csv": str(MASTER_METADATA_CSV),
        "metadata_sha256": metadata_sha,
        "expected_metadata_sha256": EXPECTED_METADATA_SHA256,
        "archive_sha256": verified,
        "inventory_name_size_sha256": hashlib.sha256(inventory_lines.encode()).hexdigest(),
        "reconstructed_index_csv": str(destination),
        "reconstructed_index_sha256": sha256_file(destination),
        "image_rows": int(len(index)),
        "unique_records": int(index["record"].nunique()),
        "images_by_split": {k: int(v) for k, v in index.groupby("split").size().items()},
        "records_by_split": {k: int(v) for k, v in index.groupby("split")["record"].nunique().items()},
        "images_by_class": {k: int(v) for k, v in index.groupby("cls").size().items()},
        "records_by_source": {
            k: int(v) for k, v in index.drop_duplicates("record").groupby("source").size().items()
        },
        "positive_images_by_split": {
            split: {
                label: int(value)
                for label, value in index.loc[index["split"].eq(split), LABELS].sum(axis=0).items()
            }
            for split in ("train", "val", "test")
        },
        "anonymous_mapping_endpoints": {
            str(record_map.iloc[0]["record"]): str(record_map.iloc[0]["record_id"]),
            str(record_map.iloc[-1]["record"]): str(record_map.iloc[-1]["record_id"]),
        },
        "annotation_sidecars_found": sidecars,
        "checks": {
            "exact_metadata_hash": True,
            "all_archive_hashes": True,
            "exact_known_counts": True,
            "one_to_one_record_mapping": True,
            "physical_split_matches_metadata": True,
            "expected_variant_sets": True,
            "normal_pathology_semantics": True,
        },
    }
    temporary_report = RECONSTRUCTION_REPORT.with_suffix(RECONSTRUCTION_REPORT.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True))
    os.replace(temporary_report, RECONSTRUCTION_REPORT)
    return index

metadata_record_map = None
metadata_sha = None
if MASTER_METADATA_CSV.is_file():
    candidate_sha = sha256_file(MASTER_METADATA_CSV)
    if candidate_sha == EXPECTED_METADATA_SHA256:
        metadata_record_map, metadata_sha = build_metadata_record_map(MASTER_METADATA_CSV)
    elif not PROVIDED_INDEX_CSV.is_file():
        # This call emits the detailed, safety-critical mismatch explanation.
        metadata_record_map, metadata_sha = build_metadata_record_map(MASTER_METADATA_CSV)
    else:
        warnings.warn(
            "A provided index.csv will be used, but the optional metadata CSV does not have "
            "the generator's exact SHA-256, so metadata cross-checking is disabled."
        )

if PROVIDED_INDEX_CSV.is_file():
    INDEX_CSV = PROVIDED_INDEX_CSV
    INDEX_ORIGIN = "provided_index"
    print("Using provided index:", INDEX_CSV)
else:
    if metadata_record_map is None:
        raise RuntimeError("No safe metadata record map is available for reconstruction")
    reconstruct_index(metadata_record_map, RECONSTRUCTED_INDEX_CSV)
    INDEX_CSV = RECONSTRUCTED_INDEX_CSV
    INDEX_ORIGIN = "reconstructed_from_exact_v3_metadata"
    print("Reconstructed index:", INDEX_CSV)
    print("Verification report:", RECONSTRUCTION_REPORT)
    print(RECONSTRUCTION_REPORT.read_text())
    if S3_UPLOAD_ENABLED:
        upload_file_to_s3(
            INDEX_CSV,
            f"{S3_EXPERIMENT_PREFIX}/{INDEX_CSV.name}",
            required=True,
        )
        upload_file_to_s3(
            RECONSTRUCTION_REPORT,
            f"{S3_EXPERIMENT_PREFIX}/{RECONSTRUCTION_REPORT.name}",
            required=True,
        )
"""),
    md(r"""## 5. Audit the master manifest and compute training-only `pos_weight`

The paths saved by a provided index can point to its former Colab work directory,
so this notebook intentionally keeps only each basename and resolves it under
the EC2 extraction root. Labels always come from the four binary columns in the
active provided/reconstructed index, never from `cls` or a directory name.

The audit enforces patient/record split isolation, including a source-scoped
`source::original_id` identity, one image per validation/test ECG, valid NORMAL
semantics, and exact file availability before any GPU work.
"""),
    code(r"""index_df = pd.read_csv(INDEX_CSV, low_memory=False)
required_columns = {
    "image_path", "record", "record_id", "original_id", "cls", "split", "source",
    "render_k", "patient_id", *LABELS,
}
missing_columns = sorted(required_columns - set(index_df.columns))
if missing_columns:
    raise ValueError(f"index.csv is missing required columns: {missing_columns}")

index_df["split"] = index_df["split"].astype(str).str.lower().str.strip()
if set(index_df["split"].unique()) != {"train", "val", "test"}:
    raise ValueError(f"Expected train/val/test only; got {sorted(index_df['split'].unique())}")

index_df["original_id"] = index_df["original_id"].fillna("").astype(str).str.strip()
if index_df["original_id"].eq("").any():
    raise ValueError("index.csv contains an empty original_id")
index_df["source_original_id"] = (
    index_df["source"].astype(str).str.strip() + "::" + index_df["original_id"]
)

for label in LABELS:
    index_df[label] = pd.to_numeric(index_df[label], errors="raise").astype(np.int8)
    if not index_df[label].isin([0, 1]).all():
        raise ValueError(f"{label} contains values outside 0/1")

pathology_sum = index_df[PATHOLOGY_LABELS].sum(axis=1)
normal_semantics = index_df["NORMAL"].eq(1) == pathology_sum.eq(0)
if not normal_semantics.all():
    bad = index_df.loc[~normal_semantics, ["record", *LABELS]].head()
    raise ValueError("NORMAL/pathology target contradiction in index.csv:\n" + bad.to_string(index=False))

# Exact master-corpus provenance checks, taken from the completed generator log.
require_exact_dict(
    "index images by split",
    index_df.groupby("split").size().to_dict(),
    EXPECTED_IMAGES_BY_SPLIT,
)
require_exact_dict(
    "index records by split",
    index_df.groupby("split")["record"].nunique().to_dict(),
    EXPECTED_RECORDS_BY_SPLIT,
)
require_exact_dict(
    "index images by class",
    index_df.groupby("cls").size().to_dict(),
    EXPECTED_CLASS_IMAGES,
)
require_exact_dict(
    "index records by source",
    index_df.drop_duplicates("record").groupby("source").size().to_dict(),
    EXPECTED_SOURCE_RECORDS,
)
for split, expected in EXPECTED_LABEL_IMAGES_BY_SPLIT.items():
    observed = index_df.loc[index_df["split"].eq(split), LABELS].sum(axis=0).astype(int).to_dict()
    require_exact_dict(f"index {split} positive images by label", observed, expected)

for key in ("record", "record_id", "patient_id", "source_original_id"):
    split_counts = index_df.groupby(key)["split"].nunique()
    if (split_counts > 1).any():
        examples = split_counts[split_counts > 1].index[:10].tolist()
        raise ValueError(f"{key} crosses dataset splits: {examples}")

# Each anonymous record must carry one consistent identity/target across its variants.
record_fields = [
    "record_id", "original_id", "cls", "split", "source", "patient_id",
    "source_original_id", *GENERATOR_LABEL_ORDER,
]
for field in record_fields:
    inconsistent = index_df.groupby("record")[field].nunique(dropna=False)
    if inconsistent.gt(1).any():
        examples = inconsistent[inconsistent.gt(1)].index[:10].tolist()
        raise RuntimeError(f"{field} changes across variants of records: {examples}")

# When the exact v3 metadata is available, independently compare every record's
# identity, split, source, patient group, class string, and four targets.
if metadata_record_map is not None:
    observed_map = index_df.drop_duplicates("record").set_index("record")
    expected_map = metadata_record_map.set_index("record")
    if set(observed_map.index) != set(expected_map.index):
        raise RuntimeError("Index record set does not match the exact v3 metadata mapping")
    for field in [field for field in record_fields if field != "source_original_id"]:
        if field in GENERATOR_LABEL_ORDER:
            left = pd.to_numeric(observed_map[field], errors="raise").astype(np.int8)
            right = expected_map[field].astype(np.int8)
        else:
            left = observed_map[field].fillna("").astype(str).str.strip()
            right = expected_map[field].fillna("").astype(str).str.strip()
        right = right.reindex(left.index)
        unequal = ~left.eq(right)
        if unequal.any():
            examples = left.index[unequal][:10].tolist()
            raise RuntimeError(
                f"Index field {field} disagrees with exact v3 metadata for records: {examples}"
            )

for split in ("val", "test"):
    per_record = index_df[index_df.split == split].groupby("record").size()
    if not per_record.eq(1).all():
        raise ValueError(f"{split} must contain exactly one image per ECG record")

def resolve_image_path(row) -> str:
    basename = Path(str(row.image_path)).name
    if not basename.lower().endswith((".jpg", ".jpeg", ".png")):
        basename = f"{row.record}_a{int(row.render_k)}.jpg"
    return str((images_root / row.split / basename).resolve())

index_df["local_path"] = index_df.apply(resolve_image_path, axis=1)
missing_files = [p for p in index_df.local_path if not Path(p).is_file()]
if missing_files:
    raise FileNotFoundError(
        f"{len(missing_files):,} indexed images are missing after extraction; first: {missing_files[:5]}"
    )
if index_df["local_path"].duplicated().any():
    raise ValueError("Duplicate local image paths found in index.csv")

# Prove bidirectional coverage: every indexed file exists (above), and every
# extracted JPG is indexed exactly once (below). This catches stale extraction
# markers, orphan images, and accidentally incomplete manifests.
physical_inventory = inventory_extracted_images()
physical_paths = set(physical_inventory["image_path"])
indexed_paths = set(index_df["local_path"])
if physical_paths != indexed_paths:
    raise RuntimeError(
        "Index/extracted-file coverage differs: "
        f"{len(indexed_paths - physical_paths)} indexed paths absent on disk; "
        f"{len(physical_paths - indexed_paths)} extracted images absent from index"
    )

train_df = index_df[index_df.split == "train"].reset_index(drop=True)
val_df = index_df[index_df.split == "val"].reset_index(drop=True)
test_df = index_df[index_df.split == "test"].reset_index(drop=True)

# Exactly the requested formula, independently per output, using train rows only.
train_positive = train_df[LABELS].sum(axis=0).astype(np.int64)
train_negative = len(train_df) - train_positive
if (train_positive == 0).any():
    raise ValueError(f"Training split has a label with zero positives: {train_positive.to_dict()}")
pos_weight_series = train_negative / train_positive
POS_WEIGHT = torch.tensor(pos_weight_series.to_numpy(), dtype=torch.float32)

split_table = index_df.groupby("split").agg(images=("record", "size"), records=("record", "nunique"))
label_table = pd.concat({
    split: frame[LABELS].sum(axis=0).astype(int)
    for split, frame in (("train", train_df), ("val", val_df), ("test", test_df))
}, axis=1)
print("split counts")
display(split_table)
print("positive image counts")
display(label_table)
print("training-only pos_weight = negatives / positives")
display(pd.DataFrame({
    "positive": train_positive,
    "negative": train_negative,
    "pos_weight": pos_weight_series,
}).round(4))
print("source × split")
display(pd.crosstab(index_df.source, index_df.split))

# Count independent ECG records, not repeated rendered images, to expose source
# and label associations that can affect external generalization.
record_level = index_df.drop_duplicates("record")
source_label_rows = []
for (split, source), frame in record_level.groupby(["split", "source"], sort=True):
    row = {"split": split, "source": source, "records": int(len(frame))}
    row.update({f"positive_{label}": int(frame[label].sum()) for label in LABELS})
    source_label_rows.append(row)
source_label_support = pd.DataFrame(source_label_rows)
source_label_support.to_csv(RUN_DIR / "source_label_support_by_split.csv", index=False)
print("source × label support (unique ECG records)")
display(source_label_support)

INDEX_SHA256 = sha256_file(INDEX_CSV)
print("index origin:", INDEX_ORIGIN)
print("index rows:", f"{len(index_df):,}")
print("index sha256:", INDEX_SHA256)
if metadata_sha is not None:
    print("metadata sha256:", metadata_sha)
"""),
    md(r"""## 6. Dataset and transforms

The generator supplies four independently styled photographs for each training
ECG and one render for each validation/test ECG. Consequently, image rows are
the optimization samples, while the ECG record (and patient group where
available) is the independent split unit. The audit above reports both counts.
The generator applies grid, perspective, tilt, paper, lighting, blur, sensor,
and JPEG variation. Additional random augmentation is therefore disabled.
Horizontal flip would reverse time, vertical flip would invert voltage/ST
polarity, and random crops could remove diagnostic leads. Validation and test
use the identical deterministic resize/normalization path.
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
            "record": str(row.record),
            "record_id": str(row.record_id),
            "patient_id": str(row.patient_id),
            "source": str(row.source),
            "split": str(row.split),
            "cls": str(row.cls),
            "render_k": int(row.render_k),
        }
        return tensor, target, metadata

train_dataset = ECGImageDataset(train_df)
val_dataset = ECGImageDataset(val_df)
test_dataset = ECGImageDataset(test_df)

# Decode a few real samples now, before the long run.
for dataset_name, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
    image, target, metadata = dataset[0]
    print(dataset_name, tuple(image.shape), target.tolist(), metadata)
"""),
    code(r"""# Optional visual QA: one example for each reporting class string.
import matplotlib.pyplot as plt

examples = []
for class_name in ("NORMAL", "STEMI", "AF", "LVH", "STEMI+AF+LVH"):
    rows = train_df[train_df.cls == class_name]
    if len(rows):
        examples.append(rows.iloc[0])

fig, axes = plt.subplots(1, len(examples), figsize=(5 * len(examples), 4))
axes = np.atleast_1d(axes)
for ax, row in zip(axes, examples):
    with Image.open(row.local_path) as image:
        ax.imshow(image.convert("RGB"))
    ax.set_title(f"{row.cls}\n{row.source}")
    ax.axis("off")
plt.tight_layout()
plt.show()
"""),
    md(r"""## 7. Model and loss

The model emits raw logits. Sigmoid is applied only for probabilities and the
contradiction term; using `BCEWithLogitsLoss` preserves numerical stability.
The contradiction term is the probability of NORMAL multiplied by the soft
union probability of at least one pathology:

```text
p(NORMAL) × [1 - (1-p(STEMI))(1-p(AF))(1-p(LVH))]
```

This directly targets `NORMAL AND any pathology`, without dividing a single
strong contradiction by the number of pathology labels. There are no
pathology×pathology penalties, so `STEMI + AF + LVH` remains valid.
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
assert model.get_classifier().out_features == 4

class CardioSentryLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, contradiction_lambda: float):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight.clone().float())
        self.contradiction_lambda = float(contradiction_lambda)
        self.normal_index = LABELS.index("NORMAL")
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
dummy_logits = torch.zeros(2, 4)
dummy_targets = torch.tensor([[1, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.float32)
dummy_loss, dummy_parts = criterion_cpu(dummy_logits, dummy_targets)
assert torch.isclose(dummy_parts["contradiction"], torch.tensor(0.4375))
_, pathology_only_parts = criterion_cpu(
    torch.tensor([[-20.0, 20.0, 20.0, 20.0]]),
    torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
)
assert pathology_only_parts["contradiction"] < 1e-6
print("loss smoke test:", float(dummy_loss), {k: float(v) for k, v in dummy_parts.items()})
"""),
    md(r"""## 8. Automatically choose the largest safe micro-batch

The probe includes the same layer-wise AdamW parameter groups and a real
optimizer step, so optimizer state is counted.
It tries divisors of effective batch 32 from largest to smallest and accepts the
first whose measured peak allocation is at most 90% of actual VRAM. Gradient
accumulation then restores effective batch 32. This is safer than assuming the
nominal A10G memory and lets the same notebook use a larger GPU later.
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
print("loader batches:", len(train_loader), len(val_loader), len(test_loader))
print("DataLoader workers/start method:", NUM_WORKERS, DATALOADER_START_METHOD if NUM_WORKERS else "single-process")
print("optimizer steps per epoch:", OPTIMIZER_STEPS_PER_EPOCH)
"""),
    md(r"""## 9. Metrics

Every split receives per-label AUROC, Average Precision (AP), sensitivity, specificity,
precision, NPV, F1, FPR, FNR, accuracy, balanced accuracy, support, and full
TP/FP/FN/TN counts. Macro and micro summaries, Hamming accuracy, and exact-match
accuracy are also retained. Macro AP is the unweighted mean of sklearn's
per-label `average_precision_score`, is threshold-free, and is the only
checkpoint-selection signal. It is not labeled as trapezoidal PR-AUC.

Per-source reports include all the same per-label rows. A label absent from a
source is reported as undefined (`NaN`) and excluded from that source's macro;
`valid_ranking_labels` makes that denominator explicit.
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
    normal_prediction = prediction[:, LABELS.index("NORMAL")].astype(bool)
    pathology_prediction = prediction[:, [LABELS.index(label) for label in PATHOLOGY_LABELS]].astype(bool)
    result["normal_pathology_contradiction_rate"] = float(
        (normal_prediction & pathology_prediction.any(axis=1)).mean()
    )
    result["predicted_multi_pathology_rate"] = float((pathology_prediction.sum(axis=1) >= 2).mean())
    pathology_probability = y_probability[:, [LABELS.index(label) for label in PATHOLOGY_LABELS]]
    any_pathology_probability = 1.0 - np.prod(1.0 - pathology_probability, axis=1)
    result["mean_soft_normal_any_pathology_product"] = float(
        (y_probability[:, LABELS.index("NORMAL")] * any_pathology_probability).mean()
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
    if bits in VALID_COMBINATION_LOOKUP:
        return VALID_COMBINATION_LOOKUP[bits]
    active = [label for label, value in zip(LABELS, bits) if value]
    return "NO_LABEL" if not active else " + ".join(active)

def evaluate_exact_combinations(y_true, y_probability, thresholds):
    '''Treat each four-bit label pattern as one exact-combination category.'''
    y_true = np.asarray(y_true, dtype=np.int8)
    thresholds = np.asarray(thresholds, dtype=float)
    prediction = (np.asarray(y_probability) >= thresholds[None, :]).astype(np.int8)
    true_names = np.asarray([combination_name(row) for row in y_true], dtype=object)
    predicted_names = np.asarray([combination_name(row) for row in prediction], dtype=object)
    invalid_true = sorted(set(true_names) - set(VALID_COMBINATION_LOOKUP.values()))
    if invalid_true:
        raise RuntimeError(f"Ground truth contains invalid exact combinations: {invalid_true}")

    incorrect_labels = np.not_equal(prediction, y_true).sum(axis=1)
    metric_rows = []
    for name, _ in VALID_COMBINATIONS:
        truth_mask = true_names == name
        support = int(truth_mask.sum())
        if support == 0:
            continue
        prediction_mask = predicted_names == name
        correct_exact = int((truth_mask & prediction_mask).sum())
        predicted_count = int(prediction_mask.sum())
        precision = divide(correct_exact, predicted_count)
        recall = divide(correct_exact, support)
        f1 = (
            0.0 if correct_exact == 0
            else divide(2 * precision * recall, precision + recall)
        )
        metric_rows.append({
            "combination": name,
            "support": support,
            "predicted_count": predicted_count,
            "correct_exact": correct_exact,
            "exact_match_accuracy": recall,
            "hamming_loss": float(incorrect_labels[truth_mask].sum() / (support * len(LABELS))),
            "mean_incorrect_labels_per_sample": float(incorrect_labels[truth_mask].mean()),
            "combination_precision": precision,
            "combination_recall": recall,
            "combination_f1": f1,
            "precision_status": "defined" if predicted_count else "N/A — no predictions for combination",
            "support_flag": (
                f"LOW SUPPORT (<{COMBINATION_LOW_SUPPORT_THRESHOLD})"
                if support < COMBINATION_LOW_SUPPORT_THRESHOLD else "adequate"
            ),
        })

    normal_prediction = prediction[:, LABELS.index("NORMAL")].astype(bool)
    pathology_prediction = prediction[:, [LABELS.index(label) for label in PATHOLOGY_LABELS]].astype(bool)
    contradiction_mask = normal_prediction & pathology_prediction.any(axis=1)
    no_label_mask = prediction.sum(axis=1) == 0
    valid_prediction_names = set(VALID_COMBINATION_LOOKUP.values())
    invalid_mask = np.asarray([name not in valid_prediction_names for name in predicted_names])

    invalid_rows = []
    for name in sorted(set(predicted_names[invalid_mask])):
        pattern_mask = predicted_names == name
        example_bits = prediction[np.flatnonzero(pattern_mask)[0]]
        is_contradiction = bool(example_bits[LABELS.index("NORMAL")] and example_bits[1:].any())
        invalid_rows.append({
            "predicted_combination": name,
            "count": int(pattern_mask.sum()),
            "rate": float(pattern_mask.mean()),
            "invalid_reason": "NORMAL + pathology contradiction" if is_contradiction else "no label predicted",
        })

    summary = {
        "n_samples": int(len(y_true)),
        "exact_match_accuracy": float(np.equal(prediction, y_true).all(axis=1).mean()),
        "hamming_loss": float(np.not_equal(prediction, y_true).mean()),
        "mean_incorrect_labels_per_sample": float(incorrect_labels.mean()),
        "normal_pathology_contradiction_count": int(contradiction_mask.sum()),
        "normal_pathology_contradiction_rate": float(contradiction_mask.mean()),
        "no_label_prediction_count": int(no_label_mask.sum()),
        "no_label_prediction_rate": float(no_label_mask.mean()),
        "invalid_prediction_count": int(invalid_mask.sum()),
        "invalid_prediction_rate": float(invalid_mask.mean()),
    }
    return {"summary": summary, "per_combination": metric_rows, "invalid_patterns": invalid_rows}

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

The test split is not touched here. At each epoch, validation Macro Average
Precision (Macro AP) is computed from probabilities and the best checkpoint is
replaced whenever it strictly improves. Six consecutive non-improving epochs
stop the run. `last.pt` contains optimizer, scheduler, scaler, RNG state, and
history for safe notebook restarts. When FP16 scaling skips a non-finite
optimizer update, the learning-rate scheduler does not advance; both successful
and skipped update counts are written to the epoch history. Before updating
`config_snapshot.json`, the notebook validates the fingerprints of any existing
snapshot, `best.pt`, and `last.pt`; an incompatible run directory is left
untouched and must use a different `CS_RUN_NAME`.
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

@torch.inference_mode()
def predict_loader(active_model, loader, active_criterion=None):
    active_model.eval()
    probabilities, targets = [], []
    metadata = {key: [] for key in ("row_index", "record", "record_id", "patient_id", "source", "split", "cls", "render_k")}
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
    "monitor": "validation_macro_average_precision",
    "monitor_mode": "max",
    "reported_threshold_policies": ["fixed_0_5", "validation_tuned_per_label_f1"],
    "combination_low_support_threshold": COMBINATION_LOW_SUPPORT_THRESHOLD,
    "exact_ground_truth_combinations": [name for name, _ in VALID_COMBINATIONS],
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
    "loss": "BCEWithLogitsLoss(pos_weight=train negatives/train positives) + contradiction",
    "train_pos_weight": dict(zip(LABELS, map(float, POS_WEIGHT.tolist()))),
    "contradiction_lambda": CONTRADICTION_LAMBDA,
    "contradiction_formula": "p(NORMAL) * (1 - product(1 - p(pathology)))",
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
    "epoch_monitoring_columns": [
        "val_macro_average_precision",
        "val_pathology_only_macro_average_precision",
        *[f"val_ap_{label}" for label in LABELS],
        "val_macro_auroc",
        "val_normal_pathology_contradiction_rate_fixed_0_5",
    ],
    "seed": SEED,
    "index_csv": str(INDEX_CSV),
    "index_origin": INDEX_ORIGIN,
    "index_sha256": INDEX_SHA256,
    "master_metadata_csv": str(MASTER_METADATA_CSV) if MASTER_METADATA_CSV.is_file() else None,
    "master_metadata_sha256": metadata_sha,
    "index_reconstruction_report": (
        str(RECONSTRUCTION_REPORT) if INDEX_ORIGIN.startswith("reconstructed") else None
    ),
    "archive_sha256": verified,
    "s3_upload_enabled": S3_UPLOAD_ENABLED,
    "s3_bucket": S3_BUCKET if S3_UPLOAD_ENABLED else None,
    "s3_run_prefix": S3_RUN_PREFIX if S3_UPLOAD_ENABLED else None,
    "s3_experiment_prefix": S3_EXPERIMENT_PREFIX if S3_UPLOAD_ENABLED else None,
    "s3_upload_every_n_epochs": S3_UPLOAD_EVERY_N_EPOCHS if S3_UPLOAD_ENABLED else None,
    "split_image_counts": {split: int((index_df.split == split).sum()) for split in ("train", "val", "test")},
    "split_record_counts": {split: int(index_df.loc[index_df.split == split, "record"].nunique()) for split in ("train", "val", "test")},
}
CONFIG_FINGERPRINT = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]
CONFIG["fingerprint"] = CONFIG_FINGERPRINT

config_snapshot_path = RUN_DIR / "config_snapshot.json"
best_path = RUN_DIR / "best.pt"
last_path = RUN_DIR / "last.pt"
history_path = RUN_DIR / "training_history.csv"
completion_path = RUN_DIR / "training_complete.json"

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

if best_path.is_file():
    existing_best = torch.load(best_path, map_location="cpu", weights_only=False)
    require_matching_config_fingerprint(
        "best.pt", existing_best.get("config_fingerprint")
    )
    del existing_best

checkpoint = None
if last_path.is_file():
    checkpoint = torch.load(last_path, map_location=DEVICE, weights_only=False)
    require_matching_config_fingerprint(
        "last.pt", checkpoint.get("config_fingerprint")
    )

start_epoch, best_macro_average_precision, bad_epochs, history = 1, -float("inf"), 0, []
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
    best_macro_average_precision = float(checkpoint["best_macro_average_precision"])
    bad_epochs = int(checkpoint["bad_epochs"])
    history = list(checkpoint["history"])
    print(
        f"resuming after epoch {start_epoch - 1}; "
        f"best validation Macro AP={best_macro_average_precision:.6f}"
    )

# Only a fully validated compatible run may create or refresh this file.
atomic_json_save(CONFIG, config_snapshot_path)
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
        validation_metrics = evaluate_predictions(
            validation_output["y_true"], validation_output["y_probability"], [0.5] * len(LABELS)
        )
        validation_macro_average_precision = validation_metrics["macro_average_precision"]
        validation_pathology_only_macro_average_precision = validation_metrics[
            "pathology_only_macro_average_precision"
        ]
        improved = validation_macro_average_precision > best_macro_average_precision
        if improved:
            best_macro_average_precision = validation_macro_average_precision
            bad_epochs = 0
            atomic_torch_save({
                "epoch": epoch,
                "model": model.state_dict(),
                "validation_macro_average_precision": validation_macro_average_precision,
                "validation_pathology_only_macro_average_precision": validation_pathology_only_macro_average_precision,
                "validation_average_precision_per_condition": {
                    label: validation_metrics["per_label"][label]["average_precision"]
                    for label in LABELS
                },
                "config": CONFIG,
                "config_fingerprint": CONFIG_FINGERPRINT,
            }, best_path)
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
            "val_contradiction_loss_fraction": validation_output["weighted_contradiction"] / max(validation_output["loss"], 1e-12),
            "val_macro_average_precision": validation_macro_average_precision,
            "val_pathology_only_macro_average_precision": validation_pathology_only_macro_average_precision,
            "val_ap_NORMAL": validation_metrics["per_label"]["NORMAL"]["average_precision"],
            "val_ap_STEMI": validation_metrics["per_label"]["STEMI"]["average_precision"],
            "val_ap_AF": validation_metrics["per_label"]["AF"]["average_precision"],
            "val_ap_LVH": validation_metrics["per_label"]["LVH"]["average_precision"],
            "val_macro_auroc": validation_metrics["macro_auroc"],
            "val_normal_pathology_contradiction_rate_fixed_0_5": validation_metrics["normal_pathology_contradiction_rate"],
            "learning_rate_min": min(group["lr"] for group in optimizer.param_groups),
            "learning_rate_max": max(group["lr"] for group in optimizer.param_groups),
            "epoch_minutes": (time.time() - epoch_started) / 60,
            "peak_vram_gib": gib(torch.cuda.max_memory_allocated()),
            "best_checkpoint": int(improved),
            "non_improving_epochs": bad_epochs,
        }
        history.append(epoch_row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        atomic_torch_save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_macro_average_precision": best_macro_average_precision,
            "bad_epochs": bad_epochs,
            "history": history,
            "train_generator_state": TRAIN_GENERATOR.get_state(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state_all": torch.cuda.get_rng_state_all(),
            "config_fingerprint": CONFIG_FINGERPRINT,
        }, last_path)
        if S3_UPLOAD_ENABLED and (epoch % S3_UPLOAD_EVERY_N_EPOCHS == 0 or improved):
            epoch_uploads = [
                RUN_DIR / "config_snapshot.json",
                history_path,
                last_path,
            ]
            if improved:
                epoch_uploads.append(best_path)
            # A transient upload problem should not discard GPU progress. It is
            # warned here and retried by the required sync after training.
            upload_run_files(epoch_uploads, required=False)
        print(
            f"epoch {epoch:02d}/{MAX_EPOCHS}  train={train_values['loss']:.4f}  "
            f"val={validation_output['loss']:.4f}  val Macro AP={validation_macro_average_precision:.5f}  "
            f"pathology-only Macro AP={validation_pathology_only_macro_average_precision:.5f}  "
            f"AMP skips={train_values['amp_skipped_optimizer_steps']}  "
            f"best={best_macro_average_precision:.5f}  {'SAVED BEST' if improved else f'patience {bad_epochs}/{EARLY_STOPPING_PATIENCE}'}"
        )
        if bad_epochs >= EARLY_STOPPING_PATIENCE:
            stopped_early = True
            print("early stopping: validation Macro AP did not improve for", EARLY_STOPPING_PATIENCE, "epochs")
            break

    atomic_json_save({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_epoch": int(history[-1]["epoch"]),
        "best_validation_macro_average_precision": float(best_macro_average_precision),
        "stopped_early": stopped_early,
        "selection_metric": "validation_macro_average_precision",
    }, completion_path)

# Required retry/sync of all recovery-critical state. This also runs when the
# cell is re-entered after training_complete.json was already created.
if S3_UPLOAD_ENABLED:
    upload_run_files([
        RUN_DIR / "config_snapshot.json",
        history_path,
        last_path,
        best_path,
        completion_path,
    ], required=True)
    print("Recovery checkpoints persisted to:", s3_uri(S3_RUN_PREFIX) + "/")
"""),
    md(r"""## 11. Final validation, test, and per-source evaluation

This cell reloads `best.pt`, tunes one decision threshold per label on validation
only, then freezes those thresholds for test. Fixed-0.5 metrics are saved beside
the tuned-threshold metrics, including the raw NORMAL/pathology contradiction
rate under each policy. Exact disease-combination performance is evaluated on
all eight valid ground-truth patterns. Invalid predictions, including
`NORMAL + pathology` and `NO_LABEL`, are counted rather than discarded.

Source×combination rows are published only at support ≥30; excluded low-support
rows are recorded in `result.json`. Validation/test samples are never augmented.
Raw probabilities are retained for future calibration or subgroup analysis.
"""),
    code(r"""if not best_path.is_file():
    raise FileNotFoundError("No best.pt found; complete at least one training epoch first")
best_checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(best_checkpoint["model"])
model.eval()
print(
    "loaded best epoch", best_checkpoint["epoch"],
    "with validation Macro AP", best_checkpoint["validation_macro_average_precision"],
)

validation_output = predict_loader(model, val_loader, criterion)
tuned_thresholds = tune_thresholds_on_validation(
    validation_output["y_true"], validation_output["y_probability"]
)
test_output = predict_loader(model, test_loader, criterion)
fixed_thresholds = [0.5] * len(LABELS)

evaluation_policies = {
    "fixed_0_5": fixed_thresholds,
    "validation_tuned": tuned_thresholds,
}
evaluations = {}
combination_evaluations = {}
combination_rows = []
combination_summary_rows = []
invalid_pattern_rows = []
source_combination_rows = []
source_combination_summary_rows = []
for policy, policy_thresholds in evaluation_policies.items():
    validation_metrics_for_policy = evaluate_predictions(
        validation_output["y_true"], validation_output["y_probability"], policy_thresholds
    )
    test_metrics_for_policy = evaluate_predictions(
        test_output["y_true"], test_output["y_probability"], policy_thresholds
    )
    evaluations[policy] = {
        "validation": validation_metrics_for_policy,
        "test": test_metrics_for_policy,
        "validation_per_source": evaluate_by_source(
            validation_output["y_true"], validation_output["y_probability"],
            validation_output["metadata"]["source"], policy_thresholds,
        ),
        "test_per_source": evaluate_by_source(
            test_output["y_true"], test_output["y_probability"],
            test_output["metadata"]["source"], policy_thresholds,
        ),
    }

    combination_evaluations[policy] = {}
    for split, output in (("val", validation_output), ("test", test_output)):
        overall = evaluate_exact_combinations(
            output["y_true"], output["y_probability"], policy_thresholds
        )
        combination_evaluations[policy][split] = {
            "overall": overall,
            "by_source": {},
        }
        combination_summary_rows.append({
            "split": split, "source": "ALL", "threshold_policy": policy,
            **overall["summary"],
        })
        combination_rows.extend({
            "split": split, "source": "ALL", "threshold_policy": policy, **row,
        } for row in overall["per_combination"])
        invalid_pattern_rows.extend({
            "split": split, "source": "ALL", "threshold_policy": policy, **row,
        } for row in overall["invalid_patterns"])

        sources = np.asarray(output["metadata"]["source"], dtype=str)
        for source in sorted(np.unique(sources)):
            source_mask = sources == source
            source_result = evaluate_exact_combinations(
                output["y_true"][source_mask],
                output["y_probability"][source_mask],
                policy_thresholds,
            )
            supported = [
                row for row in source_result["per_combination"]
                if row["support"] >= COMBINATION_LOW_SUPPORT_THRESHOLD
            ]
            excluded = [
                {"combination": row["combination"], "support": row["support"]}
                for row in source_result["per_combination"]
                if row["support"] < COMBINATION_LOW_SUPPORT_THRESHOLD
            ]
            combination_evaluations[policy][split]["by_source"][source] = {
                "summary": source_result["summary"],
                "per_combination_supported": supported,
                "excluded_low_support_combinations": excluded,
                "invalid_patterns": source_result["invalid_patterns"],
            }
            source_combination_summary_rows.append({
                "split": split, "source": source, "threshold_policy": policy,
                "reported_combination_min_support": COMBINATION_LOW_SUPPORT_THRESHOLD,
                **source_result["summary"],
            })
            source_combination_rows.extend({
                "split": split, "source": source, "threshold_policy": policy, **row,
            } for row in supported)
            invalid_pattern_rows.extend({
                "split": split, "source": source, "threshold_policy": policy, **row,
            } for row in source_result["invalid_patterns"])

# Tuned-threshold aliases keep the human-readable tables concise. Checkpoint
# selection remains threshold-free and is unaffected by this policy.
validation_metrics = evaluations["validation_tuned"]["validation"]
test_metrics = evaluations["validation_tuned"]["test"]
validation_per_source = evaluations["validation_tuned"]["validation_per_source"]
test_per_source = evaluations["validation_tuned"]["test_per_source"]

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
    metric_rows.extend(flatten_metrics(policy_results["validation"], "val", threshold_policy=policy))
    metric_rows.extend(flatten_metrics(policy_results["test"], "test", threshold_policy=policy))
    for split, source_key in (("val", "validation_per_source"), ("test", "test_per_source")):
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

pd.DataFrame(combination_rows).to_csv(RUN_DIR / "combination_metrics.csv", index=False)
pd.DataFrame(combination_summary_rows).to_csv(RUN_DIR / "combination_summary.csv", index=False)
pd.DataFrame(source_combination_rows).to_csv(
    RUN_DIR / "per_source_combination_metrics.csv", index=False
)
pd.DataFrame(source_combination_summary_rows).to_csv(
    RUN_DIR / "per_source_combination_summary.csv", index=False
)
invalid_pattern_columns = [
    "split", "source", "threshold_policy", "predicted_combination",
    "count", "rate", "invalid_reason",
]
pd.DataFrame(invalid_pattern_rows, columns=invalid_pattern_columns).to_csv(
    RUN_DIR / "invalid_prediction_patterns.csv", index=False
)

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

benchmark = {
    "batch_1": benchmark_inference(model, 1),
    "selected_micro_batch": benchmark_inference(model, MICRO_BATCH_SIZE),
}
model_size_mib = best_path.stat().st_size / 2**20

RESULT = {
    "config": CONFIG,
    "best_epoch": int(best_checkpoint["epoch"]),
    "selection_metric": "validation_macro_average_precision",
    "best_checkpoint_validation_macro_average_precision": float(
        best_checkpoint["validation_macro_average_precision"]
    ),
    "threshold_selection": "per-label maximum F1 on validation only",
    "validation_tuned_thresholds": dict(zip(LABELS, map(float, tuned_thresholds))),
    "evaluations": evaluations,
    "combination_evaluations": combination_evaluations,
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
        "benchmark": benchmark,
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
"""),
    code(r"""# 11. Human-readable report and tables.
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
    (per_source_per_label.split == "test")
    & (per_source_per_label.threshold_policy == "validation_tuned")
    & (per_source_per_label.label.isin(LABELS))
]
display(per_source_per_label[[
    "source", "label", "n", "support", "negative_support", "ranking_metric_status",
    "average_precision", "auroc", "sensitivity",
    "specificity", "precision", "npv", "f1", "threshold", "tp", "fp", "fn", "tn",
]].sort_values(["source", "label"]).round(4))

test_combination_result = combination_evaluations["validation_tuned"]["test"]["overall"]
test_combination_table = pd.DataFrame(test_combination_result["per_combination"])
display(test_combination_table[[
    "combination", "support", "support_flag", "predicted_count", "correct_exact",
    "exact_match_accuracy", "hamming_loss", "mean_incorrect_labels_per_sample",
    "combination_precision", "combination_recall", "combination_f1", "precision_status",
]].round(4))

test_invalid_patterns = pd.DataFrame(test_combination_result["invalid_patterns"])
if len(test_invalid_patterns):
    display(test_invalid_patterns)
else:
    print("No invalid prediction patterns on the test set at validation-tuned thresholds.")

source_combination_display = pd.DataFrame(source_combination_rows)
if len(source_combination_display):
    source_combination_display = source_combination_display[
        (source_combination_display["split"] == "test")
        & (source_combination_display["threshold_policy"] == "validation_tuned")
    ]
if len(source_combination_display):
    display(source_combination_display[[
        "source", "combination", "support", "correct_exact", "exact_match_accuracy",
        "hamming_loss", "combination_precision", "combination_recall", "combination_f1",
    ]].sort_values(["source", "combination"]).round(4))

def value(value):
    return "NA" if value is None or not np.isfinite(value) else f"{value:.4f}"

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
    "# CardioSentry ConvNeXt-V2 Tiny master-corpus report",
    "",
    f"- Best epoch: {best_checkpoint['epoch']}",
    f"- Selection rule: maximum validation Macro Average Precision (never accuracy)",
    f"- Validation Macro Average Precision: {validation_metrics['macro_average_precision']:.4f}",
    f"- Validation pathology-only Macro Average Precision: {validation_metrics['pathology_only_macro_average_precision']:.4f}",
    f"- Test Macro Average Precision: {test_metrics['macro_average_precision']:.4f}",
    f"- Test pathology-only Macro Average Precision: {test_metrics['pathology_only_macro_average_precision']:.4f}",
    f"- Test Macro F1 (fixed 0.5): {evaluations['fixed_0_5']['test']['macro_f1']:.4f}",
    f"- Test macro AUROC: {test_metrics['macro_auroc']:.4f}",
    f"- Test Macro F1 (validation-tuned thresholds): {test_metrics['macro_f1']:.4f}",
    f"- Test exact-match accuracy: {test_metrics['exact_match_accuracy']:.4f}",
    f"- Test Hamming accuracy: {test_metrics['hamming_accuracy']:.4f}",
    f"- Validation weighted BCE at best checkpoint: {validation_output['bce']:.4f}",
    f"- Validation raw contradiction penalty at best checkpoint: {validation_output['contradiction']:.4f}",
    f"- Validation weighted contradiction contribution at best checkpoint: {validation_output['weighted_contradiction']:.4f}",
    f"- Test NORMAL/pathology contradiction rate (validation-tuned): {test_metrics['normal_pathology_contradiction_rate']:.4f}",
    f"- Test NORMAL/pathology contradiction rate (fixed 0.5): {evaluations['fixed_0_5']['test']['normal_pathology_contradiction_rate']:.4f}",
    f"- Test NORMAL/pathology contradiction count (validation-tuned): {test_combination_summary['normal_pathology_contradiction_count']}",
    f"- Test invalid exact-pattern count (validation-tuned, contradictions plus NO_LABEL): {test_combination_summary['invalid_prediction_count']}",
    f"- Test predicted multi-pathology rate (valid combinations retained): {test_metrics['predicted_multi_pathology_rate']:.4f}",
    "",
    "## Configuration",
    "",
] + [f"- {key}: `{entry}`" for key, entry in CONFIG.items()] + [
    "", "## Test metrics by label", "",
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
    "", "## Test metrics by source", "",
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
    "", "## Test source × condition metrics", "",
    "`STEMI` is the ACS/STEMI source identifier. Undefined ranking metrics are never converted to zero.",
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
    "", "## Test exact disease-combination performance", "",
    f"Rows below `{COMBINATION_LOW_SUPPORT_THRESHOLD}` samples are retained but explicitly flagged.",
    "Combination precision is N/A when the model never predicts that exact pattern.",
    "",
    "| ground-truth combination | support | support flag | predicted count | correct exact | exact-match accuracy | Hamming loss | mean incorrect labels | precision | recall | F1 |",
    "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in test_combination_result["per_combination"]:
    report_lines.append(
        f"| {row['combination']} | {row['support']} | {row['support_flag']} | "
        f"{row['predicted_count']} | {row['correct_exact']} | {value(row['exact_match_accuracy'])} | "
        f"{value(row['hamming_loss'])} | {value(row['mean_incorrect_labels_per_sample'])} | "
        f"{value(row['combination_precision'])} | {value(row['combination_recall'])} | "
        f"{value(row['combination_f1'])} |"
    )

report_lines += [
    "", "## Invalid test prediction patterns", "",
    f"- NORMAL + pathology contradiction count: {test_combination_summary['normal_pathology_contradiction_count']}",
    f"- NORMAL + pathology contradiction rate: {test_combination_summary['normal_pathology_contradiction_rate']:.4f}",
    f"- NO_LABEL count: {test_combination_summary['no_label_prediction_count']}",
    f"- Total invalid-pattern count: {test_combination_summary['invalid_prediction_count']}",
    "",
]
if test_combination_result["invalid_patterns"]:
    report_lines += [
        "| predicted pattern | count | rate | reason |",
        "|---|---:|---:|---|",
    ]
    for row in test_combination_result["invalid_patterns"]:
        report_lines.append(
            f"| {row['predicted_combination']} | {row['count']} | {row['rate']:.4f} | {row['invalid_reason']} |"
        )
else:
    report_lines.append("No invalid prediction patterns were observed.")

report_lines += [
    "", "## Source × exact-combination performance", "",
    f"Only source/combination rows with support ≥{COMBINATION_LOW_SUPPORT_THRESHOLD} are shown; all exclusions and supports are recorded in `result.json`.",
    "",
]
if len(source_combination_display):
    report_lines += [
        "| source | combination | support | correct exact | exact-match accuracy | Hamming loss | precision | recall | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in source_combination_display.sort_values(["source", "combination"]).iterrows():
        report_lines.append(
            f"| {row['source']} | {row['combination']} | {int(row['support'])} | "
            f"{int(row['correct_exact'])} | {value(row['exact_match_accuracy'])} | "
            f"{value(row['hamming_loss'])} | {value(row['combination_precision'])} | "
            f"{value(row['combination_recall'])} | {value(row['combination_f1'])} |"
        )
else:
    report_lines.append("No source/combination group met the reporting threshold.")
(RUN_DIR / "report.md").write_text("\n".join(report_lines) + "\n")
print((RUN_DIR / "report.md").read_text())
"""),
    code(r"""# 12. Learning curves and per-label precision-recall curves.
import seaborn as sns
sns.set_theme(style="whitegrid")
history_frame = pd.read_csv(history_path)
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes = axes.ravel()
axes[0].plot(history_frame.epoch, history_frame.train_loss, marker="o", label="train")
axes[0].plot(history_frame.epoch, history_frame.val_loss, marker="o", label="validation")
axes[0].set_title("loss"); axes[0].legend()
axes[1].plot(history_frame.epoch, history_frame.val_macro_average_precision, marker="o", label="all labels")
axes[1].plot(history_frame.epoch, history_frame.val_pathology_only_macro_average_precision, marker="o", label="pathology only")
axes[1].axvline(best_checkpoint["epoch"], color="tab:red", linestyle="--", label="best")
axes[1].set_title("validation Macro Average Precision"); axes[1].legend()
for label in LABELS:
    axes[2].plot(history_frame.epoch, history_frame[f"val_ap_{label}"], marker="o", label=label)
axes[2].set_title("validation AP by condition"); axes[2].legend()
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
        test_output["y_true"][:, label_index], test_output["y_probability"][:, label_index]
    )
    ap = test_metrics["per_label"][label]["average_precision"]
    axis.plot(recall, precision, label=f"{label} AP={ap:.3f}")
axis.set(xlabel="recall", ylabel="precision", title="Test precision-recall curves")
axis.legend()
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
<CS_RUN_ROOT>/convnextv2_tiny_master_seed42/
├── best.pt                              best validation Macro-AP checkpoint
├── last.pt                              interruption/resume checkpoint
├── config_snapshot.json                 full training and data identity
├── training_history.csv                 losses, AP, contradiction, optimizer/AMP-step counts by epoch
├── training_complete.json               terminal state
├── source_label_support_by_split.csv    unique-ECG source/label audit
├── validation_tuned_thresholds.json     validation-only F1 thresholds
├── validation_predictions.csv           labels + four probabilities
├── test_predictions.csv                 labels + four probabilities
├── metrics.csv                          fixed-0.5 + tuned aggregate/per-label metrics
├── per_source_metrics.csv               both threshold policies by source
├── source_condition_metrics.csv         source × label metrics with undefined-status reason
├── combination_metrics.csv              exact ground-truth combination performance
├── combination_summary.csv              overall exact/invalid-pattern summaries
├── invalid_prediction_patterns.csv      observed invalid patterns and counts
├── per_source_combination_metrics.csv   source combinations with support ≥30
├── per_source_combination_summary.csv   source exact/invalid-pattern summaries
├── result.json                           complete machine-readable report
├── report.md                             human-readable settings and results
├── learning_curves.png
├── test_precision_recall_curves.png
└── s3_upload_manifest.json               S3 URIs, sizes, and local SHA-256 hashes
```

Only validation Macro Average Precision controls checkpointing and early stopping. Accuracy
is reported for context but never participates in selection. The test split is
evaluated only after the best validation checkpoint has been fixed.

With the default configuration, the complete directory above is also persisted
to:

```text
s3://cardiosentry-fcmae-758821198776-us-east-1-an/CardioSentry/runs/convnextv2_tiny_master_v1/convnextv2_tiny_master_seed42/
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
