"""Build the single Colab notebook for the CardioSentry combination study."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "CardioSentry_ConvNeXtV2_Multilabel_Combination_Study_Colab.ipynb"


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
    md(r"""# CardioSentry — ConvNeXt-V2 Tiny multi-label combination study

This single end-to-end Google Colab notebook compares four serious models on
the **same frozen 9,398-record train/validation/test split**:

1. independent four-label baseline;
2. pairwise auxiliary supervision;
3. pairwise plus triple auxiliary supervision; and
4. explicit label-query cross-attention plus label self-attention.

The four deployment outputs always remain `NORMAL`, `STEMI`, `AF`, and `LVH`.
Auxiliary outputs are training supervision only and never rewrite primary
probabilities or decisions. No new split, image, augmentation, crop, filter, or
post-hoc contradiction repair is introduced.

## Project behavior reused after inspection

The implementation deliberately follows
`convnextv2_tiny_master_v1/CardioSentry_ConvNeXtV2_Tiny_Master_Colab_L4_High_RAM.ipynb`:

| item | reused behavior |
|---|---|
| model | `convnextv2_tiny.fcmae_ft_in22k_in1k_384` |
| input | 768×1024 RGB, bicubic resize, ImageNet mean/std |
| primary loss | `BCEWithLogitsLoss`, training-only negatives/positives `pos_weight` |
| contradiction | `0.20 × p(NORMAL) × [1 − ∏(1 − p(pathology))]` |
| optimizer | AdamW, weight decay 0.05, ConvNeXt layer-wise LR decay 0.90 |
| schedule | three-epoch linear warm-up then cosine decay |
| selection | maximum validation Macro Average Precision (Macro AP) |
| stopping | at most 30 epochs, patience 6 around validation Macro AP |
| thresholds | fixed 0.5 and per-label validation-F1 optimum, frozen for test |
| acceleration | BF16 on L4 (FP16 fallback), TF32, channels-last, fused AdamW when available |

The existing master contradiction term penalizes only `NORMAL AND any
pathology`; it does **not** penalize pathology/pathology coexistence. It is used
unchanged in experiments A–D.

## Important pre-registered limitation

The preserved split has 38 triple-positive ECGs in train, 5 in validation, and
**0 in test**. The notebook does not alter that split. Triple-head train versus
validation behavior is reported, but held-out triple recall/F1 are correctly
left undefined. Any triple conclusion is therefore exploratory until a test
set with triple-positive support is available.

Run every cell in order on a Colab **NVIDIA L4** runtime. Test data are first
evaluated only after each best validation checkpoint and its validation-only
thresholds have been fixed.
"""),
    code(r"""# 1. Mount Drive.
from google.colab import drive
drive.mount("/content/drive", force_remount=False)
"""),
    code(r"""# 2. Install only the notebook dependencies; keep Colab's CUDA-compatible PyTorch build.
%pip install -q "timm>=1.0.15,<1.1" "scikit-learn>=1.4" "pandas>=2.1" \
    "pillow>=10" "matplotlib>=3.8" "seaborn>=0.13" "tqdm>=4.66" \
    "psutil>=5.9" "tabulate>=0.9"
"""),
    md(r"""## Storage contract

The directory-map rules are enforced:

```text
MyDrive/CardioSentry/
├── codes/convnextv2_multilabel_combinations_v1/   # put this notebook here
├── datasets/                                       # read-only source tar (optional checksum sidecar)
├── experiments/convnextv2_multilabel_combinations_v1/<run>/
├── hf_cache/                                       # managed model cache
└── runs/convnextv2_multilabel_combinations_v1/<run>/
    ├── baseline/
    ├── pair_aux/
    ├── pair_triple_aux/
    └── label_dependency/
```

The exact input is pinned to
`prototype_1152_plus_af_lvh_stemi_combinations_v1_20260911T171251Z.tar`.
It is extracted to fast ephemeral `/content` storage. Source archives are never
modified or unpacked inside `datasets/`.
"""),
    code(r"""# 3. Imports, fixed data identity, independent experiment configs, and L4 setup.
import contextlib
import copy
import hashlib
import inspect
import io
import json
import math
import os
import platform
import random
import shutil
import sys
import tarfile
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import sklearn
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from timm.optim import param_groups_layer_decay
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

PROJECT_ROOT = Path(os.environ.get(
    "CARDIOSENTRY_PROJECT_ROOT", "/content/drive/MyDrive/CardioSentry"
)).expanduser().resolve()
DATASET_STEM = "prototype_1152_plus_af_lvh_stemi_combinations_v1_20260911T171251Z"
DATASET_ARCHIVE = Path(os.environ.get(
    "CS_DATASET_ARCHIVE", str(PROJECT_ROOT / "datasets" / f"{DATASET_STEM}.tar")
)).expanduser().resolve()
EXPECTED_ARCHIVE_SHA256 = "fb46fc1b5336d899611ae8c1875efea90133e9b7c7c1aeb2c44718c854a55487"
DATASET_ARCHIVE_SHA256 = Path(os.environ.get(
    "CS_DATASET_ARCHIVE_SHA256", str(DATASET_ARCHIVE) + ".sha256"
)).expanduser().resolve()
LOCAL_DATA_PARENT = Path(os.environ.get(
    "CS_LOCAL_DATA_PARENT", "/content/cardiosentry_scratch"
)).expanduser().resolve()
DATASET_ROOT = Path(os.environ.get(
    "CS_DATASET_ROOT", str(LOCAL_DATA_PARENT / DATASET_STEM)
)).expanduser().resolve()

TOPIC = "convnextv2_multilabel_combinations_v1"
RUN_ID = os.environ.get("CS_RUN_ID", "primary_seed42").strip()
if not RUN_ID or any(part in RUN_ID for part in ("/", "\\")):
    raise ValueError("CS_RUN_ID must be one non-empty path component")
CODES_DIR = PROJECT_ROOT / "codes" / TOPIC
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / TOPIC / RUN_ID
RUN_DIR = PROJECT_ROOT / "runs" / TOPIC / RUN_ID
HF_HOME = PROJECT_ROOT / "hf_cache"
os.environ["HF_HOME"] = str(HF_HOME)
for directory in (CODES_DIR, EXPERIMENT_DIR, RUN_DIR, HF_HOME, LOCAL_DATA_PARENT):
    directory.mkdir(parents=True, exist_ok=True)

LABELS = ["NORMAL", "STEMI", "AF", "LVH"]
PATHOLOGY_LABELS = ["STEMI", "AF", "LVH"]
PAIR_LABELS = ["AF_LVH", "STEMI_AF", "STEMI_LVH"]
TRIPLE_LABEL = "STEMI_AF_LVH"
COMBINATION_ORDER = [
    "NORMAL", "AF", "LVH", "STEMI", "AF+LVH", "STEMI+AF",
    "STEMI+LVH", "STEMI+AF+LVH",
]
BITS_TO_COMBINATION = {
    (1, 0, 0, 0): "NORMAL",
    (0, 0, 1, 0): "AF",
    (0, 0, 0, 1): "LVH",
    (0, 1, 0, 0): "STEMI",
    (0, 0, 1, 1): "AF+LVH",
    (0, 1, 1, 0): "STEMI+AF",
    (0, 1, 0, 1): "STEMI+LVH",
    (0, 1, 1, 1): "STEMI+AF+LVH",
}
EXPECTED_COMBINATIONS = {
    "NORMAL": 1693, "AF": 1664, "LVH": 1606, "STEMI": 1442,
    "AF+LVH": 2376, "STEMI+AF": 173, "STEMI+LVH": 401,
    "STEMI+AF+LVH": 43,
}
EXPECTED_PRIMARY_SUPPORT = {"NORMAL": 1693, "STEMI": 2059, "AF": 4256, "LVH": 4426}
EXPECTED_AUX_SUPPORT = {"AF_LVH": 2419, "STEMI_AF": 216, "STEMI_LVH": 444, TRIPLE_LABEL: 43}
EXPECTED_SPLIT_COUNTS = {"train": 7122, "val": 1136, "test": 1140}

MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
PRETRAINING = "FCMAE ImageNet-1K -> supervised ImageNet-22K -> ImageNet-1K at 384px"
INPUT_SIZE = (768, 1024)
MASTER_REFERENCE = {
    "model": MODEL_NAME, "input_size_hw": list(INPUT_SIZE), "base_lr": 1e-4,
    "weight_decay": 0.05, "layer_decay": 0.90, "drop_path_rate": 0.20,
    "warmup_epochs": 3, "max_epochs": 30, "early_stopping_patience": 6,
    "contradiction_lambda": 0.20, "effective_batch_size": 32,
}

# These configs are independent and architecture-aware. The baseline preserves
# the master recipe; auxiliary/dependency heads may use their own peak head LR.
COMMON = dict(
    max_epochs=30, patience=6, warmup_epochs=3, weight_decay=0.05,
    layer_decay=0.90, drop_path_rate=0.20, contradiction_lambda=0.20,
    effective_batch_size=32, grad_clip_norm=1.0,
)
EXPERIMENT_CONFIGS = {
    "baseline": {**COMMON, "backbone_lr": 1e-4, "head_lr": 1e-4,
                 "pair_lambda": 0.0, "triple_lambda": 0.0},
    "pair_aux": {**COMMON, "backbone_lr": 8e-5, "head_lr": 1.5e-4,
                 "pair_lambda": 0.25, "triple_lambda": 0.0},
    "pair_triple_aux": {**COMMON, "backbone_lr": 8e-5, "head_lr": 1.5e-4,
                        "pair_lambda": 0.25, "triple_lambda": 0.05},
    "label_dependency": {**COMMON, "backbone_lr": 6e-5, "head_lr": 2e-4,
                         "pair_lambda": 0.0, "triple_lambda": 0.0,
                         "attention_dim": 256, "attention_heads": 8,
                         "attention_depth": 1, "attention_dropout": 0.10},
}
REQUIRED_EXPERIMENTS = list(EXPERIMENT_CONFIGS)
SEED = int(os.environ.get("CS_SEED", "42"))
NUM_WORKERS = int(os.environ.get("CS_NUM_WORKERS", str(min(8, max(1, (os.cpu_count() or 2) - 2)))))
PREFETCH_FACTOR = int(os.environ.get("CS_PREFETCH_FACTOR", "2"))
AUTO_BATCH_CANDIDATES = [int(value) for value in os.environ.get(
    "CS_BATCH_CANDIDATES", "16,12,8,4,2,1"
).split(",")]
VRAM_SAFETY_FRACTION = float(os.environ.get("CS_VRAM_SAFETY_FRACTION", "0.90"))
TRY_TORCH_COMPILE = os.environ.get("CS_TRY_TORCH_COMPILE", "1").lower() not in {"0", "false", "no"}
BOOTSTRAP_REPETITIONS = int(os.environ.get("CS_BOOTSTRAP_REPETITIONS", "1000"))

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Select an NVIDIA L4 Colab runtime and reconnect.")
DEVICE = torch.device("cuda:0")
GPU = torch.cuda.get_device_properties(0)
if "L4" not in GPU.name.upper():
    warnings.warn(f"Expected NVIDIA L4; assigned {GPU.name}. Batch probing will adapt.")
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
USE_GRAD_SCALER = AMP_DTYPE == torch.float16
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
print("Dataset archive:", DATASET_ARCHIVE)
print("Run directory:  ", RUN_DIR)
print("Experiment work:", EXPERIMENT_DIR)
print("GPU / VRAM:     ", GPU.name, f"{GPU.total_memory / 2**30:.1f} GiB")
print("AMP dtype:      ", str(AMP_DTYPE).removeprefix("torch."))
print("PyTorch / timm: ", torch.__version__, timm.__version__)
"""),
    md(r"""## Dataset verification and extraction

The independently verified archive checksum is pinned in this notebook; an
adjacent checksum sidecar, when present, must agree. Extraction is path-safe and
goes only to local scratch. The archive's internal `checksums.sha256` is checked
once per runtime extraction. The authoritative labels and split come only from
the embedded `index.csv`.
"""),
    code(r"""# 3. Verify the pinned archive, extract safely, and verify internal files.
def sha256_file(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()

def atomic_json_save(payload, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, destination)

if not DATASET_ARCHIVE.is_file():
    raise FileNotFoundError(
        f"Pinned dataset archive is missing: {DATASET_ARCHIVE}. Do not substitute prototype_1152.tar."
    )
expected_archive_sha256 = EXPECTED_ARCHIVE_SHA256
if DATASET_ARCHIVE_SHA256.is_file():
    checksum_parts = DATASET_ARCHIVE_SHA256.read_text(encoding="utf-8").split()
    if not checksum_parts:
        raise ValueError("Archive checksum sidecar is empty")
    sidecar_sha256 = checksum_parts[0].lower()
    if sidecar_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(
            f"Checksum sidecar disagrees with the pinned digest: {sidecar_sha256}"
        )
    print("Adjacent checksum sidecar agrees with pinned digest")
else:
    print("No adjacent checksum sidecar; using the independently verified pinned digest")
actual_archive_sha256 = sha256_file(DATASET_ARCHIVE)
if actual_archive_sha256 != expected_archive_sha256:
    raise RuntimeError(
        f"Archive SHA-256 mismatch: expected {expected_archive_sha256}, observed {actual_archive_sha256}"
    )
print("Archive SHA-256 verified:", actual_archive_sha256)

extract_marker = DATASET_ROOT / ".cardiosentry_extract_verified.json"
marker_ok = False
if extract_marker.is_file() and (DATASET_ROOT / "index.csv").is_file():
    marker = json.loads(extract_marker.read_text(encoding="utf-8"))
    marker_ok = marker.get("archive_sha256") == actual_archive_sha256

if not marker_ok:
    if DATASET_ROOT.exists():
        raise RuntimeError(
            f"Unverified extraction already exists at {DATASET_ROOT}. Remove that local scratch directory "
            "or set CS_LOCAL_DATA_PARENT to a clean local path; Drive data were not touched."
        )
    LOCAL_DATA_PARENT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(DATASET_ARCHIVE, "r:") as archive:
        members = archive.getmembers()
        root = LOCAL_DATA_PARENT.resolve()
        for member in members:
            destination = (LOCAL_DATA_PARENT / member.name).resolve()
            if root not in destination.parents and destination != root:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"Archive contains unsupported link/device: {member.name}")
        print(f"Extracting {len(members):,} members to local scratch...")
        try:
            archive.extractall(LOCAL_DATA_PARENT, members=members, filter="data")
        except TypeError:  # Python <3.12 compatibility
            archive.extractall(LOCAL_DATA_PARENT, members=members)
    if not (DATASET_ROOT / "index.csv").is_file():
        raise FileNotFoundError(f"Extracted archive did not create {DATASET_ROOT / 'index.csv'}")

    checksum_manifest = DATASET_ROOT / "checksums.sha256"
    if not checksum_manifest.is_file():
        raise FileNotFoundError("Extracted dataset is missing checksums.sha256")
    checksum_rows = []
    for line in checksum_manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        checksum_rows.append((digest.lower(), relative.strip()))

    def verify_internal(item):
        expected, relative = item
        path = DATASET_ROOT / relative
        return relative, path.is_file() and sha256_file(path) == expected

    failures = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, NUM_WORKERS))) as executor:
        for completed, (relative, ok) in enumerate(executor.map(verify_internal, checksum_rows), 1):
            if not ok:
                failures.append(relative)
            if completed % 1000 == 0 or completed == len(checksum_rows):
                print(f"Verified {completed:,}/{len(checksum_rows):,} internal files", flush=True)
    if failures:
        raise RuntimeError(f"Internal checksum failures: {failures[:10]}")
    atomic_json_save({
        "archive": str(DATASET_ARCHIVE),
        "archive_sha256": actual_archive_sha256,
        "verified_internal_files": len(checksum_rows),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }, extract_marker)
else:
    print("Reusing checksum-verified local extraction:", DATASET_ROOT)
"""),
    md(r"""## Frozen-split audit

All audits are performed before model training. A genuine integrity failure
raises immediately; the notebook never repairs, resamples, or re-splits data.
The zero-support triple test row is retained in every table and confusion
matrix.
"""),
    code(r"""# 4. Audit records, labels, combinations, split isolation, and file coverage.
index_path = DATASET_ROOT / "index.csv"
index_df = pd.read_csv(index_path, low_memory=False)
required_columns = {"rel_path", "record", "record_id", "cls", "split", "source", "patient_id", *LABELS}
missing = sorted(required_columns - set(index_df.columns))
if missing:
    raise ValueError(f"index.csv is missing required columns: {missing}")
if len(index_df) != 9398:
    raise ValueError(f"Expected 9,398 rows/records, observed {len(index_df):,}")

index_df["split"] = index_df["split"].astype(str).str.lower().str.strip()
if set(index_df["split"]) != {"train", "val", "test"}:
    raise ValueError(f"Expected train/val/test only; observed {sorted(index_df['split'].unique())}")
for label in LABELS:
    index_df[label] = pd.to_numeric(index_df[label], errors="raise").astype(np.int8)
    if not index_df[label].isin([0, 1]).all():
        raise ValueError(f"{label} contains values outside 0/1")

index_df["AF_LVH"] = index_df["AF"] & index_df["LVH"]
index_df["STEMI_AF"] = index_df["STEMI"] & index_df["AF"]
index_df["STEMI_LVH"] = index_df["STEMI"] & index_df["LVH"]
index_df[TRIPLE_LABEL] = index_df["STEMI"] & index_df["AF"] & index_df["LVH"]

pathology_count = index_df[PATHOLOGY_LABELS].sum(axis=1)
valid_normal = index_df["NORMAL"].eq(1) == pathology_count.eq(0)
if not valid_normal.all():
    raise ValueError("Ground truth violates NORMAL = no pathology semantics")

def exact_name(row):
    bits = tuple(int(row[label]) for label in LABELS)
    return BITS_TO_COMBINATION.get(bits, "INVALID:" + "+".join(
        label for label, value in zip(LABELS, bits) if value
    ))

index_df["exact_combination"] = index_df.apply(exact_name, axis=1)
class_text = index_df["cls"].astype(str).str.replace(" ", "", regex=False)
if not class_text.eq(index_df["exact_combination"]).all():
    bad = index_df.loc[~class_text.eq(index_df["exact_combination"]), ["record_id", "cls", "exact_combination"]].head()
    raise RuntimeError("cls disagrees with binary labels:\n" + bad.to_string(index=False))
observed_combinations = index_df["exact_combination"].value_counts().reindex(COMBINATION_ORDER, fill_value=0).astype(int).to_dict()
observed_primary = index_df[LABELS].sum().astype(int).to_dict()
observed_auxiliary = index_df[[*PAIR_LABELS, TRIPLE_LABEL]].sum().astype(int).to_dict()
observed_splits = index_df["split"].value_counts().astype(int).to_dict()
for name, observed, expected in (
    ("exact combination", observed_combinations, EXPECTED_COMBINATIONS),
    ("primary label", observed_primary, EXPECTED_PRIMARY_SUPPORT),
    ("auxiliary label", observed_auxiliary, EXPECTED_AUX_SUPPORT),
    ("split", observed_splits, EXPECTED_SPLIT_COUNTS),
):
    if observed != expected:
        raise RuntimeError(f"{name} support mismatch: expected {expected}, observed {observed}")

for identity in ("record", "record_id", "rel_path"):
    if index_df[identity].astype(str).duplicated().any():
        raise ValueError(f"Duplicate {identity} values detected")
    if index_df.groupby(identity)["split"].nunique().gt(1).any():
        raise ValueError(f"{identity} crosses splits")
patient = index_df["patient_id"].fillna("").astype(str).str.strip()
identified = index_df.loc[patient.ne("")].copy()
patient_leaks = identified.groupby("patient_id")["split"].nunique()
patient_leaks = patient_leaks[patient_leaks > 1]
if len(patient_leaks):
    raise RuntimeError(f"Patient leakage across existing splits: {patient_leaks.index[:10].tolist()}")

index_df["local_path"] = index_df["rel_path"].map(lambda value: str((DATASET_ROOT / str(value)).resolve()))
missing_images = [path for path in index_df["local_path"] if not Path(path).is_file()]
if missing_images:
    raise FileNotFoundError(f"{len(missing_images):,} indexed images are missing; first: {missing_images[:5]}")

split_combination = pd.crosstab(index_df["exact_combination"], index_df["split"]).reindex(
    index=COMBINATION_ORDER, columns=["train", "val", "test"], fill_value=0
).astype(int)
split_primary = pd.concat({
    split: frame[LABELS].sum().astype(int)
    for split, frame in index_df.groupby("split", sort=False)
}, axis=1).reindex(columns=["train", "val", "test"])
split_auxiliary = pd.concat({
    split: frame[[*PAIR_LABELS, TRIPLE_LABEL]].sum().astype(int)
    for split, frame in index_df.groupby("split", sort=False)
}, axis=1).reindex(columns=["train", "val", "test"])

display(split_combination.rename_axis("Combination").reset_index())
display(split_primary.rename_axis("Primary label").reset_index())
display(split_auxiliary.rename_axis("Auxiliary label").reset_index())
print(f"Patient IDs available: {len(identified):,}/{len(index_df):,}; cross-split patients: {len(patient_leaks)}")
print("Triple-positive support:", split_combination.loc["STEMI+AF+LVH"].to_dict())

provenance_path = DATASET_ROOT / "MERGE_PROVENANCE.json"
if not provenance_path.is_file():
    raise FileNotFoundError("Merged dataset is missing MERGE_PROVENANCE.json")
merge_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
if merge_provenance.get("counts", {}).get("final_images") != 9398:
    raise RuntimeError("Merge provenance does not declare 9,398 final images")
if merge_provenance.get("split_integrity", {}).get("patients_spanning_splits_after_merge") != 0:
    raise RuntimeError("Merge provenance reports patient leakage")

dataset_audit = {
    "dataset_stem": DATASET_STEM,
    "archive": str(DATASET_ARCHIVE),
    "archive_sha256": actual_archive_sha256,
    "index_sha256": sha256_file(index_path),
    "merge_provenance_sha256": sha256_file(provenance_path),
    "records": int(len(index_df)),
    "split_counts": observed_splits,
    "exact_combination_counts": observed_combinations,
    "primary_label_support": observed_primary,
    "auxiliary_label_support": observed_auxiliary,
    "patient_ids_available": int(len(identified)),
    "patient_cross_split_leaks": int(len(patient_leaks)),
    "normal_pathology_target_contradictions": int((~valid_normal).sum()),
    "split_combination_counts": split_combination.to_dict(),
    "split_primary_support": split_primary.to_dict(),
    "split_auxiliary_support": split_auxiliary.to_dict(),
    "triple_test_support": int(split_combination.loc["STEMI+AF+LVH", "test"]),
    "triple_test_inference_limit": "Held-out triple recall/F1 are undefined because test support is zero.",
}
for destination in (EXPERIMENT_DIR / "dataset_audit.json", RUN_DIR / "dataset_audit.json"):
    atomic_json_save(dataset_audit, destination)
for destination in (EXPERIMENT_DIR / "dataset_audit.csv", RUN_DIR / "dataset_audit.csv"):
    split_combination.to_csv(destination)
for directory in (EXPERIMENT_DIR, RUN_DIR):
    split_primary.to_csv(directory / "primary_support_by_split.csv")
    split_auxiliary.to_csv(directory / "auxiliary_support_by_split.csv")
"""),
    md(r"""## Deterministic input pipeline

The input path is identical for train, validation, and test: RGB conversion,
fixed 768×1024 bicubic resize, tensor conversion, and ImageNet normalization.
There is deliberately no online augmentation. Each of the 9,398 records has
one already-existing image.
"""),
    code(r"""# 5. Dataset and deterministic preprocessing.
ImageFile.LOAD_TRUNCATED_IMAGES = False
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
image_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE, interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class ECGDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True).copy()
        self.primary = self.frame[LABELS].to_numpy(np.float32)
        self.pair = self.frame[PAIR_LABELS].to_numpy(np.float32)
        self.triple = self.frame[[TRIPLE_LABEL]].to_numpy(np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, item):
        row = self.frame.iloc[item]
        with Image.open(row.local_path) as image:
            tensor = image_transform(image.convert("RGB"))
        return {
            "image": tensor,
            "primary": torch.from_numpy(self.primary[item]),
            "pair": torch.from_numpy(self.pair[item]),
            "triple": torch.from_numpy(self.triple[item]),
            "record_id": str(row.record_id),
            "record": str(row.record),
            "patient_id": str(row.patient_id),
            "source": str(row.source),
            "split": str(row.split),
            "true_exact_combination": str(row.exact_combination),
        }

train_df = index_df[index_df.split.eq("train")].reset_index(drop=True)
val_df = index_df[index_df.split.eq("val")].reset_index(drop=True)
test_df = index_df[index_df.split.eq("test")].reset_index(drop=True)
train_dataset, val_dataset, test_dataset = map(ECGDataset, (train_df, val_df, test_df))

for name, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
    sample = dataset[0]
    assert sample["image"].shape == (3, *INPUT_SIZE)
    print(name, len(dataset), tuple(sample["image"].shape), sample["primary"].tolist(), sample["record_id"])
"""),
    md(r"""## Architectures

Experiments A–C use the same pooled ConvNeXt-V2 representation and separate
linear heads. Experiment D retains the spatial ConvNeXt feature map, projects
it to 256 dimensions, lets four learned label queries cross-attend to spatial
features, then passes the four label representations through a lightweight
self-attention encoder. A distinct classifier vector produces each label logit.
PyTorch multi-head attention uses scaled-dot-product attention on supported L4
kernels.
"""),
    code(r"""# 6. Model definitions.
def initialize_linear(layer):
    nn.init.trunc_normal_(layer.weight, std=0.02)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

class PooledMultiHeadModel(nn.Module):
    def __init__(self, experiment, pretrained, drop_path_rate):
        super().__init__()
        self.experiment = experiment
        self.backbone = timm.create_model(
            MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool="avg",
            drop_path_rate=drop_path_rate,
        )
        width = int(self.backbone.num_features)
        self.primary_head = nn.Linear(width, len(LABELS))
        initialize_linear(self.primary_head)
        self.pair_head = None
        self.triple_head = None
        if experiment in {"pair_aux", "pair_triple_aux"}:
            self.pair_head = nn.Linear(width, len(PAIR_LABELS))
            initialize_linear(self.pair_head)
        if experiment == "pair_triple_aux":
            self.triple_head = nn.Linear(width, 1)
            initialize_linear(self.triple_head)

    def forward(self, images):
        representation = self.backbone(images)
        output = {"primary_logits": self.primary_head(representation)}
        if self.pair_head is not None:
            output["pair_logits"] = self.pair_head(representation)
        if self.triple_head is not None:
            output["triple_logits"] = self.triple_head(representation)
        return output

class LabelDependencyModel(nn.Module):
    def __init__(self, pretrained, drop_path_rate, attention_dim, attention_heads,
                 attention_depth, attention_dropout):
        super().__init__()
        self.backbone = timm.create_model(
            MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool="",
            drop_path_rate=drop_path_rate,
        )
        backbone_dim = int(self.backbone.num_features)
        self.feature_projection = nn.Conv2d(backbone_dim, attention_dim, kernel_size=1)
        self.label_queries = nn.Parameter(torch.empty(1, len(LABELS), attention_dim))
        self.cross_norm_q = nn.LayerNorm(attention_dim)
        self.cross_norm_kv = nn.LayerNorm(attention_dim)
        self.cross_attention = nn.MultiheadAttention(
            attention_dim, attention_heads, dropout=attention_dropout, batch_first=True
        )
        self.cross_output_norm = nn.LayerNorm(attention_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attention_dim,
            nhead=attention_heads,
            dim_feedforward=attention_dim * 4,
            dropout=attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.label_interaction = nn.TransformerEncoder(
            encoder_layer, num_layers=attention_depth, norm=nn.LayerNorm(attention_dim)
        )
        self.label_classifier_weight = nn.Parameter(torch.empty(len(LABELS), attention_dim))
        self.label_classifier_bias = nn.Parameter(torch.zeros(len(LABELS)))
        nn.init.trunc_normal_(self.feature_projection.weight, std=0.02)
        nn.init.zeros_(self.feature_projection.bias)
        nn.init.trunc_normal_(self.label_queries, std=0.02)
        nn.init.trunc_normal_(self.label_classifier_weight, std=0.02)

    def forward(self, images):
        spatial = self.backbone.forward_features(images)
        if spatial.ndim != 4:
            raise RuntimeError(f"Expected a 4D ConvNeXt spatial map, received {tuple(spatial.shape)}")
        if spatial.shape[1] != self.backbone.num_features and spatial.shape[-1] == self.backbone.num_features:
            spatial = spatial.permute(0, 3, 1, 2).contiguous()
        spatial = self.feature_projection(spatial).flatten(2).transpose(1, 2)
        queries = self.label_queries.expand(images.shape[0], -1, -1)
        attended, _ = self.cross_attention(
            self.cross_norm_q(queries), self.cross_norm_kv(spatial),
            self.cross_norm_kv(spatial), need_weights=False,
        )
        labels = self.cross_output_norm(queries + attended)
        labels = self.label_interaction(labels)
        logits = (labels * self.label_classifier_weight.unsqueeze(0)).sum(dim=-1)
        logits = logits + self.label_classifier_bias.unsqueeze(0)
        return {"primary_logits": logits}

def build_model(experiment, pretrained=True):
    cfg = EXPERIMENT_CONFIGS[experiment]
    if experiment == "label_dependency":
        return LabelDependencyModel(
            pretrained=pretrained,
            drop_path_rate=cfg["drop_path_rate"],
            attention_dim=cfg["attention_dim"],
            attention_heads=cfg["attention_heads"],
            attention_depth=cfg["attention_depth"],
            attention_dropout=cfg["attention_dropout"],
        )
    return PooledMultiHeadModel(
        experiment=experiment, pretrained=pretrained, drop_path_rate=cfg["drop_path_rate"]
    )
"""),
    md(r"""## Losses and imbalance handling

Primary `pos_weight` is exactly training negatives / positives, matching the
master notebook. Auxiliary weights use `sqrt(negatives/positives)` capped at
10. This avoids letting 38 triple-positive training examples dominate through
a raw inverse-frequency weight near 186. The pair and triple loss coefficients
are recorded independently for each experiment.
"""),
    code(r"""# 7. Training-only weights, multi-task objective, and scalar metrics.
def training_pos_weight(frame, columns, sqrt_scale=False, cap=None):
    positive = frame[columns].sum().astype(float)
    negative = len(frame) - positive
    if positive.eq(0).any():
        raise ValueError(f"Training support is zero: {positive.to_dict()}")
    weight = negative / positive
    if sqrt_scale:
        weight = np.sqrt(weight)
    if cap is not None:
        weight = weight.clip(upper=float(cap))
    return torch.tensor(weight.to_numpy(), dtype=torch.float32), pd.DataFrame({
        "positive": positive.astype(int), "negative": negative.astype(int), "pos_weight": weight
    })

PRIMARY_POS_WEIGHT, primary_weight_table = training_pos_weight(train_df, LABELS)
PAIR_POS_WEIGHT, pair_weight_table = training_pos_weight(train_df, PAIR_LABELS, sqrt_scale=True, cap=10)
TRIPLE_POS_WEIGHT, triple_weight_table = training_pos_weight(train_df, [TRIPLE_LABEL], sqrt_scale=True, cap=10)
print("Primary training weights (master formula)")
display(primary_weight_table.round(4))
print("Auxiliary training weights (sqrt inverse-frequency, cap=10)")
display(pd.concat([pair_weight_table, triple_weight_table]).round(4))

class CombinationAwareLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer("primary_pos_weight", PRIMARY_POS_WEIGHT.clone())
        self.register_buffer("pair_pos_weight", PAIR_POS_WEIGHT.clone())
        self.register_buffer("triple_pos_weight", TRIPLE_POS_WEIGHT.clone())
        self.pair_lambda = float(config["pair_lambda"])
        self.triple_lambda = float(config["triple_lambda"])
        self.contradiction_lambda = float(config["contradiction_lambda"])

    def forward(self, outputs, batch):
        logits = outputs["primary_logits"]
        primary_bce = F.binary_cross_entropy_with_logits(
            logits, batch["primary"], pos_weight=self.primary_pos_weight
        )
        probabilities = torch.sigmoid(logits)
        normal = probabilities[:, :1]
        pathology = probabilities[:, 1:]
        any_pathology = 1.0 - torch.prod(1.0 - pathology, dim=1, keepdim=True)
        contradiction = (normal * any_pathology).mean()
        pair_bce = logits.new_zeros(())
        triple_bce = logits.new_zeros(())
        if self.pair_lambda:
            pair_bce = F.binary_cross_entropy_with_logits(
                outputs["pair_logits"], batch["pair"], pos_weight=self.pair_pos_weight
            )
        if self.triple_lambda:
            triple_bce = F.binary_cross_entropy_with_logits(
                outputs["triple_logits"], batch["triple"], pos_weight=self.triple_pos_weight
            )
        weighted_pair = self.pair_lambda * pair_bce
        weighted_triple = self.triple_lambda * triple_bce
        weighted_contradiction = self.contradiction_lambda * contradiction
        total = primary_bce + weighted_pair + weighted_triple + weighted_contradiction
        return total, {
            "primary_bce": primary_bce.detach(), "pair_bce": pair_bce.detach(),
            "triple_bce": triple_bce.detach(), "contradiction": contradiction.detach(),
            "weighted_pair": weighted_pair.detach(), "weighted_triple": weighted_triple.detach(),
            "weighted_contradiction": weighted_contradiction.detach(),
        }

def safe_rank(metric, truth, probability):
    truth = np.asarray(truth)
    return float(metric(truth, probability)) if np.unique(truth).size == 2 else float("nan")

def divide(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")

def binary_metrics(truth, probability, threshold):
    truth = np.asarray(truth, dtype=np.int8)
    prediction = (np.asarray(probability) >= threshold).astype(np.int8)
    tp = int(((prediction == 1) & (truth == 1)).sum())
    tn = int(((prediction == 0) & (truth == 0)).sum())
    fp = int(((prediction == 1) & (truth == 0)).sum())
    fn = int(((prediction == 0) & (truth == 1)).sum())
    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    precision = divide(tp, tp + fp)
    npv = divide(tn, tn + fn)
    return {
        "support": int(truth.sum()), "negative_support": int(len(truth) - truth.sum()),
        "ap": safe_rank(average_precision_score, truth, probability),
        "auroc": safe_rank(roc_auc_score, truth, probability),
        "sensitivity": sensitivity, "specificity": specificity, "precision": precision,
        "npv": npv, "f1": divide(2 * tp, 2 * tp + fp + fn),
        "threshold": float(threshold), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

def primary_metrics(truth, probability, thresholds):
    truth = np.asarray(truth, dtype=np.int8)
    probability = np.asarray(probability, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    prediction = (probability >= thresholds[None, :]).astype(np.int8)
    per_label = {
        label: binary_metrics(truth[:, i], probability[:, i], thresholds[i])
        for i, label in enumerate(LABELS)
    }
    macro_names = ["ap", "auroc", "sensitivity", "specificity", "precision", "f1"]
    result = {
        f"macro_{metric}": float(np.nanmean([per_label[label][metric] for label in LABELS]))
        for metric in macro_names
    }
    pooled_tp = sum(per_label[label]["tp"] for label in LABELS)
    pooled_fp = sum(per_label[label]["fp"] for label in LABELS)
    pooled_fn = sum(per_label[label]["fn"] for label in LABELS)
    normal = prediction[:, 0].astype(bool)
    pathology = prediction[:, 1:].astype(bool)
    contradiction = normal & pathology.any(axis=1)
    no_label = prediction.sum(axis=1) == 0
    result.update({
        "micro_f1": divide(2 * pooled_tp, 2 * pooled_tp + pooled_fp + pooled_fn),
        "hamming_accuracy": float((prediction == truth).mean()),
        "hamming_loss": float((prediction != truth).mean()),
        "exact_match_accuracy": float((prediction == truth).all(axis=1).mean()),
        "contradiction_count": int(contradiction.sum()),
        "contradiction_rate": float(contradiction.mean()),
        "no_label_prediction_count": int(no_label.sum()),
        "no_label_prediction_rate": float(no_label.mean()),
        "n": int(len(truth)), "per_label": per_label,
    })
    return result

def tune_thresholds_on_validation(truth, probability):
    thresholds, details = [], {}
    for i, label in enumerate(LABELS):
        y, p = truth[:, i], probability[:, i]
        precision, recall, candidates = precision_recall_curve(y, p)
        if len(candidates) == 0:
            threshold = 0.5
        else:
            f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
            threshold = float(candidates[int(np.nanargmax(f1))])
        thresholds.append(threshold)
        details[label] = binary_metrics(y, p, threshold)
    return thresholds, details
"""),
    code(r"""# 8. Exact-combination and auxiliary evaluation. No contradiction post-processing is used.
def names_from_bits(bits):
    names = []
    for row in np.asarray(bits, dtype=np.int8):
        key = tuple(int(value) for value in row)
        if key in BITS_TO_COMBINATION:
            names.append(BITS_TO_COMBINATION[key])
        elif sum(key) == 0:
            names.append("NO_LABEL")
        else:
            names.append(" + ".join(label for label, value in zip(LABELS, key) if value))
    return np.asarray(names, dtype=object)

def exact_combination_evaluation(truth, probability, thresholds):
    truth = np.asarray(truth, dtype=np.int8)
    prediction = (np.asarray(probability) >= np.asarray(thresholds)[None, :]).astype(np.int8)
    true_names = names_from_bits(truth)
    predicted_names = names_from_bits(prediction)
    rows = []
    for name in COMBINATION_ORDER:
        truth_mask = true_names == name
        prediction_mask = predicted_names == name
        support = int(truth_mask.sum())
        predicted_count = int(prediction_mask.sum())
        correct = int((truth_mask & prediction_mask).sum())
        wrong = Counter(predicted_names[truth_mask & ~prediction_mask])
        rows.append({
            "combination": name, "support": support, "predicted_count": predicted_count,
            "correct_predictions": correct, "incorrect_predictions": support - correct,
            "precision": divide(correct, predicted_count), "recall": divide(correct, support),
            # With no true support, recall and F1 are not estimable even if the
            # model emits false-positive predictions for that class.
            "f1": float("nan") if support == 0 else divide(2 * correct, support + predicted_count),
            "most_common_incorrect_predicted_combinations": "; ".join(
                f"{label} ({count})" for label, count in wrong.most_common(3)
            ) if wrong else "",
        })
    valid_confusion = confusion_matrix(true_names, predicted_names, labels=COMBINATION_ORDER)
    invalid = pd.Series(predicted_names[~np.isin(predicted_names, COMBINATION_ORDER)]).value_counts()
    invalid_rows = [{"predicted_combination": str(name), "count": int(count)} for name, count in invalid.items()]
    row_frame = pd.DataFrame(rows)
    supported = row_frame.support.gt(0)
    cooccurrence = row_frame.combination.isin(["AF+LVH", "STEMI+AF", "STEMI+LVH", "STEMI+AF+LVH"]) & supported
    single = row_frame.combination.isin(["NORMAL", "AF", "LVH", "STEMI"]) & supported
    summary = {
        "exact_combo_macro_f1_supported": float(row_frame.loc[supported, "f1"].mean()),
        "cooccurrence_macro_f1_supported": float(row_frame.loc[cooccurrence, "f1"].mean()),
        "single_state_macro_f1_supported": float(row_frame.loc[single, "f1"].mean()),
        "invalid_prediction_count": int((~np.isin(predicted_names, COMBINATION_ORDER)).sum()),
    }
    return row_frame, valid_confusion, invalid_rows, summary, true_names, predicted_names

def auxiliary_metrics(truth, probability, threshold=0.5):
    truth = np.asarray(truth, dtype=np.int8)
    probability = np.asarray(probability, dtype=float)
    row = binary_metrics(truth, probability, threshold)
    if row["support"] == 0:
        row["f1"] = float("nan")
    if np.unique(truth).size == 2:
        precision, recall, _ = precision_recall_curve(truth, probability)
        row["pr_auc_trapezoidal"] = float(auc(recall[::-1], precision[::-1]))
    else:
        row["pr_auc_trapezoidal"] = float("nan")
    return row

def save_confusion_plot(matrix, destination, title):
    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=COMBINATION_ORDER, yticklabels=COMBINATION_ORDER, ax=axis)
    axis.set(xlabel="Predicted exact combination", ylabel="True exact combination", title=title)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
"""),
    md(r"""## L4 batch probing, optimizer grouping, and compile benchmark

Every architecture performs a real forward/backward/AdamW update probe and
selects the largest candidate below 90% peak allocation. Gradient accumulation
targets effective batch 32. `torch.compile` is benchmarked briefly on an actual
forward/backward workload and retained only when stable and at least 3% faster.
"""),
    code(r"""# 9. Loaders, optimizer, AMP, physical-batch probe, and torch.compile benchmark.
def autocast_context():
    return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)

def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=USE_GRAD_SCALER)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=USE_GRAD_SCALER)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def make_loaders(batch_size, experiment_seed):
    options = dict(
        batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        worker_init_fn=seed_worker,
    )
    generator = torch.Generator().manual_seed(experiment_seed)
    return {
        "train": DataLoader(train_dataset, shuffle=True, generator=generator, drop_last=False, **options),
        "train_eval": DataLoader(train_dataset, shuffle=False, drop_last=False, **options),
        "val": DataLoader(val_dataset, shuffle=False, drop_last=False, **options),
        "test": DataLoader(test_dataset, shuffle=False, drop_last=False, **options),
        "generator": generator,
    }

def build_optimizer(model, config):
    backbone_groups = param_groups_layer_decay(
        model.backbone, weight_decay=config["weight_decay"], layer_decay=config["layer_decay"]
    )
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters() if parameter.requires_grad}
    for group in backbone_groups:
        group["lr"] = config["backbone_lr"] * float(group.get("lr_scale", 1.0))
    head_decay, head_no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in backbone_ids:
            continue
        if parameter.ndim <= 1 or name.endswith("bias") or "label_queries" in name:
            head_no_decay.append(parameter)
        else:
            head_decay.append(parameter)
    groups = list(backbone_groups)
    if head_decay:
        groups.append({"params": head_decay, "lr": config["head_lr"], "weight_decay": config["weight_decay"], "lr_scale": "head"})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": config["head_lr"], "weight_decay": 0.0, "lr_scale": "head_no_decay"})
    grouped = [id(parameter) for group in groups for parameter in group["params"]]
    expected = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    if len(grouped) != len(set(grouped)) or set(grouped) != set(expected):
        raise RuntimeError("Optimizer groups do not cover every trainable parameter exactly once")
    kwargs = dict(lr=config["head_lr"], betas=(0.9, 0.999), weight_decay=config["weight_decay"])
    if "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(groups, **kwargs)
        fused = bool(kwargs.get("fused"))
    except (TypeError, RuntimeError):
        kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(groups, **kwargs)
        fused = False
    return optimizer, fused

def synthetic_batch(batch_size, device):
    indices = torch.arange(batch_size, device=device) % len(BITS_TO_COMBINATION)
    bits = torch.tensor(list(BITS_TO_COMBINATION), device=device, dtype=torch.float32)[indices]
    return {
        "primary": bits,
        "pair": torch.stack((bits[:, 2] * bits[:, 3], bits[:, 1] * bits[:, 2], bits[:, 1] * bits[:, 3]), dim=1),
        "triple": (bits[:, 1] * bits[:, 2] * bits[:, 3]).unsqueeze(1),
    }

def probe_physical_batch(experiment, initial_state, config):
    rows, accepted = [], None
    for batch_size in AUTO_BATCH_CANDIDATES:
        probe = criterion = optimizer = scaler = images = None
        try:
            torch.cuda.empty_cache()
            probe = build_model(experiment, pretrained=False)
            probe.load_state_dict(initial_state, strict=True)
            probe = probe.to(DEVICE, memory_format=torch.channels_last).train()
            criterion = CombinationAwareLoss(config).to(DEVICE)
            optimizer, _ = build_optimizer(probe, config)
            scaler = make_scaler()
            images = torch.randn(batch_size, 3, *INPUT_SIZE, device=DEVICE).contiguous(memory_format=torch.channels_last)
            batch = synthetic_batch(batch_size, DEVICE)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            with autocast_context():
                outputs = probe(images)
                loss, _ = criterion(outputs, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            fraction = peak / GPU.total_memory
            status = "fits" if fraction <= VRAM_SAFETY_FRACTION else "rejected_headroom"
            rows.append({"physical_batch_size": batch_size, "peak_vram_gib": peak / 2**30,
                         "vram_fraction": fraction, "status": status})
            print(experiment, rows[-1])
            if status == "fits":
                accepted = batch_size
                break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if "out of memory" not in str(error).lower() and not isinstance(error, torch.cuda.OutOfMemoryError):
                raise
            rows.append({"physical_batch_size": batch_size, "peak_vram_gib": float("nan"),
                         "vram_fraction": float("nan"), "status": "OOM"})
            print(experiment, "batch", batch_size, "OOM")
        finally:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            del probe, criterion, optimizer, scaler, images
            torch.cuda.empty_cache()
    if accepted is None:
        raise RuntimeError(f"No physical batch candidate fits for {experiment}")
    return accepted, pd.DataFrame(rows)

def benchmark_compile(model, criterion, experiment_dir, batch_size):
    probe_batch_size = min(batch_size, 4)
    images = torch.randn(probe_batch_size, 3, *INPUT_SIZE, device=DEVICE).contiguous(memory_format=torch.channels_last)
    batch = synthetic_batch(probe_batch_size, DEVICE)

    def time_steps(active, warmup=2, repetitions=3):
        active.train()
        for _ in range(warmup):
            active.zero_grad(set_to_none=True)
            with autocast_context():
                loss, _ = criterion(active(images), batch)
            loss.backward()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repetitions):
            active.zero_grad(set_to_none=True)
            with autocast_context():
                loss, _ = criterion(active(images), batch)
            loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return repetitions * probe_batch_size / elapsed

    eager_ips = time_steps(model)
    compiled_model, compiled_ips, status = model, None, "not_attempted"
    if TRY_TORCH_COMPILE and hasattr(torch, "compile"):
        try:
            candidate = torch.compile(model, dynamic=True)
            compiled_ips = time_steps(candidate)
            if compiled_ips >= eager_ips * 1.03:
                compiled_model, status = candidate, "enabled"
            else:
                status = "rejected_no_3pct_gain"
        except Exception as error:
            status = f"failed:{type(error).__name__}"
            warnings.warn(f"torch.compile benchmark failed for {experiment_dir.name}: {error}")
    row = pd.DataFrame([{
        "probe_batch_size": probe_batch_size, "eager_images_per_second": eager_ips,
        "compiled_images_per_second": compiled_ips, "status": status,
    }])
    row.to_csv(experiment_dir / "torch_compile_benchmark.csv", index=False)
    model.zero_grad(set_to_none=True)
    del images, batch
    torch.cuda.empty_cache()
    return compiled_model, row.iloc[0].to_dict()
"""),
    md(r"""## Pre-training sanity gate

Before any long run, this cell decodes real images and sends a real batch
through each architecture, verifies output and target shapes, every loss term,
finite backward propagation into ConvNeXt, the contradiction formula, Macro AP,
Drive checkpoint write/read, and prediction CSV export. Any failure stops all
training.
"""),
    code(r"""# 10. Mandatory sanity checks for all four architectures.
perfect_truth = np.asarray(list(BITS_TO_COMBINATION), dtype=np.int8)
perfect_probability = perfect_truth * 0.98 + (1 - perfect_truth) * 0.02
assert math.isclose(primary_metrics(perfect_truth, perfect_probability, [0.5] * 4)["macro_ap"], 1.0)
loss_probe = CombinationAwareLoss(EXPERIMENT_CONFIGS["baseline"])
_, contradiction_parts = loss_probe(
    {"primary_logits": torch.zeros(2, 4)},
    {"primary": torch.tensor([[1, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.float32),
     "pair": torch.zeros(2, 3), "triple": torch.zeros(2, 1)},
)
assert torch.isclose(contradiction_parts["contradiction"], torch.tensor(0.4375))

sanity_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
real_batch = next(iter(sanity_loader))
for key in ("image", "primary", "pair", "triple"):
    real_batch[key] = real_batch[key].to(DEVICE)
for experiment in REQUIRED_EXPERIMENTS:
    model = build_model(experiment, pretrained=True).to(DEVICE, memory_format=torch.channels_last).train()
    criterion = CombinationAwareLoss(EXPERIMENT_CONFIGS[experiment]).to(DEVICE)
    model.zero_grad(set_to_none=True)
    with autocast_context():
        outputs = model(real_batch["image"].contiguous(memory_format=torch.channels_last))
        loss, parts = criterion(outputs, real_batch)
    assert outputs["primary_logits"].shape == (1, 4)
    if experiment in {"pair_aux", "pair_triple_aux"}:
        assert outputs["pair_logits"].shape == (1, 3)
    if experiment == "pair_triple_aux":
        assert outputs["triple_logits"].shape == (1, 1)
    assert torch.isfinite(loss) and all(torch.isfinite(value) for value in parts.values())
    loss.backward()
    backbone_gradient = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.backbone.parameters() if parameter.grad is not None
    )
    assert backbone_gradient > 0, f"No ConvNeXt gradient in {experiment}"
    print(experiment, "sanity passed; loss=", float(loss.detach()), "backbone |grad|=", backbone_gradient)
    del model, criterion, outputs, loss
    torch.cuda.empty_cache()

checkpoint_probe = EXPERIMENT_DIR / ".checkpoint_write_probe.pt"
torch.save({"ok": True, "tensor": torch.arange(3)}, checkpoint_probe)
assert torch.load(checkpoint_probe, map_location="cpu", weights_only=False)["ok"] is True
checkpoint_probe.unlink()
prediction_probe = EXPERIMENT_DIR / ".prediction_export_probe.csv"
pd.DataFrame({"record_id": ["sanity"], "prob_NORMAL": [0.5]}).to_csv(prediction_probe, index=False)
assert pd.read_csv(prediction_probe).shape == (1, 2)
prediction_probe.unlink()
atomic_json_save({"status": "passed", "at_utc": datetime.now(timezone.utc).isoformat()},
                 EXPERIMENT_DIR / "pretraining_sanity_checks.json")
print("All pre-training sanity checks passed. Full training may begin.")
"""),
    md(r"""## Training engine

Each experiment has its own directory, configuration fingerprint, physical
batch probe, compile decision, optimizer, scheduler, checkpoint, and history.
`last.pt`, `best.pt`, and history are written to Drive after every epoch. Only
validation Macro AP selects `best.pt`; test data never enter this loop.
"""),
    code(r"""# 11. Prediction and training loops with resumable Drive checkpoints.
LOSS_KEYS = [
    "loss", "primary_bce", "pair_bce", "triple_bce", "contradiction",
    "weighted_pair", "weighted_triple", "weighted_contradiction",
]

def move_batch(batch):
    for key in ("image", "primary", "pair", "triple"):
        batch[key] = batch[key].to(DEVICE, non_blocking=True)
    batch["image"] = batch["image"].contiguous(memory_format=torch.channels_last)
    return batch

@torch.inference_mode()
def predict_loader(active_model, loader, criterion=None):
    active_model.eval()
    primary_probability, primary_truth = [], []
    pair_probability, pair_truth, triple_probability, triple_truth = [], [], [], []
    metadata = {key: [] for key in (
        "record_id", "record", "patient_id", "source", "split", "true_exact_combination"
    )}
    sums = torch.zeros(len(LOSS_KEYS), device=DEVICE, dtype=torch.float64)
    samples = 0
    for batch in tqdm(loader, leave=False):
        batch = move_batch(batch)
        with autocast_context():
            outputs = active_model(batch["image"])
            if criterion is not None:
                loss, parts = criterion(outputs, batch)
        count = len(batch["image"])
        if criterion is not None:
            values = [loss.detach(), *[parts[key] for key in LOSS_KEYS[1:]]]
            sums += torch.stack(values).double() * count
        samples += count
        primary_probability.append(torch.sigmoid(outputs["primary_logits"]).float().cpu().numpy())
        primary_truth.append(batch["primary"].byte().cpu().numpy())
        if "pair_logits" in outputs:
            pair_probability.append(torch.sigmoid(outputs["pair_logits"]).float().cpu().numpy())
            pair_truth.append(batch["pair"].byte().cpu().numpy())
        if "triple_logits" in outputs:
            triple_probability.append(torch.sigmoid(outputs["triple_logits"]).float().cpu().numpy())
            triple_truth.append(batch["triple"].byte().cpu().numpy())
        for key in metadata:
            metadata[key].extend(list(batch[key]))
    averages = (sums / max(samples, 1)).cpu().numpy()
    result = {
        "primary_probability": np.concatenate(primary_probability),
        "primary_truth": np.concatenate(primary_truth),
        "metadata": metadata,
        **dict(zip(LOSS_KEYS, averages)),
    }
    if pair_probability:
        result["pair_probability"] = np.concatenate(pair_probability)
        result["pair_truth"] = np.concatenate(pair_truth)
    if triple_probability:
        result["triple_probability"] = np.concatenate(triple_probability)
        result["triple_truth"] = np.concatenate(triple_truth)
    return result

def train_one_epoch(active_model, loader, criterion, optimizer, scaler, scheduler,
                    accumulation_steps, grad_clip_norm):
    active_model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(len(LOSS_KEYS), device=DEVICE, dtype=torch.float64)
    samples = updates = skipped = 0
    started = time.perf_counter()
    loader_length = len(loader)
    group_start = 0
    group_size = min(accumulation_steps, loader_length)
    progress = tqdm(enumerate(loader), total=loader_length, leave=False)
    for batch_index, batch in progress:
        if batch_index == group_start:
            group_size = min(accumulation_steps, loader_length - group_start)
        batch = move_batch(batch)
        with autocast_context():
            outputs = active_model(batch["image"])
            loss, parts = criterion(outputs, batch)
            scaled_loss = loss / group_size
        old_scale = scaler.get_scale()
        scaler.scale(scaled_loss).backward()
        end_group = (batch_index - group_start + 1) == group_size
        if end_group:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(active_model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= old_scale:
                scheduler.step()
                updates += 1
            else:
                skipped += 1
            group_start = batch_index + 1
        count = len(batch["image"])
        values = [loss.detach(), *[parts[key] for key in LOSS_KEYS[1:]]]
        totals += torch.stack(values).double() * count
        samples += count
        progress.set_postfix(loss=f"{(totals[0] / samples).item():.4f}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        **dict(zip(LOSS_KEYS, (totals / samples).cpu().numpy())),
        "samples": samples, "optimizer_updates": updates, "amp_skipped_steps": skipped,
        "train_seconds": elapsed, "images_per_second": samples / elapsed,
    }

def scheduler_multiplier(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return max((step + 1) / max(warmup_steps, 1), 1e-8)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

def torch_save_atomic(payload, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
"""),
    code(r"""# 12. Train one experiment independently; save recovery state after every epoch.
def train_experiment(experiment):
    experiment_dir = RUN_DIR / experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(EXPERIMENT_CONFIGS[experiment])
    seed_everything(SEED)

    model = build_model(experiment, pretrained=True)
    pretrained_cfg = dict(getattr(model.backbone, "pretrained_cfg", {}) or {})
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    physical_batch, probe_table = probe_physical_batch(experiment, initial_state, config)
    probe_table.to_csv(experiment_dir / "batch_size_probe.csv", index=False)
    accumulation_steps = math.ceil(config["effective_batch_size"] / physical_batch)
    effective_batch = physical_batch * accumulation_steps
    loaders = make_loaders(physical_batch, SEED)
    steps_per_epoch = math.ceil(len(loaders["train"]) / accumulation_steps)

    model = model.to(DEVICE, memory_format=torch.channels_last)
    criterion = CombinationAwareLoss(config).to(DEVICE)
    active_model, compile_result = benchmark_compile(
        model, criterion, experiment_dir, physical_batch
    )
    optimizer, fused_adamw = build_optimizer(model, config)
    total_steps = config["max_epochs"] * steps_per_epoch
    warmup_steps = config["warmup_epochs"] * steps_per_epoch
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: scheduler_multiplier(step, warmup_steps, total_steps),
    )
    scaler = make_scaler()

    resolved_config = {
        "experiment": experiment, "model": MODEL_NAME, "pretraining": PRETRAINING,
        "pretrained_checkpoint": pretrained_cfg.get("hf_hub_id", MODEL_NAME),
        "labels": LABELS, "pair_labels": PAIR_LABELS, "triple_label": TRIPLE_LABEL,
        "dataset": DATASET_STEM, "dataset_archive_sha256": actual_archive_sha256,
        "dataset_index_sha256": dataset_audit["index_sha256"],
        "input_size_hw": list(INPUT_SIZE), "resize": "bicubic_antialias",
        "normalization_mean": list(IMAGENET_MEAN), "normalization_std": list(IMAGENET_STD),
        "online_augmentation": "none", "primary_pos_weight": PRIMARY_POS_WEIGHT.tolist(),
        "pair_pos_weight": PAIR_POS_WEIGHT.tolist(), "triple_pos_weight": TRIPLE_POS_WEIGHT.tolist(),
        "auxiliary_weighting": "sqrt(train negatives/train positives), cap=10",
        "contradiction_formula": "p(NORMAL) * (1 - product(1 - p(STEMI/AF/LVH)))",
        "checkpoint_monitor": "validation_macro_average_precision", "threshold_tuning": "per-label max validation F1",
        "physical_batch_size": physical_batch, "accumulation_steps": accumulation_steps,
        "effective_batch_size_actual": effective_batch, "steps_per_epoch": steps_per_epoch,
        "amp_dtype": str(AMP_DTYPE).removeprefix("torch."), "channels_last": True,
        "tf32": True, "fused_adamw": fused_adamw, "torch_compile": compile_result,
        "optimizer": "AdamW", "optimizer_parameter_groups": len(optimizer.param_groups),
        "schedule": "three-epoch linear warm-up then cosine decay",
        "classifier_initialization": "truncated normal std=0.02; zero bias",
        "head_weight_decay_strategy": "no decay for biases, 1D parameters, and label queries",
        "parameters": parameter_count, "trainable_parameters": trainable_count,
        "pytorch_version": torch.__version__, "timm_version": timm.__version__,
        "gpu": GPU.name, "gpu_vram_gib": GPU.total_memory / 2**30,
        "seed": SEED, "architecture_config": config, "master_reference": MASTER_REFERENCE,
    }
    # Compile availability and measured throughput are hardware/runtime
    # evidence, not optimizer identity; both may vary after a Colab reconnect.
    fingerprint_config = copy.deepcopy(resolved_config)
    fingerprint_config.pop("torch_compile", None)
    fingerprint_payload = json.dumps(fingerprint_config, sort_keys=True, default=str)
    fingerprint = hashlib.sha256(fingerprint_payload.encode()).hexdigest()[:16]
    resolved_config["fingerprint"] = fingerprint

    config_path = experiment_dir / "configuration.json"
    last_path = experiment_dir / "last.pt"
    best_path = experiment_dir / "best.pt"
    history_path = experiment_dir / "training_history.csv"
    completion_path = experiment_dir / "training_complete.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"{experiment_dir} contains a different configuration. Change CS_RUN_ID; nothing was overwritten."
            )
    else:
        atomic_json_save(resolved_config, config_path)

    start_epoch, best_macro_ap, bad_epochs, history = 1, -float("inf"), 0, []
    if last_path.is_file() and not completion_path.is_file():
        checkpoint = torch.load(last_path, map_location=DEVICE, weights_only=False)
        if checkpoint.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Checkpoint fingerprint mismatch in {last_path}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        loaders["generator"].set_state(checkpoint["train_generator_state"].cpu())
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_random_state_all"]])
        start_epoch = checkpoint["epoch"] + 1
        best_macro_ap = checkpoint["best_macro_ap"]
        bad_epochs = checkpoint["bad_epochs"]
        history = checkpoint["history"]
        print(experiment, "resuming at epoch", start_epoch)

    if completion_path.is_file():
        print(experiment, "already complete; skipping training")
        return resolved_config, active_model, model, criterion, loaders

    for epoch in range(start_epoch, config["max_epochs"] + 1):
        torch.cuda.reset_peak_memory_stats()
        epoch_started = time.perf_counter()
        train_values = train_one_epoch(
            active_model, loaders["train"], criterion, optimizer, scaler, scheduler,
            accumulation_steps, config["grad_clip_norm"],
        )
        validation = predict_loader(active_model, loaders["val"], criterion)
        validation_metrics = primary_metrics(
            validation["primary_truth"], validation["primary_probability"], [0.5] * len(LABELS)
        )
        validation_combo, _, _, _, _, _ = exact_combination_evaluation(
            validation["primary_truth"], validation["primary_probability"], [0.5] * len(LABELS)
        )
        macro_ap = validation_metrics["macro_ap"]
        improved = macro_ap > best_macro_ap
        if improved:
            best_macro_ap, bad_epochs = macro_ap, 0
            torch_save_atomic({
                "epoch": epoch, "model": model.state_dict(), "validation_macro_ap": macro_ap,
                "fingerprint": fingerprint,
            }, best_path)
        else:
            bad_epochs += 1
        epoch_seconds = time.perf_counter() - epoch_started
        combo_recall = validation_combo.set_index("combination")["recall"].to_dict()
        row = {
            "epoch": epoch, **{f"train_{key}": value for key, value in train_values.items() if key not in {"samples"}},
            **{f"val_{key}": float(validation[key]) for key in LOSS_KEYS},
            "val_macro_ap": macro_ap, "val_macro_auroc": validation_metrics["macro_auroc"],
            "val_macro_f1_fixed_0_5": validation_metrics["macro_f1"],
            "val_exact_match_fixed_0_5": validation_metrics["exact_match_accuracy"],
            "val_stemi_sensitivity_fixed_0_5": validation_metrics["per_label"]["STEMI"]["sensitivity"],
            "val_contradiction_rate_fixed_0_5": validation_metrics["contradiction_rate"],
            "val_recall_AF_LVH_fixed_0_5": combo_recall["AF+LVH"],
            "val_recall_STEMI_AF_fixed_0_5": combo_recall["STEMI+AF"],
            "val_recall_STEMI_LVH_fixed_0_5": combo_recall["STEMI+LVH"],
            "val_recall_STEMI_AF_LVH_fixed_0_5": combo_recall["STEMI+AF+LVH"],
            "learning_rate_min": min(group["lr"] for group in optimizer.param_groups),
            "learning_rate_max": max(group["lr"] for group in optimizer.param_groups),
            "epoch_seconds_total": epoch_seconds,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
            "best_checkpoint": int(improved), "non_improving_epochs": bad_epochs,
        }
        if "pair_probability" in validation:
            for i, label in enumerate(PAIR_LABELS):
                row[f"val_aux_ap_{label}"] = safe_rank(
                    average_precision_score, validation["pair_truth"][:, i], validation["pair_probability"][:, i]
                )
        if "triple_probability" in validation:
            row[f"val_aux_ap_{TRIPLE_LABEL}"] = safe_rank(
                average_precision_score, validation["triple_truth"][:, 0], validation["triple_probability"][:, 0]
            )
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        torch_save_atomic({
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "best_macro_ap": best_macro_ap, "bad_epochs": bad_epochs, "history": history,
            "train_generator_state": loaders["generator"].get_state(),
            "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(), "cuda_random_state_all": torch.cuda.get_rng_state_all(),
            "fingerprint": fingerprint,
        }, last_path)
        print(
            f"{experiment} epoch {epoch:02d}: val Macro AP={macro_ap:.5f}, "
            f"STEMI sens={validation_metrics['per_label']['STEMI']['sensitivity']:.3f}, "
            f"{train_values['images_per_second']:.1f} img/s, "
            + ("SAVED BEST" if improved else f"patience {bad_epochs}/{config['patience']}")
        )
        if bad_epochs >= config["patience"]:
            break
    atomic_json_save({
        "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_validation_macro_ap": best_macro_ap, "last_epoch": history[-1]["epoch"],
        "selection_metric": "validation_macro_average_precision",
    }, completion_path)
    return resolved_config, active_model, model, criterion, loaders
"""),
    md(r"""## Frozen-threshold test evaluation and complete exports

The best checkpoint is reloaded. Per-label thresholds are selected only from
validation predictions, frozen, and then applied to the test set. Both fixed
0.5 and validation-tuned results are saved. Exact-combination matrices contain
the eight valid states; invalid/no-label prediction patterns are saved
separately rather than silently projected onto valid states.
"""),
    code(r"""# 13. Evaluate one completed experiment and persist all required artifacts.
def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def markdown_table(frame, digits=4):
    return frame.to_markdown(index=False, floatfmt=f".{digits}f")

def prediction_frame(output, tuned_thresholds):
    frame = pd.DataFrame(output["metadata"])
    truth = output["primary_truth"]
    probability = output["primary_probability"]
    tuned_prediction = (probability >= np.asarray(tuned_thresholds)[None, :]).astype(np.int8)
    fixed_prediction = (probability >= 0.5).astype(np.int8)
    for i, label in enumerate(LABELS):
        frame[f"true_{label}"] = truth[:, i]
        frame[f"prob_{label}"] = probability[:, i]
        frame[f"threshold_{label}"] = tuned_thresholds[i]
        frame[f"pred_{label}"] = tuned_prediction[:, i]
        frame[f"pred_fixed_0_5_{label}"] = fixed_prediction[:, i]
    frame["true_exact_combination"] = names_from_bits(truth)
    frame["predicted_exact_combination"] = names_from_bits(tuned_prediction)
    frame["predicted_exact_combination_fixed_0_5"] = names_from_bits(fixed_prediction)
    if "pair_probability" in output:
        for i, label in enumerate(PAIR_LABELS):
            frame[f"true_aux_{label}"] = output["pair_truth"][:, i]
            frame[f"prob_aux_{label}"] = output["pair_probability"][:, i]
    if "triple_probability" in output:
        frame[f"true_aux_{TRIPLE_LABEL}"] = output["triple_truth"][:, 0]
        frame[f"prob_aux_{TRIPLE_LABEL}"] = output["triple_probability"][:, 0]
    return frame

def evaluate_and_export(experiment, resolved_config, active_model, model, criterion, loaders):
    experiment_dir = RUN_DIR / experiment
    best = torch.load(experiment_dir / "best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    validation = predict_loader(active_model, loaders["val"], criterion)
    tuned_thresholds, threshold_details = tune_thresholds_on_validation(
        validation["primary_truth"], validation["primary_probability"]
    )
    atomic_json_save(json_ready({
        "selection_split": "validation", "objective": "maximum per-label F1",
        "thresholds": dict(zip(LABELS, tuned_thresholds)), "validation_metrics": threshold_details,
    }), experiment_dir / "thresholds.json")

    # This is the first test-set inference in the experiment.
    test = predict_loader(active_model, loaders["test"], criterion)
    policies = {"fixed_0_5": [0.5] * len(LABELS), "validation_tuned": tuned_thresholds}
    primary_rows, label_rows, combination_rows, invalid_rows = [], [], [], []
    tuned_test_transitions = None
    tuned_test_transition_analysis = None
    evaluation_payload = {}
    for split, output in (("validation", validation), ("test", test)):
        evaluation_payload[split] = {}
        for policy, thresholds in policies.items():
            metrics = primary_metrics(output["primary_truth"], output["primary_probability"], thresholds)
            combinations, matrix, invalid, combo_summary, true_names, predicted_names = exact_combination_evaluation(
                output["primary_truth"], output["primary_probability"], thresholds
            )
            evaluation_payload[split][policy] = {
                "primary": metrics, "combination_summary": combo_summary,
            }
            primary_rows.append({
                "split": split, "threshold_policy": policy,
                **{key: value for key, value in metrics.items() if key != "per_label"},
                **combo_summary,
                "stemi_false_negatives": metrics["per_label"]["STEMI"]["fn"],
            })
            for label in LABELS:
                label_rows.append({"split": split, "threshold_policy": policy, "label": label,
                                   **metrics["per_label"][label]})
            combination_rows.extend({
                "split": split, "threshold_policy": policy, **row
            } for row in combinations.to_dict("records"))
            invalid_rows.extend({
                "split": split, "threshold_policy": policy, **row
            } for row in invalid)
            if split == "test":
                suffix = "" if policy == "validation_tuned" else "_fixed_0_5"
                matrix_frame = pd.DataFrame(matrix, index=COMBINATION_ORDER, columns=COMBINATION_ORDER)
                matrix_frame.to_csv(experiment_dir / f"exact_combination_confusion_matrix{suffix}.csv")
                save_confusion_plot(
                    matrix, experiment_dir / f"exact_combination_confusion_matrix{suffix}.png",
                    f"{experiment}: test exact combinations ({policy})",
                )
                if policy == "validation_tuned":
                    transition_frame = pd.DataFrame({"true": true_names, "predicted": predicted_names})
                    tuned_test_transitions = (
                        transition_frame.groupby(["true", "predicted"]).size().rename("count").reset_index()
                    )
                    tuned_test_transitions["true_support"] = tuned_test_transitions["true"].map(
                        transition_frame["true"].value_counts()
                    )
                    tuned_test_transitions["rate_within_true_state"] = (
                        tuned_test_transitions["count"] / tuned_test_transitions["true_support"]
                    )
                    tuned_test_transitions.to_csv(
                        experiment_dir / "exact_combination_transitions.csv", index=False
                    )
                    targeted_pairs = [
                        ("STEMI+AF", "STEMI"), ("STEMI+AF", "AF"),
                        ("STEMI+LVH", "STEMI"), ("STEMI+LVH", "LVH"),
                        ("AF+LVH", "AF"), ("AF+LVH", "LVH"),
                        ("STEMI+AF+LVH", "STEMI+AF"),
                        ("STEMI+AF+LVH", "STEMI+LVH"),
                        ("STEMI+AF+LVH", "AF+LVH"),
                    ]
                    analysis_rows = []
                    true_support_counts = transition_frame["true"].value_counts()
                    for true_state, predicted_state in targeted_pairs:
                        count = int(((true_names == true_state) & (predicted_names == predicted_state)).sum())
                        support = int(true_support_counts.get(true_state, 0))
                        analysis_rows.append({
                            "transition": f"{true_state} -> {predicted_state}", "count": count,
                            "source_state_support": support, "rate_within_source_state": divide(count, support),
                        })
                    pathology_true = true_names != "NORMAL"
                    pathology_to_normal = int((pathology_true & (predicted_names == "NORMAL")).sum())
                    normal_true = true_names == "NORMAL"
                    normal_to_pathology = int((normal_true & (predicted_names != "NORMAL") & (predicted_names != "NO_LABEL")).sum())
                    analysis_rows.extend([
                        {"transition": "any pathology -> NORMAL", "count": pathology_to_normal,
                         "source_state_support": int(pathology_true.sum()),
                         "rate_within_source_state": divide(pathology_to_normal, int(pathology_true.sum()))},
                        {"transition": "NORMAL -> any pathology/contradiction", "count": normal_to_pathology,
                         "source_state_support": int(normal_true.sum()),
                         "rate_within_source_state": divide(normal_to_pathology, int(normal_true.sum()))},
                    ])
                    tuned_test_transition_analysis = pd.DataFrame(analysis_rows)
                    tuned_test_transition_analysis.to_csv(
                        experiment_dir / "transition_analysis.csv", index=False
                    )

    pd.DataFrame(primary_rows).to_csv(experiment_dir / "primary_metrics.csv", index=False)
    pd.DataFrame(label_rows).to_csv(experiment_dir / "per_label_metrics.csv", index=False)
    pd.DataFrame(combination_rows).to_csv(experiment_dir / "combination_metrics.csv", index=False)
    pd.DataFrame(invalid_rows, columns=["split", "threshold_policy", "predicted_combination", "count"]).to_csv(
        experiment_dir / "invalid_prediction_patterns.csv", index=False
    )
    prediction_frame(validation, tuned_thresholds).to_csv(experiment_dir / "validation_predictions.csv", index=False)
    prediction_frame(test, tuned_thresholds).to_csv(experiment_dir / "predictions.csv", index=False)

    aux_rows = []
    if "pair_probability" in validation:
        train_output = predict_loader(active_model, loaders["train_eval"], criterion)
        for split, output in (("train", train_output), ("validation", validation), ("test", test)):
            for i, label in enumerate(PAIR_LABELS):
                aux_rows.append({"split": split, "auxiliary_label": label,
                                 **auxiliary_metrics(output["pair_truth"][:, i], output["pair_probability"][:, i])})
            if "triple_probability" in output:
                aux_rows.append({"split": split, "auxiliary_label": TRIPLE_LABEL,
                                 **auxiliary_metrics(output["triple_truth"][:, 0], output["triple_probability"][:, 0])})
        auxiliary_frame = pd.DataFrame(aux_rows)
        if (auxiliary_frame.auxiliary_label == TRIPLE_LABEL).any():
            triple = auxiliary_frame[auxiliary_frame.auxiliary_label.eq(TRIPLE_LABEL)].set_index("split")
            train_ap, val_ap = triple.loc["train", "ap"], triple.loc["validation", "ap"]
            gap = train_ap - val_ap
            auxiliary_frame["triple_train_minus_validation_ap_gap"] = gap
            likely_memorization = bool(train_ap >= 0.90 and gap >= 0.15)
            auxiliary_frame["triple_likely_memorization"] = likely_memorization
        auxiliary_frame.to_csv(experiment_dir / "auxiliary_metrics.csv", index=False)

    history = pd.read_csv(experiment_dir / "training_history.csv")
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()
    axes[0].plot(history.epoch, history.train_loss, label="train")
    axes[0].plot(history.epoch, history.val_loss, label="validation")
    axes[0].set_title("Total loss"); axes[0].legend()
    axes[1].plot(history.epoch, history.val_macro_ap)
    axes[1].axvline(best["epoch"], color="tab:red", linestyle="--")
    axes[1].set_title("Validation Macro AP")
    axes[2].plot(history.epoch, history.val_stemi_sensitivity_fixed_0_5)
    axes[2].set_title("Validation STEMI sensitivity @0.5")
    for label in ("AF_LVH", "STEMI_AF", "STEMI_LVH", "STEMI_AF_LVH"):
        column = f"val_recall_{label}_fixed_0_5"
        if column in history:
            axes[3].plot(history.epoch, history[column], label=label)
    axes[3].set_title("Validation exact-combination recall @0.5"); axes[3].legend()
    for axis in axes:
        axis.set_xlabel("Epoch"); axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(experiment_dir / "learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    tuned_test = evaluation_payload["test"]["validation_tuned"]["primary"]
    tuned_combo_summary = evaluation_payload["test"]["validation_tuned"]["combination_summary"]
    tuned_combo = pd.DataFrame(combination_rows)
    tuned_combo = tuned_combo[
        tuned_combo["split"].eq("test") & tuned_combo["threshold_policy"].eq("validation_tuned")
    ].set_index("combination")
    tuned_labels = pd.DataFrame(label_rows)
    tuned_labels = tuned_labels[
        tuned_labels["split"].eq("test") & tuned_labels["threshold_policy"].eq("validation_tuned")
    ].set_index("label")
    fixed_test = evaluation_payload["test"]["fixed_0_5"]["primary"]
    summary = {
        "experiment": experiment,
        "parameters": resolved_config["parameters"],
        "trainable_parameters": resolved_config["trainable_parameters"],
        "best_epoch": int(best["epoch"]),
        "best_validation_macro_ap": float(best["validation_macro_ap"]),
        "training_time_seconds": float(history.epoch_seconds_total.sum()),
        "peak_vram_gib": float(history.peak_vram_gib.max()),
        "throughput_images_per_second": float(history.train_images_per_second.mean()),
        "physical_batch_size": resolved_config["physical_batch_size"],
        "effective_batch_size": resolved_config["effective_batch_size_actual"],
        "backbone_lr": resolved_config["architecture_config"]["backbone_lr"],
        "head_lr": resolved_config["architecture_config"]["head_lr"],
        "weight_decay": resolved_config["architecture_config"]["weight_decay"],
        "pair_lambda": resolved_config["architecture_config"]["pair_lambda"],
        "triple_lambda": resolved_config["architecture_config"]["triple_lambda"],
        "contradiction_lambda": resolved_config["architecture_config"]["contradiction_lambda"],
        "attention_configuration": {
            key: resolved_config["architecture_config"].get(key)
            for key in ("attention_dim", "attention_heads", "attention_depth", "attention_dropout")
            if key in resolved_config["architecture_config"]
        } or None,
        "macro_ap": tuned_test["macro_ap"], "macro_auroc": tuned_test["macro_auroc"],
        "macro_sensitivity": tuned_test["macro_sensitivity"],
        "macro_specificity": tuned_test["macro_specificity"],
        "macro_precision": tuned_test["macro_precision"], "macro_f1": tuned_test["macro_f1"],
        "micro_f1": tuned_test["micro_f1"], "hamming_accuracy": tuned_test["hamming_accuracy"],
        "hamming_loss": tuned_test["hamming_loss"], "exact_match": tuned_test["exact_match_accuracy"],
        "contradiction_count_fixed_0_5": fixed_test["contradiction_count"],
        "contradiction_rate_fixed_0_5": fixed_test["contradiction_rate"],
        "contradiction_count": tuned_test["contradiction_count"],
        "contradiction_rate": tuned_test["contradiction_rate"],
        "no_label_prediction_count": tuned_test["no_label_prediction_count"],
        "stemi_f1": tuned_labels.loc["STEMI", "f1"],
        "stemi_sensitivity": tuned_labels.loc["STEMI", "sensitivity"],
        "stemi_false_negatives_fixed_0_5": fixed_test["per_label"]["STEMI"]["fn"],
        "stemi_false_negatives": tuned_test["per_label"]["STEMI"]["fn"],
        "af_f1": tuned_labels.loc["AF", "f1"], "lvh_f1": tuned_labels.loc["LVH", "f1"],
        "af_lvh_f1": tuned_combo.loc["AF+LVH", "f1"],
        "stemi_af_f1": tuned_combo.loc["STEMI+AF", "f1"],
        "stemi_lvh_f1": tuned_combo.loc["STEMI+LVH", "f1"],
        "triple_support": int(tuned_combo.loc["STEMI+AF+LVH", "support"]),
        "triple_recall": tuned_combo.loc["STEMI+AF+LVH", "recall"],
        "triple_f1": tuned_combo.loc["STEMI+AF+LVH", "f1"],
        **tuned_combo_summary,
    }
    atomic_json_save(json_ready(summary), experiment_dir / "summary.json")
    atomic_json_save(json_ready({
        "config": resolved_config, "best_checkpoint": {"epoch": best["epoch"], "validation_macro_ap": best["validation_macro_ap"]},
        "thresholds": dict(zip(LABELS, tuned_thresholds)), "evaluations": evaluation_payload,
        "auxiliary_metrics": aux_rows,
    }), experiment_dir / "result.json")

    label_report = pd.DataFrame(label_rows)
    label_report = label_report[label_report["split"].eq("test") & label_report["threshold_policy"].eq("validation_tuned")]
    combo_report = pd.DataFrame(combination_rows)
    combo_report = combo_report[combo_report["split"].eq("test") & combo_report["threshold_policy"].eq("validation_tuned")]
    error_transitions = tuned_test_transitions[
        tuned_test_transitions["true"] != tuned_test_transitions["predicted"]
    ].sort_values("count", ascending=False).head(20)
    report = [
        f"# {experiment} final report", "",
        f"Best epoch: {best['epoch']} (validation Macro AP {best['validation_macro_ap']:.5f}).", "",
        "## Test primary metrics", "", markdown_table(pd.DataFrame(primary_rows).query("split == 'test'")), "",
        "## Test per-label metrics — validation-tuned thresholds", "", markdown_table(label_report), "",
        "## Test exact-combination metrics — validation-tuned thresholds", "", markdown_table(combo_report), "",
        "The triple test row is undefined because its support is 0; it is not converted to a zero score.", "",
        "## Most frequent exact-state errors", "", markdown_table(error_transitions), "",
        "## Pre-specified transition analysis", "", markdown_table(tuned_test_transition_analysis), "",
        "## Configuration", "", "```json", json.dumps(json_ready(resolved_config), indent=2), "```", "",
    ]
    if aux_rows:
        report += ["## Auxiliary-head diagnostics", "", markdown_table(pd.DataFrame(aux_rows)), ""]
    (experiment_dir / "final_report.md").write_text("\n".join(report), encoding="utf-8")
    return summary
"""),
    code(r"""# 14. Required A–D run. Each completed experiment is durable before the next begins.
EXPERIMENT_SUMMARIES = {}
for experiment in REQUIRED_EXPERIMENTS:
    experiment_dir = RUN_DIR / experiment
    summary_path = experiment_dir / "summary.json"
    if summary_path.is_file():
        print(experiment, "final evaluation already exists; loading it")
        EXPERIMENT_SUMMARIES[experiment] = json.loads(summary_path.read_text(encoding="utf-8"))
        continue
    resolved_config, active_model, base_model, criterion, loaders = train_experiment(experiment)
    EXPERIMENT_SUMMARIES[experiment] = evaluate_and_export(
        experiment, resolved_config, active_model, base_model, criterion, loaders
    )
    del active_model, base_model, criterion, loaders
    torch.cuda.empty_cache()
    print(experiment, "complete and persisted to", experiment_dir)
"""),
    md(r"""## Master comparison, paired uncertainty, and automated interpretation

The comparison uses the same test records, primary labels, evaluator, and
frozen validation-tuned thresholds. A paired record bootstrap estimates 95%
intervals for Macro AP, exact match, and supported co-occurrence macro F1
deltas versus the baseline. This is an uncertainty diagnostic, not a substitute
for multi-seed training. Triple test conclusions remain impossible with zero
support.
"""),
    code(r"""# 15. Combined tables, plots, paired bootstrap, and the 15-question interpretation.
comparison_columns = [
    "experiment", "macro_ap", "macro_auroc", "macro_sensitivity", "macro_specificity",
    "macro_precision", "macro_f1", "micro_f1", "hamming_accuracy", "hamming_loss",
    "exact_match", "contradiction_count_fixed_0_5", "contradiction_rate_fixed_0_5",
    "contradiction_count", "contradiction_rate", "no_label_prediction_count",
    "stemi_f1", "stemi_sensitivity", "stemi_false_negatives_fixed_0_5", "stemi_false_negatives",
    "af_f1", "lvh_f1", "af_lvh_f1", "stemi_af_f1", "stemi_lvh_f1",
    "triple_support", "triple_recall", "triple_f1", "exact_combo_macro_f1_supported",
    "cooccurrence_macro_f1_supported", "single_state_macro_f1_supported",
    "parameters", "trainable_parameters", "best_epoch", "best_validation_macro_ap",
    "training_time_seconds", "peak_vram_gib", "throughput_images_per_second",
    "physical_batch_size", "effective_batch_size", "backbone_lr", "head_lr", "weight_decay",
    "pair_lambda", "triple_lambda", "contradiction_lambda", "attention_configuration",
]
comparison = pd.DataFrame([EXPERIMENT_SUMMARIES[name] for name in REQUIRED_EXPERIMENTS])
comparison = comparison.reindex(columns=comparison_columns)
comparison.to_csv(RUN_DIR / "comparison.csv", index=False)
display(comparison)

prediction_tables = {
    name: pd.read_csv(RUN_DIR / name / "predictions.csv") for name in REQUIRED_EXPERIMENTS
}
reference = prediction_tables["baseline"]
truth_columns = [f"true_{label}" for label in LABELS]
for name, frame in prediction_tables.items():
    if not frame.record_id.equals(reference.record_id):
        raise RuntimeError(f"Test record order differs for {name}")
    if not np.array_equal(frame[truth_columns].to_numpy(), reference[truth_columns].to_numpy()):
        raise RuntimeError(f"Test targets differ for {name}")

def supported_cooccurrence_macro_f1(truth_names, predicted_names):
    scores = []
    for label in ("AF+LVH", "STEMI+AF", "STEMI+LVH", "STEMI+AF+LVH"):
        true_mask = truth_names == label
        if not true_mask.any():
            continue
        predicted_mask = predicted_names == label
        correct = int((true_mask & predicted_mask).sum())
        scores.append(divide(2 * correct, int(true_mask.sum()) + int(predicted_mask.sum())))
    return float(np.nanmean(scores))

truth = reference[truth_columns].to_numpy(np.int8)
truth_names = reference["true_exact_combination"].to_numpy(str)
rng = np.random.default_rng(SEED + 991)
bootstrap_rows = []
for name in REQUIRED_EXPERIMENTS[1:]:
    deltas = {"macro_ap": [], "exact_match": [], "cooccurrence_macro_f1": []}
    for _ in tqdm(range(BOOTSTRAP_REPETITIONS), desc=f"paired bootstrap {name}", leave=False):
        indices = rng.integers(0, len(reference), size=len(reference))
        for candidate, sign in ((name, 1), ("baseline", -1)):
            frame = prediction_tables[candidate]
            probability = frame[[f"prob_{label}" for label in LABELS]].to_numpy()[indices]
            predicted = frame[[f"pred_{label}" for label in LABELS]].to_numpy(np.int8)[indices]
            metrics = primary_metrics(truth[indices], probability,
                                      frame[[f"threshold_{label}" for label in LABELS]].iloc[0].to_numpy())
            if candidate == name:
                candidate_values = (
                    metrics["macro_ap"],
                    float((predicted == truth[indices]).all(axis=1).mean()),
                    supported_cooccurrence_macro_f1(truth_names[indices], names_from_bits(predicted)),
                )
            else:
                baseline_values = (
                    metrics["macro_ap"],
                    float((predicted == truth[indices]).all(axis=1).mean()),
                    supported_cooccurrence_macro_f1(truth_names[indices], names_from_bits(predicted)),
                )
        for metric, candidate_value, baseline_value in zip(deltas, candidate_values, baseline_values):
            deltas[metric].append(candidate_value - baseline_value)
    for metric, values in deltas.items():
        low, high = np.nanpercentile(values, [2.5, 97.5])
        bootstrap_rows.append({
            "experiment": name, "reference": "baseline", "metric": metric,
            "mean_delta": float(np.nanmean(values)), "ci_2_5": float(low), "ci_97_5": float(high),
            "ci_excludes_zero": bool(low > 0 or high < 0), "repetitions": BOOTSTRAP_REPETITIONS,
        })
bootstrap_frame = pd.DataFrame(bootstrap_rows)
bootstrap_frame.to_csv(RUN_DIR / "paired_bootstrap_comparisons.csv", index=False)
display(bootstrap_frame)

figure, axes = plt.subplots(1, 2, figsize=(16, 6))
metric_plot = comparison.set_index("experiment")[["macro_ap", "macro_auroc", "macro_f1", "exact_match"]]
metric_plot.plot(kind="bar", ax=axes[0], ylim=(0, 1), title="Primary test comparison")
combo_plot = comparison.set_index("experiment")[["af_lvh_f1", "stemi_af_f1", "stemi_lvh_f1"]]
combo_plot.plot(kind="bar", ax=axes[1], ylim=(0, 1), title="Supported co-occurrence exact F1")
for axis in axes:
    axis.set_xlabel(""); axis.tick_params(axis="x", rotation=25); axis.grid(axis="y", alpha=0.25)
figure.tight_layout()
figure.savefig(RUN_DIR / "combined_model_comparison.png", dpi=180, bbox_inches="tight")
plt.close(figure)

metric_rows = [
    ("Macro AP", "macro_ap"), ("Macro AUROC", "macro_auroc"), ("Macro F1", "macro_f1"),
    ("Micro F1", "micro_f1"), ("Exact Match", "exact_match"),
    ("Contradiction Rate", "contradiction_rate"), ("STEMI F1", "stemi_f1"),
    ("STEMI Sensitivity", "stemi_sensitivity"), ("AF F1", "af_f1"), ("LVH F1", "lvh_f1"),
    ("AF+LVH F1", "af_lvh_f1"), ("STEMI+AF F1", "stemi_af_f1"),
    ("STEMI+LVH F1", "stemi_lvh_f1"), ("Triple Recall", "triple_recall"),
    ("Triple F1", "triple_f1"),
]
pretty_names = {"baseline": "Baseline", "pair_aux": "Pair Aux",
                "pair_triple_aux": "Pair+Triple Aux", "label_dependency": "Label Dependency"}
metric_matrix = pd.DataFrame({
    "Metric": [label for label, _ in metric_rows],
    **{pretty_names[name]: [comparison.set_index("experiment").loc[name, key] for _, key in metric_rows]
       for name in REQUIRED_EXPERIMENTS},
})
training_keys = [
    ("Parameters", "parameters"), ("Trainable parameters", "trainable_parameters"),
    ("Best epoch", "best_epoch"), ("Best validation Macro AP", "best_validation_macro_ap"),
    ("Training time (s)", "training_time_seconds"), ("Peak VRAM (GiB)", "peak_vram_gib"),
    ("Throughput (img/s)", "throughput_images_per_second"),
    ("Batch size", "physical_batch_size"), ("Effective batch", "effective_batch_size"),
    ("Backbone LR", "backbone_lr"), ("Head LR", "head_lr"), ("Weight decay", "weight_decay"),
    ("Pair lambda", "pair_lambda"), ("Triple lambda", "triple_lambda"),
    ("Contradiction lambda", "contradiction_lambda"), ("Attention configuration", "attention_configuration"),
]
training_matrix = pd.DataFrame({
    "Training property": [label for label, _ in training_keys],
    **{pretty_names[name]: [comparison.set_index("experiment").loc[name, key] for _, key in training_keys]
       for name in REQUIRED_EXPERIMENTS},
})

def value(name, key):
    return comparison.set_index("experiment").loc[name, key]

def delta(name, key, reference_name="baseline"):
    return value(name, key) - value(reference_name, key)

best_macro = comparison.loc[comparison.macro_ap.idxmax(), "experiment"]
best_combo = comparison.loc[comparison.cooccurrence_macro_f1_supported.idxmax(), "experiment"]
best_stemi = comparison.loc[comparison.stemi_sensitivity.idxmax(), "experiment"]
worst_contradiction = comparison.loc[comparison.contradiction_rate.idxmax(), "experiment"]
rank_metrics = comparison.set_index("experiment")[[
    "macro_ap", "macro_f1", "exact_match", "stemi_sensitivity", "cooccurrence_macro_f1_supported"
]].rank(ascending=False, method="average")
rank_metrics["contradiction_rate"] = comparison.set_index("experiment")["contradiction_rate"].rank(
    ascending=True, method="average"
)
balance = rank_metrics.mean(axis=1)
best_balance = balance.idxmin()

baseline_predictions = prediction_tables["baseline"]
baseline_errors = baseline_predictions[
    baseline_predictions.true_exact_combination != baseline_predictions.predicted_exact_combination
]
baseline_transitions = baseline_errors.groupby(
    ["true_exact_combination", "predicted_exact_combination"]
).size().sort_values(ascending=False)
transition_text = "; ".join(
    f"{true} → {predicted}: {count}"
    for (true, predicted), count in baseline_transitions.head(8).items()
)

triple_aux_path = RUN_DIR / "pair_triple_aux" / "auxiliary_metrics.csv"
triple_diagnostic = "not available"
triple_memorization_flag = None
if triple_aux_path.is_file():
    aux = pd.read_csv(triple_aux_path)
    triple = aux[aux.auxiliary_label.eq(TRIPLE_LABEL)].set_index("split")
    gap = triple.loc["train", "ap"] - triple.loc["validation", "ap"]
    triple_diagnostic = (
        f"train AP {triple.loc['train', 'ap']:.4f} versus validation AP "
        f"{triple.loc['validation', 'ap']:.4f} (gap {gap:+.4f}; validation support "
        f"{int(triple.loc['validation', 'support'])})"
    )
    flag_values = aux.loc[aux.auxiliary_label.eq(TRIPLE_LABEL), "triple_likely_memorization"].dropna()
    if len(flag_values):
        triple_memorization_flag = bool(flag_values.iloc[0])

single_tradeoffs = "; ".join(
    f"{pretty_names[name]}: co-occurrence F1 {delta(name, 'cooccurrence_macro_f1_supported'):+.4f}, "
    f"single-state F1 {delta(name, 'single_state_macro_f1_supported'):+.4f}"
    for name in REQUIRED_EXPERIMENTS[1:]
)
selection_warning = ""
if best_macro != "baseline":
    warnings_for_winner = []
    if delta(best_macro, "stemi_sensitivity") <= -0.10:
        warnings_for_winner.append("STEMI sensitivity fell by at least 0.10")
    if delta(best_macro, "cooccurrence_macro_f1_supported") <= -0.10:
        warnings_for_winner.append("supported co-occurrence macro F1 fell by at least 0.10")
    if warnings_for_winner:
        selection_warning = " Caution: " + "; ".join(warnings_for_winner) + "."

bootstrap_by_model = bootstrap_frame.groupby("experiment")
uncertain_models = []
for name in REQUIRED_EXPERIMENTS[1:]:
    row = bootstrap_frame[(bootstrap_frame.experiment == name) & (bootstrap_frame.metric == "macro_ap")].iloc[0]
    if not row.ci_excludes_zero:
        uncertain_models.append(name)

interpretation = [
    "# Evidence-based interpretation", "",
    "1. **Baseline strength.** The ordinary baseline achieved "
    f"test Macro AP {value('baseline', 'macro_ap'):.4f}, Macro F1 {value('baseline', 'macro_f1'):.4f}, "
    f"exact match {value('baseline', 'exact_match'):.4f}, and supported co-occurrence macro F1 "
    f"{value('baseline', 'cooccurrence_macro_f1_supported'):.4f}. Its leading error transitions were: {transition_text}.", "",
    "2. **Pair supervision and AF+LVH.** Pair Aux changed AF+LVH exact F1 by "
    f"{delta('pair_aux', 'af_lvh_f1'):+.4f} versus baseline.", "",
    "3. **Pair supervision and STEMI+AF.** Pair Aux changed STEMI+AF exact F1 by "
    f"{delta('pair_aux', 'stemi_af_f1'):+.4f}; raw counts are retained in each combination_metrics.csv.", "",
    "4. **Pair supervision and STEMI+LVH.** Pair Aux changed STEMI+LVH exact F1 by "
    f"{delta('pair_aux', 'stemi_lvh_f1'):+.4f}.", "",
    "5. **Triple objective.** The held-out test has 0 triple-positive ECGs, so whether the triple head improves "
    "triple recognition beyond Pair Aux is not testable. Exploratory head behavior is: " + triple_diagnostic + ".", "",
    "6. **Triple overfit.** The pre-specified memorization flag is "
    + ("not evaluable" if triple_memorization_flag is None else ("ON" if triple_memorization_flag else "off"))
    + ". Treat the train-validation AP gap together with only five validation positives; no test claim is made.", "",
    "7. **Explicit dependency versus independent heads.** Label Dependency changed Macro AP by "
    f"{delta('label_dependency', 'macro_ap'):+.4f}, exact match by {delta('label_dependency', 'exact_match'):+.4f}, "
    f"and supported co-occurrence macro F1 by {delta('label_dependency', 'cooccurrence_macro_f1_supported'):+.4f} versus baseline.", "",
    f"8. **Highest test Macro AP.** {pretty_names[best_macro]} ({value(best_macro, 'macro_ap'):.4f})."
    + selection_warning, "",
    f"9. **Strongest supported exact-combination performance.** {pretty_names[best_combo]} "
    f"(supported co-occurrence macro F1 {value(best_combo, 'cooccurrence_macro_f1_supported'):.4f}).", "",
    f"10. **Best STEMI sensitivity.** {pretty_names[best_stemi]} ({value(best_stemi, 'stemi_sensitivity'):.4f}); "
    "fixed-0.5 and tuned false-negative counts are both in comparison.csv.", "",
    "11. **Combination gains versus single states.** Deltas versus baseline were " + single_tradeoffs
    + ". A co-occurrence gain accompanied by a single-state decline is a trade-off, not a free gain.", "",
    "12. **NORMAL contradictions.** Validation-tuned contradiction rates were " + ", ".join(
        f"{pretty_names[name]} {value(name, 'contradiction_rate'):.4f}" for name in REQUIRED_EXPERIMENTS
    ) + f". {pretty_names[worst_contradiction]} had the highest tuned contradiction rate. "
    "No predictions were silently repaired.", "",
    f"13. **Best overall balance.** Under the explicitly reported equal-rank summary of Macro AP, Macro F1, exact match, "
    f"STEMI sensitivity, supported co-occurrence F1, and low contradiction rate, {pretty_names[best_balance]} ranked best. "
    "This is interpretive; Macro AP remains the formal checkpoint criterion.", "",
    "14. **Magnitude and confidence.** Paired bootstrap intervals are in paired_bootstrap_comparisons.csv. "
    + ("Macro-AP differences whose intervals include zero: " + ", ".join(pretty_names[name] for name in uncertain_models) + ". " if uncertain_models else "All non-baseline Macro-AP intervals excluded zero. ")
    + "These intervals capture test-record sampling only, not training-seed uncertainty; small differences require multi-seed replication.", "",
    "15. **Next test on the 120k master set.** Replicate the leading simple and combination-aware candidates across multiple seeds; "
    "ensure meaningful triple-positive validation/test support without changing existing master identities; retain Macro-AP checkpointing, "
    "report source/patient subgroups, calibrate frozen thresholds, and perform external validation. Do not scale the triple objective until "
    "its generalization is supported beyond five validation examples.", "",
]
(RUN_DIR / "interpretation.md").write_text("\n".join(interpretation), encoding="utf-8")

comparison_markdown = [
    "# CardioSentry ConvNeXt-V2 multi-label comparison", "",
    "All values use the same held-out test set; thresholds are tuned on validation only unless named fixed 0.5.", "",
    "## Model metrics", "", markdown_table(metric_matrix), "",
    "## Training properties", "", markdown_table(training_matrix), "",
    "## Paired bootstrap versus baseline", "", markdown_table(bootstrap_frame), "",
    *interpretation,
]
(RUN_DIR / "comparison.md").write_text("\n".join(comparison_markdown), encoding="utf-8")
print((RUN_DIR / "interpretation.md").read_text(encoding="utf-8"))

environment = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
    "platform": platform.platform(), "pytorch": torch.__version__, "timm": timm.__version__,
    "sklearn": sklearn.__version__, "pandas": pd.__version__, "cuda": torch.version.cuda,
    "gpu": GPU.name, "gpu_vram_gib": GPU.total_memory / 2**30, "amp_dtype": str(AMP_DTYPE),
    "cpu_count": os.cpu_count(), "ram_gib": psutil.virtual_memory().total / 2**30,
    "model_identifier": MODEL_NAME, "pretraining": PRETRAINING, "input_size_hw": list(INPUT_SIZE),
}
atomic_json_save(environment, RUN_DIR / "environment.json")
atomic_json_save({
    "status": "complete", "required_experiments": REQUIRED_EXPERIMENTS,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "selection_metric": "validation_macro_average_precision",
    "triple_test_support": 0,
}, RUN_DIR / "STUDY_COMPLETE.json")
print("Study complete:", RUN_DIR)
"""),
    md(r"""## Durable outputs

Each experiment directory contains:

- `configuration.json`, `best.pt`, `last.pt`, `training_history.csv`;
- `thresholds.json`, `primary_metrics.csv`, `per_label_metrics.csv`;
- `combination_metrics.csv`, `predictions.csv`, validation predictions;
- exact-combination confusion-matrix CSVs/images for fixed 0.5 and tuned thresholds;
- auxiliary metrics where applicable, learning curves, `result.json`, and
  `final_report.md`.

The study root contains `comparison.csv`, `comparison.md`, combined plots,
paired bootstrap intervals, `dataset_audit.json/csv`, `environment.json`,
`interpretation.md`, and `STUDY_COMPLETE.json`. Results are saved after every
experiment; optional Experiment E is intentionally not run before or mixed into
the required A–D comparison.
"""),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"wrote {OUTPUT} ({len(cells)} cells)")
