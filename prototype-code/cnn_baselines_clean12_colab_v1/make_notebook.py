"""Build the self-contained CardioSentry CNN baselines (ResNet-50, DenseNet-121, EfficientNet-B3) clean12 Google Colab (A100) notebook."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "CardioSentry_CNN_Baselines_Clean12_Colab.ipynb"


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
    md(r"""# CardioSentry — CNN baselines on the HEEDB clean12 ECG images (Google Colab, A100)

This notebook fine-tunes **three classic CNNs one after another** on the clean12
ECG image corpus, in one Colab session:

1. **ResNet-50** (`resnet50.a1_in1k`)
2. **DenseNet-121** (`densenet121.ra_in1k`)
3. **EfficientNet-B3** (`efficientnet_b3.ra2_in1k`)

The slow shared work runs **once** for all three: installing packages, mounting
Drive, streaming and verifying the ~104.5 GiB archive from Google Drive, auditing the
manifests, rebuilding the patient split, and tuning the PNG decoder workers.
Then each architecture gets its own batch probe, eager-vs-compiled benchmark,
training run, final evaluation and report. A comparison table, with paired
patient-bootstrap intervals, is written at the end.

`torch.compile` cannot be shared between architectures: every network is a
different graph, so each one compiles once (in its benchmark; the real model
then mostly hits Inductor's local cache). This notebook uses the `default`
compile mode, which skips the long kernel autotuning of `max-autotune`.

The corpus and the evaluation are those of the ConvNeXt clean12 runs
(`clean_convnext_codes/`): 113,358 physician-verified HEEDB records (native 500
Hz, 10 s, 12 leads) rendered by the real ECG-image-kit as 1024×768 PNG sheets,
the same **patient-level split** (SHA-256 `4eba80c7…`), the same 12 labels,
the same checkpoint rule and the same test metrics. So the CNN numbers sit
directly next to the ConvNeXt ones.

## Goal: raw performance, one plain recipe for all three

The ConvNeXt notebooks were used as a reference for the plumbing only. Their
hyperparameters (layer decay, head-init scale, drop-path, the V1/V2 LRs) are
**not** copied. Every CNN trains with the same plain fine-tuning recipe below,
untuned per architecture, so the comparison shows what each backbone does out of
the box.

| setting | choice (identical for the three CNNs) |
|---|---|
| task | multi-label; twelve independent logits in `LABELS` order (NORM + SB, STACH, LBBB, LQT, AF, AFL, 3AVB, 2AVB, 1AVB, STEMI, STE) |
| pretrained weights | timm's default ImageNet-1K tag of each model (above); a new 12-way linear head |
| input | 768×1024 RGB (H×W), the full native sheet, bicubic resize (a no-op on 1024×768 sheets), each model's own normalization |
| sampling | uniform epochs (no rare-label resampling) |
| loss | per-label `BCEWithLogitsLoss(pos_weight=sqrt(negatives / positives))`, training counts |
| optimizer | AdamW, one LR for all layers (`3e-4`, no layer decay), weight decay `0.01` on conv/linear weights only (none on BatchNorm or biases), gradient clip `1.0` |
| schedule | 1-epoch linear warm-up, then cosine to 1% of the peak within 20 epochs |
| training | at most 20 epochs; early stopping patience 5, eligible from epoch 10 |
| regularization | none beyond weight decay: no dropout, no drop-path; training-only brightness/contrast ±6% |
| batch | effective batch 64; the **largest** micro-batch that fits 90% of VRAM (BatchNorm statistics improve with batch size) |
| EMA | decay `0.9998` with warm-up; raw and EMA weights both evaluated every epoch |
| checkpoint monitor | best of raw/EMA **validation pathology-only Macro AP** (the 11 non-NORM labels) |
| thresholds | fixed 0.5, plus per-label F1 thresholds tuned on validation only, frozen for test |
| uncertainty | 95% patient-bootstrap intervals for test AP; paired bootstrap for the differences between models |
| acceleration | BF16 AMP, fused AdamW, TF32, channels-last, uint8 pinned batches, GPU augmentation and normalization |

The test split never selects epochs, raw versus EMA weights, or thresholds.

## Patient-level split, rebuilt at training time

The archive is unsplit. Section 4b assigns every **patient** (`patient_key`, the
HEEDB `BDSPPatientID`) to one of train/validation/test (80/10/10) by multi-label
iterative stratification, seeded by `CS_SPLIT_SEED` (default 20261002). It must
reproduce the frozen split SHA-256 `4eba80c7…` that every clean12 model uses.
`CS_SPLIT_FILE` (a `patient_key,split` CSV path, e.g. on Drive) reuses a frozen
copy instead.

References: [ResNet](https://arxiv.org/abs/1512.03385),
[DenseNet](https://arxiv.org/abs/1608.06993),
[EfficientNet](https://arxiv.org/abs/1905.11946),
[ResNet strikes back (timm a1 weights)](https://arxiv.org/abs/2110.00476).
"""),
    md(r"""## 0. Before you run (once per Google account)

**Runtime.** Runtime → Change runtime type → **A100 GPU**, and turn on
**High-RAM** if offered. Then Runtime → Run all.

**Dataset on Google Drive.** The archive is read from your Drive, not from S3,
so there is no AWS key and no S3 egress. It must be at:

```text
MyDrive/CardioSentry/new_dataset/clean_ecg_signal_images_clean12.tar          (~104.5 GiB)
MyDrive/CardioSentry/new_dataset/clean_ecg_signal_images_clean12.tar.sha256   (its SHA-256)
```

Section 3 reads it through the Drive API, which asks for **one extra Google
sign-in** (choose the account that owns `CardioSentry`). `CS_DATASET_READ_MODE=mount`
reads it through the mounted folder instead, with no extra sign-in (see section 3).
`CS_DATASET_DRIVE_DIR` points at a different folder.

**Google Drive.** The config cell mounts Drive (one consent prompt) and writes
all three runs into one group folder:

```text
MyDrive/CardioSentry/new_runs/<CS_RUN_GROUP>/     default cnn_baselines_clean12_colab_a100_seed42
├── patient_split.csv, split_audit.json, ...      shared by the three models
├── resnet50/                                     one complete run per architecture
├── densenet121/
├── efficientnet_b3/
└── comparison.csv, comparison.md, ...            written after the last model
```

**Time budget.** Expect roughly the data staging (tens of minutes), then for
each model a few minutes of probes and compilation plus up to 20 epochs. The
benchmark cell of each model prints its measured epoch time and the projected
20-epoch total before training starts. A Colab session has a maximum lifetime
(at most 24 h, on paid plans), so the three models together may not fit in one
session. See the resume notes below.

## Data path: Drive → Colab disk (streamed) → DataLoader → GPU

The archive is ~104.5 GiB. Section 3 streams it from Drive with parallel ranged
reads straight into `tar -x`, hashing every byte on the way, so the archive is
**never stored** on the Colab disk; only the extracted images are (~104 GiB).
The SHA-256 is compared with `clean_ecg_signal_images_clean12.tar.sha256`
after the last byte; on any mismatch the extracted tree is deleted and the cell
raises. The cell checks free disk first and needs ~113 GiB free under
`/content`. Training reads images only from the local disk.

**Each new Colab runtime streams the archive again** (the local disk is wiped
when the runtime is recycled). Reading from your own Drive costs nothing; it
only takes time (section 3 prints the measured MiB/s).

## Resume after a disconnect

Checkpoints go to Drive after every epoch (`<arch>/last.pt`; `<arch>/best.pt`
on every new best). After a disconnect, reconnect and **Run all** again with the
same `CS_RUN_GROUP`: the dataset is re-staged once, every architecture that
already finished is **skipped** (its results are reloaded from Drive for the
comparison), and the interrupted one continues from its last finished epoch.
A run never resumes onto a different configuration or split: both are
fingerprinted per architecture.

To run only some models, set `CS_ARCHITECTURES` before the config cell, e.g.
`os.environ["CS_ARCHITECTURES"] = "densenet121,efficientnet_b3"`. If one model
fails (out of memory, a training collapse), the error is recorded in
`<arch>/failure.json` and the next model still runs (`CS_CONTINUE_ON_ERROR=0`
stops instead).

## Benchmark only: what would a full run cost on this GPU?

Set these in a cell above the config cell, then Run all:

```python
import os
os.environ["CS_BENCHMARK_ONLY"] = "1"
os.environ["CS_COMPUTE_UNITS_PER_HOUR"] = "1.71"   # optional: the rate in Colab's Resources panel
```

The shared work runs as usual: the dataset download, audit, split and worker
tuning. Then each model runs only its batch probe and eager-vs-compiled
benchmark, and nothing is trained. Section 10d prints one table per GPU:

- the micro-batch and images per second;
- minutes per epoch;
- projected hours for a run stopped early at epoch 10 and for all 20 epochs;
- the compute units for each, when the rate is set.

It is written to `new_runs/cnn_baselines_benchmark_<gpu>/benchmark_comparison.csv`,
separate from any training run. Expect the dataset download plus about 10–20
minutes for the three models.

Nothing irreplaceable lives on the Colab disk: checkpoints, logs, metrics,
configs, the patient split, and final evaluation outputs are copied to Drive.

Leak note: this kit render prints no record-ID header (only lead names and the
25 mm/s / 10 mm/mV legend), so the full sheet is used with no content crop.
"""),
    code(r"""# 1. Install only what Colab lacks. Colab's own PyTorch/CUDA build is kept.
%pip install -q "timm==1.0.28" "psutil>=5.9" "seaborn>=0.13" \
    "scikit-learn>=1.4" "pandas>=2.1" "pillow>=10" "tqdm>=4.66"
"""),
    code(r"""# 2. Configuration, Colab setup (Google Drive), and GPU preflight.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import contextlib, csv, gc, hashlib, itertools, json, math, multiprocessing as mp, platform
import random, re, shutil, subprocess, sys, threading, time, warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.optim import param_groups_layer_decay
from timm.utils import ModelEmaV3
from PIL import Image, ImageFile
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
)
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

def env_flag(name, default):
    return os.environ.get(name, default).strip().lower() not in {
        "0", "false", "no", "off", "",
    }

HARDWARE_PROFILE = "colab-a100"
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ----------------------------------------------------------------------------
# Google Drive holds every run artifact. Colab's local disk is scratch only.
# ----------------------------------------------------------------------------
DRIVE_MOUNT_POINT = Path("/content/drive")
if IN_COLAB and not (DRIVE_MOUNT_POINT / "MyDrive").is_dir():
    from google.colab import drive
    drive.mount(str(DRIVE_MOUNT_POINT))
CARDIOSENTRY_DRIVE_ROOT = Path(os.environ.get(
    "CS_DRIVE_PROJECT_ROOT", str(DRIVE_MOUNT_POINT / "MyDrive" / "CardioSentry")
)).expanduser()
if not CARDIOSENTRY_DRIVE_ROOT.is_dir():
    raise RuntimeError(
        f"{CARDIOSENTRY_DRIVE_ROOT} is not available. On Colab this cell mounts Google "
        "Drive; elsewhere set CS_DRIVE_PROJECT_ROOT to the CardioSentry folder."
    )
DRIVE_RUNS_ROOT = Path(os.environ.get(
    "CS_DRIVE_RUNS_DIR", str(CARDIOSENTRY_DRIVE_ROOT / "new_runs")
)).expanduser()
DRIVE_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
DRIVE_SYNC_ENABLED = env_flag("CS_DRIVE_SYNC", "1")
DRIVE_SYNC_EVERY_N_EPOCHS = int(os.environ.get("CS_DRIVE_SYNC_EVERY_N_EPOCHS", "1"))
if DRIVE_SYNC_EVERY_N_EPOCHS < 1:
    raise ValueError("CS_DRIVE_SYNC_EVERY_N_EPOCHS must be at least 1")
# After the last cell succeeds, flush Drive and release the Colab runtime so it
# stops using compute units (the run is already on Drive). CS_RELEASE_RUNTIME=0 keeps it.
RELEASE_RUNTIME_WHEN_DONE = env_flag("CS_RELEASE_RUNTIME", "1")

# ----------------------------------------------------------------------------
# Dataset archive on Google Drive: copied there once from S3, so a new runtime
# costs no S3 egress. Read in section 3 (streamed into tar, never stored).
# ----------------------------------------------------------------------------
DATASET_TAR_NAME = os.environ.get(
    "CS_DATASET_TAR_NAME", "clean_ecg_signal_images_clean12.tar"
).strip()
DATASET_DRIVE_DIR = Path(os.environ.get(
    "CS_DATASET_DRIVE_DIR", str(CARDIOSENTRY_DRIVE_ROOT / "new_dataset")
)).expanduser()
DATASET_DRIVE_PATH = DATASET_DRIVE_DIR / DATASET_TAR_NAME
DATASET_SIDECAR_PATH = DATASET_DRIVE_DIR / f"{DATASET_TAR_NAME}.sha256"
# Strict archive provenance: a missing SHA-256 sidecar fails.
# CS_REQUIRE_TAR_SHA256=0 allows a size-only check.
REQUIRE_TAR_SHA256 = env_flag("CS_REQUIRE_TAR_SHA256", "1")
# "api": parallel ranged downloads through the Drive API (one extra Google
# sign-in; nothing cached on disk). "mount": parallel reads through the
# /content/drive mount (no extra sign-in; Drive's FUSE cache may use extra disk).
DATASET_READ_MODE = os.environ.get("CS_DATASET_READ_MODE", "api").strip().lower()
if DATASET_READ_MODE not in {"api", "mount"}:
    raise ValueError("CS_DATASET_READ_MODE must be api or mount")
# Optional: the archive's Drive file id, if it is not under MyDrive (e.g. a shared drive).
DATASET_DRIVE_FILE_ID = os.environ.get("CS_DATASET_DRIVE_FILE_ID", "").strip()
DATASET_DOWNLOAD_WORKERS = int(os.environ.get(
    "CS_DATASET_DOWNLOAD_WORKERS", "16" if DATASET_READ_MODE == "api" else "8"
))
DATASET_DOWNLOAD_CHUNK_BYTES = int(float(os.environ.get("CS_DATASET_DOWNLOAD_CHUNK_MIB", "32")) * 2**20)
if DATASET_DOWNLOAD_WORKERS < 1 or DATASET_DOWNLOAD_CHUNK_BYTES < 2**20:
    raise ValueError("CS_DATASET_DOWNLOAD_WORKERS must be >= 1 and CS_DATASET_DOWNLOAD_CHUNK_MIB >= 1")

# ----------------------------------------------------------------------------
# Local scratch (Colab's VM disk): extracted images, local run copy, caches.
# ----------------------------------------------------------------------------
SCRATCH_ROOT = Path(os.environ.get(
    "CS_SCRATCH_ROOT", "/content/cs_scratch" if IN_COLAB else str(Path.home() / "cs_scratch")
)).expanduser().resolve()
SCRATCH_MARGIN_GIB = float(os.environ.get("CS_SCRATCH_MARGIN_GIB", "8"))
EST_BYTES_PER_IMAGE = 1.0 * 2**20  # above the measured ~0.94 MiB mean
# Mid-epoch snapshots never drive resume (only epoch-boundary checkpoints do),
# so on Colab they are off by default; a positive value writes one every N minutes.
INTRA_EPOCH_SYNC_MINUTES = float(os.environ.get("CS_INTRA_EPOCH_SYNC_MINUTES", "0"))
EXTRACT_ROOT = Path(os.environ.get(
    "CS_EXTRACT_ROOT", str(SCRATCH_ROOT / "dataset")
)).expanduser().resolve()
RUN_ROOT = Path(os.environ.get(
    "CS_RUN_ROOT", str(SCRATCH_ROOT / "runs" / "cnn_baselines_clean12_colab_v1")
)).expanduser().resolve()
# Hugging Face cache on local disk: the ~350 MB weights download in seconds,
# and the Drive FUSE mount handles the hub's lock files and symlinks poorly.
# CS_HF_HOME=/content/drive/MyDrive/CardioSentry/hf_cache keeps it on Drive.
HF_HOME = Path(os.environ.get(
    "CS_HF_HOME", str(SCRATCH_ROOT / "hf_cache")
)).expanduser().resolve()
os.environ["HF_HOME"] = str(HF_HOME)
SCRATCH_TMP = SCRATCH_ROOT / "tmp"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(SCRATCH_ROOT / "torchinductor_cache"))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
for key in ("TMPDIR", "TEMP", "TMP"):
    os.environ.setdefault(key, str(SCRATCH_TMP))

# ----------------------------------------------------------------------------
# Architectures, trained one after another in this order. Each one gets its own
# subfolder of the run group (locally and on Drive) and its own fingerprint.
# CS_MODEL_<ARCH> (e.g. CS_MODEL_RESNET50=resnet50.tv_in1k) swaps the weights.
# ----------------------------------------------------------------------------
ARCHITECTURE_SPECS = {
    "resnet50": {
        "display_name": "ResNet-50",
        "model_name": os.environ.get("CS_MODEL_RESNET50", "resnet50.a1_in1k"),
        "pretraining": "timm ResNet strikes back A1 recipe, ImageNet-1K",
    },
    "densenet121": {
        "display_name": "DenseNet-121",
        "model_name": os.environ.get("CS_MODEL_DENSENET121", "densenet121.ra_in1k"),
        "pretraining": "timm RandAugment recipe, ImageNet-1K",
    },
    "efficientnet_b3": {
        "display_name": "EfficientNet-B3",
        "model_name": os.environ.get("CS_MODEL_EFFICIENTNET_B3", "efficientnet_b3.ra2_in1k"),
        "pretraining": "timm RandAugment v2 recipe, ImageNet-1K",
    },
}
ARCHITECTURES = [name.strip().lower() for name in os.environ.get(
    "CS_ARCHITECTURES", "resnet50,densenet121,efficientnet_b3"
).split(",") if name.strip()]
_unknown_architectures = sorted(set(ARCHITECTURES) - set(ARCHITECTURE_SPECS))
if _unknown_architectures or not ARCHITECTURES or len(set(ARCHITECTURES)) != len(ARCHITECTURES):
    raise ValueError(
        f"CS_ARCHITECTURES must list distinct names from {sorted(ARCHITECTURE_SPECS)}; "
        f"got {ARCHITECTURES}"
    )
# One failing model (OOM, collapse) is recorded and the next one still trains.
CONTINUE_ON_ERROR = env_flag("CS_CONTINUE_ON_ERROR", "1")

# Benchmark-only mode: every model runs its batch probe and eager-vs-compiled
# benchmark, then the notebook moves on without training; section 10d writes
# one table (time per epoch, projected hours, and compute units when
# CS_COMPUTE_UNITS_PER_HOUR is set, from Colab's Resources panel).
BENCHMARK_ONLY = env_flag("CS_BENCHMARK_ONLY", "0")
COMPUTE_UNITS_PER_HOUR = float(os.environ.get("CS_COMPUTE_UNITS_PER_HOUR", "0"))
if COMPUTE_UNITS_PER_HOUR < 0:
    raise ValueError("CS_COMPUTE_UNITS_PER_HOUR must be nonnegative")

# One run group holds the shared split files plus one subfolder per model.
# RUN_DIR / DRIVE_RUN_DIR below are the group folders; every Drive helper maps
# a local path under RUN_DIR to the same relative path under DRIVE_RUN_DIR.
# A benchmark-only run gets its own group, named after the GPU, so it never
# touches a training run.
_gpu_slug = (re.sub(r"[^a-z0-9]+", "_", torch.cuda.get_device_name(0).lower()).strip("_")
             if torch.cuda.is_available() else "nogpu")
RUN_NAME = os.environ.get(
    "CS_RUN_GROUP",
    f"cnn_baselines_benchmark_{_gpu_slug}" if BENCHMARK_ONLY
    else f"cnn_baselines_clean12_colab_a100_seed{os.environ.get('CS_SEED', '42')}",
)
RUN_DIR = RUN_ROOT / RUN_NAME
DRIVE_RUN_DIR = DRIVE_RUNS_ROOT / RUN_NAME
for path in (SCRATCH_ROOT, EXTRACT_ROOT, RUN_DIR, HF_HOME, SCRATCH_TMP, DRIVE_RUN_DIR):
    path.mkdir(parents=True, exist_ok=True)
RUN_LOG_PATH = RUN_DIR / "run.log"

def log_event(event, **fields):
    '''Append one JSONL event to the run log (copied to Drive with the run).'''
    try:
        with RUN_LOG_PATH.open("a") as handle:
            handle.write(json.dumps({
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **{str(key): value for key, value in fields.items()},
            }, default=str) + "\n")
    except OSError as exc:
        warnings.warn(f"run.log write failed: {exc}")

# ----------------------------------------------------------------------------
# Drive persistence: every run file is written locally first, then copied to
# DRIVE_RUN_DIR under the same relative path (write to *.partial, then rename).
# ----------------------------------------------------------------------------
def drive_path_for(local_path):
    return DRIVE_RUN_DIR / Path(local_path).relative_to(RUN_DIR)

def copy_to_drive(local_path, required=False):
    '''Copy one completed local artifact to Drive, then verify the byte count.'''
    local_path = Path(local_path)
    if not DRIVE_SYNC_ENABLED:
        return None
    if not local_path.is_file():
        message = f"Cannot copy missing local artifact: {local_path}"
        if required:
            raise FileNotFoundError(message)
        warnings.warn(message)
        return None
    destination = drive_path_for(local_path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        shutil.copyfile(local_path, partial)
        try:
            os.replace(partial, destination)
        except OSError:
            # Some FUSE mounts refuse rename-over-existing: remove, then rename.
            destination.unlink(missing_ok=True)
            os.replace(partial, destination)
        if destination.stat().st_size != local_path.stat().st_size:
            raise RuntimeError(
                f"Drive size check failed for {destination}: "
                f"local={local_path.stat().st_size}, drive={destination.stat().st_size}"
            )
    except Exception as exc:
        message = f"Drive copy failed for {local_path} -> {destination}: {exc}"
        if required:
            raise RuntimeError(message) from exc
        warnings.warn(message)
        return None
    print("saved to Drive:", destination)
    return destination

def copy_run_files(paths, required=False):
    copied = []
    for path in paths:
        path = Path(path)
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        destination = copy_to_drive(path, required=required)
        if destination:
            copied.append(destination)
    return copied

_sync_threads = []
_sync_threads_lock = threading.Lock()
_copy_key_threads = {}
_copy_key_lock = threading.Lock()

def _prune_sync_threads():
    with _sync_threads_lock:
        _sync_threads[:] = [t for t in _sync_threads if t.is_alive()]
        return len(_sync_threads)

def spawn_drive_copy(local_path):
    '''Copy one finished artifact to Drive on a daemon thread; never raises.

    Same-file copies are serialized: if a copy of this file is already in
    flight, the new request is skipped so an older copy can never land last.
    '''
    if not DRIVE_SYNC_ENABLED:
        return None
    key = Path(local_path).relative_to(RUN_DIR).as_posix()
    with _copy_key_lock:
        existing = _copy_key_threads.get(key)
        if existing is not None and existing.is_alive():
            warnings.warn(f"skipping Drive copy of {local_path}: a copy is already in flight")
            return existing
    _prune_sync_threads()
    def _target():
        try:
            copy_to_drive(Path(local_path), required=False)
        except Exception as exc:
            warnings.warn(f"background Drive copy failed for {local_path}: {exc}")
    thread = threading.Thread(target=_target, name=f"drive-copy-{Path(local_path).name}", daemon=True)
    with _sync_threads_lock:
        _sync_threads.append(thread)
    with _copy_key_lock:
        _copy_key_threads[key] = thread
    thread.start()
    return thread

def join_drive_copies(timeout=None):
    '''Wait for outstanding background copies (epoch boundaries, run end).'''
    with _sync_threads_lock:
        pending = list(_sync_threads)
    for thread in pending:
        thread.join(timeout=timeout)
    remaining = _prune_sync_threads()
    with _copy_key_lock:
        for finished in [k for k, t in _copy_key_threads.items() if not t.is_alive()]:
            del _copy_key_threads[finished]
    return remaining

def restore_from_drive(name, local_path):
    '''Copy one run file back from Drive. Returns None when Drive has no such file.'''
    local_path = Path(local_path)
    if not DRIVE_SYNC_ENABLED:
        return None
    source = DRIVE_RUN_DIR / name
    if not DRIVE_RUN_DIR.is_dir():
        raise RuntimeError(
            f"Drive run folder {DRIVE_RUN_DIR} is not reachable; refusing to assume a fresh run."
        )
    if not source.is_file():
        return None
    temporary = local_path.with_suffix(local_path.suffix + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size != source.stat().st_size:
            raise RuntimeError("byte count differs after copy")
        os.replace(temporary, local_path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        # The file exists on Drive, so a failed copy must never read as "missing":
        # a half-restored run directory would restart over existing checkpoints.
        raise RuntimeError(f"Drive restore failed for {source} -> {local_path}: {exc}") from exc
    print("restored from Drive:", source)
    return local_path

def restore_missing_run_files_from_drive(filenames):
    '''Fresh-runtime resume: fetch run files absent locally; never overwrites.'''
    restored = []
    for name in filenames:
        local_path = RUN_DIR / name
        if local_path.is_file():
            continue
        if restore_from_drive(name, local_path) is not None:
            restored.append(name)
    if restored:
        print("restored from Drive:", ", ".join(restored))
    return restored

def sync_complete_run_to_drive(required=True, root=None, recursive=True):
    '''Copy every final artifact under root to Drive and write a SHA-256 manifest last.

    root defaults to the run group; an architecture passes its own subfolder.
    recursive=False copies only root's own files (the group-level summaries).
    '''
    root = Path(root or RUN_DIR)
    if not DRIVE_SYNC_ENABLED:
        print("Drive sync disabled; final artifacts remain local at", root)
        return None
    paths = sorted(
        path for path in (root.rglob("*") if recursive else root.iterdir())
        if path.is_file()
        and not path.name.endswith(".tmp")
        and path.name != "drive_sync_manifest.json"
    )
    if not paths:
        raise RuntimeError(f"No final artifacts found under {root}")
    records = []
    for path in paths:
        destination = copy_to_drive(path, required=required)
        if destination:
            records.append({
                "relative_path": path.relative_to(RUN_DIR).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "drive_path": str(destination),
            })
    manifest_path = root / "drive_sync_manifest.json"
    manifest = {
        "synced_at_utc": datetime.now(timezone.utc).isoformat(),
        "drive_run_dir": str(drive_path_for(root)),
        "artifacts": records,
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(temporary, manifest_path)
    copy_to_drive(manifest_path, required=required)
    print(f"Final Drive sync verified: {len(records):,} artifacts plus sync manifest")
    print("Drive folder:", drive_path_for(root))
    return manifest

# clean12 corpus identity (unsplit HEEDB 500 Hz records + kit render, one image
# per record). Label names are the record_manifest.csv column names; NORM first.
NORMAL_LABEL = "NORM"
LABELS = ["NORM", "SB", "STACH", "LBBB", "LQT", "AF", "AFL",
          "3AVB", "2AVB", "1AVB", "STEMI", "STE"]
LABEL_NAMES = {
    "NORM": "Normal ECG", "SB": "Sinus bradycardia", "STACH": "Sinus tachycardia",
    "LBBB": "Left bundle branch block", "LQT": "QT/QTc prolongation",
    "AF": "Atrial fibrillation", "AFL": "Atrial flutter",
    "3AVB": "Third-degree AV block", "2AVB": "Second-degree AV block",
    "1AVB": "First-degree AV block", "STEMI": "STEMI", "STE": "ST elevation",
}
PATHOLOGY_LABELS = [label for label in LABELS if label != NORMAL_LABEL]
# The clean12 archive carries exactly one clean image per ECG record.
TRAIN_VARIANTS = ["clean"]
EVAL_VARIANTS = ["clean"]
SUBSETS = ("train", "validation", "test")
SITES = ("I0001", "I0006")
KIT_TOPDIR = "clean_ecg_signal_images_clean12"
EXPECTED_RECORDS = 113_358
EXPECTED_RECORDS_BY_SITE = {"I0001": 74_356, "I0006": 39_002}

# Patient-level split, identical to the V1 Base clean12 notebook's: a
# deterministic function of record_manifest.csv and CS_SPLIT_SEED only, never
# of CS_SEED. CS_SPLIT_FILE (local or Drive path of a patient_key,split CSV)
# reuses a frozen split instead.
SPLIT_VERSION = "clean12_patient_split_v1"
SPLIT_SEED = int(os.environ.get("CS_SPLIT_SEED", "20261002"))
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
SPLIT_FILE = os.environ.get("CS_SPLIT_FILE", "").strip()
# The default seed must reproduce the frozen split every clean12 model uses, so
# every clean12 model (ConvNeXt V1/V2, fusion, these CNNs) is on the same patients. Checked in section 4b.
EXPECTED_SPLIT_SHA256 = "4eba80c71bba27e8bde0ed1a510de9d4819ede5139946b48526461a7bc4841cd"
# Acceptance gates: every split within 1 point of its record share, and every
# label within 1.5 points of its target share in every split.
SPLIT_RECORD_TOLERANCE = 0.01
SPLIT_LABEL_TOLERANCE = 0.015

# One plain fine-tuning recipe, identical for every CNN (the point is raw
# backbone performance, not a per-model tuned optimum). Every value below is in
# each architecture's fingerprint; change one only with a new CS_RUN_GROUP.
INPUT_SIZE = (768, 1024)                 # (height, width): the full native sheet
MAX_EPOCHS = int(os.environ.get("CS_MAX_EPOCHS", "20"))
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_EPOCHS = min(10, MAX_EPOCHS)
WARMUP_EPOCHS = 1
# One LR for every layer (no layer-wise decay), AdamW.
BASE_LR = float(os.environ.get("CS_BASE_LR", "3e-4"))
MIN_LR_RATIO = 0.01                     # LR floor 3e-6
# Decoupled AdamW weight decay on conv/linear weights; BatchNorm and biases get none.
WEIGHT_DECAY = 0.01
# No extra regularization: timm's classifier dropout and stochastic depth off.
DROP_RATE = 0.0
DROP_PATH_RATE = 0.0
EFFECTIVE_BATCH_SIZE = 64
EMA_DECAY = 0.9998
# Collapse guard. From the second epoch on, a rolling window of training BCE
# at >= COLLAPSE_LOSS_RATIO x the best constant predictor's loss for
# COLLAPSE_PATIENCE_WINDOWS consecutive windows means the model has stopped
# using the image; so does a raw-weight validation macro AUROC below
# COLLAPSE_MIN_VAL_MACRO_AUROC. Either stops training before last.pt is
# replaced. CS_COLLAPSE_GUARD=0 disables both checks.
COLLAPSE_GUARD_ENABLED = env_flag("CS_COLLAPSE_GUARD", "1")
COLLAPSE_LOSS_RATIO = 0.97
COLLAPSE_CHECK_EVERY_N_BATCHES = 100
COLLAPSE_PATIENCE_WINDOWS = 5
COLLAPSE_MIN_VAL_MACRO_AUROC = 0.60
TRAIN_BRIGHTNESS_JITTER = 0.06
TRAIN_CONTRAST_JITTER = 0.06
# Rare-label repeat-factor sampling (Gupta et al., LVIS 2019), multi-label form:
# label repeat r = max(1, sqrt(threshold / training frequency)); an image is
# drawn max(r over its labels) times per epoch with stochastic rounding. With
# 0.10 that is 3AVB 2.72x, 2AVB 1.76x, STEMI 1.12x, other labels 1x (+5.7% per
# epoch). pos_weight is then computed on the resampled counts (section 4c).
# Off by default here: uniform epochs are the raw baseline. CS_RARE_SAMPLING=1
# turns it on for every architecture (it is in the fingerprint).
RARE_SAMPLING = env_flag("CS_RARE_SAMPLING", "0")
RARE_SAMPLING_THRESHOLD = float(os.environ.get("CS_RARE_SAMPLING_THRESHOLD", "0.10"))
if not 0.0 < RARE_SAMPLING_THRESHOLD < 1.0:
    raise ValueError("CS_RARE_SAMPLING_THRESHOLD must be in (0, 1)")
# Patient-bootstrap confidence intervals for the final test metrics.
BOOTSTRAP_ROUNDS = int(os.environ.get("CS_BOOTSTRAP_ROUNDS", "1000"))
BOOTSTRAP_CONFIDENCE = 0.95
# Baseline uses plain weighted BCE. The noisy-OR contradiction term saturates
# with many pathologies and would mostly suppress NORM; re-enable it only
# as an explicit ablation with its own CS_RUN_GROUP.
CONTRADICTION_LAMBDA = 0.0
GRAD_CLIP_NORM = 1.0
SEED = int(os.environ.get("CS_SEED", "42"))
if not 0 <= SEED < 2**32:
    raise ValueError("CS_SEED must be in [0, 2**32)")
NUM_WORKERS = int(os.environ.get(
    "CS_NUM_WORKERS", str(min(6, max(1, (os.cpu_count() or 2) - 2)))
))
AUTO_TUNE_WORKERS = env_flag("CS_AUTO_TUNE_WORKERS", "1") and "CS_NUM_WORKERS" not in os.environ
# Colab's A100 runtime has 12 vCPUs; leave one for the training loop.
WORKER_CANDIDATES = [value for value in (4, 6, 8, 10)
                     if value <= max(1, (os.cpu_count() or 1) - 1)] or [0]
if NUM_WORKERS < 0:
    raise ValueError("CS_NUM_WORKERS must be nonnegative")
DATALOADER_START_METHOD = os.environ.get("CS_DATALOADER_START_METHOD", "fork").strip()
if NUM_WORKERS > 0 and DATALOADER_START_METHOD not in mp.get_all_start_methods():
    raise ValueError(
        f"Unsupported CS_DATALOADER_START_METHOD={DATALOADER_START_METHOD!r}; "
        f"available methods: {mp.get_all_start_methods()}"
    )
PREFETCH_FACTOR = int(os.environ.get("CS_PREFETCH_FACTOR", "4"))
if PREFETCH_FACTOR < 1:
    raise ValueError("CS_PREFETCH_FACTOR must be positive")
PROGRESS_UPDATE_EVERY_N_BATCHES = int(os.environ.get(
    "CS_PROGRESS_UPDATE_EVERY_N_BATCHES", "50"
))
if PROGRESS_UPDATE_EVERY_N_BATCHES < 1:
    raise ValueError("CS_PROGRESS_UPDATE_EVERY_N_BATCHES must be at least 1")
# Training micro-batch: the largest of these that fits wins (BatchNorm
# statistics are computed per micro-batch, so bigger is better for the CNNs).
AUTO_BATCH_CANDIDATES = [64, 32, 16, 8, 4, 2]
EVAL_BATCH_CANDIDATES = [128, 64, 32, 16, 8, 4, 2, 1]
# PNG decoder workers are tuned once, at this batch, and shared by all models.
WORKER_TUNING_BATCH = 16
BATCH_PROBE_TIMED_UPDATES = 2
EVAL_PROBE_TIMED_BATCHES = 2
VRAM_SAFETY_FRACTION = 0.90
USE_AMP = True
DETERMINISTIC = env_flag("CS_DETERMINISTIC", "0")
USE_FUSED_ADAMW = env_flag("CS_FUSED_ADAMW", "1") and not DETERMINISTIC
TORCH_COMPILE_ENABLED = env_flag("CS_TORCH_COMPILE", "1") and not DETERMINISTIC
# "default" compiles each CNN in minutes; "max-autotune-no-cudagraphs" also
# autotunes every convolution at 768x1024, which can take far longer per model.
TORCH_COMPILE_MODE = os.environ.get("CS_TORCH_COMPILE_MODE", "default").strip()
COMPILE_SELECTED = False
if DETERMINISTIC:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Benchmark size (number of timed training/validation batches after warm-up).
BENCHMARK_TRAIN_BATCHES = int(os.environ.get("CS_BENCHMARK_TRAIN_BATCHES", "64"))
BENCHMARK_VAL_BATCHES = int(os.environ.get("CS_BENCHMARK_VAL_BATCHES", "10"))
BENCHMARK_WARMUP_BATCHES = 3
WORKER_BENCHMARK_BATCHES = 16
COMBINATION_LOW_SUPPORT_THRESHOLD = 30

def gib(value):
    return value / 2**30

print("dataset mode:       clean only (single mode); clean12, 12 labels")
print("Python:             ", sys.version.split()[0])
print("PyTorch:            ", torch.__version__)
print("timm:               ", timm.__version__)
print("running in Colab:   ", IN_COLAB)
print("CPU count:          ", os.cpu_count())
print(f"RAM:                 {gib(psutil.virtual_memory().total):.1f} GiB")
print("dataset on Drive:   ", DATASET_DRIVE_PATH, f"(read mode: {DATASET_READ_MODE})")
print("scratch root:       ", SCRATCH_ROOT)
print("dataset path:       ", EXTRACT_ROOT / KIT_TOPDIR)
print("split:              ", f"{SPLIT_VERSION} seed={SPLIT_SEED}",
      f"(frozen file {SPLIT_FILE})" if SPLIT_FILE else "(computed from record_manifest.csv)")
print("local run directory:", RUN_DIR)
print("Drive run folder:   ", DRIVE_RUN_DIR, "" if DRIVE_SYNC_ENABLED else "(Drive sync DISABLED)")
print("HF cache:           ", HF_HOME)
print("rare-label sampling:", f"on, threshold {RARE_SAMPLING_THRESHOLD}" if RARE_SAMPLING else "off")
if BENCHMARK_ONLY:
    print("BENCHMARK ONLY:      probes + eager/compiled benchmark per model, no training",
          f"({COMPUTE_UNITS_PER_HOUR:g} compute units/hour)" if COMPUTE_UNITS_PER_HOUR else "")
scratch_free_gib = gib(shutil.disk_usage(EXTRACT_ROOT).free)
print(f"free scratch disk:   {scratch_free_gib:.1f} GiB "
      f"(the streamed dataset needs ~{EXPECTED_RECORDS * EST_BYTES_PER_IMAGE / 2**30 + SCRATCH_MARGIN_GIB:.0f} GiB "
      "unless it is already extracted)")

# Prove the archive and its checksum are on Drive before anything long starts.
if not DATASET_DRIVE_PATH.is_file():
    raise FileNotFoundError(
        f"{DATASET_DRIVE_PATH} is missing. Copy the clean12 archive (and its .sha256) "
        "into MyDrive/CardioSentry/new_dataset, or set CS_DATASET_DRIVE_DIR."
    )
if REQUIRE_TAR_SHA256 and not DATASET_SIDECAR_PATH.is_file():
    raise FileNotFoundError(f"{DATASET_SIDECAR_PATH} is missing (CS_REQUIRE_TAR_SHA256=1)")
print(f"Drive dataset preflight: OK ({gib(DATASET_DRIVE_PATH.stat().st_size):.2f} GiB archive)")

# Prove the Drive run folder can be written and read back.
if DRIVE_SYNC_ENABLED:
    preflight_path = RUN_DIR / "upload_preflight.json"
    preflight_path.write_text(json.dumps({
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "purpose": "CardioSentry clean12 Drive write preflight",
    }, indent=2))
    copied = copy_to_drive(preflight_path, required=True)
    if copied.read_bytes() != preflight_path.read_bytes():
        raise RuntimeError(f"Drive preflight read-back differs: {copied}")
    print("Drive write preflight: OK ->", copied)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. In Colab choose Runtime -> Change runtime type -> "
        "A100 GPU, then run all cells again."
    )
gpu = torch.cuda.get_device_properties(0)
print(f"GPU:                 {gpu.name}")
print(f"actual GPU VRAM:     {gib(gpu.total_memory):.1f} GiB")
print("CUDA runtime:        ", torch.version.cuda)
if os.cpu_count() and os.cpu_count() < 8:
    warnings.warn("Fewer than 8 logical CPUs are visible; image decoding may bottleneck the GPU.")
if psutil.virtual_memory().total < 50 * 2**30:
    warnings.warn("Less than 50 GiB RAM is visible; turn on High-RAM in Colab's runtime settings if offered.")
if "A100" not in gpu.name:
    warnings.warn(f"This notebook targets a Colab A100; the visible GPU is {gpu.name}. "
                  "Batch sizes and workers are still measured on it.")
torch.set_num_threads(1)

torch.backends.cudnn.benchmark = not DETERMINISTIC
torch.backends.cudnn.deterministic = DETERMINISTIC
torch.use_deterministic_algorithms(DETERMINISTIC)
torch.backends.cuda.matmul.allow_tf32 = not DETERMINISTIC
torch.backends.cudnn.allow_tf32 = not DETERMINISTIC
torch.set_float32_matmul_precision("highest" if DETERMINISTIC else "high")
DEVICE = torch.device("cuda:0")
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print("AMP dtype:           ", AMP_DTYPE)
if AMP_DTYPE == torch.float16:
    warnings.warn(
        "BF16 is unavailable on this GPU; falling back to FP16 with loss scaling. "
        "This notebook was only validated for BF16 (the A100 supports it)."
    )
print("fused AdamW:         ", USE_FUSED_ADAMW)
print("compiler comparison: ", TORCH_COMPILE_ENABLED)
C_COMPILER = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
print("C compiler:          ", C_COMPILER or "missing")
if TORCH_COMPILE_ENABLED and not C_COMPILER:
    # Triton builds its kernel launchers with a host C compiler; without one
    # the compiled benchmark fails and the run silently stays eager.
    warnings.warn(
        "No C compiler found, so torch.compile will fail and training will run eager. "
        "Install one and restart the runtime: apt-get install -y build-essential."
    )
print("compile mode:        ", TORCH_COMPILE_MODE)
print("architectures:       ", " -> ".join(
    f"{name} ({ARCHITECTURE_SPECS[name]['model_name']})" for name in ARCHITECTURES))
print("continue on error:   ", CONTINUE_ON_ERROR)
print("collapse guard:      ", COLLAPSE_GUARD_ENABLED)
print(f"peak LR (all layers): {BASE_LR:.2e}; weight decay {WEIGHT_DECAY}; {MAX_EPOCHS} epochs max")

def seed_global_rngs(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_global_rngs(SEED)
print("reproducibility:     ", "deterministic opt-in" if DETERMINISTIC else "seeded performance mode")

# Exact package provenance for this run. timm is hard-pinned in the install
# cell (the head-init and weight-tag behavior above was verified against
# 1.0.28); torch/torchvision are Colab's and are recorded, not pinned. The
# versions are stored in the run config and a drift warning fires on resume,
# so a new Colab image can never silently change the implementation mid-run.
from importlib.metadata import version as _dist_version, PackageNotFoundError
def _installed_version(distribution):
    try:
        return _dist_version(distribution)
    except PackageNotFoundError:
        return "missing"
PACKAGE_VERSIONS = {
    "python": platform.python_version(),
    **{name: _installed_version(name) for name in (
        "timm", "torch", "torchvision", "scikit-learn", "pandas", "pillow",
        "matplotlib", "seaborn", "tqdm", "psutil", "requests", "google-auth", "numpy",
    )},
}
print("package versions:    ", json.dumps(PACKAGE_VERSIONS, sort_keys=True))
"""),
    md(r"""## 3. Stream the clean12 archive from Drive, verify it, and extract it

The archive (~104.5 GiB) is never stored on the Colab disk. Parallel ranged
reads (`CS_DATASET_DOWNLOAD_WORKERS` × 32 MiB) are reassembled in order and
piped into `tar -x`, while every byte is hashed. After the last byte the
SHA-256 must equal the `.sha256` sidecar next to the archive on Drive. On a
mismatch, a `tar` error, or an interruption, the partly extracted tree is
deleted and the cell raises. Re-running it then starts a clean stream.

Two read modes (`CS_DATASET_READ_MODE`):

- **`api`** (default): ranged downloads through the Google Drive API, 16 in
  parallel. Colab asks for one extra Google sign-in here; choose the account
  that owns `CardioSentry`. Nothing is cached on the local disk. Before
  downloading, Drive's own SHA-256 of the file is compared with the sidecar.
- **`mount`**: parallel reads through `/content/drive`, with no extra sign-in.
  Drive's FUSE layer may cache what it reads on the local disk, which competes
  with the extracted images for space. Use it if the API sign-in fails.

A completion marker records the archive SHA-256, so re-running the notebook on
the same runtime skips the download (and the sign-in). A new Colab runtime has
an empty disk and streams again; reading your own Drive costs nothing.
"""),
    code(r"""def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()

def expected_tar_sha256():
    '''The archive's SHA-256 from its Drive sidecar, or None when allowed to be absent.'''
    if not DATASET_SIDECAR_PATH.is_file():
        if REQUIRE_TAR_SHA256:
            raise RuntimeError(
                f"No checksum sidecar at {DATASET_SIDECAR_PATH} and CS_REQUIRE_TAR_SHA256=1: "
                "refusing to train without archive-hash authority."
            )
        warnings.warn(f"No checksum sidecar at {DATASET_SIDECAR_PATH}; verifying by byte count only.")
        return None
    text = DATASET_SIDECAR_PATH.read_text().split()[0].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"Malformed SHA-256 sidecar {DATASET_SIDECAR_PATH}")
    return text

DRIVE_API_FILES = "https://www.googleapis.com/drive/v3/files"
_drive_credentials = None
_drive_credentials_lock = threading.Lock()

def drive_api_token(force_refresh=False):
    '''A valid OAuth token for the Drive API (Colab asks to sign in on first use).'''
    global _drive_credentials
    import google.auth
    from google.auth.transport.requests import Request as GoogleAuthRequest
    with _drive_credentials_lock:
        if _drive_credentials is None:
            if IN_COLAB:
                from google.colab import auth as colab_auth
                colab_auth.authenticate_user()
            _drive_credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/drive"])
        if force_refresh or not _drive_credentials.valid:
            _drive_credentials.refresh(GoogleAuthRequest())
        return _drive_credentials.token

def drive_api_get(url, params=None, headers=None, timeout=(30, 300)):
    '''GET a Drive API URL; one retry with a refreshed token on 401.'''
    for force_refresh in (False, True):
        response = requests.get(url, params=params, timeout=timeout, headers={
            **(headers or {}), "Authorization": f"Bearer {drive_api_token(force_refresh)}"})
        if response.status_code != 401:
            break
    return response

def drive_file_id_for(path):
    '''Resolve a path under MyDrive to its Drive file id, one folder level at a time.'''
    if DATASET_DRIVE_FILE_ID:
        return DATASET_DRIVE_FILE_ID
    try:
        parts = path.relative_to(DRIVE_MOUNT_POINT / "MyDrive").parts
    except ValueError:
        raise RuntimeError(f"{path} is not under MyDrive; set CS_DATASET_DRIVE_FILE_ID") from None
    parent = "root"
    for part in parts:
        quoted = part.replace("\\", "\\\\").replace("'", "\\'")
        response = drive_api_get(DRIVE_API_FILES, params={
            "q": f"name = '{quoted}' and '{parent}' in parents and trashed = false",
            "fields": "files(id,name)", "pageSize": 10})
        response.raise_for_status()
        found = response.json()["files"]
        if len(found) != 1:
            raise RuntimeError(
                f"Expected exactly one {part!r} on Drive while resolving {path}, found {len(found)}. "
                "Remove the duplicate or set CS_DATASET_DRIVE_FILE_ID."
            )
        parent = found[0]["id"]
    return parent

def iter_ordered_chunks(fetch, total, chunk_bytes, workers):
    '''Yield fetch(offset) for every chunk of [0, total) in order, `workers` at a time.

    At most 2 x workers chunks are in flight or buffered.
    '''
    offsets = iter(range(0, total, chunk_bytes))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dataset-read") as pool:
        window = deque(pool.submit(fetch, offset) for offset in itertools.islice(offsets, 2 * workers))
        try:
            while window:
                data = window.popleft().result()
                following = next(offsets, None)
                if following is not None:
                    window.append(pool.submit(fetch, following))
                yield data
        finally:
            for future in window:
                future.cancel()

def drive_api_fetcher(file_id, total, chunk_bytes):
    '''fetch(offset) -> bytes, a ranged Drive API download with retries.'''
    url = f"{DRIVE_API_FILES}/{file_id}"
    def fetch(offset):
        end = min(offset + chunk_bytes, total) - 1
        problem = None
        for attempt in range(8):
            try:
                response = drive_api_get(url, params={"alt": "media", "supportsAllDrives": "true"},
                                         headers={"Range": f"bytes={offset}-{end}"})
                code = response.status_code
                whole_file = code == 200 and offset == 0 and end == total - 1
                if (code == 206 or whole_file) and len(response.content) == end - offset + 1:
                    return response.content
                if code in (400, 404):
                    raise RuntimeError(f"Drive refused the download ({code}): {response.text[:300]}")
                problem = f"HTTP {code} ({len(response.content):,} bytes)"  # 403/429 rate limits, 5xx
            except requests.RequestException as exc:
                problem = repr(exc)
            time.sleep(min(60, 2 ** attempt))
        raise IOError(f"Drive read at byte {offset:,} kept failing: {problem}")
    return fetch

def mounted_file_fetcher(path, total, chunk_bytes):
    '''fetch(offset) -> bytes, a positional read through the Drive mount.'''
    descriptor = os.open(path, os.O_RDONLY)
    def fetch(offset):
        wanted = min(offset + chunk_bytes, total) - offset
        parts, received = [], 0
        while received < wanted:
            for attempt in range(6):
                try:
                    data = os.pread(descriptor, wanted - received, offset + received)
                    break
                except OSError:
                    if attempt == 5:
                        raise
                    time.sleep(2 ** attempt)
            if not data:
                raise IOError(f"{path} ended at byte {offset + received:,} of {total:,}")
            parts.append(data)
            received += len(data)
        return b"".join(parts)
    fetch.close = lambda: os.close(descriptor)
    return fetch

def stream_extract_archive(expected_sha, total, chunks):
    '''Pipe the archive's chunks into tar while hashing them; return (sha256, seconds).'''
    command = ["tar", "-x", "-f", "-", "-C", str(EXTRACT_ROOT),
               "--no-same-owner", "--no-same-permissions"]
    tar_log_path = SCRATCH_TMP / "tar_extract.log"
    digest = hashlib.sha256()
    received = 0
    started = time.time()
    progress = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                    desc=f"Drive ({DATASET_READ_MODE}) -> tar (streamed)")
    with tar_log_path.open("wb") as tar_log:
        # stderr goes to a file: an undrained pipe could fill and deadlock tar.
        extractor = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=tar_log)
        try:
            for data in chunks:
                digest.update(data)
                extractor.stdin.write(data)
                received += len(data)
                progress.update(len(data))
            extractor.stdin.close()
            returncode = extractor.wait()
        except BaseException as exc:
            extractor.kill()
            extractor.wait()
            shutil.rmtree(KIT_ROOT, ignore_errors=True)
            if isinstance(exc, KeyboardInterrupt):
                raise
            tail = tar_log_path.read_text(errors="replace")[-2000:]
            raise RuntimeError(
                f"Streaming extraction failed after {gib(received):.1f} of {gib(total):.1f} GiB; "
                f"the partial tree was deleted. tar said:\n{tail}"
            ) from exc
        finally:
            progress.close()
    tail = tar_log_path.read_text(errors="replace")[-2000:]
    actual = digest.hexdigest()
    problems = []
    if returncode != 0:
        problems.append(f"tar exited with {returncode}: {tail}")
    if received != total:
        problems.append(f"received {received:,} bytes, Drive reports {total:,}")
    if expected_sha is not None and actual != expected_sha:
        problems.append(f"SHA-256 {actual} != sidecar {expected_sha}")
    if problems:
        shutil.rmtree(KIT_ROOT, ignore_errors=True)
        raise RuntimeError("Archive verification failed; the extracted tree was deleted:\n- "
                           + "\n- ".join(problems))
    return actual, time.time() - started

def open_archive_chunks():
    '''The archive as an ordered chunk iterator, plus a close() for the mount reader.'''
    if DATASET_READ_MODE == "mount":
        fetch = mounted_file_fetcher(DATASET_DRIVE_PATH, ARCHIVE_BYTES, DATASET_DOWNLOAD_CHUNK_BYTES)
        closer = fetch.close
    else:
        file_id = drive_file_id_for(DATASET_DRIVE_PATH)
        response = drive_api_get(f"{DRIVE_API_FILES}/{file_id}", params={
            "fields": "id,name,size,sha256Checksum", "supportsAllDrives": "true"})
        response.raise_for_status()
        meta = response.json()
        print(f"Drive file {meta['name']} id={meta['id']} sha256={meta.get('sha256Checksum')}")
        if int(meta["size"]) != ARCHIVE_BYTES:
            raise RuntimeError(f"Drive API size {int(meta['size']):,} != mounted size {ARCHIVE_BYTES:,}")
        if EXPECTED_TAR_SHA256 and meta.get("sha256Checksum") not in (None, EXPECTED_TAR_SHA256):
            raise RuntimeError(
                f"Drive's SHA-256 of {DATASET_DRIVE_PATH} is {meta['sha256Checksum']}, but the sidecar "
                f"says {EXPECTED_TAR_SHA256}: the copy on Drive is damaged. Copy it again."
            )
        fetch = drive_api_fetcher(meta["id"], ARCHIVE_BYTES, DATASET_DOWNLOAD_CHUNK_BYTES)
        closer = lambda: None
    chunks = iter_ordered_chunks(fetch, ARCHIVE_BYTES, DATASET_DOWNLOAD_CHUNK_BYTES,
                                 DATASET_DOWNLOAD_WORKERS)
    return chunks, closer

EXPECTED_TAR_SHA256 = expected_tar_sha256()
print("expected archive sha256:", EXPECTED_TAR_SHA256 or "none published (byte count only)")
_archive_stat = DATASET_DRIVE_PATH.stat()
ARCHIVE_BYTES = int(_archive_stat.st_size)
# Identifies the Drive copy when no SHA-256 sidecar exists.
ARCHIVE_VERSION = f"{ARCHIVE_BYTES}:{int(_archive_stat.st_mtime)}"

KIT_ROOT = EXTRACT_ROOT / KIT_TOPDIR
MANIFEST_CSV = KIT_ROOT / "manifest.csv"
RECORD_MANIFEST_CSV = KIT_ROOT / "record_manifest.csv"
FINAL_COUNTS_CSV = KIT_ROOT / "final_counts.csv"
EXTRACTION_MARKER = EXTRACT_ROOT / ".clean12_extract_complete.json"

def extraction_complete():
    if not EXTRACTION_MARKER.is_file() or not KIT_ROOT.is_dir():
        return False
    try:
        payload = json.loads(EXTRACTION_MARKER.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (payload.get("sha256") == EXPECTED_TAR_SHA256
            and (EXPECTED_TAR_SHA256 is not None or payload.get("version") == ARCHIVE_VERSION))

dataset_copy_started = time.time()
if extraction_complete():
    print("dataset already extracted and verified on this runtime; skipping the download")
else:
    if KIT_ROOT.exists():
        print("removing an unverified or partial extraction ...")
        shutil.rmtree(KIT_ROOT)
    EXTRACTION_MARKER.unlink(missing_ok=True)
    need_gib = gib(ARCHIVE_BYTES) + SCRATCH_MARGIN_GIB
    free_gib = gib(shutil.disk_usage(EXTRACT_ROOT).free)
    print(f"{EXTRACT_ROOT}: free={free_gib:.1f} GiB; need ~{need_gib:.1f} GiB "
          f"({gib(ARCHIVE_BYTES):.1f} GiB extracted + {SCRATCH_MARGIN_GIB:.0f} GiB margin)")
    if free_gib < need_gib:
        raise RuntimeError(
            f"Colab disk has {free_gib:.1f} GiB free but the extracted dataset needs "
            f"~{need_gib:.1f} GiB. Use an A100 runtime (larger disk) and remove other "
            "files under /content, then re-run."
        )
    archive_chunks, close_archive = open_archive_chunks()
    try:
        actual_sha, stream_seconds = stream_extract_archive(
            EXPECTED_TAR_SHA256, ARCHIVE_BYTES, archive_chunks)
    finally:
        archive_chunks.close()
        close_archive()
    print(f"archive verified: sha256 {actual_sha} "
          f"({gib(ARCHIVE_BYTES) / max(stream_seconds, 1e-9) * 1024:.0f} MiB/s average)")
    temporary = EXTRACTION_MARKER.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "sha256": actual_sha if EXPECTED_TAR_SHA256 is not None else None,
        "observed_sha256": actual_sha, "version": ARCHIVE_VERSION, "bytes": ARCHIVE_BYTES,
        "source": str(DATASET_DRIVE_PATH), "read_mode": DATASET_READ_MODE,
        "seconds": round(stream_seconds, 1),
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    os.replace(temporary, EXTRACTION_MARKER)

for required in [MANIFEST_CSV, RECORD_MANIFEST_CSV, FINAL_COUNTS_CSV,
                 KIT_ROOT / "summary.json"]:
    if not required.is_file():
        raise FileNotFoundError(
            f"Expected {required} after extraction. If the archive layout changed, "
            "inspect the TAR listing before proceeding."
        )

# Light count check here; the audit cell joins manifest.csv against
# record_manifest.csv and checks every image file exists.
on_disk_images = {
    path.relative_to(KIT_ROOT).as_posix()
    for site in SITES
    if (KIT_ROOT / "images" / site).is_dir()
    for path in (KIT_ROOT / "images" / site).rglob("*.png")
}
if not on_disk_images:
    raise RuntimeError(f"No PNG images found under {KIT_ROOT / 'images'}")
print(f"extracted images on disk: {len(on_disk_images):,}")
FULL_METADATA_CSV = RECORD_MANIFEST_CSV
copy_seconds = time.time() - dataset_copy_started
scratch_free_gib = gib(shutil.disk_usage(EXTRACT_ROOT).free)
print(f"dataset staging time: {copy_seconds / 60:.1f} min")
print("final dataset path:", KIT_ROOT)
print(f"scratch free after staging: {scratch_free_gib:.1f} GiB")
log_event("dataset_ready", dataset_path=str(KIT_ROOT),
          copy_seconds=round(copy_seconds, 1), archive_sha256=EXPECTED_TAR_SHA256,
          scratch_device="colab-local-disk", scratch_free_gib=round(scratch_free_gib, 1))
print("Dataset is ready at", EXTRACT_ROOT)
"""),
    md(r"""## 4. Audit the clean12 manifests

This cell joins `manifest.csv` (render inventory) against `record_manifest.csv`
(the clean12 manifest: site, patient, physician-code labels) and checks the
corpus: 113,358 records, one image per record, twelve 0/1 label columns that
agree with the `labels` text and `n_labels`, per-label totals equal to the
verified `final_counts.csv`, NORM exclusive with every other label, and a
non-empty `patient_key` on every record. Records the renderer could not draw
(listed in `failures.csv`) are reported and left out. Every row is primary
(`primary_metric = 1`), so the checkpoint-selection metric covers all
validation rows.
"""),
    code(r"""manifest_df = pd.read_csv(MANIFEST_CSV, low_memory=False)
record_df = pd.read_csv(
    RECORD_MANIFEST_CSV, low_memory=False,
    dtype={"record_path": str, "site": str, "patient_key": str, "labels": str},
    keep_default_na=False,
)
for col in ("source_header", "image_path", "status"):
    if col not in manifest_df.columns:
        raise ValueError(f"manifest.csv is missing required column: {col}")
for col in ("record_path", "site", "patient_key", "labels", "n_labels", *LABELS):
    if col not in record_df.columns:
        raise ValueError(f"record_manifest.csv is missing required column: {col}")
if len(record_df) != EXPECTED_RECORDS:
    raise ValueError(f"record_manifest.csv has {len(record_df):,} records; expected {EXPECTED_RECORDS:,}")
if record_df["record_path"].duplicated().any():
    raise ValueError("record_manifest.csv contains duplicate record_path values")
record_df["patient_key"] = record_df["patient_key"].astype(str).str.strip()
if record_df["patient_key"].eq("").any() or record_df["patient_key"].str.lower().isin({"nan", "none"}).any():
    raise ValueError("record_manifest.csv has records without a patient_key; the split would leak")
unexpected_sites = sorted(set(record_df["site"]) - set(SITES))
if unexpected_sites:
    raise ValueError(f"Unexpected sites in record_manifest.csv: {unexpected_sites}")
observed_sites = record_df["site"].value_counts().to_dict()
if observed_sites != EXPECTED_RECORDS_BY_SITE:
    raise ValueError(f"Records per site {observed_sites} != expected {EXPECTED_RECORDS_BY_SITE}")

# The twelve label columns are the target. Cross-check them against the
# `labels` text, `n_labels`, and the verified totals from the download check.
for label in LABELS:
    values = pd.to_numeric(record_df[label], errors="raise")
    if not values.isin([0, 1]).all():
        raise ValueError(f"record_manifest.csv column {label} is not 0/1")
    record_df[label] = values.astype(np.int8)
column_sets = record_df[LABELS].apply(lambda row: frozenset(row.index[row.eq(1)]), axis=1)
text_sets = record_df["labels"].map(lambda text: frozenset(p.strip() for p in text.split(";") if p.strip()))
if not column_sets.eq(text_sets).all():
    bad = record_df.loc[~column_sets.eq(text_sets), ["record_path", "labels"] + LABELS].head()
    raise ValueError("Label columns disagree with the labels text:\n" + bad.to_string())
if not record_df[LABELS].sum(axis=1).eq(pd.to_numeric(record_df["n_labels"])).all():
    raise ValueError("Label columns disagree with n_labels")
final_counts = pd.read_csv(FINAL_COUNTS_CSV).set_index("condition")["total"].astype(int).to_dict()
observed_counts = {label: int(record_df[label].sum()) for label in LABELS}
if final_counts != observed_counts:
    raise ValueError(f"Label totals {observed_counts} differ from final_counts.csv {final_counts}")
cross_site_patients = int((record_df.groupby("patient_key")["site"].nunique() > 1).sum())
print(f"record_manifest: {len(record_df):,} records, {record_df['patient_key'].nunique():,} patients, "
      f"sites {observed_sites}; patient keys at both sites: {cross_site_patients}")
print("label totals match final_counts.csv:", observed_counts)

# Usable image rows: generation statuses success/skipped carry a real PNG.
# Anything else (invalid/failed) has no image and cannot train.
manifest_df["status"] = manifest_df["status"].astype(str).str.strip().str.lower()
usable_manifest = manifest_df[manifest_df["status"].isin(["success", "skipped"])].copy()
unusable = manifest_df[~manifest_df.index.isin(usable_manifest.index)]
print(f"manifest rows: {len(usable_manifest):,} usable, {len(unusable):,} without image")
if len(unusable):
    print(unusable["status"].value_counts().to_string())
    print("first unusable rows:")
    print(unusable.head(10).to_string())

# Join images to their record labels on the record (header) path.
record_key = record_df["record_path"].astype(str)
manifest_key = usable_manifest["source_header"].astype(str)
if manifest_key.duplicated().any():
    raise ValueError("manifest.csv contains duplicate source_header values")
joined = usable_manifest.merge(
    record_df, left_on="source_header", right_on="record_path",
    how="inner", suffixes=("", "_record"),
)
missing_labels = set(manifest_key) - set(record_key)
missing_images = set(record_key) - set(manifest_key)
print(f"joined image+label rows: {len(joined):,}")
if missing_labels:
    raise RuntimeError(f"{len(missing_labels):,} manifest images have no record_manifest entry")
# Records without a usable image are expected only when generation reported
# failures; they keep their patient's split slot but cannot train or evaluate.
if missing_images:
    print(f"WARNING: {len(missing_images):,} records have no usable image (see failures.csv)")
    pd.DataFrame({"record_path": sorted(missing_images)}).to_csv(
        RUN_DIR / "records_without_image.csv", index=False)

# NORM must be exclusive with every other label (cleaned upstream).
pathology_sum = joined[PATHOLOGY_LABELS].sum(axis=1)
normal_ok = joined[NORMAL_LABEL].eq(1) == pathology_sum.eq(0)
normal_ok = normal_ok | (joined[LABELS].sum(axis=1) == 0)
if not normal_ok.all():
    bad = joined.loc[~normal_ok, ["source_header", NORMAL_LABEL] + PATHOLOGY_LABELS].head()
    raise ValueError("NORM/pathology target contradiction in record_manifest.csv: " + str(bad.to_dict()))

# Every clean12 record carries at least one label; exclude (and save) any row
# that does not, so every training row carries >= 1 positive.
excluded_no_label = joined[joined[LABELS].sum(axis=1) == 0].copy()
joined = joined[joined[LABELS].sum(axis=1) > 0].reset_index(drop=True)
print(f"excluded no-target rows: {len(excluded_no_label):,}; kept rows: {len(joined):,}")
if len(excluded_no_label):
    excluded_no_label.to_csv(RUN_DIR / "excluded_no_target_label.csv", index=False)

# Frame columns used by the harness below.
# record_id/dup_group_id: one image per record, each record its own group.
# source_dataset: HEEDB acquisition site (I0001/I0006). primary_metric: all rows primary.
joined["record_id"] = joined["record_path"].astype(str)
joined["dup_group_id"] = joined["record_id"]
joined["patient_id"] = joined["patient_key"].astype(str)
joined["source_dataset"] = joined["site"].astype(str)
joined["variant_id"] = "clean"
joined["variant_type"] = "clean"
joined["primary_metric"] = 1
joined["metric_note"] = "clean12: single source, every row primary"
joined["image_path"] = joined["image_path"].astype(str)
metadata_df = joined
METADATA_SHA256 = sha256_file(RECORD_MANIFEST_CSV)
MANIFEST_SHA256 = sha256_file(MANIFEST_CSV)
"""),
    md(r"""## 4b. Patient-level train/validation/test split

The archive is unsplit, so the split is built here, over **patients**: every
`patient_key` goes to exactly one split and takes all its ECGs with it. The
split is computed from all 113,358 records in `record_manifest.csv` (not only
the rendered ones), so a render failure never moves a patient.

Method: multi-label iterative stratification (Sechidis et al., 2011) applied to
patient groups. Repeatedly take the label with the fewest unassigned positive
records; give each unassigned patient carrying it to the split that still needs
the most of that label (ties: most records still needed, then a seeded hash).
Visiting order and ties use `sha256(seed:patient_key)`, so the result depends
only on the manifest, the 80/10/10 ratios, and `CS_SPLIT_SEED` — not on row
order, library versions, or the training `CS_SEED`.

Checks (the cell raises if any fails): no patient in two splits; every split
within 1 point of its record share; every label within 1.5 points of its target
share in every split; and, with the default seed, the split's SHA-256 equals
`4eba80c7…`, the frozen split every clean12 run uses. So all models are
evaluated on exactly the same patients.

Source: `CS_SPLIT_FILE` (a local or Drive path of a `patient_key,split` CSV) when
set, else computed here. `patient_split.csv`, `record_split.csv`, and `split_audit.json`
are saved in the run folder on Drive. The split's SHA-256 is part of the
training identity, so a run cannot resume on a different split.
"""),
    code(r"""def split_tiebreak(seed, keys):
    # Version-independent per-patient order: no NumPy RNG stream involved.
    return np.array([
        int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")
        for key in keys
    ], dtype=np.uint64)

def stratified_patient_split(records, labels, ratios, seed):
    '''Assign every patient to one split; return a patient_key -> split Series.

    Multi-label iterative stratification over patient groups. All records of a
    patient move together, so no patient can span splits.
    '''
    names = list(ratios)
    target = np.asarray([ratios[name] for name in names], dtype=np.float64)
    if not math.isclose(float(target.sum()), 1.0) or (target <= 0).any():
        raise ValueError(f"Split ratios must be positive and sum to 1: {ratios}")
    grouped = records.groupby("patient_key", sort=True)
    counts = grouped[list(labels)].sum()
    keys = counts.index.to_numpy()
    positives = counts.to_numpy(np.float64)
    sizes = grouped.size().reindex(counts.index).to_numpy(np.float64)
    tie = split_tiebreak(seed, keys)
    need = target[:, None] * positives.sum(axis=0)[None, :]
    need_records = target * sizes.sum()
    assignment = np.full(len(keys), -1, dtype=np.int64)
    unassigned = np.ones(len(keys), dtype=bool)
    while unassigned.any():
        remaining = positives[unassigned].sum(axis=0)
        if (remaining > 0).any():
            column = int(np.argmin(np.where(remaining > 0, remaining, np.inf)))
            members = np.flatnonzero(unassigned & (positives[:, column] > 0))
        else:
            column = None  # patients without any label: balance records only
            members = np.flatnonzero(unassigned)
        members = members[np.argsort(tie[members], kind="stable")]
        for index in members:
            candidates = np.arange(len(names))
            if column is not None:
                score = need[:, column]
                candidates = candidates[score == score.max()]
            if len(candidates) > 1:
                score = need_records[candidates]
                candidates = candidates[score == score.max()]
            chosen = int(candidates[int(tie[index]) % len(candidates)])
            assignment[index] = chosen
            need[chosen] -= positives[index]
            need_records[chosen] -= sizes[index]
            unassigned[index] = False
    return pd.Series(np.asarray(names, dtype=object)[assignment],
                     index=pd.Index(keys, name="patient_key"), name="split")

def split_sha256(patient_split):
    # Canonical identity of an assignment: sorted "patient_key,split" lines.
    lines = [f"{key},{value}" for key, value in sorted(patient_split.items())]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()

def read_split_csv(path, label):
    frame = pd.read_csv(path, dtype={"patient_key": str, "split": str}, keep_default_na=False)
    if not {"patient_key", "split"} <= set(frame.columns):
        raise ValueError(f"{label} needs patient_key and split columns")
    if frame["patient_key"].duplicated().any():
        raise ValueError(f"{label} assigns a patient more than once")
    return frame.set_index("patient_key")["split"].rename("split")

def load_frozen_split(location):
    return read_split_csv(Path(location).expanduser(), location)

def split_support(records, patient_split):
    assigned = records.assign(split=records["patient_key"].map(patient_split))
    rows = {}
    for name in SUBSETS:
        frame = assigned[assigned["split"].eq(name)]
        rows[name] = {
            "records": int(len(frame)),
            "patients": int(frame["patient_key"].nunique()),
            "record_share": len(frame) / len(assigned),
            "sites": {site: int(n) for site, n in frame["site"].value_counts().sort_index().items()},
            "positives": {label: int(frame[label].sum()) for label in LABELS},
            "label_share": {label: float(frame[label].sum() / max(assigned[label].sum(), 1))
                            for label in LABELS},
        }
    return rows

split_started = time.time()
# Source: CS_SPLIT_FILE when set, else computed (checked against the frozen SHA-256 below).
if SPLIT_FILE:
    patient_split = load_frozen_split(SPLIT_FILE)
    split_source = f"frozen file {SPLIT_FILE}"
else:
    patient_split = stratified_patient_split(record_df, LABELS, SPLIT_RATIOS, SPLIT_SEED)
    split_source = "computed from record_manifest.csv"
all_patients = set(record_df["patient_key"])
missing_patients = all_patients - set(patient_split.index)
if missing_patients:
    raise ValueError(f"{len(missing_patients):,} patients have no split, e.g. {sorted(missing_patients)[:5]}")
extra_patients = set(patient_split.index) - all_patients
if extra_patients:
    raise ValueError(f"The split lists {len(extra_patients):,} patients absent from record_manifest.csv")
patient_split = patient_split.loc[sorted(all_patients)]
if not patient_split.isin(SUBSETS).all():
    raise ValueError(f"Unexpected split values: {sorted(set(patient_split) - set(SUBSETS))}")
SPLIT_SHA256 = split_sha256(patient_split)
if (not SPLIT_FILE and SPLIT_VERSION == "clean12_patient_split_v1" and SPLIT_SEED == 20261002
        and SPLIT_SHA256 != EXPECTED_SPLIT_SHA256):
    raise RuntimeError(
        f"Patient split SHA-256 {SPLIT_SHA256} != the frozen clean12 split "
        f"{EXPECTED_SPLIT_SHA256}. The results would not be on the same patients as the other clean12 runs "
        "patients; inspect record_manifest.csv before training."
    )

# Acceptance gates: shares per split and per label, before anything trains.
split_rows = split_support(record_df, patient_split)
split_problems = []
for name in SUBSETS:
    row = split_rows[name]
    if abs(row["record_share"] - SPLIT_RATIOS[name]) > SPLIT_RECORD_TOLERANCE:
        split_problems.append(f"{name}: {row['record_share']:.4f} of records, target {SPLIT_RATIOS[name]}")
    for label in LABELS:
        share = row["label_share"][label]
        if abs(share - SPLIT_RATIOS[name]) > SPLIT_LABEL_TOLERANCE:
            split_problems.append(f"{name}/{label}: {share:.4f} of positives, target {SPLIT_RATIOS[name]}")
        if row["positives"][label] == 0:
            split_problems.append(f"{name}/{label}: no positive records")
if split_problems:
    raise RuntimeError("Patient split failed its balance checks:\n- " + "\n- ".join(split_problems))

# A run directory keeps exactly one split: refuse to overwrite a different one.
# On a fresh runtime, first pull this run's split back from Drive (if it exists)
# so a resumed run is compared against its real split, never overwritten.
patient_split_path = RUN_DIR / "patient_split.csv"
split_audit_path = RUN_DIR / "split_audit.json"
if not patient_split_path.is_file():
    restore_from_drive(patient_split_path.name, patient_split_path)
if patient_split_path.is_file():
    existing_split = pd.read_csv(patient_split_path, dtype={"patient_key": str, "split": str},
                                 keep_default_na=False).set_index("patient_key")["split"]
    if split_sha256(existing_split) != SPLIT_SHA256:
        raise RuntimeError(
            f"{patient_split_path} holds a different patient split. Choose a new "
            "CS_RUN_GROUP, or pass that file as CS_SPLIT_FILE to reuse it."
        )
patient_table = pd.DataFrame({
    "patient_key": patient_split.index,
    "split": patient_split.to_numpy(),
}).merge(
    record_df.groupby("patient_key").agg(
        site=("site", lambda values: ";".join(sorted(set(values)))),
        records=("record_path", "size"),
    ).reset_index(),
    on="patient_key", how="left",
)
temporary = patient_split_path.with_suffix(".csv.tmp")
patient_table.to_csv(temporary, index=False)
os.replace(temporary, patient_split_path)
SPLIT_AUDIT = {
    "split_version": SPLIT_VERSION,
    "method": "multi-label iterative stratification over patient_key groups (Sechidis et al., 2011); "
              "visit order and ties by sha256(seed:patient_key)",
    "source": split_source,
    "split_seed": SPLIT_SEED,
    "ratios": SPLIT_RATIOS,
    "split_sha256": SPLIT_SHA256,
    "record_manifest_sha256": METADATA_SHA256,
    "records": int(len(record_df)),
    "patients": int(len(patient_split)),
    "patients_with_multiple_records": int((record_df.groupby("patient_key").size() > 1).sum()),
    "patient_keys_at_both_sites": cross_site_patients,
    "tolerances": {"record_share": SPLIT_RECORD_TOLERANCE, "label_share": SPLIT_LABEL_TOLERANCE},
    "splits": split_rows,
    "computed_at_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = split_audit_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(SPLIT_AUDIT, indent=2))
os.replace(temporary, split_audit_path)
# Record-level view (record_path -> split) for any other notebook, and a
# sha256sum-format sidecar for the patient table (the authority).
record_split_path = RUN_DIR / "record_split.csv"
temporary = record_split_path.with_suffix(".csv.tmp")
record_df.assign(split=record_df["patient_key"].map(patient_split))[
    ["record_path", "patient_key", "site", "split"]
].sort_values("record_path").to_csv(temporary, index=False)
os.replace(temporary, record_split_path)
patient_split_sidecar_path = RUN_DIR / "patient_split.csv.sha256"
patient_split_sidecar_path.write_text(f"{sha256_file(patient_split_path)}  {patient_split_path.name}\n")
print(f"patient split ({split_source}) in {time.time() - split_started:.1f}s; sha256 {SPLIT_SHA256}")
display(pd.DataFrame({
    name: {"records": row["records"], "patients": row["patients"],
           "record share": round(row["record_share"], 4), **row["sites"]}
    for name, row in split_rows.items()
}))
print("positive records per split (share of each label's total)")
display(pd.DataFrame({
    name: {label: f"{row['positives'][label]:,} ({row['label_share'][label]:.3f})" for label in LABELS}
    for name, row in split_rows.items()
}))

# Attach the split to the image rows; the frames below follow it.
metadata_df["split"] = metadata_df["patient_id"].map(patient_split)
if metadata_df["split"].isna().any():
    raise RuntimeError("Some image rows have no split; the patient split is incomplete")
metadata_df["subset"] = metadata_df["split"]

# Split integrity: one image per record, patient-disjoint splits.
observed_records = metadata_df.groupby("split")["record_id"].nunique().to_dict()
print("image records by split:", observed_records)
for key in ("record_id", "dup_group_id"):
    counts = metadata_df.groupby(key)["split"].nunique()
    if (counts > 1).any():
        raise ValueError(f"{key} crosses dataset splits")
patient_splits = metadata_df.groupby("patient_id")["split"].nunique()
if (patient_splits > 1).any():
    examples = patient_splits[patient_splits > 1].index[:10].tolist()
    raise ValueError(f"patient_id crosses dataset splits: {examples}")
split_patient_sets = {name: set(metadata_df.loc[metadata_df["split"].eq(name), "patient_id"])
                      for name in SUBSETS}
for first_index, first in enumerate(SUBSETS):
    for second in SUBSETS[first_index + 1:]:
        shared = split_patient_sets[first] & split_patient_sets[second]
        if shared:
            raise ValueError(f"{len(shared)} patients in both {first} and {second}")
print(f"patients: {metadata_df['patient_id'].nunique():,}; no patient spans splits")
log_event("patient_split", split_sha256=SPLIT_SHA256, source=split_source, seed=SPLIT_SEED,
          records_by_split=observed_records)
copy_run_files([patient_split_path, patient_split_sidecar_path, split_audit_path,
                record_split_path], required=True)
"""),
    md(r"""## 4c. Select the split frames and compute training-only class weights
"""),
    code(r"""# Selection: one clean image per record in every split.
train_df = metadata_df[metadata_df["split"].eq("train")].reset_index(drop=True)
val_df = metadata_df[metadata_df["split"].eq("validation")].reset_index(drop=True)
test_df = metadata_df[metadata_df["split"].eq("test")].reset_index(drop=True)
selected_frames = {"train": train_df, "validation": val_df, "test": test_df}
for name, frame in selected_frames.items():
    frame["local_path"] = frame["image_path"].map(lambda rel: str((KIT_ROOT / rel).resolve()))
    missing_files = [p for p in frame["local_path"] if not Path(p).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files):,} {name} images are missing after extraction; "
            f"first: {missing_files[:5]}. Re-run the extraction cell."
        )
    if frame["local_path"].duplicated().any():
        raise ValueError(f"Duplicate local image paths in the {name} selection")
    if frame["record_id"].duplicated().any():
        raise ValueError(f"Duplicate records in the {name} selection")

# Compute each output's class weight using training rows only.
train_positive = train_df[LABELS].sum(axis=0).astype(np.int64)
train_negative = len(train_df) - train_positive
if (train_positive == 0).any():
    raise ValueError(f"Training split has a label with zero positives: {train_positive.to_dict()}")

# Rare-label repeat factors (training rows only). An image's factor is the
# largest factor among its labels; every clean12 row has at least one label.
train_label_frequency = train_positive / len(train_df)
if RARE_SAMPLING:
    label_repeat_factor = np.maximum(
        1.0, np.sqrt(RARE_SAMPLING_THRESHOLD / train_label_frequency.astype(np.float64))
    )
else:
    label_repeat_factor = pd.Series(1.0, index=train_positive.index)
_train_targets = train_df[LABELS].to_numpy(dtype=np.float64)
TRAIN_REPEAT_FACTORS = np.maximum(
    1.0, (_train_targets * label_repeat_factor.to_numpy()[None, :]).max(axis=1)
)
# Expected positives/negatives per epoch under the sampler. pos_weight uses
# these, so rarity is not compensated twice (sampling AND loss weight).
effective_total = float(TRAIN_REPEAT_FACTORS.sum())
effective_positive = pd.Series(
    (_train_targets * TRAIN_REPEAT_FACTORS[:, None]).sum(axis=0), index=LABELS
)
effective_negative = effective_total - effective_positive
print(f"rare-label sampling: {'on' if RARE_SAMPLING else 'off'}; "
      f"expected draws per epoch {effective_total:,.0f} for {len(train_df):,} images "
      f"(+{effective_total / len(train_df) - 1:.1%})")

POS_WEIGHT_MODE = os.environ.get("CS_POS_WEIGHT_MODE", "sqrt").strip().lower()
if POS_WEIGHT_MODE not in ("full", "sqrt", "none"):
    raise ValueError(
        f"CS_POS_WEIGHT_MODE={POS_WEIGHT_MODE!r} must be one of full|sqrt|none."
    )
pos_weight_series = effective_negative / effective_positive
if POS_WEIGHT_MODE == "sqrt":
    pos_weight_series = np.sqrt(pos_weight_series)
elif POS_WEIGHT_MODE == "none":
    pos_weight_series = pd.Series(
        np.ones(len(train_positive)), index=train_positive.index
    )
print(f"pos_weight mode: {POS_WEIGHT_MODE} "
      f"(median weight {float(np.median(pos_weight_series.to_numpy())):.2f})")
POS_WEIGHT = torch.tensor(pos_weight_series.to_numpy(), dtype=torch.float32)

# Weighted BCE of the best constant predictor: per label, p = w*pi / (w*pi + 1 - pi)
# for pos_weight w and the prevalence pi the sampler actually feeds. A model
# that ignores the image cannot go below this, so the collapse guard in
# section 10 compares against it.
_prevalence = (effective_positive / effective_total).to_numpy(dtype=np.float64)
_label_weights = pos_weight_series.to_numpy(dtype=np.float64)
_constant_probability = _label_weights * _prevalence / (
    _label_weights * _prevalence + 1.0 - _prevalence
)
CONSTANT_PREDICTOR_LOSS = float(np.mean(-(
    _label_weights * _prevalence * np.log(_constant_probability)
    + (1.0 - _prevalence) * np.log1p(-_constant_probability)
)))
print(f"constant-predictor weighted BCE (collapse reference): {CONSTANT_PREDICTOR_LOSS:.4f}")

DATASET_MODE = "clean-only"
print("dataset mode:", DATASET_MODE)
split_table = pd.DataFrame([
    {"selection": name, "images": len(frame),
     "records": int(frame["record_id"].nunique()),
     "patients": int(frame["patient_id"].nunique())}
    for name, frame in selected_frames.items()
]).set_index("selection")
display(split_table)
print("positive image counts")
display(pd.concat(
    {name: frame[LABELS].sum(axis=0).astype(int)
     for name, frame in selected_frames.items()}, axis=1
))
print(f"training-only pos_weight policy: {POS_WEIGHT_MODE} (on resampled counts)")
display(pd.DataFrame({
    "positive": train_positive,
    "negative": train_negative,
    "prevalence": train_positive / len(train_df),
    "label_repeat_factor": label_repeat_factor,
    "sampled_prevalence": effective_positive / effective_total,
    "pos_weight": pos_weight_series,
}).round(4))
print("source x split (unique ECG records)")
display(pd.crosstab(
    metadata_df.drop_duplicates("record_id")["source_dataset"],
    metadata_df.drop_duplicates("record_id")["split"],
))

# Hard label-support gate: every label needs a positive and a negative row in
# each evaluation split, else ranking metrics would be undefined.
for name in ("validation", "test"):
    frame = selected_frames[name]
    support_table = pd.DataFrame({
        "positive": frame[LABELS].sum(axis=0).astype(int),
        "negative": (len(frame) - frame[LABELS].sum(axis=0)).astype(int),
    })
    support_table["prevalence"] = support_table["positive"] / len(frame)
    print(f"{name} label support")
    display(support_table.round(4))
    unsupported = support_table[
        (support_table["positive"] == 0) | (support_table["negative"] == 0)
    ].index.tolist()
    if unsupported:
        raise RuntimeError(
            f"{name} split lacks positive or negative support for "
            f"{unsupported}; refusing to train with undefined ranking metrics."
        )

# Record-level source/label audit (unique ECGs).
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

print("metadata rows:", f"{len(metadata_df):,}")
print("record_manifest sha256:", METADATA_SHA256)
print("manifest.csv sha256:", MANIFEST_SHA256)
print("archive sha256:", EXPECTED_TAR_SHA256)
print("patient split sha256:", SPLIT_SHA256)

"""),
    md(r"""## 5. Dataset and transforms

Every sample is decoded to RGB, resized to 768×1024 (H×W) with bicubic
interpolation, and batched as uint8 in pinned CPU memory. Conversion to float,
training brightness/contrast changes, and ImageNet normalization run on the
GPU. Training adds mild brightness and contrast changes (±6%) before normalization;
validation and test use deterministic transforms. No flips, crops, rotations,
mixup, or cutmix: lead positions, polarity, and the complete traces are retained.

**Rare-label sampling.** `RepeatFactorSampler` draws each training image
`floor(r)` times per epoch plus one extra draw with probability `r - floor(r)`
(systematic sampling over a seeded shuffle), so an epoch always has the same
length (~95.9k draws) and the LR schedule stays exact. The order depends only
on the seed and the epoch, so a resumed run replays it exactly. The sampler
yields *occurrence indices* `occurrence × N + row`: the dataset decodes the row,
and the augmentation draw is keyed by the occurrence, so the repeats of one
image get different brightness/contrast. With sampling off every image is drawn
once per epoch and occurrence = row.

Augmentation draws depend on the seed, epoch, and occurrence index, making them
stable across worker counts, persistent-worker reuse, and epoch-boundary resumes.
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
    transforms.PILToTensor(),  # keep uint8: 2.25 MiB per native sheet, not 9 MiB
])
GPU_MEAN = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
GPU_STD = torch.tensor(IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
GPU_LUMINANCE = torch.tensor((0.2989, 0.5870, 0.1140), device=DEVICE).view(1, 3, 1, 1)

EVAL_METADATA_KEYS = ("record_id", "source", "split", "subset",
                      "variant_id", "primary_metric")

def training_augmentation_factors(index, epoch):
    # Local NumPy generator never changes the global RNG or a worker RNG.
    # The epoch is shared with forked workers; per-row factors are replayable.
    row_seed = (SEED + (int(index) + 1) * 1000003 + int(epoch) * 1000033) % 2**32
    rng = np.random.default_rng(row_seed)
    brightness = float(rng.uniform(1 - TRAIN_BRIGHTNESS_JITTER, 1 + TRAIN_BRIGHTNESS_JITTER))
    contrast = float(rng.uniform(1 - TRAIN_CONTRAST_JITTER, 1 + TRAIN_CONTRAST_JITTER))
    return torch.tensor((brightness, contrast), dtype=torch.float32)

def prepare_image_batch(images, augmentation_factors=None):
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

class ECGImageDataset(Dataset):
    # Paths/targets/metadata are precomputed: no pandas in the hot path.
    # Training rows return (uint8 image, target, two augmentation factors).
    # Metadata dicts are only returned by validation/test rows:
    # (image, target, metadata) assembled from precomputed lists, so
    # evaluation reports keep full provenance at negligible per-sample cost.
    # The frame column is `source_dataset`; the metadata key is `source` to
    # match PREDICT_METADATA_KEYS and every downstream consumer
    # (predict_loader, per-source evaluation, prediction frames).
    def __init__(self, frame: pd.DataFrame, return_metadata: bool, training: bool = False):
        frame = frame.reset_index(drop=True)
        self.paths = frame["local_path"].astype(str).tolist()
        self.targets = frame[LABELS].to_numpy(dtype=np.float32, copy=True)
        self.return_metadata = return_metadata
        self.training = bool(training)
        self.epoch = torch.zeros(1, dtype=torch.int64).share_memory_()
        if return_metadata:
            self.meta = {}
            for key in EVAL_METADATA_KEYS:
                column = "source_dataset" if key == "source" else key
                self.meta[key] = frame[column].tolist()
            self.meta["primary_metric"] = [int(v) for v in self.meta["primary_metric"]]
            for key in ("record_id", "source", "split", "subset", "variant_id"):
                self.meta[key] = [str(v) for v in self.meta[key]]
        else:
            self.meta = None

    def __len__(self):
        return len(self.targets)

    def set_epoch(self, epoch):
        self.epoch[0] = int(epoch)

    def __getitem__(self, index):
        # Training indices may be occurrence indices from RepeatFactorSampler
        # (occurrence * N + row): decode the row, key augmentation by the
        # occurrence so repeats of one image differ. For rows < N both match.
        occurrence = int(index)
        index = occurrence % len(self.targets)
        with Image.open(self.paths[index]) as image:
            image = image.convert(IMAGE_COLOR_MODE)
            tensor = image_transform(image)
        target = torch.from_numpy(self.targets[index])
        if not self.return_metadata:
            factors = training_augmentation_factors(occurrence, self.epoch[0].item()) if self.training else torch.ones(2)
            return tensor, target, factors
        return tensor, target, {
            "row_index": int(index),
            **{key: self.meta[key][index] for key in EVAL_METADATA_KEYS},
        }

class RepeatFactorSampler(Sampler):
    '''Epoch-keyed repeat-factor sampler with a fixed epoch length.

    Each row is drawn floor(r) times, plus once more for exactly
    floor(sum(r - floor(r))) rows picked by systematic sampling over a shuffled
    order, so row i gets its extra draw with probability (r_i - floor(r_i))
    scaled by floor(F)/F for F = sum of fractions (a < 1-draw shortfall spread
    over all rows). Yields occurrence * N + row, shuffled. The order is a pure
    function of (seed, epoch): resumes replay it exactly.
    '''

    def __init__(self, repeat_factors, seed):
        factors = np.asarray(repeat_factors, dtype=np.float64)
        if factors.ndim != 1 or len(factors) == 0 or (factors < 1.0).any():
            raise ValueError("repeat factors must be a non-empty 1-D array of values >= 1")
        self.rows = len(factors)
        self.whole = np.floor(factors).astype(np.int64)
        self.fraction = factors - self.whole
        self.extra = int(math.floor(self.fraction.sum() + 1e-9))
        self.length = int(self.whole.sum()) + self.extra
        self.seed = int(seed)
        self.epoch = 1

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.length

    def epoch_indices(self, epoch):
        rng = np.random.default_rng([self.seed, 104729, int(epoch)])
        counts = self.whole.copy()
        if self.extra:
            # Systematic sampling: points u, u+1, ... over the cumulative
            # fractions of a random order. Each fraction is < 1, so no row is
            # hit twice, and a row is hit with probability equal to its fraction.
            order = rng.permutation(self.rows)
            cumulative = np.cumsum(self.fraction[order])
            points = rng.random() + np.arange(self.extra)
            counts[order[np.searchsorted(cumulative, points, side="right")]] += 1
        rows = np.repeat(np.arange(self.rows, dtype=np.int64), counts)
        first = np.repeat(np.cumsum(counts) - counts, counts)
        occurrence = np.arange(len(rows), dtype=np.int64) - first
        return (occurrence * self.rows + rows)[rng.permutation(len(rows))]

    def __iter__(self):
        return iter(self.epoch_indices(self.epoch).tolist())

train_dataset = ECGImageDataset(train_df, return_metadata=False, training=True)
train_sampler = RepeatFactorSampler(TRAIN_REPEAT_FACTORS, SEED)
print(f"training draws per epoch: {len(train_sampler):,} "
      f"({len(train_sampler) / len(train_dataset) - 1:+.1%} vs one pass over {len(train_dataset):,} images)")
val_dataset = ECGImageDataset(val_df, return_metadata=True)
test_dataset = ECGImageDataset(test_df, return_metadata=True)

# Decode a few real samples now, before the long run.
for dataset_name, dataset in (
    [("train", train_dataset), ("validation", val_dataset), ("test", test_dataset)]
):
    sample = dataset[0]
    image, target = sample[0], sample[1]
    metadata = sample[2] if len(sample) > 2 else {"rows": len(dataset)}
    print(dataset_name, tuple(image.shape), target.tolist(), metadata)
"""),
    code(r"""# Per-label visual QA: one training image per target label.
# Run after the download/audit cells to inspect one full sheet per label;
# save the sample grid to the run group folder and Drive.
import math
import matplotlib.pyplot as plt

samples = []
for label in LABELS:
    hits = train_df[train_df[label].eq(1)]
    if hits.empty:
        raise RuntimeError(f"No training image carries label {label!r}")
    row = hits.iloc[0]
    samples.append((label, row))

ncols = 3
nrows = math.ceil(len(samples) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
axes = axes.ravel()
for axis, (label, row) in zip(axes, samples):
    path = Path(row.local_path)
    if not path.is_file():
        raise FileNotFoundError(f"Sample image missing: {path}")
    axis.imshow(np.asarray(Image.open(path).convert("RGB")))
    axis.set_title(f"{label} :: {row.record_id.split('/')[-1][:28]} ({row.source_dataset})", fontsize=9)
    axis.axis("off")
for axis in axes[len(samples):]:
    axis.axis("off")
fig.suptitle(f"clean12 training samples: one image per label (n={len(samples)})", fontsize=13)
plt.tight_layout()
sample_grid_path = RUN_DIR / "sample_grid_per_label.png"
fig.savefig(sample_grid_path, dpi=140, bbox_inches="tight")
plt.show()
print(f"saved {sample_grid_path} ({sample_grid_path.stat().st_size / 2**20:.1f} MiB)")
print("label -> record_id -> image_path")
for label, row in samples:
    print(f"- {label}: {row.record_id} :: {row.image_path}")
copy_to_drive(sample_grid_path, required=True)
print("Inspect the sample grid; it is shared by every architecture below.")
"""),
    md(r"""## 6. Architectures, loss, and optimizer

The three CNNs come from timm with their default ImageNet-1K weights and a new
12-way linear head (timm's default head init). This cell downloads all three
weight files up front, so a Hugging Face problem shows up now and not hours
later, and records for each model its parameter count, normalization, and
compute per 768×1024 sheet.

Loss: square-root-weighted BCE (`CONTRADICTION_LAMBDA = 0.0`), the same as the
ConvNeXt clean12 runs. Optimizer: AdamW with one LR for every layer. Two
parameter groups: conv/linear weights with weight decay, and BatchNorm
weights plus all biases without it.

`torch.compile` caches are per architecture, so dynamo's recompile limit is
raised for the run (train mode, eval mode and the partial last evaluation
batch are each a separate graph, for the model and its EMA copy).
"""),
    code(r"""from types import SimpleNamespace
from torch.utils.flop_counter import FlopCounterMode
from timm.optim import param_groups_weight_decay

import torch._dynamo
torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
    torch._dynamo.config.accumulated_cache_size_limit = max(
        512, torch._dynamo.config.accumulated_cache_size_limit
    )

def set_input_normalization(mean, std):
    '''Point prepare_image_batch (section 5) at one model's normalization.'''
    global GPU_MEAN, GPU_STD
    GPU_MEAN = torch.tensor(mean, device=DEVICE, dtype=torch.float32).view(1, 3, 1, 1)
    GPU_STD = torch.tensor(std, device=DEVICE, dtype=torch.float32).view(1, 3, 1, 1)

def build_model(spec, pretrained: bool) -> nn.Module:
    return timm.create_model(
        spec["model_name"],
        pretrained=pretrained,
        # Explicit cache_dir also covers libraries imported before HF_HOME was set.
        cache_dir=str(HF_HOME) if pretrained else None,
        num_classes=len(LABELS),
        drop_rate=DROP_RATE,
    )

def build_model_ema(active_model):
    ema = ModelEmaV3(active_model, decay=EMA_DECAY, use_warmup=True, foreach=True)
    ema.num_updates = 0
    return ema

def update_model_ema(ema, active_model):
    ema.num_updates += 1
    ema.update(active_model, step=ema.num_updates)

def configure_model_compilation(active_model, enabled, backend=None):
    # Compile in place: checkpoint keys and optimizer parameter identities stay
    # unchanged. Construct EMA BEFORE calling this so its copy stays independent.
    if not enabled:
        active_model._compiled_call_impl = None
        return
    if not hasattr(active_model, "compile"):
        raise RuntimeError("In-place module compilation requires PyTorch >=2.1")
    options = {"dynamic": False}
    if backend is None:
        options["mode"] = TORCH_COMPILE_MODE
    else:
        options["backend"] = backend
    active_model.compile(**options)

class CardioSentryLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, contradiction_lambda: float):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight.clone().float())
        self.contradiction_lambda = float(contradiction_lambda)
        self.normal_index = LABELS.index(NORMAL_LABEL)
        self.pathology_indices = [LABELS.index(label) for label in PATHOLOGY_LABELS]

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="mean"
        )
        probabilities = torch.sigmoid(logits.float())
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
    # One LR for every layer. Weight decay on conv/linear weights only:
    # timm puts every 1-D parameter (BatchNorm weight/bias, conv/linear bias)
    # in the no-decay group.
    parameter_groups = param_groups_weight_decay(active_model, weight_decay=WEIGHT_DECAY)
    expected = {id(parameter) for parameter in active_model.parameters() if parameter.requires_grad}
    grouped_ids = [id(parameter) for group in parameter_groups for parameter in group["params"]]
    if set(grouped_ids) != expected or len(grouped_ids) != len(expected):
        raise RuntimeError("Optimizer grouping does not cover every trainable parameter exactly")
    for group in parameter_groups:
        group["lr"] = BASE_LR
    return torch.optim.AdamW(
        parameter_groups, lr=BASE_LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY,
        fused=USE_FUSED_ADAMW and next(active_model.parameters()).is_cuda,
    )

def make_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=USE_AMP and AMP_DTYPE == torch.float16)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=USE_AMP and AMP_DTYPE == torch.float16)

def autocast_context():
    return torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP)

def optimizer_update(active_model, active_optimizer, active_scaler, active_ema):
    active_scaler.unscale_(active_optimizer)
    torch.nn.utils.clip_grad_norm_(
        active_model.parameters(), GRAD_CLIP_NORM,
        error_if_nonfinite=not active_scaler.is_enabled(),
    )
    previous_scale = active_scaler.get_scale()
    active_scaler.step(active_optimizer)
    active_scaler.update()
    if active_scaler.get_scale() >= previous_scale:
        update_model_ema(active_ema, active_model)
    active_optimizer.zero_grad(set_to_none=True)

def release_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@torch.no_grad()
def count_forward_gmacs(active_model):
    '''Multiply-accumulates of one 768x1024 forward pass (FLOPs / 2), on the GPU.'''
    active_model = active_model.to(DEVICE).eval()
    sample = torch.zeros(1, 3, *INPUT_SIZE, device=DEVICE)
    counter = FlopCounterMode(display=False)
    with counter:
        active_model(sample)
    return counter.get_total_flops() / 2 / 1e9

criterion_cpu = CardioSentryLoss(POS_WEIGHT, CONTRADICTION_LAMBDA)
dummy_logits = torch.zeros(2, len(LABELS))
dummy_targets = torch.zeros(2, len(LABELS))
dummy_targets[0, LABELS.index(NORMAL_LABEL)] = 1.0
dummy_targets[1, 1:] = 1.0
dummy_loss, dummy_parts = criterion_cpu(dummy_logits, dummy_targets)
# All-zero logits give p=0.5 everywhere: 0.5 * (1 - 0.5**11).
assert torch.isclose(
    dummy_parts["contradiction"], torch.tensor(0.5 * (1.0 - 0.5 ** len(PATHOLOGY_LABELS)))
)
print("loss smoke test:", float(dummy_loss), {k: float(v) for k, v in dummy_parts.items()})

# Download every selected model's weights now and describe each architecture.
ARCHITECTURE_INFO = {}
for arch in ARCHITECTURES:
    spec = ARCHITECTURE_SPECS[arch]
    started = time.time()
    candidate = build_model(spec, pretrained=True)
    cfg = candidate.pretrained_cfg
    if candidate.get_classifier().out_features != len(LABELS):
        raise RuntimeError(f"{arch}: classifier has {candidate.get_classifier().out_features} outputs")
    try:
        gmacs = count_forward_gmacs(candidate)
    except Exception as exc:  # informational only
        warnings.warn(f"{arch}: MAC count failed ({exc})")
        gmacs = float("nan")
    ARCHITECTURE_INFO[arch] = {
        "display_name": spec["display_name"],
        "model_name": spec["model_name"],
        "pretrained_tag": cfg.get("tag"),
        "pretrained_hf_hub_id": cfg.get("hf_hub_id"),
        "pretraining": spec["pretraining"],
        "pretrained_input_size": list(cfg.get("input_size", ())),
        "normalization_mean": [float(v) for v in cfg.get("mean", IMAGENET_MEAN)],
        "normalization_std": [float(v) for v in cfg.get("std", IMAGENET_STD)],
        "parameters_m": sum(p.numel() for p in candidate.parameters()) / 1e6,
        "gmacs_per_sheet": gmacs,
        "classifier": type(candidate.get_classifier()).__name__,
    }
    print(f"{arch:16s} {spec['model_name']:28s} {ARCHITECTURE_INFO[arch]['parameters_m']:6.2f} M params, "
          f"{gmacs:6.1f} GMACs per 768x1024 sheet, mean {ARCHITECTURE_INFO[arch]['normalization_mean']} "
          f"({time.time() - started:.0f}s)")
    del candidate
    release_cuda_memory()
architecture_info_path = RUN_DIR / "architectures.json"
architecture_info_path.write_text(json.dumps(ARCHITECTURE_INFO, indent=2))
copy_to_drive(architecture_info_path, required=True)
"""),
    md(r"""## 7. Tune the PNG decoder workers (once, shared by every model)

Decoding speed does not depend on the network, so the 4/6/8/10-worker timing
runs once here at a 16-image batch (uint8 batches, prefetch 4); an explicit
`CS_NUM_WORKERS` skips it. The training and evaluation batches depend on each
network's memory use, so they are probed per architecture in section 9.
"""),
    code(r"""def seed_data_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.set_num_threads(1)

def make_loader_options(batch_size, workers, persistent=True):
    return dict(
        batch_size=batch_size, num_workers=workers, pin_memory=DEVICE.type == "cuda",
        persistent_workers=persistent and workers > 0,
        prefetch_factor=PREFETCH_FACTOR if workers > 0 else None,
        worker_init_fn=seed_data_worker,
        multiprocessing_context=DATALOADER_START_METHOD if workers > 0 else None,
    )

def tune_decoder_workers(dataset, batch_size, candidates):
    rows = []
    for workers in candidates:
        loader = iterator = None
        batch = None
        try:
            generator = torch.Generator().manual_seed(SEED)
            loader = DataLoader(dataset, shuffle=True, generator=generator,
                                **make_loader_options(batch_size, workers, persistent=False))
            iterator = iter(loader)
            warmup = min(2, max(0, len(loader) - 1))
            for _ in range(warmup):
                batch = next(iterator)
            timed = min(WORKER_BENCHMARK_BATCHES, len(loader) - warmup)
            if timed < 1:
                raise RuntimeError("Too few batches to benchmark image decoding")
            started = time.perf_counter()
            images = 0
            for _ in range(timed):
                batch = next(iterator)
                images += len(batch[0])
            elapsed = time.perf_counter() - started
            rows.append({"workers": workers, "decoded_images_per_second": images / elapsed,
                         "timed_batches": timed})
        finally:
            # Non-persistent iterator destruction joins its workers.
            del batch, iterator, loader
            gc.collect()
    best = max(rows, key=lambda row: row["decoded_images_per_second"])
    return best["workers"], pd.DataFrame(rows)

if AUTO_TUNE_WORKERS:
    NUM_WORKERS, worker_benchmark = tune_decoder_workers(train_dataset, WORKER_TUNING_BATCH, WORKER_CANDIDATES)
else:
    worker_benchmark = pd.DataFrame([{"workers": NUM_WORKERS, "selection": "explicit_or_fixed"}])
display(worker_benchmark)
print("selected decoder workers:", NUM_WORKERS)
WORKER_TUNING = {"num_workers": NUM_WORKERS, "tuning_batch": WORKER_TUNING_BATCH,
                 "prefetch_factor": PREFETCH_FACTOR,
                 "worker_benchmark": worker_benchmark.to_dict("records")}
(RUN_DIR / "worker_benchmark.json").write_text(json.dumps(WORKER_TUNING, indent=2))
copy_to_drive(RUN_DIR / "worker_benchmark.json", required=True)
seed_global_rngs(SEED)
"""),
    md(r"""## 8. Metrics

Threshold-free ranking metrics (Average Precision, AUROC) carry checkpoint
selection; thresholded metrics use the fixed-0.5 baseline plus per-label F1
thresholds tuned on validation only. Undefined metrics stay missing with an
explicit reason — they are never converted to zero. Exact-combination
reporting covers the most common test patterns with support and a low-support
flag. clean12 has ~78k single-label and ~35k multi-label records, so the common
patterns are single conditions and their frequent pairs (e.g. STEMI + STE).
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
    normal_prediction = prediction[:, LABELS.index(NORMAL_LABEL)].astype(bool)
    pathology_prediction = prediction[:, [LABELS.index(label) for label in PATHOLOGY_LABELS]].astype(bool)
    result["normal_pathology_contradiction_rate"] = float(
        (normal_prediction & pathology_prediction.any(axis=1)).mean()
    )
    result["predicted_multi_pathology_rate"] = float((pathology_prediction.sum(axis=1) >= 2).mean())
    pathology_probability = y_probability[:, [LABELS.index(label) for label in PATHOLOGY_LABELS]]
    any_pathology_probability = 1.0 - np.prod(1.0 - pathology_probability, axis=1)
    result["mean_soft_normal_any_pathology_product"] = float(
        (y_probability[:, LABELS.index(NORMAL_LABEL)] * any_pathology_probability).mean()
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
    normal_prediction = prediction[:, LABELS.index(NORMAL_LABEL)].astype(bool)
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
    md(r"""## 9. Per-architecture pipeline

The next four cells only **define** the pipeline; section 10 runs it once per
architecture. For one model, `run_architecture(arch)` does:

1. **Skip check.** It copies the model's small result files back from Drive.
   If `evaluation_complete.json` matches this configuration's fingerprint, the
   model is finished and its results are reloaded for the comparison. Nothing
   retrains.
2. **Probes.** It picks the largest training micro-batch (64/32/16/…) whose
   full step (forward, backward, AdamW, EMA) stays under 90% of VRAM. Then a
   forward-only probe picks the evaluation batch with the training state still
   resident.
3. **Loaders.** It builds the train/validation/test loaders at those batch
   sizes, with the shared worker count.
4. **Eager vs `torch.compile`.** It times both on real batches with disposable
   copies. Compiled is kept only if it fits memory and is at least 2% faster
   per projected epoch.
5. **Training with resume.** This is the clean12 harness of the ConvNeXt
   notebooks: per-epoch `last.pt`, `best.pt` on every new best, raw vs EMA
   selection on validation pathology Macro AP, early stopping, collapse guard,
   and resume from Drive after a disconnect.
6. **Final evaluation.** Thresholds are tuned on validation only. Then the test
   split is scored once, with patient-bootstrap CIs, per-site and
   combination tables, `inference.pt`, `report.md` and the curves.
7. **Clean-up.** It frees the GPU, the loader workers and the compiled graphs
   before the next model.

A model that fails writes `<arch>/failure.json`. With `CS_CONTINUE_ON_ERROR=1`
(the default) the next model still runs, and the comparison lists the failure.

The micro-batch is runtime metadata, not identity, so a model may resume on an
A100 of the other size. With BatchNorm, a different micro-batch changes the
batch statistics, so a resume at a different micro-batch prints a warning.
"""),
    code(r"""class TrainingCollapseError(RuntimeError):
    pass

ARCH_RESULTS = {}
ARCH_FAILURES = {}
LOSS_COMPONENT_KEYS = ("loss", "bce", "contradiction", "weighted_contradiction")
PREDICT_METADATA_KEYS = ("row_index", "record_id", "source", "split", "subset",
                         "variant_id", "primary_metric")
# Restored for every model, finished or not: enough for the skip check and
# the comparison in section 11 without copying checkpoints back.
SMALL_RESULT_FILES = ("evaluation_complete.json", "summary.json", "result.json",
                      "test_bootstrap_draws.npy", "test_bootstrap_ci.csv", "training_history.csv",
                      "training_complete.json", "config_snapshot.json")
CHECKPOINT_FILES = ("last.pt", "best.pt", "last_intra.pt", "benchmark.json",
                    "input_pipeline_benchmark.json")

def format_duration(seconds):
    seconds = float(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes):02d}m"
    if minutes >= 1:
        return f"{int(minutes)}m {int(secs):02d}s"
    return f"{secs:.1f}s"

def atomic_torch_save(payload, destination: Path):
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

def architecture_identity(arch):
    '''Everything that defines one model's training; hashed into its fingerprint.'''
    spec = ARCHITECTURE_SPECS[arch]
    info = ARCHITECTURE_INFO[arch]
    return {
        "run_group": RUN_NAME,
        "architecture": arch,
        "model": spec["model_name"],
        "pretraining": spec["pretraining"],
        "pretrained_hf_hub_id": info["pretrained_hf_hub_id"],
        "head": "timm default classifier, re-initialised for 12 outputs",
        "label_order": LABELS,
        "dataset": "CardioSentry HEEDB clean12 ECG images (1024x768 PNG, 500 Hz records, 12 labels)",
        "dataset_archive": DATASET_TAR_NAME,
        "dataset_archive_sha256": EXPECTED_TAR_SHA256,
        "dataset_metadata_sha256": METADATA_SHA256,
        "dataset_manifest_sha256": MANIFEST_SHA256,
        "dataset_mode": DATASET_MODE,
        "split": {"version": SPLIT_VERSION, "unit": "patient_key", "seed": SPLIT_SEED,
                  "ratios": SPLIT_RATIOS, "sha256": SPLIT_SHA256},
        "train_variants": TRAIN_VARIANTS,
        "rare_label_sampling": {
            "enabled": RARE_SAMPLING,
            "threshold": RARE_SAMPLING_THRESHOLD if RARE_SAMPLING else None,
            "label_repeat_factors": {label: float(value) for label, value in label_repeat_factor.items()},
            "draws_per_epoch": len(train_sampler),
        },
        "input_size_hw": list(INPUT_SIZE),
        "input_color_mode": IMAGE_COLOR_MODE,
        "resize_interpolation": RESIZE_INTERPOLATION.value,
        "resize_antialias": RESIZE_ANTIALIAS,
        "normalization_mean": info["normalization_mean"],
        "normalization_std": info["normalization_std"],
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_epochs": EARLY_STOPPING_MIN_EPOCHS,
        "monitor": "validation_primary_pathology_only_macro_average_precision",
        "monitor_mode": "max",
        "reported_threshold_policies": ["fixed_0_5", "validation_tuned_per_label_f1"],
        "optimizer": "AdamW, one LR for all layers",
        "base_lr": BASE_LR,
        "min_lr_ratio": MIN_LR_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "weight_decay_policy": "conv/linear weights only; no decay on 1-D params (BatchNorm, biases)",
        "warmup_epochs": WARMUP_EPOCHS,
        "schedule": "linear_warmup_then_cosine",
        "scheduler_step_policy": "advance only after a successful optimizer update",
        "drop_rate": DROP_RATE,
        "drop_path_rate": DROP_PATH_RATE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "ema": {"decay": EMA_DECAY, "use_warmup": True,
                "update_policy": "successful_optimizer_steps_only", "select_raw_or_ema": True},
        "loss": {"full": "BCEWithLogitsLoss(pos_weight=train negatives/positives)",
                 "sqrt": "BCEWithLogitsLoss(pos_weight=sqrt(train negatives/positives))",
                 "none": "BCEWithLogitsLoss(unweighted)"}[POS_WEIGHT_MODE],
        "pos_weight_mode": POS_WEIGHT_MODE,
        "train_pos_weight": dict(zip(LABELS, map(float, POS_WEIGHT.tolist()))),
        "contradiction_lambda": CONTRADICTION_LAMBDA,
        "online_augmentation": {
            "brightness": TRAIN_BRIGHTNESS_JITTER, "contrast": TRAIN_CONTRAST_JITTER,
            "geometry": "complete native sheet; no spatial augmentation",
        },
        "amp": "BF16 if supported, otherwise FP16 with dynamic scaling",
        "tf32": not DETERMINISTIC,
        "deterministic_algorithms": DETERMINISTIC,
        "channels_last": True,
        "seed": SEED,
    }

def fingerprint_of(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]

def new_architecture_context(arch):
    run_dir = RUN_DIR / arch
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = architecture_identity(arch)
    return SimpleNamespace(
        arch=arch, spec=ARCHITECTURE_SPECS[arch], info=ARCHITECTURE_INFO[arch],
        run_dir=run_dir, identity=identity, fingerprint=fingerprint_of(identity),
        best_path=run_dir / "best.pt", last_path=run_dir / "last.pt",
        intra_path=run_dir / "last_intra.pt", history_path=run_dir / "training_history.csv",
        completion_path=run_dir / "training_complete.json",
        config_path=run_dir / "config_snapshot.json",
        evaluation_path=run_dir / "evaluation_complete.json",
        summary_path=run_dir / "summary.json",
        model=None, model_ema=None, criterion=None, optimizer=None, scheduler=None, scaler=None,
        base_state=None, train_loader=None, val_loader=None, test_loader=None,
        compile_selected=False, benchmark=None, pipeline=None, config=None,
    )

def restore_architecture_files(ctx, names):
    return restore_missing_run_files_from_drive([f"{ctx.arch}/{name}" for name in names])

def require_matching_fingerprint(ctx, artifact_name, artifact_fingerprint):
    if artifact_fingerprint != ctx.fingerprint:
        raise RuntimeError(
            f"{ctx.arch}/{artifact_name} belongs to a different configuration "
            f"({artifact_fingerprint} != {ctx.fingerprint}). Choose a new CS_RUN_GROUP; "
            "existing run files were not modified."
        )

def architecture_log(ctx, event, **fields):
    log_event(event, **{"architecture": ctx.arch, **fields})

# ---------------------------------------------------------------------------
# Probes: training micro-batch (largest that fits) and evaluation batch.
# ---------------------------------------------------------------------------
def probe_micro_batch(ctx):
    total_vram = torch.cuda.get_device_properties(0).total_memory
    rows = []
    for batch_size in AUTO_BATCH_CANDIDATES:
        if EFFECTIVE_BATCH_SIZE % batch_size:
            continue
        probe = probe_ema = probe_criterion = probe_optimizer = probe_scaler = None
        pixels = inputs = targets = factors = logits = loss = None
        fits = False
        try:
            release_cuda_memory()
            probe = build_model(ctx.spec, pretrained=False)
            probe.load_state_dict(ctx.base_state, strict=True)
            probe = probe.to(DEVICE, memory_format=torch.channels_last).train()
            probe_ema = build_model_ema(probe)
            probe_criterion = CardioSentryLoss(POS_WEIGHT, CONTRADICTION_LAMBDA).to(DEVICE)
            probe_optimizer = build_optimizer(probe)
            probe_scaler = make_grad_scaler()
            pixels = torch.randint(0, 256, (batch_size, 3, *INPUT_SIZE), dtype=torch.uint8, device=DEVICE)
            targets = torch.randint(0, 2, (batch_size, len(LABELS)), device=DEVICE).float()
            factors = torch.ones(batch_size, 2, device=DEVICE)
            accumulation = EFFECTIVE_BATCH_SIZE // batch_size
            for update in range(1 + BATCH_PROBE_TIMED_UPDATES):
                if update == 1:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                for _ in range(accumulation):
                    inputs = prepare_image_batch(pixels, factors)
                    with autocast_context():
                        logits = probe(inputs)
                        loss, _ = probe_criterion(logits, targets)
                    probe_scaler.scale(loss / accumulation).backward()
                optimizer_update(probe, probe_optimizer, probe_scaler, probe_ema)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated()
            fraction = peak / total_vram
            fits = fraction <= VRAM_SAFETY_FRACTION
            rows.append({"micro_batch": batch_size, "peak_GiB": gib(peak), "VRAM_fraction": fraction,
                         "images_per_second": EFFECTIVE_BATCH_SIZE * BATCH_PROBE_TIMED_UPDATES / elapsed,
                         "status": "fits" if fits else "over_budget"})
            print(f"  micro-batch {batch_size:2d}: {rows[-1]['images_per_second']:.1f} images/s, "
                  f"{gib(peak):.1f} GiB ({fraction:.0%}), {rows[-1]['status']}")
        except torch.cuda.OutOfMemoryError:
            rows.append({"micro_batch": batch_size, "status": "OOM"})
            print(f"  micro-batch {batch_size:2d}: OOM")
        finally:
            if probe_optimizer is not None:
                probe_optimizer.zero_grad(set_to_none=True)
            del probe, probe_ema, probe_criterion, probe_optimizer, probe_scaler
            del pixels, inputs, targets, factors, logits, loss
            release_cuda_memory()
        if fits:
            break  # candidates descend: the first that fits is the largest
    selected = next((row["micro_batch"] for row in rows if row["status"] == "fits"), None)
    if selected is None:
        raise RuntimeError(f"{ctx.arch}: no micro-batch passed the VRAM preflight")
    return selected, pd.DataFrame(rows)

def probe_evaluation_batch(ctx, training_batch):
    # A real optimizer update allocates the resident AdamW + EMA state first,
    # so the evaluation batch is measured against the memory training leaves.
    active_model = build_model(ctx.spec, pretrained=False).to(DEVICE, memory_format=torch.channels_last)
    active_ema = build_model_ema(active_model)
    active_optimizer = build_optimizer(active_model)
    active_scaler = make_grad_scaler()
    active_criterion = CardioSentryLoss(POS_WEIGHT, CONTRADICTION_LAMBDA).to(DEVICE)
    pixels = inputs = targets = logits = loss = None
    rows = []
    try:
        active_model.train()
        pixels = torch.randint(0, 256, (training_batch, 3, *INPUT_SIZE), dtype=torch.uint8, device=DEVICE)
        inputs = prepare_image_batch(pixels)
        targets = torch.zeros(training_batch, len(LABELS), device=DEVICE)
        with autocast_context():
            logits = active_model(inputs)
            loss, _ = active_criterion(logits, targets)
        active_scaler.scale(loss).backward()
        optimizer_update(active_model, active_optimizer, active_scaler, active_ema)
        pixels = inputs = targets = logits = loss = None
        active_model.eval()
        for batch_size in EVAL_BATCH_CANDIDATES:
            try:
                torch.cuda.empty_cache()
                pixels = torch.randint(0, 256, (batch_size, 3, *INPUT_SIZE), dtype=torch.uint8, device=DEVICE)
                with torch.inference_mode(), autocast_context():
                    inputs = prepare_image_batch(pixels)
                    logits = active_model(inputs)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    for _ in range(EVAL_PROBE_TIMED_BATCHES):
                        inputs = prepare_image_batch(pixels)
                        logits = active_model(inputs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated()
                fraction = peak / torch.cuda.get_device_properties(0).total_memory
                status = "fits" if fraction <= VRAM_SAFETY_FRACTION else "over_budget"
                rows.append({"eval_batch": batch_size, "peak_GiB": gib(peak),
                             "VRAM_fraction": fraction, "status": status,
                             "images_per_second": batch_size * EVAL_PROBE_TIMED_BATCHES / elapsed})
                if len([row for row in rows if row["status"] == "fits"]) >= 3:
                    break
            except torch.cuda.OutOfMemoryError:
                rows.append({"eval_batch": batch_size, "status": "OOM"})
            finally:
                pixels = inputs = logits = None
                torch.cuda.empty_cache()
    finally:
        del active_model, active_ema, active_optimizer, active_scaler, active_criterion
        del pixels, inputs, targets, logits, loss
        release_cuda_memory()
    fits = [row for row in rows if row["status"] == "fits"]
    if not fits:
        raise RuntimeError(f"{ctx.arch}: no evaluation batch passed the VRAM preflight")
    return max(fits, key=lambda row: row["images_per_second"])["eval_batch"], pd.DataFrame(rows)

def build_architecture_loaders(ctx):
    queue_bytes = NUM_WORKERS * PREFETCH_FACTOR * (ctx.micro_batch + ctx.eval_batch) * 3 * math.prod(INPUT_SIZE)
    if NUM_WORKERS > 0 and Path("/dev/shm").is_dir():
        if queue_bytes * 1.25 > shutil.disk_usage("/dev/shm").free:
            raise RuntimeError("Insufficient /dev/shm for the selected queues; reduce CS_PREFETCH_FACTOR or CS_NUM_WORKERS.")
    ctx.queue_gib = gib(queue_bytes)
    # The repeat-factor sampler fixes the epoch length and order per (seed, epoch).
    ctx.train_loader = DataLoader(train_dataset, sampler=train_sampler, drop_last=True,
                                  **make_loader_options(ctx.micro_batch, NUM_WORKERS))
    ctx.val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False,
                                **make_loader_options(ctx.eval_batch, NUM_WORKERS))
    ctx.test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                                 **make_loader_options(ctx.eval_batch, NUM_WORKERS))
    ctx.accumulation = EFFECTIVE_BATCH_SIZE // ctx.micro_batch
    ctx.usable_train_batches = len(ctx.train_loader) - (len(ctx.train_loader) % ctx.accumulation)
    if ctx.usable_train_batches <= 0:
        raise RuntimeError("Training loader is too short for the configured effective batch")
    ctx.steps_per_epoch = ctx.usable_train_batches // ctx.accumulation

def benchmark_execution_mode(ctx, compiled):
    active_model = active_ema = active_criterion = active_optimizer = active_scaler = None
    benchmark_train_loader = benchmark_val_loader = None
    train_iterator = val_iterator = images = pixels = labels = factors = logits = loss = None
    train_batches = min(max(BENCHMARK_TRAIN_BATCHES, ctx.accumulation), ctx.usable_train_batches)
    train_batches -= train_batches % ctx.accumulation
    try:
        seed_global_rngs(SEED)
        active_model = build_model(ctx.spec, pretrained=False)
        active_model.load_state_dict(ctx.base_state, strict=True)
        active_model = active_model.to(DEVICE, memory_format=torch.channels_last).train()
        active_ema = build_model_ema(active_model)
        active_criterion = CardioSentryLoss(POS_WEIGHT, CONTRADICTION_LAMBDA).to(DEVICE)
        active_optimizer = build_optimizer(active_model)
        active_scaler = make_grad_scaler()
        configure_model_compilation(active_model, compiled)
        # Disposable loaders: both modes see the same order and the real
        # persistent loaders are never advanced.
        benchmark_train_loader = DataLoader(
            train_dataset, shuffle=True, drop_last=True,
            generator=torch.Generator().manual_seed(SEED),
            **make_loader_options(ctx.micro_batch, NUM_WORKERS, persistent=False),
        )
        benchmark_val_loader = DataLoader(
            val_dataset, shuffle=False, drop_last=False,
            **make_loader_options(ctx.eval_batch, NUM_WORKERS, persistent=False),
        )
        train_iterator = iter(benchmark_train_loader)

        def run_train_batches(count):
            active_optimizer.zero_grad(set_to_none=True)
            for batch_index in range(count):
                pixels, labels, factors = next(train_iterator)
                images = prepare_image_batch(pixels, factors)
                labels = labels.to(DEVICE, non_blocking=True)
                with autocast_context():
                    logits = active_model(images)
                    loss, _ = active_criterion(logits, labels)
                active_scaler.scale(loss / ctx.accumulation).backward()
                if (batch_index + 1) % ctx.accumulation == 0:
                    optimizer_update(active_model, active_optimizer, active_scaler, active_ema)

        # Warm-up covers at least one full update (AdamW state) and, compiled,
        # the forward AND backward compilation, before any timing.
        warmup_batches = ctx.accumulation * max(1, math.ceil(BENCHMARK_WARMUP_BATCHES / ctx.accumulation))
        startup_started = time.perf_counter()
        run_train_batches(warmup_batches)
        torch.cuda.synchronize()
        startup_seconds = time.perf_counter() - startup_started
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        run_train_batches(train_batches)
        torch.cuda.synchronize()
        train_elapsed = time.perf_counter() - started
        train_peak = torch.cuda.max_memory_allocated()

        active_model.eval()
        val_iterator = iter(benchmark_val_loader)
        val_warmup = min(BENCHMARK_WARMUP_BATCHES, max(0, len(ctx.val_loader) - 1))
        val_batches = min(BENCHMARK_VAL_BATCHES, len(ctx.val_loader) - val_warmup)
        if val_batches < 1:
            raise RuntimeError("Validation loader is too short for the benchmark")
        with torch.inference_mode():
            for _ in range(val_warmup):
                pixels, _, _ = next(val_iterator)
                with autocast_context():
                    active_model(prepare_image_batch(pixels))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            val_images = 0
            for _ in range(val_batches):
                pixels, _, _ = next(val_iterator)
                images = prepare_image_batch(pixels)
                with autocast_context():
                    active_model(images)
                val_images += len(images)
            torch.cuda.synchronize()
            val_elapsed = time.perf_counter() - started
        peak = max(train_peak, torch.cuda.max_memory_allocated())
        total_vram = torch.cuda.get_device_properties(0).total_memory
        train_seconds = train_elapsed / train_batches * ctx.usable_train_batches
        val_seconds = val_elapsed / val_batches * len(ctx.val_loader)
        return {
            "mode": "compiled" if compiled else "eager",
            "status": "fits" if peak / total_vram <= VRAM_SAFETY_FRACTION else "over_budget",
            "startup_train_warmup_seconds": startup_seconds,
            "timed_train_batches": train_batches, "timed_val_batches": val_batches,
            "train_images_per_second": ctx.micro_batch * train_batches / train_elapsed,
            "eval_images_per_second": val_images / val_elapsed,
            "train_sec_per_epoch": train_seconds, "val_sec_per_epoch": val_seconds,
            # Two validation passes per epoch (raw and EMA), plus 5% overhead.
            "estimated_sec_per_epoch": (train_seconds + 2 * val_seconds) * 1.05,
            "peak_allocated_gib": gib(peak), "vram_total_gib": gib(total_vram),
            "vram_peak_fraction": peak / total_vram,
        }
    finally:
        del train_iterator, val_iterator, images, pixels, labels, factors, logits, loss
        del benchmark_train_loader, benchmark_val_loader
        del active_model, active_ema, active_criterion, active_optimizer, active_scaler
        release_cuda_memory()

def select_runtime_benchmark(rows, minimum_compile_gain=0.02):
    eager = next(row for row in rows if row["mode"] == "eager" and row["status"] == "fits")
    compiled = next((row for row in rows if row["mode"] == "compiled" and row["status"] == "fits"), None)
    if compiled is not None and compiled["estimated_sec_per_epoch"] < eager["estimated_sec_per_epoch"] * (1 - minimum_compile_gain):
        return compiled
    return eager

def prepare_architecture_runtime(ctx, need_benchmark):
    '''Probe batches, build loaders, and (for training) pick eager or compiled.'''
    print(f"[{ctx.arch}] probing the training micro-batch (largest that fits) ...")
    ctx.micro_batch, batch_probe = probe_micro_batch(ctx)
    ctx.eval_batch, eval_probe = probe_evaluation_batch(ctx, ctx.micro_batch)
    display(batch_probe)
    display(eval_probe)
    build_architecture_loaders(ctx)
    print(f"[{ctx.arch}] micro-batch {ctx.micro_batch} x {ctx.accumulation} accumulation; "
          f"eval batch {ctx.eval_batch}; {ctx.steps_per_epoch} optimizer steps per epoch; "
          f"loader queues {ctx.queue_gib:.2f} GiB")
    ctx.pipeline = {
        "micro_batch_size": ctx.micro_batch, "eval_batch_size": ctx.eval_batch,
        "accumulation_steps": ctx.accumulation, "num_workers": NUM_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR, "estimated_queue_gib": ctx.queue_gib,
        "training_batch_probe": batch_probe.to_dict("records"),
        "eval_batch_probe": eval_probe.to_dict("records"),
    }
    pipeline_path = ctx.run_dir / "input_pipeline_benchmark.json"
    atomic_json_save(ctx.pipeline, pipeline_path)
    if not need_benchmark:
        copy_to_drive(pipeline_path, required=True)
        return
    rows = []
    for compiled in ([False, True] if TORCH_COMPILE_ENABLED else [False]):
        started = time.time()
        try:
            row = benchmark_execution_mode(ctx, compiled)
            rows.append(row)
            print(f"[{ctx.arch}] {row['mode']}: {row['train_images_per_second']:.1f} train images/s, "
                  f"{row['eval_images_per_second']:.1f} eval images/s, {row['peak_allocated_gib']:.1f} GiB, "
                  f"{row['status']} (warm-up incl. compilation {format_duration(row['startup_train_warmup_seconds'])}; "
                  f"cell time {format_duration(time.time() - started)})")
            if not compiled and row["status"] != "fits":
                raise RuntimeError("Eager benchmark exceeds 90% VRAM")
        except Exception as exc:
            if not compiled:
                raise
            rows.append({"mode": "compiled", "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            warnings.warn(f"[{ctx.arch}] compiler benchmark failed; keeping eager: {exc}")
    # The benchmark's compiled copy is gone; drop its dynamo entries too. The
    # real model's compile below re-traces and reuses Inductor's on-disk cache.
    torch._dynamo.reset()
    chosen = select_runtime_benchmark(rows)
    ctx.compile_selected = chosen["mode"] == "compiled"
    ctx.benchmark = {
        **chosen, "architecture": ctx.arch, "hardware_profile": HARDWARE_PROFILE,
        "compile_selected": ctx.compile_selected,
        "compile_mode": TORCH_COMPILE_MODE if ctx.compile_selected else None,
        "mode_comparison": rows, "pipeline": ctx.pipeline,
        "usable_train_batches": ctx.usable_train_batches, "val_batches": len(ctx.val_loader),
        "estimated_total_max_epochs": chosen["estimated_sec_per_epoch"] * MAX_EPOCHS,
    }
    benchmark_path = ctx.run_dir / "benchmark.json"
    atomic_json_save(ctx.benchmark, benchmark_path)
    copy_run_files([benchmark_path, pipeline_path], required=True)
    seed_global_rngs(SEED)
    print(f"[{ctx.arch}] selected execution: {chosen['mode']}; estimated epoch "
          f"{format_duration(chosen['estimated_sec_per_epoch'])}; at most {MAX_EPOCHS} epochs "
          f"= {format_duration(ctx.benchmark['estimated_total_max_epochs'])}")
"""),
    code(r"""# ---------------------------------------------------------------------------
# Training with resume (one architecture). Same harness as the ConvNeXt
# clean12 notebooks: epoch-boundary checkpoints only drive resume.
# ---------------------------------------------------------------------------
def batch_metadata_to_lists(batch_metadata):
    output = {}
    for key, values in batch_metadata.items():
        output[key] = values.detach().cpu().tolist() if torch.is_tensor(values) else list(values)
    return output

@torch.inference_mode()
def predict_loader(active_model, loader, active_criterion=None):
    active_model.eval()
    probabilities, targets = [], []
    metadata = {key: [] for key in PREDICT_METADATA_KEYS}
    loss_sums = torch.zeros(len(LOSS_COMPONENT_KEYS), device=DEVICE, dtype=torch.float32)
    sample_count = 0
    for pixels, labels, batch_metadata in tqdm(loader, leave=False):
        images = prepare_image_batch(pixels)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast_context():
            logits = active_model(images)
            if active_criterion is not None:
                loss, parts = active_criterion(logits, labels)
        if active_criterion is not None:
            loss_sums.add_(torch.stack((
                loss.detach(), parts["bce"], parts["contradiction"], parts["weighted_contradiction"],
            )).float(), alpha=len(images))
        sample_count += len(images)
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
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

def select_validation_weights(active_model, active_ema, loader, active_criterion):
    # Both candidates see validation only. The test loader is never used here.
    best = None
    scores, macro_aurocs = {}, {}
    for weight_source, candidate_model in (("raw", active_model), ("ema", active_ema.module)):
        output = predict_loader(candidate_model, loader, active_criterion)
        mask = primary_mask(output)
        if not mask.any():
            raise RuntimeError("Validation output has no primary-metric rows")
        metrics = evaluate_predictions(output["y_true"][mask], output["y_probability"][mask], [0.5] * len(LABELS))
        score = float(metrics["pathology_only_macro_average_precision"])
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite validation pathology Macro AP for {weight_source}")
        scores[weight_source] = score
        macro_aurocs[weight_source] = float(metrics["macro_auroc"])
        if best is None or score > best["score"]:
            best = {"weight_source": weight_source, "score": score,
                    "output": output, "metrics": metrics, "mask": mask}
    best["candidate_macro_auroc"] = macro_aurocs
    return best, scores

def report_training_collapse(ctx, epoch, stage, details):
    # Called before last.pt is replaced, so it still holds the last healthy epoch.
    report = {"detected_at_utc": datetime.now(timezone.utc).isoformat(), "architecture": ctx.arch,
              "epoch": int(epoch), "stage": stage, "base_lr": BASE_LR,
              "constant_predictor_loss": CONSTANT_PREDICTOR_LOSS, **details}
    report_path = ctx.run_dir / "collapse_report.json"
    atomic_json_save(report, report_path)
    architecture_log(ctx, "training_collapse", **report)
    copy_run_files([report_path, RUN_LOG_PATH], required=False)
    raise TrainingCollapseError(
        f"{ctx.arch} collapsed during epoch {epoch} ({stage}): {json.dumps(details, default=str)}. "
        "last.pt still holds the previous healthy epoch. Re-run this model with a lower LR "
        f"(e.g. CS_BASE_LR={BASE_LR / 2:.1e}) under a new CS_RUN_GROUP."
    )

def train_one_epoch(ctx, heartbeat_fn=None):
    model, optimizer, scaler, scheduler = ctx.model, ctx.optimizer, ctx.scaler, ctx.scheduler
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_totals = torch.zeros(len(LOSS_COMPONENT_KEYS), device=DEVICE, dtype=torch.float32)
    sample_count = optimizer_updates = amp_skipped_optimizer_steps = 0
    scale_before = scaler.get_scale()
    window_bce = torch.zeros((), device=DEVICE, dtype=torch.float32)
    window_samples = collapsed_windows = 0
    progress = tqdm(enumerate(ctx.train_loader), total=ctx.usable_train_batches, leave=False,
                    desc=f"{ctx.arch} train")
    for batch_index, (pixels, labels, factors) in progress:
        if batch_index >= ctx.usable_train_batches:
            break
        images = prepare_image_batch(pixels, factors)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast_context():
            logits = model(images)
            loss, parts = ctx.criterion(logits, labels)
        scaler.scale(loss / ctx.accumulation).backward()
        if (batch_index + 1) % ctx.accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM,
                                           error_if_nonfinite=not scaler.is_enabled())
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer.zero_grad(set_to_none=True)
            # A scale decrease means GradScaler skipped the step: keep the LR
            # schedule and EMA aligned with real updates.
            if scale_after >= scale_before:
                scheduler.step()
                optimizer_updates += 1
                update_model_ema(ctx.model_ema, model)
            else:
                amp_skipped_optimizer_steps += 1
            scale_before = scale_after
        if heartbeat_fn is not None:
            heartbeat_fn()
        batch_samples = len(images)
        loss_totals.add_(torch.stack((
            loss.detach(), parts["bce"], parts["contradiction"], parts["weighted_contradiction"],
        )).float(), alpha=batch_samples)
        sample_count += batch_samples
        window_bce.add_(parts["bce"].float(), alpha=batch_samples)
        window_samples += batch_samples
        if (batch_index + 1) % COLLAPSE_CHECK_EVERY_N_BATCHES == 0:
            # Active after a full epoch of updates; earlier, a loss near the
            # constant predictor is ordinary early training.
            if COLLAPSE_GUARD_ENABLED and scheduler.last_epoch >= ctx.steps_per_epoch:
                window_mean = (window_bce / window_samples).item()
                collapsed_windows = collapsed_windows + 1 if window_mean >= COLLAPSE_LOSS_RATIO * CONSTANT_PREDICTOR_LOSS else 0
                if collapsed_windows >= COLLAPSE_PATIENCE_WINDOWS:
                    report_training_collapse(ctx, scheduler.last_epoch // ctx.steps_per_epoch + 1, "training_loss", {
                        "window_train_bce": window_mean, "consecutive_windows": collapsed_windows,
                        "batches_per_window": COLLAPSE_CHECK_EVERY_N_BATCHES,
                        "optimizer_step": int(scheduler.last_epoch),
                    })
            window_bce.zero_()
            window_samples = 0
        if (batch_index + 1) % PROGRESS_UPDATE_EVERY_N_BATCHES == 0:
            progress.set_postfix(loss=f"{(loss_totals[0] / sample_count).item():.4f}")
    averaged_losses = (loss_totals / sample_count).cpu().tolist()
    return {**dict(zip(LOSS_COMPONENT_KEYS, averaged_losses)),
            "optimizer_updates": optimizer_updates,
            "amp_skipped_optimizer_steps": amp_skipped_optimizer_steps}

def build_training_state(ctx):
    ctx.model = build_model(ctx.spec, pretrained=False)
    ctx.model.load_state_dict(ctx.base_state, strict=True)
    ctx.model = ctx.model.to(DEVICE, memory_format=torch.channels_last)
    ctx.model_ema = build_model_ema(ctx.model)
    ctx.criterion = CardioSentryLoss(POS_WEIGHT, CONTRADICTION_LAMBDA).to(DEVICE)
    ctx.optimizer = build_optimizer(ctx.model)
    warmup_steps = WARMUP_EPOCHS * ctx.steps_per_epoch
    total_steps = MAX_EPOCHS * ctx.steps_per_epoch

    def lr_multiplier(step):
        if step < warmup_steps:
            return max((step + 1) / max(warmup_steps, 1), 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

    ctx.scheduler = LambdaLR(ctx.optimizer, lr_lambda=lr_multiplier)
    ctx.scaler = make_grad_scaler()

def architecture_config(ctx):
    runtime = {
        "reproducibility_mode": "deterministic opt-in" if DETERMINISTIC else "seeded performance mode; not bitwise deterministic",
        "collapse_guard": {"enabled": COLLAPSE_GUARD_ENABLED, "constant_predictor_loss": CONSTANT_PREDICTOR_LOSS,
                           "loss_ratio": COLLAPSE_LOSS_RATIO, "check_every_n_batches": COLLAPSE_CHECK_EVERY_N_BATCHES,
                           "patience_windows": COLLAPSE_PATIENCE_WINDOWS,
                           "min_raw_val_macro_auroc": COLLAPSE_MIN_VAL_MACRO_AUROC},
        "package_versions": PACKAGE_VERSIONS,
        "hardware_profile": HARDWARE_PROFILE,
        "gpu": gpu.name,
        "fused_adamw": USE_FUSED_ADAMW,
        "compile_selected": ctx.compile_selected,
        "compile_mode": TORCH_COMPILE_MODE if ctx.compile_selected else None,
        "amp_dtype": str(AMP_DTYPE),
        "num_workers": NUM_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "micro_batch_size": ctx.micro_batch,
        "eval_batch_size": ctx.eval_batch,
        "accumulation_steps": ctx.accumulation,
        "optimizer_steps_per_epoch": ctx.steps_per_epoch,
        "benchmark_estimated_sec_per_epoch": (ctx.benchmark or {}).get("estimated_sec_per_epoch"),
        "architecture_info": ctx.info,
        "dataset_source": str(DATASET_DRIVE_PATH),
        "drive_run_dir": str(drive_path_for(ctx.run_dir)) if DRIVE_SYNC_ENABLED else None,
        "split_image_counts": {name: len(frame) for name, frame in selected_frames.items()},
    }
    return {"identity": ctx.identity, "runtime": runtime, "fingerprint": ctx.fingerprint}

def validate_existing_config(ctx):
    if not ctx.config_path.is_file():
        return
    try:
        existing = json.loads(ctx.config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{ctx.config_path} could not be read and will not be overwritten. "
                           "Choose a new CS_RUN_GROUP or repair it.") from error
    stored_identity = existing.get("identity")
    if not isinstance(stored_identity, dict) or existing.get("fingerprint") != fingerprint_of(stored_identity):
        raise RuntimeError(f"{ctx.config_path} has an invalid fingerprint and will not be overwritten.")
    require_matching_fingerprint(ctx, "config_snapshot.json", existing.get("fingerprint"))
    stored_runtime = existing.get("runtime", {})
    drifted = {name: (stored_runtime.get("package_versions", {}).get(name), PACKAGE_VERSIONS.get(name))
               for name in ("timm", "torch", "torchvision")
               if stored_runtime.get("package_versions", {}).get(name) != PACKAGE_VERSIONS.get(name)}
    if drifted:
        warnings.warn(f"[{ctx.arch}] package versions differ from the stored run config: {drifted}. Resuming anyway.")
    stored_micro = stored_runtime.get("micro_batch_size")
    if stored_micro not in (None, ctx.micro_batch):
        warnings.warn(f"[{ctx.arch}] resuming at micro-batch {ctx.micro_batch} instead of {stored_micro}: "
                      "BatchNorm sees a different batch size from here on (same effective batch and LR).")

def load_resume_checkpoint(ctx):
    '''Restore the newest epoch-boundary checkpoint; return (start_epoch, best, bad_epochs, history).'''
    checkpoint, source = None, None
    for candidate_path in (ctx.last_path, ctx.best_path):
        if not candidate_path.is_file():
            continue
        candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
        require_matching_fingerprint(ctx, candidate_path.name, candidate.get("config_fingerprint"))
        if not bool(candidate.get("epoch_complete", True)) or "optimizer" not in candidate:
            del candidate
            continue
        if checkpoint is None or int(candidate["epoch"]) > int(checkpoint["epoch"]):
            checkpoint, source = candidate, candidate_path.name
        else:
            del candidate
    if checkpoint is None:
        return 1, -float("inf"), 0, []
    ctx.model.load_state_dict(checkpoint["model"])
    ctx.model_ema.module.load_state_dict(checkpoint["model_ema"])
    ctx.model_ema.num_updates = int(checkpoint["ema_num_updates"])
    ctx.optimizer.load_state_dict(checkpoint["optimizer"])
    ctx.scheduler.load_state_dict(checkpoint["scheduler"])
    ctx.scaler.load_state_dict(checkpoint["scaler"])
    if checkpoint.get("python_random_state") is not None:
        random.setstate(checkpoint["python_random_state"])
    if checkpoint.get("numpy_random_state") is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
    if checkpoint.get("torch_random_state") is not None:
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    if checkpoint.get("cuda_random_state_all") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_random_state_all"]])
    start_epoch = int(checkpoint["epoch"]) + 1
    best = float(checkpoint["best_pathology_only_macro_average_precision"])
    bad_epochs = int(checkpoint["bad_epochs"])
    history = list(checkpoint["history"])
    print(f"[{ctx.arch}] resuming from {source} after epoch {start_epoch - 1}; best val pathology Macro AP {best:.6f}")
    del checkpoint
    return start_epoch, best, bad_epochs, history

def train_architecture(ctx):
    build_training_state(ctx)
    validate_existing_config(ctx)
    start_epoch, best_score, bad_epochs, history = load_resume_checkpoint(ctx)
    ctx.config = architecture_config(ctx)
    # Compile both copies in place, after EMA creation and the checkpoint restore.
    configure_model_compilation(ctx.model, ctx.compile_selected)
    configure_model_compilation(ctx.model_ema.module, ctx.compile_selected)
    atomic_json_save(ctx.config, ctx.config_path)
    copy_to_drive(ctx.config_path, required=True)

    def full_state(current_epoch, epoch_complete):
        return {
            "epoch": current_epoch, "epoch_complete": bool(epoch_complete),
            "architecture": ctx.arch, "model_name": ctx.spec["model_name"],
            "model": ctx.model.state_dict(), "model_ema": ctx.model_ema.module.state_dict(),
            "ema_num_updates": ctx.model_ema.num_updates,
            "optimizer": ctx.optimizer.state_dict(), "scheduler": ctx.scheduler.state_dict(),
            "scaler": ctx.scaler.state_dict(),
            "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(), "cuda_random_state_all": torch.cuda.get_rng_state_all(),
            "best_pathology_only_macro_average_precision": best_score, "bad_epochs": bad_epochs,
            "history": history, "config": ctx.config, "config_fingerprint": ctx.fingerprint,
        }

    architecture_log(ctx, "training_start", start_epoch=start_epoch, max_epochs=MAX_EPOCHS)
    stopped_early = start_epoch - 1 >= EARLY_STOPPING_MIN_EPOCHS and bad_epochs >= EARLY_STOPPING_PATIENCE
    for epoch in ([] if stopped_early else range(start_epoch, MAX_EPOCHS + 1)):
        epoch_started = time.time()
        train_dataset.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats()
        last_intra_sync = [epoch_started]

        def request_intra_epoch_sync():
            # Optional (CS_INTRA_EPOCH_SYNC_MINUTES > 0): a mid-epoch snapshot
            # to last_intra.pt for inspection; it never drives resume.
            if INTRA_EPOCH_SYNC_MINUTES <= 0 or time.time() - last_intra_sync[0] < INTRA_EPOCH_SYNC_MINUTES * 60:
                return
            with _copy_key_lock:
                pending = _copy_key_threads.get(f"{ctx.arch}/{ctx.intra_path.name}")
                if pending is not None and pending.is_alive():
                    return
            last_intra_sync[0] = time.time()
            atomic_torch_save(full_state(epoch, False), ctx.intra_path)
            spawn_drive_copy(ctx.intra_path)

        train_values = train_one_epoch(ctx, heartbeat_fn=request_intra_epoch_sync)
        choice, candidate_scores = select_validation_weights(ctx.model, ctx.model_ema, ctx.val_loader, ctx.criterion)
        validation_output, validation_metrics = choice["output"], choice["metrics"]
        raw_auroc = choice["candidate_macro_auroc"]["raw"]
        if COLLAPSE_GUARD_ENABLED and not raw_auroc >= COLLAPSE_MIN_VAL_MACRO_AUROC:
            report_training_collapse(ctx, epoch, "validation", {
                "raw_val_macro_auroc": raw_auroc, "ema_val_macro_auroc": choice["candidate_macro_auroc"]["ema"],
                "train_loss": train_values["loss"],
            })
        score = validation_metrics["pathology_only_macro_average_precision"]
        improved = score > best_score
        if improved:
            best_score, bad_epochs = score, 0
        else:
            bad_epochs += 1
        epoch_row = {
            "epoch": epoch,
            "selected_weight_source": choice["weight_source"],
            "val_raw_pathology_only_macro_average_precision": candidate_scores["raw"],
            "val_ema_pathology_only_macro_average_precision": candidate_scores["ema"],
            "val_raw_macro_auroc": raw_auroc,
            "val_ema_macro_auroc": choice["candidate_macro_auroc"]["ema"],
            "constant_predictor_loss": CONSTANT_PREDICTOR_LOSS,
            "ema_num_updates": ctx.model_ema.num_updates,
            "train_loss": train_values["loss"],
            "train_bce": train_values["bce"],
            "optimizer_updates": train_values["optimizer_updates"],
            "amp_skipped_optimizer_steps": train_values["amp_skipped_optimizer_steps"],
            "val_loss": validation_output["loss"],
            "val_bce": validation_output["bce"],
            "val_macro_average_precision": validation_metrics["macro_average_precision"],
            "val_pathology_only_macro_average_precision": score,
            **{f"val_ap_{label}": validation_metrics["per_label"][label]["average_precision"] for label in LABELS},
            "val_macro_auroc": validation_metrics["macro_auroc"],
            "val_micro_average_precision": validation_metrics["micro_average_precision"],
            "learning_rate": max(group["lr"] for group in ctx.optimizer.param_groups),
            "micro_batch_size": ctx.micro_batch,
            "epoch_seconds": time.time() - epoch_started,
            "peak_allocated_gib": gib(torch.cuda.max_memory_allocated()),
            "improved": improved,
            "bad_epochs": bad_epochs,
        }
        history.append(epoch_row)
        pd.DataFrame(history).to_csv(ctx.history_path, index=False)
        state = full_state(epoch, True)
        if improved:
            atomic_torch_save({
                **state,
                "selected_weight_source": choice["weight_source"],
                "validation_macro_average_precision": validation_metrics["macro_average_precision"],
                "validation_pathology_only_macro_average_precision": score,
                "validation_average_precision_per_condition": {
                    label: validation_metrics["per_label"][label]["average_precision"] for label in LABELS
                },
            }, ctx.best_path)
            copy_to_drive(ctx.best_path, required=True)  # a new best must survive a disconnect
            architecture_log(ctx, "best_checkpoint", epoch=epoch, val_pathology_only_macro_ap=score)
        atomic_torch_save(state, ctx.last_path)
        del state
        print(f"[{ctx.arch}] epoch {epoch:2d}/{MAX_EPOCHS} train_loss={train_values['loss']:.4f} "
              f"val_mAP={validation_metrics['macro_average_precision']:.4f} val_path_mAP={score:.4f} "
              f"weights={choice['weight_source']} bad={bad_epochs} ({epoch_row['epoch_seconds'] / 60:.1f} min)",
              flush=True)
        architecture_log(ctx, "epoch_end", epoch=epoch, train_loss=train_values["loss"],
                         val_pathology_only_macro_ap=score, improved=improved, bad_epochs=bad_epochs,
                         epoch_seconds=round(epoch_row["epoch_seconds"], 1))
        join_drive_copies()
        if epoch % DRIVE_SYNC_EVERY_N_EPOCHS == 0:
            copy_run_files([ctx.last_path, ctx.history_path, ctx.config_path, RUN_LOG_PATH]
                           + ([ctx.best_path] if ctx.best_path.is_file() else []), required=True)
        if epoch >= EARLY_STOPPING_MIN_EPOCHS and bad_epochs >= EARLY_STOPPING_PATIENCE:
            stopped_early = True
            print(f"[{ctx.arch}] early stopping after epoch {epoch} (patience {EARLY_STOPPING_PATIENCE})")
            break
    join_drive_copies()
    copy_run_files([ctx.last_path, ctx.best_path, ctx.history_path, ctx.config_path, RUN_LOG_PATH], required=True)
    completion = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "architecture": ctx.arch,
                  "stopped_early": stopped_early, "epochs_run": len(history),
                  "best_pathology_only_macro_average_precision": best_score,
                  "config_fingerprint": ctx.fingerprint}
    atomic_json_save(completion, ctx.completion_path)
    copy_to_drive(ctx.completion_path, required=True)
    architecture_log(ctx, "training_finished", **completion)
    print(f"[{ctx.arch}] training finished: {json.dumps(completion)}")
"""),
    code(r"""# ---------------------------------------------------------------------------
# Final evaluation (one architecture): validation-only thresholds, one test pass.
# ---------------------------------------------------------------------------
def prediction_frame(output, fixed_thresholds, tuned_thresholds):
    frame = pd.DataFrame(output["metadata"])
    for label_index, label in enumerate(LABELS):
        frame[f"true_{label}"] = output["y_true"][:, label_index]
        frame[f"prob_{label}"] = output["y_probability"][:, label_index]
        frame[f"pred_0_5_{label}"] = (output["y_probability"][:, label_index] >= fixed_thresholds[label_index]).astype(np.int8)
        frame[f"pred_validation_tuned_{label}"] = (
            output["y_probability"][:, label_index] >= tuned_thresholds[label_index]
        ).astype(np.int8)
    return frame

def patient_bootstrap_average_precision(y_true, y_probability, patient_ids, rounds, seed):
    '''Rounds x labels AP draws. Patients, not images, are resampled.

    The draws depend only on the seed and the sorted test patient list, so every
    architecture gets the same resamples and the differences are paired.
    '''
    patients, inverse = np.unique(np.asarray(patient_ids, dtype=str), return_inverse=True)
    rng = np.random.default_rng([int(seed), 2026, 10, 2])
    draws = np.full((rounds, len(LABELS)), np.nan)
    for round_index in range(rounds):
        counts = np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))
        weights = counts[inverse].astype(np.float64)
        keep = weights > 0
        for label_index in range(len(LABELS)):
            truth = y_true[keep, label_index]
            if truth.min() == truth.max():
                continue
            draws[round_index, label_index] = average_precision_score(
                truth, y_probability[keep, label_index], sample_weight=weights[keep]
            )
    return draws, hashlib.sha256("\n".join(patients).encode()).hexdigest()

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
    return {"batch_size": batch_size, "latency_ms_per_batch": 1000 * elapsed / repetitions,
            "latency_ms_per_image": 1000 * elapsed / (repetitions * batch_size),
            "throughput_images_per_second": repetitions * batch_size / elapsed}

def metric_text(number):
    return "NA" if number is None or not np.isfinite(number) else f"{number:.4f}"

def evaluate_architecture(ctx):
    if not ctx.completion_path.is_file():
        raise RuntimeError(f"[{ctx.arch}] finish training before the final evaluation")
    require_matching_fingerprint(ctx, "training_complete.json",
                                 json.loads(ctx.completion_path.read_text()).get("config_fingerprint"))
    if not ctx.best_path.is_file():
        raise FileNotFoundError(f"[{ctx.arch}] no best.pt; refusing to evaluate an unfinished run")
    best_checkpoint = torch.load(ctx.best_path, map_location="cpu", weights_only=False)
    require_matching_fingerprint(ctx, "best.pt", best_checkpoint.get("config_fingerprint"))
    weight_source = best_checkpoint["selected_weight_source"]
    if weight_source not in ("raw", "ema"):
        raise RuntimeError(f"Invalid selected weight source: {weight_source!r}")
    best_epoch = int(best_checkpoint["epoch"])
    best_val_path = float(best_checkpoint["validation_pathology_only_macro_average_precision"])
    best_val_all = float(best_checkpoint["validation_macro_average_precision"])
    # Evaluate eagerly on a fresh model: no recompiles for odd batch sizes.
    model = build_model(ctx.spec, pretrained=False)
    model.load_state_dict(best_checkpoint["model_ema" if weight_source == "ema" else "model"])
    del best_checkpoint
    model = model.to(DEVICE, memory_format=torch.channels_last).eval()
    criterion = ctx.criterion or CardioSentryLoss(POS_WEIGHT, CONTRADICTION_LAMBDA).to(DEVICE)
    print(f"[{ctx.arch}] evaluating best epoch {best_epoch} ({weight_source} weights; "
          f"val pathology Macro AP {best_val_path:.4f})")

    validation_output = predict_loader(model, ctx.val_loader, criterion)
    val_primary = primary_mask(validation_output)
    fixed_thresholds = [0.5] * len(LABELS)
    tuned_thresholds = tune_thresholds_on_validation(
        validation_output["y_true"][val_primary], validation_output["y_probability"][val_primary])
    # The checkpoint, weight source, and thresholds are fixed before test inference.
    test_output = predict_loader(model, ctx.test_loader, criterion)
    test_primary = primary_mask(test_output)

    evaluations = {}
    for policy, thresholds in (("fixed_0_5", fixed_thresholds), ("validation_tuned", tuned_thresholds)):
        evaluations[policy] = {
            "validation_primary": evaluate_predictions(
                validation_output["y_true"][val_primary], validation_output["y_probability"][val_primary], thresholds),
            "test_primary": evaluate_predictions(
                test_output["y_true"][test_primary], test_output["y_probability"][test_primary], thresholds),
            "validation_primary_per_source": evaluate_by_source(
                validation_output["y_true"][val_primary], validation_output["y_probability"][val_primary],
                np.asarray(validation_output["metadata"]["source"])[val_primary], thresholds),
            "test_primary_per_source": evaluate_by_source(
                test_output["y_true"][test_primary], test_output["y_probability"][test_primary],
                np.asarray(test_output["metadata"]["source"])[test_primary], thresholds),
            "test_primary_combinations": top_combination_metrics(
                test_output["y_true"][test_primary], test_output["y_probability"][test_primary], thresholds),
        }
    test_metrics = evaluations["validation_tuned"]["test_primary"]

    run_dir = ctx.run_dir
    prediction_frame(validation_output, fixed_thresholds, tuned_thresholds).to_csv(run_dir / "validation_predictions.csv", index=False)
    prediction_frame(test_output, fixed_thresholds, tuned_thresholds).to_csv(run_dir / "test_predictions.csv", index=False)
    atomic_json_save(dict(zip(LABELS, tuned_thresholds)), run_dir / "validation_tuned_thresholds.json")
    inference_path = run_dir / "inference.pt"
    atomic_torch_save({
        # timm.create_model(model_name, num_classes=len(label_order)).load_state_dict(model)
        "model": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
        "architecture": ctx.arch, "model_name": ctx.spec["model_name"],
        "label_order": list(LABELS), "input_size_hw": list(INPUT_SIZE),
        "normalization_mean": ctx.info["normalization_mean"], "normalization_std": ctx.info["normalization_std"],
        "input_color_mode": IMAGE_COLOR_MODE, "resize_interpolation": RESIZE_INTERPOLATION.value,
        "resize_antialias": RESIZE_ANTIALIAS, "thresholds": list(map(float, tuned_thresholds)),
        "selected_weight_source": weight_source, "best_epoch": best_epoch,
        "config_fingerprint": ctx.fingerprint,
    }, inference_path)

    metric_rows, source_rows = [], []
    for policy, results in evaluations.items():
        metric_rows += flatten_metrics(results["validation_primary"], "val_primary", threshold_policy=policy)
        metric_rows += flatten_metrics(results["test_primary"], "test_primary", threshold_policy=policy)
        for split, key in (("val_primary", "validation_primary_per_source"), ("test_primary", "test_primary_per_source")):
            for source, result in results[key].items():
                source_rows += flatten_metrics(result, split, source, threshold_policy=policy)
    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(source_rows).to_csv(run_dir / "per_source_metrics.csv", index=False)
    combination_rows, combination_summary_rows = [], []
    for policy, results in evaluations.items():
        combo = results["test_primary_combinations"]
        combination_summary_rows.append({"split": "test_primary", "threshold_policy": policy, **combo["summary"]})
        combination_rows += [{"split": "test_primary", "threshold_policy": policy, **row} for row in combo["top_combinations"]]
    pd.DataFrame(combination_rows).to_csv(run_dir / "top_combination_metrics.csv", index=False)
    pd.DataFrame(combination_summary_rows).to_csv(run_dir / "combination_summary.csv", index=False)

    # Patient-bootstrap 95% intervals for test AP (paired across architectures).
    test_patient_ids = pd.Series(
        test_df["patient_id"].to_numpy(), index=test_df["record_id"].astype(str)
    ).reindex(np.asarray(test_output["metadata"]["record_id"], dtype=str)[test_primary])
    if test_patient_ids.isna().any():
        raise RuntimeError("Some test predictions have no patient_id; cannot bootstrap by patient")
    bootstrap_started = time.time()
    bootstrap_draws, test_patients_sha256 = patient_bootstrap_average_precision(
        test_output["y_true"][test_primary], test_output["y_probability"][test_primary],
        test_patient_ids.to_numpy(), BOOTSTRAP_ROUNDS, SEED)
    np.save(run_dir / "test_bootstrap_draws.npy", bootstrap_draws)
    pathology_indices = [LABELS.index(label) for label in PATHOLOGY_LABELS]
    bootstrap_macro = np.nanmean(bootstrap_draws[:, pathology_indices], axis=1)
    lower_q, upper_q = 50 * (1 - BOOTSTRAP_CONFIDENCE), 100 - 50 * (1 - BOOTSTRAP_CONFIDENCE)
    bootstrap_rows = []
    for label_index, label in enumerate(LABELS):
        valid = bootstrap_draws[:, label_index][np.isfinite(bootstrap_draws[:, label_index])]
        bootstrap_rows.append({
            "label": label, "test_average_precision": test_metrics["per_label"][label]["average_precision"],
            "ci_low": float(np.percentile(valid, lower_q)) if len(valid) else float("nan"),
            "ci_high": float(np.percentile(valid, upper_q)) if len(valid) else float("nan"),
            "valid_rounds": int(len(valid)), "support": test_metrics["per_label"][label]["support"],
        })
    bootstrap_rows.append({
        "label": "PATHOLOGY_MACRO", "test_average_precision": test_metrics["pathology_only_macro_average_precision"],
        "ci_low": float(np.percentile(bootstrap_macro, lower_q)), "ci_high": float(np.percentile(bootstrap_macro, upper_q)),
        "valid_rounds": int(np.isfinite(bootstrap_macro).sum()), "support": int(test_primary.sum()),
    })
    bootstrap_ci = pd.DataFrame(bootstrap_rows)
    bootstrap_ci.to_csv(run_dir / "test_bootstrap_ci.csv", index=False)
    print(f"[{ctx.arch}] patient bootstrap: {BOOTSTRAP_ROUNDS} rounds over {test_patient_ids.nunique():,} "
          f"test patients in {time.time() - bootstrap_started:.0f}s")

    inference_benchmark = {"batch_1": benchmark_inference(model, 1),
                           "selected_micro_batch": benchmark_inference(model, ctx.micro_batch)}
    history_frame = pd.read_csv(ctx.history_path)
    completion = json.loads(ctx.completion_path.read_text())
    summary = {
        "architecture": ctx.arch, "display_name": ctx.spec["display_name"], "model_name": ctx.spec["model_name"],
        "parameters_m": ctx.info["parameters_m"], "gmacs_per_sheet": ctx.info["gmacs_per_sheet"],
        "micro_batch_size": ctx.micro_batch,
        "compiled": bool(ctx.config["runtime"]["compile_selected"]) if getattr(ctx, "config", None) else None,
        "epochs_run": int(completion["epochs_run"]), "stopped_early": bool(completion["stopped_early"]),
        "best_epoch": best_epoch, "selected_weight_source": weight_source,
        "val_pathology_macro_ap": best_val_path, "val_macro_ap": best_val_all,
        "test_pathology_macro_ap": test_metrics["pathology_only_macro_average_precision"],
        "test_pathology_macro_ap_ci_low": bootstrap_rows[-1]["ci_low"],
        "test_pathology_macro_ap_ci_high": bootstrap_rows[-1]["ci_high"],
        "test_macro_ap": test_metrics["macro_average_precision"],
        "test_macro_auroc": test_metrics["macro_auroc"],
        "test_micro_ap": test_metrics["micro_average_precision"],
        "test_macro_f1_tuned": test_metrics["macro_f1"],
        "test_macro_f1_fixed_0_5": evaluations["fixed_0_5"]["test_primary"]["macro_f1"],
        "test_exact_match_tuned": test_metrics["exact_match_accuracy"],
        "test_ap_per_label": {label: test_metrics["per_label"][label]["average_precision"] for label in LABELS},
        "test_auroc_per_label": {label: test_metrics["per_label"][label]["auroc"] for label in LABELS},
        "mean_epoch_minutes": float(history_frame["epoch_seconds"].mean() / 60),
        "train_hours": float(history_frame["epoch_seconds"].sum() / 3600),
        "inference_images_per_second_batch1": inference_benchmark["batch_1"]["throughput_images_per_second"],
        "inference_images_per_second_micro_batch": inference_benchmark["selected_micro_batch"]["throughput_images_per_second"],
        "test_patients_sha256": test_patients_sha256,
        "bootstrap_rounds": BOOTSTRAP_ROUNDS, "bootstrap_seed": SEED,
        "config_fingerprint": ctx.fingerprint,
    }
    atomic_json_save(summary, ctx.summary_path)
    atomic_json_save({
        "summary": summary,
        "config": getattr(ctx, "config", None),
        "threshold_selection": "per-label maximum F1 on primary validation rows only",
        "validation_tuned_thresholds": dict(zip(LABELS, map(float, tuned_thresholds))),
        "test_bootstrap_ci": {"method": "patient-level bootstrap, percentile interval",
                              "rounds": BOOTSTRAP_ROUNDS, "confidence": BOOTSTRAP_CONFIDENCE, "rows": bootstrap_rows},
        "evaluations": evaluations,
        "loss_components": {"validation": {k: float(validation_output[k]) for k in LOSS_COMPONENT_KEYS},
                            "test": {k: float(test_output[k]) for k in LOSS_COMPONENT_KEYS}},
        "inference_benchmark": inference_benchmark,
        "environment": {"python": sys.version, "platform": platform.platform(), "pytorch": torch.__version__,
                        "timm": timm.__version__, "cuda": torch.version.cuda, "gpu": gpu.name,
                        "gpu_vram_gib": gib(gpu.total_memory)},
    }, run_dir / "result.json")

    # Human-readable report.
    per_label = pd.DataFrame(test_metrics["per_label"]).T
    display(per_label[["average_precision", "auroc", "sensitivity", "specificity", "precision",
                       "f1", "threshold", "support"]].round(4))
    display(bootstrap_ci.round(4))
    lines = [
        f"# CardioSentry {ctx.spec['display_name']} clean12 report (Colab A100)", "",
        f"- Model: `{ctx.spec['model_name']}` ({ctx.info['parameters_m']:.2f} M parameters, "
        f"{ctx.info['gmacs_per_sheet']:.1f} GMACs per 768x1024 sheet)",
        f"- Patient split: {SPLIT_VERSION}, seed {SPLIT_SEED}, sha256 {SPLIT_SHA256}",
        f"- Recipe: AdamW LR {BASE_LR:g} (all layers), weight decay {WEIGHT_DECAY}, "
        f"{WARMUP_EPOCHS}-epoch warm-up + cosine, effective batch {EFFECTIVE_BATCH_SIZE} "
        f"(micro-batch {ctx.micro_batch}), {POS_WEIGHT_MODE} pos_weight BCE",
        f"- Epochs run: {completion['epochs_run']} (early stop: {completion['stopped_early']}); best epoch {best_epoch}; "
        f"{weight_source} weights",
        f"- Validation pathology-only Macro AP: {best_val_path:.4f}",
        f"- **Test pathology-only Macro AP: {test_metrics['pathology_only_macro_average_precision']:.4f} "
        f"(95% patient-bootstrap CI {bootstrap_rows[-1]['ci_low']:.4f}-{bootstrap_rows[-1]['ci_high']:.4f})**",
        f"- Test 12-label Macro AP: {test_metrics['macro_average_precision']:.4f}; macro AUROC {test_metrics['macro_auroc']:.4f}",
        f"- Test Macro F1: {test_metrics['macro_f1']:.4f} (validation-tuned), "
        f"{evaluations['fixed_0_5']['test_primary']['macro_f1']:.4f} (fixed 0.5)",
        f"- Test exact-match accuracy (validation-tuned): {test_metrics['exact_match_accuracy']:.4f}",
        f"- Mean epoch: {summary['mean_epoch_minutes']:.1f} min; inference {summary['inference_images_per_second_micro_batch']:.0f} images/s",
        "", "## Test AP per label with 95% patient-bootstrap intervals", "",
        "| label | AP | CI low | CI high | AUROC | F1 (tuned) | support |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bootstrap_rows[:-1]:
        label_metrics = test_metrics["per_label"][row["label"]]
        lines.append(f"| {row['label']} | {metric_text(row['test_average_precision'])} | {metric_text(row['ci_low'])} | "
                     f"{metric_text(row['ci_high'])} | {metric_text(label_metrics['auroc'])} | "
                     f"{metric_text(label_metrics['f1'])} | {row['support']} |")
    lines += ["", "## Test metrics by site (validation-tuned thresholds)", "",
              "| site | n | Macro AP | macro AUROC | macro F1 |", "|---|---:|---:|---:|---:|"]
    for source, result in evaluations["validation_tuned"]["test_primary_per_source"].items():
        lines.append(f"| {source} | {result['n_samples']} | {metric_text(result['macro_average_precision'])} | "
                     f"{metric_text(result['macro_auroc'])} | {metric_text(result['macro_f1'])} |")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    print((run_dir / "report.md").read_text())

    # Learning curves and test precision-recall curves.
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].plot(history_frame.epoch, history_frame.train_loss, marker="o", label="train")
    axes[0].plot(history_frame.epoch, history_frame.val_loss, marker="o", label="validation")
    axes[0].set_title(f"{ctx.arch}: loss"); axes[0].legend()
    axes[1].plot(history_frame.epoch, history_frame.val_pathology_only_macro_average_precision, marker="o", label="pathology only")
    axes[1].plot(history_frame.epoch, history_frame.val_macro_average_precision, marker="o", label="12 labels")
    axes[1].axvline(best_epoch, color="tab:red", linestyle="--", label="best")
    axes[1].set_title(f"{ctx.arch}: validation Macro AP"); axes[1].legend()
    for label_index, label in enumerate(LABELS):
        precision, recall, _ = precision_recall_curve(
            test_output["y_true"][test_primary][:, label_index], test_output["y_probability"][test_primary][:, label_index])
        axes[2].plot(recall, precision, label=f"{label} {test_metrics['per_label'][label]['average_precision']:.3f}")
    axes[2].set(xlabel="recall", ylabel="precision", title=f"{ctx.arch}: test PR curves")
    axes[2].legend(ncol=2, fontsize="x-small")
    for axis in axes[:2]:
        axis.set_xlabel("epoch")
    plt.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    del model
    release_cuda_memory()
    sync_complete_run_to_drive(required=True, root=run_dir)
    atomic_json_save({"completed_at_utc": datetime.now(timezone.utc).isoformat(),
                      "architecture": ctx.arch, "config_fingerprint": ctx.fingerprint}, ctx.evaluation_path)
    copy_to_drive(ctx.evaluation_path, required=True)
    architecture_log(ctx, "evaluation_finished",
                     test_pathology_macro_ap=summary["test_pathology_macro_ap"])
    return summary
"""),
    code(r"""# ---------------------------------------------------------------------------
# Orchestration: one architecture end to end, then a full clean-up.
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

def finished_summary(ctx):
    '''The stored summary when this architecture already finished, else None.'''
    restore_architecture_files(ctx, SMALL_RESULT_FILES)
    if not (ctx.evaluation_path.is_file() and ctx.summary_path.is_file()):
        return None
    marker = json.loads(ctx.evaluation_path.read_text())
    require_matching_fingerprint(ctx, "evaluation_complete.json", marker.get("config_fingerprint"))
    return json.loads(ctx.summary_path.read_text())

def shutdown_architecture(ctx):
    for name in ("train_loader", "val_loader", "test_loader", "model", "model_ema",
                 "optimizer", "scheduler", "scaler", "criterion", "base_state"):
        setattr(ctx, name, None)
    torch._dynamo.reset()
    release_cuda_memory()

def run_architecture(arch):
    if arch not in ARCHITECTURES:
        print(f"{arch}: not in CS_ARCHITECTURES; skipped")
        return None
    ctx = new_architecture_context(arch)
    started = time.time()
    print(f"===== {ctx.spec['display_name']} ({ctx.spec['model_name']}), fingerprint {ctx.fingerprint} =====")
    try:
        if BENCHMARK_ONLY:
            return benchmark_architecture(ctx, started)
        summary = finished_summary(ctx)
        if summary is not None:
            print(f"[{arch}] already finished and evaluated; reloaded from {drive_path_for(ctx.run_dir)}")
            ARCH_RESULTS[arch] = summary
            return summary
        restore_architecture_files(ctx, CHECKPOINT_FILES)
        (ctx.run_dir / "failure.json").unlink(missing_ok=True)
        seed_global_rngs(SEED)
        set_input_normalization(ctx.info["normalization_mean"], ctx.info["normalization_std"])
        ctx.base_state = build_model(ctx.spec, pretrained=True).state_dict()
        training_done = ctx.completion_path.is_file()
        if training_done:
            require_matching_fingerprint(ctx, "training_complete.json",
                                         json.loads(ctx.completion_path.read_text()).get("config_fingerprint"))
        architecture_log(ctx, "architecture_start", model=ctx.spec["model_name"], training_done=training_done)
        prepare_architecture_runtime(ctx, need_benchmark=not training_done)
        if not training_done:
            train_architecture(ctx)
        else:
            print(f"[{arch}] training already complete; running the final evaluation only")
            ctx.config = json.loads(ctx.config_path.read_text()) if ctx.config_path.is_file() else None
        # Free the training state before evaluation builds its own eager model.
        ctx.model = ctx.model_ema = ctx.optimizer = ctx.scheduler = None
        torch._dynamo.reset()
        release_cuda_memory()
        summary = evaluate_architecture(ctx)
        ARCH_RESULTS[arch] = summary
        print(f"===== {arch} done in {format_duration(time.time() - started)}: test pathology Macro AP "
              f"{summary['test_pathology_macro_ap']:.4f} "
              f"[{summary['test_pathology_macro_ap_ci_low']:.4f}, {summary['test_pathology_macro_ap_ci_high']:.4f}] =====")
        return summary
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        failure = {"failed_at_utc": datetime.now(timezone.utc).isoformat(), "architecture": arch,
                   "error_type": type(exc).__name__, "error": str(exc),
                   "elapsed_seconds": round(time.time() - started, 1)}
        ARCH_FAILURES[arch] = failure
        atomic_json_save(failure, ctx.run_dir / "failure.json")
        architecture_log(ctx, "architecture_failed", **failure)
        join_drive_copies()
        copy_run_files([ctx.run_dir / "failure.json", RUN_LOG_PATH], required=False)
        if not CONTINUE_ON_ERROR:
            raise
        import traceback
        traceback.print_exc()
        print(f"===== {arch} FAILED ({type(exc).__name__}); continuing with the next model =====")
        return None
    finally:
        shutdown_architecture(ctx)
"""),
    code(r"""# ---------------------------------------------------------------------------
# Benchmark-only mode (CS_BENCHMARK_ONLY=1): probes and the eager/compiled
# benchmark per model, no training, then one cost table (section 10d).
# ---------------------------------------------------------------------------
BENCHMARK_RESULTS = {}

def benchmark_summary_row(ctx, setup_seconds):
    '''One row of the benchmark table, from ctx.benchmark (prepare_architecture_runtime).'''
    chosen = ctx.benchmark
    modes = {row["mode"]: row for row in chosen["mode_comparison"]}
    epoch_seconds = chosen["estimated_sec_per_epoch"]
    # Final evaluation: one validation and one test pass (about the same size).
    evaluation_seconds = 2 * chosen["val_sec_per_epoch"]
    properties = torch.cuda.get_device_properties(0)
    return {
        "architecture": ctx.arch, "model_name": ctx.spec["model_name"],
        "gpu": properties.name, "vram_gib": round(gib(properties.total_memory), 1),
        "micro_batch": ctx.micro_batch, "accumulation": ctx.accumulation, "eval_batch": ctx.eval_batch,
        "execution": chosen["mode"],
        "eager_train_images_per_s": modes.get("eager", {}).get("train_images_per_second"),
        "compiled_train_images_per_s": modes.get("compiled", {}).get("train_images_per_second"),
        "train_images_per_s": chosen["train_images_per_second"],
        "eval_images_per_s": chosen["eval_images_per_second"],
        "peak_vram_gib": chosen["peak_allocated_gib"],
        "epoch_minutes": epoch_seconds / 60,
        # Weights, batch probes, and the eager + compiled benchmark (incl. compilation).
        "setup_minutes": setup_seconds / 60,
        "evaluation_minutes": evaluation_seconds / 60,
        "hours_at_min_epochs": (setup_seconds + EARLY_STOPPING_MIN_EPOCHS * epoch_seconds
                                + evaluation_seconds) / 3600,
        "hours_at_max_epochs": (setup_seconds + MAX_EPOCHS * epoch_seconds + evaluation_seconds) / 3600,
    }

def benchmark_architecture(ctx, started):
    '''Probe and benchmark one model without training; returns its table row.'''
    seed_global_rngs(SEED)
    set_input_normalization(ctx.info["normalization_mean"], ctx.info["normalization_std"])
    ctx.base_state = build_model(ctx.spec, pretrained=True).state_dict()
    architecture_log(ctx, "benchmark_start", model=ctx.spec["model_name"])
    prepare_architecture_runtime(ctx, need_benchmark=True)
    row = benchmark_summary_row(ctx, time.time() - started)
    BENCHMARK_RESULTS[ctx.arch] = row
    row_path = ctx.run_dir / "benchmark_summary.json"
    atomic_json_save(row, row_path)
    copy_run_files([row_path], required=False)
    print(f"[{ctx.arch}] benchmark only: {row['epoch_minutes']:.1f} min/epoch; "
          f"{row['hours_at_min_epochs']:.1f} h at {EARLY_STOPPING_MIN_EPOCHS} epochs, "
          f"{row['hours_at_max_epochs']:.1f} h at {MAX_EPOCHS} (setup {row['setup_minutes']:.1f} min)")
    return row

def summarize_benchmarks(results, out_dir, units_per_hour=0.0, staging_seconds=None):
    '''One table for every benchmarked model plus a TOTAL row; saved and copied to Drive.'''
    table = pd.DataFrame(list(results.values()))
    if table.empty:
        print("No model finished its benchmark.")
        return table
    staging_hours = (staging_seconds or 0.0) / 3600
    totals = {"architecture": "TOTAL (all models + dataset staging)",
              "hours_at_min_epochs": table["hours_at_min_epochs"].sum() + staging_hours,
              "hours_at_max_epochs": table["hours_at_max_epochs"].sum() + staging_hours}
    if units_per_hour:
        table["units_per_epoch"] = table["epoch_minutes"] / 60 * units_per_hour
        for column in ("hours_at_min_epochs", "hours_at_max_epochs"):
            table[column.replace("hours", "units")] = table[column] * units_per_hour
            totals[column.replace("hours", "units")] = totals[column] * units_per_hour
    table = pd.concat([table, pd.DataFrame([totals])], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out_dir / "benchmark_comparison.csv", out_dir / "benchmark_comparison.json"
    table.to_csv(csv_path, index=False)
    atomic_json_save({
        "gpu": torch.cuda.get_device_name(0), "compute_units_per_hour": units_per_hour or None,
        "dataset_staging_seconds": staging_seconds, "min_epochs": EARLY_STOPPING_MIN_EPOCHS,
        "max_epochs": MAX_EPOCHS, "rows": table.to_dict("records"),
    }, json_path)
    copy_run_files([csv_path, json_path], required=False)
    display(table.round(3))
    print(f"Hours include setup (probes + compilation), {EARLY_STOPPING_MIN_EPOCHS} or {MAX_EPOCHS} "
          "epochs (each with two validation passes, raw and EMA) and the final evaluation. "
          f"TOTAL adds this runtime's dataset staging ({staging_hours * 60:.0f} min).")
    if not units_per_hour:
        print("No compute-unit rate set (CS_COMPUTE_UNITS_PER_HOUR; Colab shows it in the "
              "Resources panel): units = hours x units/hour.")
    print("saved:", csv_path)
    return table
"""),
    md(r"""## 10. Train and evaluate the three CNNs, in order

Each of the next three cells runs one complete model. Run all goes through
them in order. A model that already finished in this run group, on Drive, is
skipped in seconds. To re-run a single model, run its cell again (the shared
cells above must have run in this session). With `CS_BENCHMARK_ONLY=1` each
cell only benchmarks its model, and 10d prints the cost table.
"""),
    code(r"""# 10a. ResNet-50
run_architecture("resnet50")
"""),
    code(r"""# 10b. DenseNet-121
run_architecture("densenet121")
"""),
    code(r"""# 10c. EfficientNet-B3
run_architecture("efficientnet_b3")
"""),
    code(r"""# 10d. Benchmark-only cost table (CS_BENCHMARK_ONLY=1); a training run skips it.
if BENCHMARK_ONLY:
    benchmark_table = summarize_benchmarks(BENCHMARK_RESULTS, RUN_DIR, COMPUTE_UNITS_PER_HOUR,
                                           globals().get("copy_seconds"))
else:
    print("Training run: each model's benchmark is in <arch>/benchmark.json.")
"""),
    md(r"""## 11. Compare the architectures

One table per run group, written next to the model folders:

- `comparison.csv`: the headline numbers per model (test pathology Macro AP with
  its 95% CI, 12-label Macro AP, macro AUROC, macro F1, epochs, time, size, speed).
- `comparison_per_label_ap.csv`: test AP per label with each model's CI.
- `comparison_paired_bootstrap.csv`: for every pair of models, the difference
  in test pathology Macro AP over the **same** 1,000 patient resamples. It
  gives the mean difference, a 95% interval, and the share of resamples in
  which the first model wins. An interval that contains 0 means the test set
  cannot separate the two models.
- `comparison.md` and `comparison_curves.png`.

To place these next to ConvNeXt, read the same `test_pathology_macro_ap` /
`test_bootstrap_ci.csv` from the ConvNeXt run folders. The same patients and
the same bootstrap seed are used if that run used `CS_SEED=42`.
"""),
    code(r"""if BENCHMARK_ONLY:
    print("CS_BENCHMARK_ONLY=1: nothing was trained, so there is nothing to compare; see the cost table in 10d.")
else:
    comparison_rows, per_label_rows, histories, draws = [], [], {}, {}
    for arch in ARCHITECTURES:
        run_dir = RUN_DIR / arch
        summary = ARCH_RESULTS.get(arch)
        if summary is None and (run_dir / "summary.json").is_file():
            summary = json.loads((run_dir / "summary.json").read_text())
        if summary is None:
            failure = ARCH_FAILURES.get(arch) or (
                json.loads((run_dir / "failure.json").read_text()) if (run_dir / "failure.json").is_file() else None)
            comparison_rows.append({"architecture": arch, "status": f"FAILED: {failure['error_type']}" if failure else "not run"})
            continue
        comparison_rows.append({
            "architecture": arch, "status": "ok", "model_name": summary["model_name"],
            "parameters_m": summary["parameters_m"], "gmacs_per_sheet": summary["gmacs_per_sheet"],
            "test_pathology_macro_ap": summary["test_pathology_macro_ap"],
            "ci_low": summary["test_pathology_macro_ap_ci_low"], "ci_high": summary["test_pathology_macro_ap_ci_high"],
            "test_macro_ap": summary["test_macro_ap"], "test_macro_auroc": summary["test_macro_auroc"],
            "test_macro_f1_tuned": summary["test_macro_f1_tuned"], "test_exact_match_tuned": summary["test_exact_match_tuned"],
            "val_pathology_macro_ap": summary["val_pathology_macro_ap"],
            "best_epoch": summary["best_epoch"], "epochs_run": summary["epochs_run"],
            "weights": summary["selected_weight_source"], "micro_batch": summary["micro_batch_size"],
            "compiled": summary["compiled"], "mean_epoch_minutes": summary["mean_epoch_minutes"],
            "train_hours": summary["train_hours"],
            "inference_images_per_second": summary["inference_images_per_second_micro_batch"],
        })
        bootstrap_table = pd.read_csv(run_dir / "test_bootstrap_ci.csv").set_index("label")
        for label in LABELS:
            per_label_rows.append({"label": label, "architecture": arch,
                                   "ap": bootstrap_table.loc[label, "test_average_precision"],
                                   "ci_low": bootstrap_table.loc[label, "ci_low"],
                                   "ci_high": bootstrap_table.loc[label, "ci_high"],
                                   "support": bootstrap_table.loc[label, "support"]})
        histories[arch] = pd.read_csv(run_dir / "training_history.csv")
        draws[arch] = (np.load(run_dir / "test_bootstrap_draws.npy"), summary["test_patients_sha256"])

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(RUN_DIR / "comparison.csv", index=False)
    display(comparison.round(4))
    per_label_ap = pd.DataFrame(per_label_rows)
    if len(per_label_ap):
        per_label_ap.to_csv(RUN_DIR / "comparison_per_label_ap.csv", index=False)
        display(per_label_ap.pivot(index="label", columns="architecture", values="ap").reindex(LABELS).round(4))

    # Paired bootstrap: identical patient resamples for every model (same seed,
    # same sorted test patients), so per-round differences are paired.
    pathology_indices = [LABELS.index(label) for label in PATHOLOGY_LABELS]
    paired_rows = []
    finished = list(draws)
    for index, first in enumerate(finished):
        for second in finished[index + 1:]:
            (first_draws, first_sha), (second_draws, second_sha) = draws[first], draws[second]
            if first_sha != second_sha or first_draws.shape != second_draws.shape:
                warnings.warn(f"{first} vs {second}: bootstrap draws are not paired (different test patients or rounds)")
                continue
            difference = (np.nanmean(first_draws[:, pathology_indices], axis=1)
                          - np.nanmean(second_draws[:, pathology_indices], axis=1))
            difference = difference[np.isfinite(difference)]
            paired_rows.append({
                "first": first, "second": second,
                "mean_difference_pathology_macro_ap": float(difference.mean()),
                "ci_low": float(np.percentile(difference, 2.5)), "ci_high": float(np.percentile(difference, 97.5)),
                "share_first_better": float((difference > 0).mean()), "rounds": int(len(difference)),
            })
    paired = pd.DataFrame(paired_rows)
    if len(paired):
        paired.to_csv(RUN_DIR / "comparison_paired_bootstrap.csv", index=False)
        print("paired patient-bootstrap differences in test pathology Macro AP (first - second)")
        display(paired.round(4))

    if histories:
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
        for arch, history_frame in histories.items():
            axes[0].plot(history_frame.epoch, history_frame.val_pathology_only_macro_average_precision, marker="o", label=arch)
            axes[0].set(title="validation pathology-only Macro AP", xlabel="epoch")
        axes[0].legend()
        if len(per_label_ap):
            pivot = per_label_ap.pivot(index="label", columns="architecture", values="ap").reindex(LABELS)
            low = per_label_ap.pivot(index="label", columns="architecture", values="ci_low").reindex(LABELS)
            high = per_label_ap.pivot(index="label", columns="architecture", values="ci_high").reindex(LABELS)
            width = 0.8 / max(len(pivot.columns), 1)
            positions = np.arange(len(LABELS))
            for offset, arch in enumerate(pivot.columns):
                axes[1].bar(positions + offset * width, pivot[arch], width, label=arch,
                            yerr=[pivot[arch] - low[arch], high[arch] - pivot[arch]], capsize=2)
            axes[1].set_xticks(positions + width * (len(pivot.columns) - 1) / 2, LABELS, rotation=45)
            axes[1].set(title="test AP per label (95% patient-bootstrap CI)", ylim=(0, 1.02))
            axes[1].legend()
        plt.tight_layout()
        fig.savefig(RUN_DIR / "comparison_curves.png", dpi=150, bbox_inches="tight")
        plt.show()

    lines = [f"# CardioSentry CNN baselines on clean12 ({RUN_NAME})", "",
             f"Patient split {SPLIT_VERSION} (sha256 {SPLIT_SHA256}); one recipe for every model: AdamW LR {BASE_LR:g}, "
             f"weight decay {WEIGHT_DECAY}, {MAX_EPOCHS} epochs max, effective batch {EFFECTIVE_BATCH_SIZE}, 768x1024 input.", "",
             "| model | status | params (M) | GMACs | test pathology Macro AP [95% CI] | test Macro AP | macro AUROC | macro F1 | best/run epochs | min/epoch |",
             "|---|---|---:|---:|---|---:|---:|---:|---|---:|"]
    for row in comparison_rows:
        if row["status"] != "ok":
            lines.append(f"| {row['architecture']} | {row['status']} | | | | | | | | |")
            continue
        lines.append(f"| {row['architecture']} | ok | {row['parameters_m']:.1f} | {row['gmacs_per_sheet']:.0f} | "
                     f"{row['test_pathology_macro_ap']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
                     f"{row['test_macro_ap']:.4f} | {row['test_macro_auroc']:.4f} | {row['test_macro_f1_tuned']:.4f} | "
                     f"{row['best_epoch']}/{row['epochs_run']} | {row['mean_epoch_minutes']:.1f} |")
    if paired_rows:
        lines += ["", "## Paired differences (same patient resamples)", "",
                  "| first - second | mean | 95% CI | first better in |", "|---|---:|---|---:|"]
        lines += [f"| {row['first']} - {row['second']} | {row['mean_difference_pathology_macro_ap']:+.4f} | "
                  f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['share_first_better']:.1%} |" for row in paired_rows]
    (RUN_DIR / "comparison.md").write_text("\n".join(lines) + "\n")
    print((RUN_DIR / "comparison.md").read_text())
    join_drive_copies()
    sync_complete_run_to_drive(required=True, root=RUN_DIR, recursive=False)
    if ARCH_FAILURES:
        print("FAILED architectures:", json.dumps(ARCH_FAILURES, indent=2))
"""),
    md(r"""## Persistent outputs

```text
MyDrive/CardioSentry/new_runs/<CS_RUN_GROUP>/
├── patient_split.csv (+ .sha256), record_split.csv, split_audit.json   shared split (sha 4eba80c7…)
├── source_label_support_by_split.csv, sample_grid_per_label.png
├── architectures.json, worker_benchmark.json, run.log
├── comparison.csv, comparison_per_label_ap.csv, comparison_paired_bootstrap.csv
├── comparison.md, comparison_curves.png, drive_sync_manifest.json
└── resnet50/ | densenet121/ | efficientnet_b3/
    ├── best.pt, last.pt                  full training state (resume)
    ├── inference.pt                      selected raw/EMA weights + preprocessing + frozen thresholds
    ├── config_snapshot.json              identity (fingerprinted) + runtime metadata
    ├── benchmark.json, input_pipeline_benchmark.json
    ├── training_history.csv, training_complete.json
    ├── validation_predictions.csv, test_predictions.csv, validation_tuned_thresholds.json
    ├── metrics.csv, per_source_metrics.csv, top_combination_metrics.csv, combination_summary.csv
    ├── test_bootstrap_ci.csv, test_bootstrap_draws.npy
    ├── summary.json, result.json, report.md, curves.png
    ├── evaluation_complete.json          written last: the model is finished
    └── drive_sync_manifest.json
```

The next cell flushes Google Drive and, by default, **releases the Colab
runtime** so it stops using compute units. Run all reaches it only if every
cell above succeeded. Set `CS_RELEASE_RUNTIME=0` to keep the runtime.
"""),
    code(r"""# 12. Flush Google Drive, then release the Colab runtime (the runs are on Drive).
if IN_COLAB and DRIVE_SYNC_ENABLED:
    join_drive_copies()
    from google.colab import drive
    drive.flush_and_unmount()
    print("Google Drive flushed; the runs are in", DRIVE_RUN_DIR)
    if RELEASE_RUNTIME_WHEN_DONE:
        from google.colab import runtime
        print("releasing the Colab runtime ...")
        runtime.unassign()
else:
    print("Not on Colab (or Drive sync off): nothing to flush or release.")
"""),
]


for index, cell in enumerate(cells):
    cell["id"] = f"cs-cnn-c12-colab-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "machine_shape": "hm"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1))
print(f"wrote {OUTPUT} ({len(cells)} cells)")
