"""Build the self-contained CardioSentry ConvNeXt-V1 Base + multi-scale feature fusion clean12 EC2 (g6e.2xlarge) notebook."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "CardioSentry_ConvNeXtV1_Base_MSFF_Clean12_EC2.ipynb"
# The fusion necks live in one file; the model cell inlines it verbatim, and
# validate_notebook.py checks the notebook copy is identical.
FUSION_SOURCE = (HERE / "fusion_necks.py").read_text()


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
    md(r"""# CardioSentry — ConvNeXt-V1 Base + multi-scale feature fusion on the HEEDB clean12 ECG images (EC2 g6e.2xlarge)

This notebook fine-tunes one **ConvNeXt-V1 Base** model with a **multi-scale
feature fusion neck** on the clean12 ECG image corpus: 113,358
physician-verified HEEDB records (native 500 Hz, 10 s, 12 leads) rendered by
`../clean12_ecg_signal_images_ec2/` with the real ECG-image-kit as 1024×768 PNG
sheets (3×4 layout plus a full 10 s Lead II rhythm strip, 25 mm/s, 10 mm/mV,
clean red grid, no printed ID header).

It is tuned for AWS **`g6e.2xlarge`**: 8 vCPUs, 64 GiB RAM, one L40S with
48 GB VRAM (the probes read the actual CUDA VRAM), and a 450 GB local NVMe
instance store. It does not depend on Colab. The dataset comes from S3
(`s3://cardiosentry/dataset/clean_ecg_signal_images_clean12.tar`, read with the
instance's IAM role), and every run file is uploaded to
`s3://cardiosentry/runs/<run name>/`, so a lost (e.g. Spot) instance loses
nothing but the current epoch.

Where it comes from:
- The dataset audit, the 12 labels, the **patient-level split** (SHA-256
  `4eba80c7…`) and the ConvNeXt-V1 Base recipe are those of
  `../convnext_v1_base_clean12_v1/`, so every fusion run is scored on the same
  patients as the plain V1 Base run.
- The EC2 plumbing (NVMe instance-store scratch, S3 download and extraction,
  S3 run persistence, Spot-safe resume) is that of `../convnext_v1_base_clean12_v1/`,
  so a machine that already staged the dataset for that run reuses it.
- The fusion necks, collapse guard, optional rare-label sampling and
  patient-bootstrap CIs are those of the Colab notebook
  `../convnext_v1_base_msff_clean12_colab_v1/`: the model and training are
  identical, only the platform differs.
- The Residual FPN is the colleague's kit-v1 design
  (`../multi_scale_feature_fusion/CardioSentry_ConvNeXtV1_Base_RFPN_KitV1_Clean_EC2.ipynb`);
  FPN, PAN and BiFPN are new here.

## Choose the fusion (`CS_FUSION`)

All necks fuse ConvNeXt stages 2–4 (strides 8/16/32: 96×128, 48×64 and 24×32
maps for a 768×1024 sheet; 256/512/1024 channels → 256 channels each, GroupNorm).
Each level is globally average-pooled and the three vectors are concatenated.

| `CS_FUSION` | neck | head | new params |
|---|---|---|---|
| `rfpn` (default) | top-down FPN | **residual**: ConvNeXt logits + zero-initialised Linear(LayerNorm(pooled FPN)); step 0 = the plain model | 2.2 M |
| `fpn` | top-down FPN (Lin et al., 2017) | **replace**: LayerNorm + Linear over the pooled levels; timm head removed | 2.2 M |
| `pan` | FPN + bottom-up path (PANet, Liu et al., 2018) | replace | 4.6 M |
| `bifpn` | 2 × BiFPN layers, fast normalized fusion, depthwise-separable nodes (EfficientDet, Tan et al., 2020), on 3 levels | replace | 1.0 M |
| `none` | — | timm ConvNeXt head: the plain V1 Base baseline | 0 |

`CS_FUSION_HEAD=residual` also works with `pan` and `bifpn` (a residual PAN or
BiFPN); `CS_FUSION_HEAD=replace` with `fpn` is the same as `CS_FUSION=fpn`.
`CS_FUSION_CHANNELS` (default 256) and `CS_BIFPN_REPEATS` (default 2) set the
neck width and depth. Every one of these is in the run name and fingerprint, so
each choice gets its own run folder on S3: run the notebook once per variant.

The backbone parameters keep the plain model's per-block layer-decay LR profile
exactly (stem ≈ 0.056× head). The neck and the new classifier train at the full
head LR, with weight decay on conv/linear weights only.

## Patient-level split, built at training time

The archive is unsplit. Section 4b assigns every **patient** (`patient_key`, the
HEEDB `BDSPPatientID`) to exactly one of train/validation/test (80/10/10), so all
ECGs of a patient land in the same split. It is multi-label iterative
stratification over patients, seeded by `CS_SPLIT_SEED` (default 20261002), and
it must reproduce the frozen split SHA-256 `4eba80c7…` shared with the V1 Base
notebook. The shared copy at `s3://cardiosentry/dataset/splits/` is loaded when
it exists (and published once, never overwritten, when it does not).

## Fixed training contract

| setting | choice |
|---|---|
| task | multi-label; twelve independent logits in `LABELS` order (NORM + 11 conditions: SB, STACH, LBBB, LQT, AF, AFL, 3AVB, 2AVB, 1AVB, STEMI, STE) |
| pretrained model | `convnext_base.fb_in22k_ft_in1k_384` (supervised ImageNet-22K → ImageNet-1K at 384 px) |
| fusion | `CS_FUSION` neck over stages 2–4 (see above); `rfpn` by default |
| input | 768×1024 RGB (H×W), bicubic resize (a no-op on native 1024×768 kit sheets) |
| sampling | uniform epochs, like the V1 Base run (`CS_RARE_SAMPLING=1` turns on repeat-factor sampling, threshold 0.10) |
| loss | per-label `BCEWithLogitsLoss(pos_weight=sqrt(negatives / positives))` on the training counts |
| contradiction | disabled (`CONTRADICTION_LAMBDA = 0.0`) |
| optimizer | AdamW, peak head LR `2e-4`, per-block layer-wise LR decay `0.925` (≈ Meta's 12-group `0.8`; stem ≈ 0.056× head), weight decay `0.02`, gradient clip `1.0` |
| schedule | 2-epoch linear warm-up, then cosine decay to 1% of each group's peak LR within the 20-epoch budget |
| training | at most 20 epochs; early stopping patience 5, eligible from epoch 10 |
| stability | collapse guard on rolling training loss vs the constant-predictor loss, and on validation macro AUROC (the `fpn`/`pan`/`bifpn` heads start from random necks) |
| checkpoint monitor | best of raw and EMA **validation pathology-only Macro Average Precision** (the 11 non-NORM labels); 12-label Macro AP tracked as secondary |
| thresholds | fixed 0.5 baseline plus per-label F1 thresholds tuned on validation only, then frozen for test |
| uncertainty | 95% patient-bootstrap intervals for test AP per label and pathology Macro AP |
| subgroup reports | per site (I0001/I0006) and site × condition; support always shown |
| regularization | drop-path `0.2`; training-only brightness/contrast ±6%, keyed by epoch and sampled occurrence; full sheet geometry retained |
| effective batch | 64; fastest safe profiled micro-batch; separately probed evaluation batch |
| EMA | decay ceiling `0.9998`, warm-up, successful optimizer updates only; raw and EMA evaluated every epoch |
| acceleration | BF16 AMP, fused AdamW, TF32, channels-last, uint8 pinned batches, GPU augmentation/normalization, tuned decoder workers |
| compiler | eager vs `torch.compile` measured on real batches; the faster safe mode is kept |
| hardware | `g6e.2xlarge` (L40S 48 GB); micro-batch and decoder workers are measured on the visible GPU/CPU |

The test split never selects epochs, raw versus EMA weights, or thresholds.
This is a reasoned recipe, not a tuned optimum. Confirm claims across
independently named seeds (the split stays fixed across them), and compare
fusion variants with their bootstrap intervals, not their point estimates.

References: [timm model card](https://huggingface.co/timm/convnext_base.fb_in22k_ft_in1k_384),
[ConvNeXt paper](https://arxiv.org/abs/2201.03545),
[FPN](https://arxiv.org/abs/1612.03144), [PANet](https://arxiv.org/abs/1803.01534),
[EfficientDet / BiFPN](https://arxiv.org/abs/1911.09070),
[repeat-factor sampling (LVIS)](https://arxiv.org/abs/1908.03195).
Hardware: [AWS G6e specifications](https://aws.amazon.com/ec2/instance-types/g6e/).

This notebook is standalone: copying this `.ipynb` to the training machine is sufficient.
"""),
    md(r"""## 0. Data expected on EC2

The clean12 image build (`../clean12_ecg_signal_images_ec2/`) streams the corpus
to S3 as one plain-TAR archive plus a SHA-256 sidecar:

```text
s3://cardiosentry/dataset/clean_ecg_signal_images_clean12.tar
s3://cardiosentry/dataset/clean_ecg_signal_images_clean12.tar.sha256
```

The archive top level is `clean_ecg_signal_images_clean12/`. Images are grouped
by HEEDB site, not by split:

```text
clean_ecg_signal_images_clean12/
├── images/I0001/WFDB/...png  images/I0006/WFDB/...png   # one 1024×768 PNG per record
├── manifest.csv            # render inventory: source_header, image_path, status, image_bytes, ...
├── record_manifest.csv     # clean12 manifest: record_path, site, patient_key, labels, 12 label columns, ...
├── final_counts.csv        # verified per-label totals from the clean12 download check
├── selection_report.json  config.json  summary.json  failures.csv  README.md  SUMMARY.md
```

Labels are the twelve 0/1 columns of `record_manifest.csv` (also listed in
`labels`). Every record has at least one label, `NORM` never co-occurs with
another label, and every STEMI record also carries STE. The audit cell checks all
of this and the per-label totals against `final_counts.csv`.

Expected corpus: 113,358 records (74,356 I0001 + 39,002 I0006) from 96,574
patients, one clean image each (~0.94 MiB; ~104 GiB archive and ~104 GiB
extracted). The 80/10/10 patient split gives about 90.7k / 11.3k / 11.3k
train/validation/test records.

## Scratch architecture: S3 → NVMe → DataLoader → GPU

Bulk data lives on the EC2 **NVMe instance-store volume** (mounted at
`/mnt/nvme` when one is detected; g5/g6e families), which is ephemeral and
fast. The setup cell downloads the archive from S3 to NVMe when the extracted
tree is not already there, verifies it against the SHA-256 sidecar, extracts all
images to `/mnt/nvme/dataset`, and deletes the local archive by default
(`CS_KEEP_LOCAL_TAR=1` keeps it). Extraction markers make re-runs incremental:
each selection class is extracted once and skipped afterwards. Training reads
**only** from NVMe — images are never streamed from S3 or EBS during epochs. If
no NVMe instance store is detected (EBS-only machine), the same layout falls back
to an EBS-backed scratch directory with a loud warning.

Nothing irreplaceable lives on NVMe: checkpoints, logs, metrics, configs, the
patient split, and final evaluation outputs sync to S3 (checkpoints first
locally, then to S3). The dataset is never synced back to S3 — S3 remains its
source of truth.

Leak note: this kit render prints no record-ID header (only lead names and the
25mm/s / 10mm/mV legend), so the full-sheet geometry is safe from the ID→STEMI
leak of the older v1/v2 sheets. No content crop is needed or applied.

## Automatic S3 persistence

At minimum, every run persists (local NVMe first, then S3):

```text
s3://cardiosentry/
└── runs/<CS_RUN_NAME>/          (default name convnext_v1_base_rfpn_clean12_g6e2xlarge_seed42)
    ├── best.pt                       synced immediately on every new best
    ├── last.pt                       synced every epoch (resume point)
    ├── last_intra.pt                 mid-epoch Spot-visibility snapshot (~15 min)
    ├── config_snapshot.json          frozen training/data/split identity
    ├── patient_split.csv (+ .sha256) frozen patient → split assignment
    ├── record_split.csv              record_path → split
    ├── split_audit.json              split method, seed, SHA-256, per-split support
    ├── training_history.csv          per-epoch losses, metrics, LR, timing
    ├── run.log                       JSONL event log (setup, syncs, epochs)
    ├── benchmark.json
    ├── collapse_report.json          only if the collapse guard stopped the run
    └── ... all final reports, predictions, bootstrap CIs, plots, and s3_upload_manifest.json
```

The notebook performs a real S3 write/read-metadata preflight before training.
During training, `last_intra.pt` is snapshotted locally and uploaded by a background
thread every `CS_INTRA_EPOCH_SYNC_MINUTES` (default 15) for Spot-interruption
visibility, while `last.pt` is written at every epoch boundary as the resume
point; `best.pt` uploads synchronously the moment a new
best validation metric lands. Resume trusts only epoch-boundary checkpoints,
so a Spot interruption redoes at most one epoch from a clean state. Every final object is byte-count checked, and
`s3_upload_manifest.json` records each local SHA-256. If training is
interrupted — or the NVMe volume vanishes with the instance — re-running the
notebook re-mounts scratch, re-stages the dataset from S3 only if missing,
pulls the missing run files back from S3, and resumes from the newest
checkpoint instead of starting over. The EC2 IAM role needs `s3:GetObject` on
`dataset/clean_ecg_signal_images_clean12.tar*`, `s3:GetObject`/`s3:PutObject` on
`dataset/splits/*` (the shared frozen split) and on `runs/*`, and `s3:ListBucket`
on the bucket so a missing object reads as 404 rather than 403.
Export `CS_S3_UPLOAD=0` only for an intentional local-only run.

**Same machine as the plain V1 Base run.** The extracted dataset lives at
`/mnt/nvme/dataset` with the same extraction markers, so if this instance (or
its NVMe) still holds the dataset staged by `../convnext_v1_base_clean12_v1/`,
section 3 skips the download. Run directories are separate
(`/mnt/nvme/runs/convnext_v1_base_msff_clean12_ec2_v1/`), so the two never mix.

## Overrides

Set them in the environment before starting the kernel, or in a cell before the
configuration cell, e.g. `import os; os.environ["CS_FUSION"] = "pan"`.

```text
CS_FUSION=rfpn                 none | fpn | pan | bifpn | rfpn
CS_FUSION_HEAD                 replace | residual (default: residual for rfpn, replace otherwise)
CS_FUSION_CHANNELS=256         CS_BIFPN_REPEATS=2
CS_RUN_NAME                    default convnext_v1_base_<fusion>_clean12_g6e2xlarge_seed<CS_SEED>
CS_SEED=42                     training seed; does not change the split
CS_BASE_LR=2e-4                CS_RARE_SAMPLING=0   CS_BOOTSTRAP_ROUNDS=1000   CS_COLLAPSE_GUARD=1
CS_S3_UPLOAD=1                 CS_S3_RUNS_PREFIX=runs   CS_INTRA_EPOCH_SYNC_MINUTES=15
CS_SPLIT_FILE  CS_SPLIT_S3_URI  CS_SPLIT_PUBLISH=1      CS_KEEP_LOCAL_TAR=0   CS_ARCHIVE_DIR
CS_TORCH_COMPILE=1  CS_NUM_WORKERS  CS_POS_WEIGHT_MODE=sqrt|full|none  CS_DETERMINISTIC=0
```
"""),
    code(r"""# 1. Install only the Python packages the training harness needs.
# Use an AWS Deep Learning AMI or another CUDA-ready environment: this cell does
# not replace its PyTorch/CUDA build. boto3 1.35.16+ is needed for the
# conditional (never-overwrite) PutObject that publishes the shared split.
%pip install -q "timm==1.0.28" "scikit-learn>=1.4" "pandas>=2.1" \
    "pillow>=10" "matplotlib>=3.8" "seaborn>=0.13" "tqdm>=4.66" "psutil>=5.9" \
    "boto3>=1.35.16"
"""),
    code(r"""# 2. Configuration and EC2/GPU preflight — edit the toggles here if needed.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import contextlib, csv, gc, hashlib, json, math, multiprocessing as mp, platform
import random, re, shutil, subprocess, sys, threading, time, warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import boto3
from botocore.exceptions import ClientError
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
            "clean12 archive, which has no v01–v03 or stress views. Unset the flag "
            "and re-run."
        )
KEEP_LOCAL_TAR_AFTER_EXTRACT = env_flag("CS_KEEP_LOCAL_TAR", "0")
HARDWARE_PROFILE = "g6e.2xlarge"

# Scratch layout: NVMe instance store when present, EBS fallback otherwise.
# Bulk data (archive, extracted dataset, local checkpoints, caches) always lives
# under SCRATCH_ROOT. EBS holds only the notebook/code checkout; S3 holds every
# persistent artifact. NVMe is ephemeral — nothing irreplaceable may exist only
# there — and the resume path below re-stages it from S3 automatically.
NVME_MOUNT_POINT = Path(os.environ.get("CS_NVME_MOUNT", "/mnt/nvme")).resolve()
NVME_DEVICE_OVERRIDE = os.environ.get("CS_NVME_DEVICE", "").strip()
SCRATCH_MARGIN_GIB = float(os.environ.get("CS_SCRATCH_MARGIN_GIB", "15"))
EST_BYTES_PER_IMAGE = 1.0 * 2**20  # above the measured ~0.94 MiB mean
INTRA_EPOCH_SYNC_MINUTES = float(os.environ.get("CS_INTRA_EPOCH_SYNC_MINUTES", "15"))

def _nvme_block_devices():
    try:
        out = subprocess.run(
            ["lsblk", "-d", "-n", "-o", "NAME,TYPE"],
            capture_output=True, text=True, check=True,
        ).stdout
    except Exception:
        return []
    return [Path("/dev") / line.split()[0] for line in out.splitlines()
            if line.strip().endswith(" disk") and line.split()[0].startswith("nvme")]

def _root_disk_name():
    try:
        src = subprocess.run(
            ["findmnt", "-n", "-o", "SOURCE", "/"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "nvme0n1"
    match = re.match(r"/dev/(nvme\d+n\d+)", src)
    return match.group(1) if match else "nvme0n1"

def _device_model(device):
    try:
        return subprocess.run(
            ["lsblk", "-d", "-n", "-o", "MODEL", str(device)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return ""

def _device_fstype(device):
    try:
        return subprocess.run(
            ["blkid", "-o", "value", "-s", "TYPE", str(device)],
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return ""

def _device_size_gib(device):
    try:
        out = subprocess.run(
            ["lsblk", "-d", "-n", "-b", "-o", "SIZE", str(device)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return int(out.split()[0]) / 2**30
    except Exception:
        return 0.0

def _mounted_source(mount):
    result = subprocess.run(
        ["findmnt", "-n", "-o", "SOURCE", str(mount)],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""

def setup_nvme_scratch():
    '''Mount the EC2 NVMe instance-store volume; return (mount, device, fallback).

    The instance-store disk is identified by its "EC2 NVMe Instance Storage"
    model string and never by position, so an attached EBS volume can never be
    mistaken for scratch. Falls back to (None, None, True) when no instance
    store exists (caller uses EBS instead).
    '''
    root_disk = _root_disk_name()
    candidates = [d for d in _nvme_block_devices() if d.name != root_disk]
    for device in candidates:
        print(f"  candidate {device}: model={_device_model(device) or 'unknown'} "
              f"size={_device_size_gib(device):.0f} GiB "
              f"fstype={_device_fstype(device) or 'none'}")
    if not candidates:
        print("  no non-root NVMe disks found")
    chosen = None
    if NVME_DEVICE_OVERRIDE:
        chosen = Path(NVME_DEVICE_OVERRIDE)
        if chosen.name == root_disk:
            raise RuntimeError(f"CS_NVME_DEVICE={chosen} is the root disk; refusing.")
        if not chosen.exists():
            raise RuntimeError(f"CS_NVME_DEVICE={chosen} does not exist.")
        print(f"using explicit CS_NVME_DEVICE={chosen}")
    else:
        store_like = [d for d in candidates
                      if "instance storage" in _device_model(d).lower()]
        if store_like:
            chosen = max(store_like, key=_device_size_gib)
            print(f"instance-store match: {chosen}")
        else:
            print("no instance-store NVMe found; scratch will fall back to EBS")
            return None, None, True
    try:
        NVME_MOUNT_POINT.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # /mnt is root-owned on DLAMI: create the mount point with sudo, then
        # hand it to the notebook user so the later mount + writes succeed.
        subprocess.run(["sudo", "mkdir", "-p", str(NVME_MOUNT_POINT)], check=True)
        _user = os.environ.get("USER") or os.environ.get("LOGNAME") or "ubuntu"
        subprocess.run(
            ["sudo", "chown", f"{_user}:{_user}", str(NVME_MOUNT_POINT)],
            check=True,
        )
    mounted = _mounted_source(NVME_MOUNT_POINT)
    if mounted and Path(mounted).name == chosen.name:
        print(f"reusing {chosen} already mounted at {NVME_MOUNT_POINT}")
    elif mounted or os.path.ismount(NVME_MOUNT_POINT):
        raise RuntimeError(
            f"{NVME_MOUNT_POINT} already mounts {mounted or 'another device'}; "
            "unmount it or set CS_NVME_MOUNT to a free path."
        )
    else:
        fstype = _device_fstype(chosen)
        if not fstype:
            print(f"formatting ephemeral {chosen} as ext4 ...")
            subprocess.run(["sudo", "mkfs.ext4", "-F", str(chosen)], check=True)
        else:
            print(f"{chosen} already carries filesystem {fstype}; mounting as-is")
        subprocess.run(["sudo", "mount", str(chosen), str(NVME_MOUNT_POINT)], check=True)
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or "ubuntu"
        subprocess.run(
            ["sudo", "chown", "-R", f"{user}:{user}", str(NVME_MOUNT_POINT)],
            check=True,
        )
    free_gib = shutil.disk_usage(NVME_MOUNT_POINT).free / 2**30
    print(f"NVMe device: {chosen}; mount: {NVME_MOUNT_POINT}; free: {free_gib:.1f} GiB")
    return NVME_MOUNT_POINT, chosen.name, False

print("detecting NVMe instance store ...")
SCRATCH_MOUNT, SCRATCH_DEVICE, SCRATCH_FALLBACK = setup_nvme_scratch()

_PROJECT_ROOT_OVERRIDE = os.environ.get("CARDIOSENTRY_PROJECT_ROOT")
if _PROJECT_ROOT_OVERRIDE:
    PROJECT_ROOT = Path(_PROJECT_ROOT_OVERRIDE).expanduser().resolve()
else:
    PROJECT_ROOT = Path("/mnt/cardiosentry").resolve()

if SCRATCH_FALLBACK:
    warnings.warn(
        "No NVMe instance store detected: bulk data will use EBS-backed scratch "
        f"under {PROJECT_ROOT}. Training still reads local disk (never S3), but "
        "epoch I/O is slower and the volume must fit the archive plus extracted "
        "images (the download cell estimates the remaining need before copying)."
    )
    if not os.path.ismount(PROJECT_ROOT):
        raise RuntimeError(
            f"Fallback project root {PROJECT_ROOT} is not a mounted volume. Mount "
            "a large (600 GiB) EBS volume there, or export CARDIOSENTRY_PROJECT_ROOT "
            "to use another location."
        )
    SCRATCH_ROOT = PROJECT_ROOT / "scratch-nvme-fallback"
else:
    SCRATCH_ROOT = SCRATCH_MOUNT
print("scratch root:", SCRATCH_ROOT,
      f"(NVMe {SCRATCH_DEVICE})" if not SCRATCH_FALLBACK else "(EBS fallback)")

ARCHIVE_DIR = Path(os.environ.get(
    "CS_ARCHIVE_DIR", str(SCRATCH_ROOT / "archive")
)).expanduser().resolve()
EXTRACT_ROOT = Path(os.environ.get(
    "CS_EXTRACT_ROOT", str(SCRATCH_ROOT / "dataset")
)).expanduser().resolve()
RUN_ROOT = Path(os.environ.get(
    "CS_RUN_ROOT", str(SCRATCH_ROOT / "runs" / "convnext_v1_base_msff_clean12_ec2_v1")
)).expanduser().resolve()
HF_HOME = Path(os.environ.get(
    "HF_HOME", str(SCRATCH_ROOT / "hf_cache")
)).expanduser().resolve()
os.environ["HF_HOME"] = str(HF_HOME)
SCRATCH_TMP = SCRATCH_ROOT / "tmp"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(SCRATCH_ROOT / "torchinductor_cache"))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
for key in ("TMPDIR", "TEMP", "TMP"):
    os.environ.setdefault(key, str(SCRATCH_TMP))

# ----------------------------------------------------------------------------
# Multi-scale feature fusion (section 6). CS_FUSION picks the neck; rfpn is the
# residual FPN (fpn neck + residual head). Each choice is in the default run
# name and the fingerprint, so every variant gets its own run folder.
# ----------------------------------------------------------------------------
FUSION = os.environ.get("CS_FUSION", "rfpn").strip().lower()
FUSION_HEAD_OVERRIDE = os.environ.get("CS_FUSION_HEAD", "").strip().lower()
FUSION_CHANNELS = int(os.environ.get("CS_FUSION_CHANNELS", "256"))
BIFPN_REPEATS = int(os.environ.get("CS_BIFPN_REPEATS", "2"))
if FUSION_CHANNELS < 32 or FUSION_CHANNELS % 32:
    raise ValueError("CS_FUSION_CHANNELS must be a positive multiple of 32 (GroupNorm(32))")
if BIFPN_REPEATS < 1:
    raise ValueError("CS_BIFPN_REPEATS must be at least 1")
# Same rules as resolve_fusion() in section 6, checked here so a bad value
# fails before a run folder is named after it.
if FUSION not in ("none", "fpn", "pan", "bifpn", "rfpn"):
    raise ValueError(f"CS_FUSION must be none, fpn, pan, bifpn or rfpn, got {FUSION!r}")
if FUSION_HEAD_OVERRIDE not in ("", "replace", "residual"):
    raise ValueError(f"CS_FUSION_HEAD must be replace or residual, got {FUSION_HEAD_OVERRIDE!r}")
if FUSION == "none" and FUSION_HEAD_OVERRIDE:
    raise ValueError("CS_FUSION_HEAD needs a fusion neck (CS_FUSION=fpn|pan|bifpn|rfpn)")
if FUSION == "rfpn" and FUSION_HEAD_OVERRIDE == "replace":
    raise ValueError("CS_FUSION=rfpn is the residual FPN; use CS_FUSION=fpn for a replace head")
_fusion_residual = FUSION == "rfpn" or (FUSION != "none" and FUSION_HEAD_OVERRIDE == "residual")
# rfpn = residual FPN; rpan / rbifpn = residual PAN / BiFPN.
_fusion_tag = FUSION if FUSION in ("none", "rfpn") else ("r" if _fusion_residual else "") + FUSION
if FUSION_CHANNELS != 256 and FUSION != "none":
    _fusion_tag += f"{FUSION_CHANNELS}"
if BIFPN_REPEATS != 2 and FUSION == "bifpn":
    _fusion_tag += f"x{BIFPN_REPEATS}"
RUN_NAME = os.environ.get(
    "CS_RUN_NAME",
    f"convnext_v1_base_{_fusion_tag}_clean12_g6e2xlarge_seed{os.environ.get('CS_SEED', '42')}",
)
RUN_DIR = RUN_ROOT / RUN_NAME
for path in (PROJECT_ROOT, ARCHIVE_DIR, EXTRACT_ROOT, RUN_DIR, HF_HOME, SCRATCH_TMP):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Any path under root-owned /mnt needs sudo to create on a fresh AMI.
        subprocess.run(["sudo", "mkdir", "-p", str(path)], check=True)
        _user = os.environ.get("USER") or os.environ.get("LOGNAME") or "ubuntu"
        subprocess.run(
            ["sudo", "chown", "-R", f"{_user}:{_user}", str(path)],
            check=True,
        )
RUN_LOG_PATH = RUN_DIR / "run.log"

def log_event(event, **fields):
    '''Append one JSONL event to the run log (synced to S3 with the run).'''
    try:
        with RUN_LOG_PATH.open("a") as handle:
            handle.write(json.dumps({
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **{str(key): value for key, value in fields.items()},
            }, default=str) + "\n")
    except OSError as exc:
        warnings.warn(f"run.log write failed: {exc}")

# Persistent S3 destinations. Disable only for an intentional local-only run by
# exporting CS_S3_UPLOAD=0 before starting the notebook kernel.
S3_UPLOAD_ENABLED = env_flag("CS_S3_UPLOAD", "1")
S3_BUCKET = os.environ.get(
    "CS_S3_BUCKET", "cardiosentry"
).strip()
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
S3_DATASET_PREFIX = os.environ.get("CS_S3_DATASET_PREFIX", "dataset").strip("/")
DATASET_TAR_NAME = os.environ.get(
    "CS_DATASET_TAR_NAME", "clean_ecg_signal_images_clean12.tar"
).strip()
# Strict archive provenance: a missing TAR SHA-256 sidecar fails instead of
# falling back to byte-count plus post-extract manifest checks. The clean12
# render notebook always publishes the sidecar, so strict is the default here;
# CS_REQUIRE_TAR_SHA256=0 allows the weaker fallback.
REQUIRE_TAR_SHA256 = env_flag("CS_REQUIRE_TAR_SHA256", "1")
S3_RUNS_PREFIX = os.environ.get(
    "CS_S3_RUNS_PREFIX", "runs"
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

_sync_threads = []
_sync_threads_lock = threading.Lock()
_upload_key_threads = {}
_upload_key_lock = threading.Lock()

def _prune_sync_threads():
    with _sync_threads_lock:
        _sync_threads[:] = [t for t in _sync_threads if t.is_alive()]
        return len(_sync_threads)

def spawn_s3_upload(local_path, key):
    '''Start a daemon-thread upload of one finished local artifact.

    Failures warn but never raise, so training never blocks on S3. Call
    join_s3_uploads() before any upload that must observe a quiesced state.
    Same-key uploads are serialized: if an upload of this key is already in
    flight, the new request is skipped so an older completion can never
    overwrite a newer checkpoint under the same key.
    '''
    if not S3_UPLOAD_ENABLED:
        return None
    key = key.lstrip("/")
    with _upload_key_lock:
        existing = _upload_key_threads.get(key)
        if existing is not None and existing.is_alive():
            warnings.warn(
                f"skipping S3 upload of {local_path}: an upload of {key} "
                "is already in flight"
            )
            return existing
    _prune_sync_threads()
    def _target():
        try:
            upload_file_to_s3(Path(local_path), key, required=False)
        except Exception as exc:
            warnings.warn(f"background S3 upload failed for {local_path}: {exc}")
    thread = threading.Thread(
        target=_target, name=f"s3-upload-{Path(local_path).name}", daemon=True,
    )
    with _sync_threads_lock:
        _sync_threads.append(thread)
    with _upload_key_lock:
        _upload_key_threads[key] = thread
    thread.start()
    return thread

def join_s3_uploads(timeout=None):
    '''Wait for outstanding background uploads (epoch boundaries, run end).'''
    with _sync_threads_lock:
        pending = list(_sync_threads)
    for thread in pending:
        thread.join(timeout=timeout)
    remaining = _prune_sync_threads()
    with _upload_key_lock:
        for finished in [k for k, t in _upload_key_threads.items() if not t.is_alive()]:
            del _upload_key_threads[finished]
    return remaining

def download_file_from_s3(key, local_path, required=False):
    '''Download one run artifact, then verify the local byte count.

    Only a genuine absent object (404 / NoSuchKey) is treated as "does not
    exist". Any other S3 failure (permissions, network, throttling) raises
    loudly: silently treating it as absent would make an existing run look
    fresh and restart training over its checkpoints.
    '''
    local_path = Path(local_path)
    if not S3_UPLOAD_ENABLED:
        return None
    key = key.lstrip("/")
    try:
        expected_bytes = int(S3_CLIENT.head_object(Bucket=S3_BUCKET, Key=key)["ContentLength"])
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound", "NoSuchBucket"):
            if required:
                raise FileNotFoundError(f"S3 object not found: {s3_uri(key)}")
            return None
        raise RuntimeError(
            f"S3 HEAD failed for {s3_uri(key)} (code={code}): check the EC2 IAM "
            "role, region, and network instead of assuming a fresh run."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"S3 HEAD failed for {s3_uri(key)}: {exc}. Refusing to assume a "
            "fresh run when S3 itself may be unreachable."
        ) from exc
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
        # The object is known to exist (HEAD succeeded), so a failed GET must
        # never degrade to "missing": a half-restored run directory would
        # resume incorrectly or restart over existing checkpoints.
        raise RuntimeError(
            f"S3 download failed for {s3_uri(key)} -> {local_path}: {exc}"
        ) from exc
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

# Patient-level split, built at training time (the archive is unsplit). It is
# a deterministic function of record_manifest.csv and CS_SPLIT_SEED only, never
# of CS_SEED, so every training seed and every model sees the same split.
# CS_SPLIT_FILE (local path or s3:// URI of a patient_key,split CSV, e.g. a
# previous run's patient_split.csv) reuses a frozen split instead.
SPLIT_VERSION = "clean12_patient_split_v1"
SPLIT_SEED = int(os.environ.get("CS_SPLIT_SEED", "20261002"))
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
SPLIT_FILE = os.environ.get("CS_SPLIT_FILE", "").strip()
# The default seed must reproduce the frozen split every clean12 model uses, so
# every clean12 model (V1, V2, fusion) is on the same patients. Checked in section 4b.
EXPECTED_SPLIT_SHA256 = "4eba80c71bba27e8bde0ed1a510de9d4819ede5139946b48526461a7bc4841cd"
# Shared frozen copy of the split, stored next to the dataset (not inside it).
# The first run that computes the split publishes it here and never overwrites
# it; every later run (any model, any CS_SEED) loads and verifies this copy, so
# the split never has to be shipped in another TAR. CS_SPLIT_S3_URI="" turns
# the shared copy off; CS_SPLIT_PUBLISH=0 reads it but never publishes.
SPLIT_S3_URI = os.environ.get(
    "CS_SPLIT_S3_URI", f"s3://{S3_BUCKET}/{S3_DATASET_PREFIX}/splits/{SPLIT_VERSION}.csv"
).strip()
SPLIT_PUBLISH = env_flag("CS_SPLIT_PUBLISH", "1")
# Acceptance gates: every split within 1 point of its record share, and every
# label within 1.5 points of its target share in every split.
SPLIT_RECORD_TOLERANCE = 0.01
SPLIT_LABEL_TOLERANCE = 0.015

# The ConvNeXt-V1 Base clean12 recipe (../convnext_v1_base_clean12_v1/),
# unchanged, so a fusion run differs from the plain V1 Base run only in the neck.
MODEL_NAME = "convnext_base.fb_in22k_ft_in1k_384"
INPUT_SIZE = (768, 1024)                 # (height, width)
# A 20-epoch cosine lets the LR actually anneal: with a 40-epoch cosine,
# early stopping fired around epoch 20-25 while LR was still ~60-75% of peak.
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_EPOCHS = 10
WARMUP_EPOCHS = 2
# Peak LR of the classifier and the fusion neck. Override only with a new
# CS_RUN_NAME (e.g. after a collapse-guard stop); it is part of the fingerprint.
BASE_LR = float(os.environ.get("CS_BASE_LR", "2e-4"))
MIN_LR_RATIO = 0.01                     # head LR floor 2e-6
WEIGHT_DECAY = 0.02
# timm's param_groups_layer_decay gives every ConvNeXt block its own layer
# (38 levels for depths 3/3/27/3), unlike Meta's recipe, which groups stage-3
# blocks in threes (13 levels). 0.8 per block would leave the stem at
# 0.8**37 ~ 2.6e-4x the head LR (effectively frozen); 0.925 per block puts it
# at ~0.056x, matching Meta's 0.8**13 = 0.055.
LAYER_DECAY = 0.925
DROP_PATH_RATE = 0.20
HEAD_INIT_SCALE = float(os.environ.get("CS_HEAD_INIT_SCALE", "0.001"))
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
# Off by default here, so a fusion run differs from the plain V1 Base clean12
# run (uniform epochs) only in the neck. CS_RARE_SAMPLING=1 for the V2 Colab setting.
RARE_SAMPLING = env_flag("CS_RARE_SAMPLING", "0")
RARE_SAMPLING_THRESHOLD = float(os.environ.get("CS_RARE_SAMPLING_THRESHOLD", "0.10"))
if not 0.0 < RARE_SAMPLING_THRESHOLD < 1.0:
    raise ValueError("CS_RARE_SAMPLING_THRESHOLD must be in (0, 1)")
# Patient-bootstrap confidence intervals for the final test metrics.
BOOTSTRAP_ROUNDS = int(os.environ.get("CS_BOOTSTRAP_ROUNDS", "1000"))
BOOTSTRAP_CONFIDENCE = 0.95
# Baseline uses plain weighted BCE. The noisy-OR contradiction term saturates
# with many pathologies and would mostly suppress NORM; re-enable it only
# as an explicit ablation with its own CS_RUN_NAME.
CONTRADICTION_LAMBDA = 0.0
GRAD_CLIP_NORM = 1.0
SEED = int(os.environ.get("CS_SEED", "42"))
if not 0 <= SEED < 2**32:
    raise ValueError("CS_SEED must be in [0, 2**32)")
NUM_WORKERS = int(os.environ.get(
    "CS_NUM_WORKERS", str(min(6, max(1, (os.cpu_count() or 2) - 2)))
))
AUTO_TUNE_WORKERS = env_flag("CS_AUTO_TUNE_WORKERS", "1") and "CS_NUM_WORKERS" not in os.environ
WORKER_CANDIDATES = [value for value in (4, 6, 8) if value <= (os.cpu_count() or 1)] or [0]
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
AUTO_BATCH_CANDIDATES = [32, 16, 8, 4, 2, 1]
EVAL_BATCH_CANDIDATES = [64, 32, 16, 8, 4, 2, 1]
BATCH_PROBE_TIMED_UPDATES = 2
EVAL_PROBE_TIMED_BATCHES = 2
VRAM_SAFETY_FRACTION = 0.90
USE_AMP = True
DETERMINISTIC = env_flag("CS_DETERMINISTIC", "0")
USE_FUSED_ADAMW = env_flag("CS_FUSED_ADAMW", "1") and not DETERMINISTIC
TORCH_COMPILE_ENABLED = env_flag("CS_TORCH_COMPILE", "1") and not DETERMINISTIC
TORCH_COMPILE_MODE = "max-autotune-no-cudagraphs"
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
print("CPU count:          ", os.cpu_count())
print(f"RAM:                 {gib(psutil.virtual_memory().total):.1f} GiB")
print("dataset S3 source:  ", s3_uri(S3_DATASET_KEY))
print("NVMe device:        ", SCRATCH_DEVICE if not SCRATCH_FALLBACK else "none (EBS fallback)")
print("scratch root:       ", SCRATCH_ROOT)
print("local archive:      ", LOCAL_TAR)
print("dataset path:       ", EXTRACT_ROOT / KIT_TOPDIR)
print("split:              ", f"{SPLIT_VERSION} seed={SPLIT_SEED}",
      f"(frozen file {SPLIT_FILE})" if SPLIT_FILE
      else f"(shared copy {SPLIT_S3_URI}; computed and published if absent)" if SPLIT_S3_URI
      else "(computed from record_manifest.csv)")
print("run directory:      ", RUN_DIR)
print("HF cache:           ", HF_HOME)
print("rare-label sampling:", f"on, threshold {RARE_SAMPLING_THRESHOLD}" if RARE_SAMPLING else "off")
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
        "purpose": "CardioSentry clean12 training artifact upload preflight",
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
if "L40S" not in gpu.name or psutil.virtual_memory().total < 60 * 2**30:
    warnings.warn("This notebook targets g6e.2xlarge (L40S, 64 GiB RAM); measured tuning will use the visible hardware.")
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
        "This notebook was only validated for BF16 (the L40S supports it)."
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
        "Install one and restart the kernel: sudo apt-get install -y build-essential."
    )
print("fusion:              ", FUSION,
      f"(head override {FUSION_HEAD_OVERRIDE})" if FUSION_HEAD_OVERRIDE else "",
      f"channels {FUSION_CHANNELS}" if FUSION != "none" else "",
      f"repeats {BIFPN_REPEATS}" if FUSION == "bifpn" else "")
print("collapse guard:      ", COLLAPSE_GUARD_ENABLED)
print(f"peak head LR:         {BASE_LR:.2e}")


def seed_global_rngs(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_global_rngs(SEED)
print("reproducibility:     ", "deterministic opt-in" if DETERMINISTIC else "seeded performance mode")

# Exact package provenance for this run. timm is hard-pinned in the install
# cell (the head-init and weight-tag behavior above was verified against
# 1.0.28); torch/torchvision ride with the DLAMI and are recorded, not
# pinned, so a working CUDA build is never replaced by pip. The versions are
# stored in the run config and a drift warning fires on resume, so a Spot
# replacement can never silently train on a different implementation.
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
        "matplotlib", "seaborn", "tqdm", "psutil", "boto3", "numpy",
    )},
}
print("package versions:    ", json.dumps(PACKAGE_VERSIONS, sort_keys=True))
"""),
    md(r"""## 3. Download, verify, and extract the clean12 archive

This cell ensures the corpus is on local disk: it downloads the archive from
S3 when the local copy is missing, verifies its SHA-256 against the sidecar
(required by default; `CS_REQUIRE_TAR_SHA256=0` falls back to byte count plus
post-extract manifests), and extracts the metadata plus all images under
`clean_ecg_signal_images_clean12/`. Each selection class (metadata, I0001
images, I0006 images) writes a marker recording the archive identity, so
re-running extracts only what is still missing. The clean12 TAR is ~104 GiB and
extracts to ~104 GiB; the space check asks for ~225 GiB free on one volume
(archive + images + 15 GiB margin). The 450 GB g6e.2xlarge NVMe has room.
"""),
    code(r"""def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()

def expected_tar_sha256():
    '''SHA-256 authority when published, else None (size + manifests verify).'''
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
            message = str(exc)
            if any(key in message for key in ("NoSuchKey", "404", "NotFound", "NoSuchBucket")):
                if REQUIRE_TAR_SHA256:
                    raise RuntimeError(
                        f"No checksum sidecar at {s3_uri(S3_DATASET_KEY + '.sha256')} "
                        "and CS_REQUIRE_TAR_SHA256=1: refusing to train without "
                        "archive-hash authority."
                    ) from exc
                warnings.warn(
                    f"No checksum sidecar at {s3_uri(S3_DATASET_KEY + '.sha256')}; "
                    "verifying by S3 byte count plus post-extract manifests instead."
                )
                return None
            raise RuntimeError(
                f"Cannot read the dataset checksum sidecar "
                f"{s3_uri(S3_DATASET_KEY + '.sha256')}. The EC2 IAM role needs "
                f"s3:GetObject on {S3_DATASET_PREFIX}/, or place "
                f"{local_sidecar.name} beside the archive."
            ) from exc
    if len(text) != 64:
        raise ValueError("Malformed SHA-256 sidecar for the dataset archive")
    return text

def download_dataset_tar(expected_sha) -> None:
    if expected_sha is not None and LOCAL_TAR.is_file() and sha256_file(LOCAL_TAR) == expected_sha:
        print("local archive already verified; skipping download")
        return
    if LOCAL_TAR.is_file() and expected_sha is not None:
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
    progress = tqdm(total=total, unit="B", unit_scale=True, desc="downloading clean12 archive")
    def callback(chunk):
        progress.update(chunk)
    temporary = LOCAL_TAR.with_suffix(LOCAL_TAR.suffix + ".tmp")
    try:
        dataset_client.download_file(
            S3_BUCKET, S3_DATASET_KEY, str(temporary), Callback=callback
        )
    finally:
        progress.close()
    print("verifying download ...", flush=True)
    if temporary.stat().st_size != total:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Downloaded archive byte count differs from S3 ContentLength")
    if expected_sha is not None:
        if sha256_file(temporary) != expected_sha:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Downloaded archive SHA-256 mismatch; partial file removed")
        print("download hash: OK")
    else:
        print("download size: OK (no sidecar; manifests verify after extraction)")
    os.replace(temporary, LOCAL_TAR)
    print(f"download verified: {LOCAL_TAR} ({gib(LOCAL_TAR.stat().st_size):.1f} GiB)")

EXPECTED_TAR_SHA256 = expected_tar_sha256()
if EXPECTED_TAR_SHA256 is not None:
    print("expected archive sha256:", EXPECTED_TAR_SHA256[:16] + "...")
else:
    print("expected archive sha256: none published (size + manifest verification)")

# Extraction selection classes. The clean12 plain-TAR holds exactly these, so
# every class is always required. Images are grouped by HEEDB site (the archive
# is unsplit). Markers make re-runs incremental. Patterns are relative to the
# archive root and always carry the KIT_TOPDIR prefix.
EXTRACTION_CLASSES = {
    "meta": {"patterns": [
        f"{KIT_TOPDIR}/manifest.csv",
        f"{KIT_TOPDIR}/record_manifest.csv",
        f"{KIT_TOPDIR}/final_counts.csv",
        f"{KIT_TOPDIR}/selection_report.json",
        f"{KIT_TOPDIR}/summary.json",
        f"{KIT_TOPDIR}/config.json",
        f"{KIT_TOPDIR}/failures.csv",
        f"{KIT_TOPDIR}/README.md",
        f"{KIT_TOPDIR}/SUMMARY.md",
    ], "exclude": []},
    **{f"{site}-images": {"patterns": [f"{KIT_TOPDIR}/images/{site}/*"], "exclude": []}
       for site in SITES},
}
REQUIRED_CLASSES = ["meta"] + [f"{site}-images" for site in SITES]
KIT_ROOT = EXTRACT_ROOT / KIT_TOPDIR
MANIFEST_CSV = KIT_ROOT / "manifest.csv"
RECORD_MANIFEST_CSV = KIT_ROOT / "record_manifest.csv"
FINAL_COUNTS_CSV = KIT_ROOT / "final_counts.csv"

def extraction_marker(name):
    return EXTRACT_ROOT / f".clean12_extract_{name}.json"

def extraction_complete(name):
    marker = extraction_marker(name)
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("sha256") == EXPECTED_TAR_SHA256

dataset_copy_started = time.time()
print(f"scratch for dataset: {SCRATCH_ROOT} "
      f"(device={SCRATCH_DEVICE if not SCRATCH_FALLBACK else 'EBS fallback'})")
def estimate_remaining_scratch_need_gib(missing):
    # Conservative bytes still needed: tar (if absent) + missing PNGs + margin.
    need_gib, archive_need_gib, detail = 0.0, 0.0, {}
    image_classes = [c for c in missing if c != "meta"]
    if missing and not LOCAL_TAR.is_file():
        # Dataset download needs only S3 GetObject, independent of the
        # CS_S3_UPLOAD artifact-upload flag, so use a dedicated client here
        # (S3_CLIENT is None for local-only runs) and never gate on it.
        try:
            dataset_client = boto3.client("s3", region_name=AWS_REGION)
            tar_bytes = int(dataset_client.head_object(
                Bucket=S3_BUCKET, Key=S3_DATASET_KEY)["ContentLength"])
        except Exception:
            tar_bytes = int(110 * 2**30)  # conservative fallback
        archive_need_gib = tar_bytes / 2**30
        need_gib += archive_need_gib
        detail["archive_gib"] = round(tar_bytes / 2**30, 1)
    if image_classes:
        if "meta" not in missing and MANIFEST_CSV.is_file():
            # Exact: sum manifest bytes for PNGs of missing classes absent disk.
            manifest_need = pd.read_csv(
                MANIFEST_CSV, usecols=["image_path", "image_bytes"],
                low_memory=False)
            for cls in image_classes:
                site = cls.split("-")[0]
                rows = manifest_need[manifest_need["image_path"].str.startswith(
                    f"images/{site}/")]
                missing_bytes = sum(
                    int(n) for path, n in zip(rows["image_path"], rows["image_bytes"])
                    if not (KIT_ROOT / path).is_file())
                need_gib += missing_bytes / 2**30
                detail[f"{cls}_gib"] = round(missing_bytes / 2**30, 1)
        else:
            # Manifests not staged yet: conservative per-class estimates.
            for cls in image_classes:
                site = cls.split("-")[0]
                est = (EXPECTED_RECORDS_BY_SITE[site] * EST_BYTES_PER_IMAGE
                       / 2**30)
                need_gib += est
                detail[f"{cls}_gib_est"] = round(est, 1)
    need_gib += SCRATCH_MARGIN_GIB
    detail["margin_gib"] = SCRATCH_MARGIN_GIB
    return need_gib, archive_need_gib, detail

missing_classes = [name for name in REQUIRED_CLASSES if not extraction_complete(name)]
if missing_classes:
    need_gib, archive_need_gib, need_detail = estimate_remaining_scratch_need_gib(missing_classes)
    if ARCHIVE_DIR.stat().st_dev == EXTRACT_ROOT.stat().st_dev:
        volume_requirements = [(EXTRACT_ROOT, need_gib)]
    else:
        # A mounted EBS archive volume permits staging on a smaller NVMe disk.
        # Never count archive bytes against the extraction volume in this case.
        volume_requirements = [
            (ARCHIVE_DIR, archive_need_gib + SCRATCH_MARGIN_GIB),
            (EXTRACT_ROOT, need_gib - archive_need_gib),
        ]
    for volume_path, volume_need_gib in volume_requirements:
        free_gib = gib(shutil.disk_usage(volume_path).free)
        print(f"{volume_path}: free={free_gib:.1f} GiB; "
              f"remaining need={volume_need_gib:.1f} GiB {need_detail}")
        if free_gib < volume_need_gib:
            raise RuntimeError(
                f"{volume_path} has {free_gib:.1f} GiB free but needs "
                f"~{volume_need_gib:.1f} GiB. Free space or mount a larger volume "
                "and set CS_ARCHIVE_DIR/CS_EXTRACT_ROOT."
            )
else:
    print("extracted dataset already present on scratch; skipping S3 copy")
if missing_classes and not LOCAL_TAR.is_file():
    print("local archive missing but extraction is incomplete; downloading ...")
    download_dataset_tar(EXPECTED_TAR_SHA256)
elif LOCAL_TAR.is_file():
    if EXPECTED_TAR_SHA256 is not None:
        print("hashing local archive ...", flush=True)
        actual_sha = sha256_file(LOCAL_TAR)
        if actual_sha != EXPECTED_TAR_SHA256:
            raise RuntimeError(
                f"Local archive SHA-256 mismatch: expected {EXPECTED_TAR_SHA256}, "
                f"got {actual_sha}. Delete it and re-run to re-download."
            )
        print("local archive hash: OK")
    else:
        print(f"local archive present ({gib(LOCAL_TAR.stat().st_size):.1f} GiB); "
              "no sidecar, manifests verify after extraction")
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

for required in [MANIFEST_CSV, RECORD_MANIFEST_CSV, FINAL_COUNTS_CSV,
                 KIT_ROOT / "summary.json"]:
    if not required.is_file():
        raise FileNotFoundError(
            f"Expected {required} after extraction. If the archive layout changed, "
            "inspect the TAR listing before proceeding."
        )

# Light count check here; the audit cell joins manifest.csv against
# record_manifest.csv and verifies every image file byte-existence.
on_disk_images = {
    path.relative_to(KIT_ROOT).as_posix()
    for site in SITES
    if (KIT_ROOT / "images" / site).is_dir()
    for path in (KIT_ROOT / "images" / site).rglob("*.png")
}
if not on_disk_images:
    raise RuntimeError(f"No PNG images found under {KIT_ROOT / 'images'}")
print(f"extracted images on disk: {len(on_disk_images):,} "
      f"(classes: {', '.join(sorted(REQUIRED_CLASSES))})")
FULL_METADATA_CSV = RECORD_MANIFEST_CSV

if LOCAL_TAR.is_file() and not KEEP_LOCAL_TAR_AFTER_EXTRACT:
    print(f"removing local archive to free {gib(LOCAL_TAR.stat().st_size):.1f} GiB ...")
    LOCAL_TAR.unlink()
    print("archive removed; S3 remains the source of truth for re-downloads")
copy_seconds = time.time() - dataset_copy_started
scratch_free_gib = gib(shutil.disk_usage(EXTRACT_ROOT).free)
print(f"dataset copy/extract time: {copy_seconds / 60:.1f} min")
print("final dataset path:", KIT_ROOT)
print(f"scratch free after staging: {scratch_free_gib:.1f} GiB")
log_event("dataset_ready", dataset_path=str(KIT_ROOT),
          copy_seconds=round(copy_seconds, 1),
          scratch_device=SCRATCH_DEVICE if not SCRATCH_FALLBACK else "ebs-fallback",
          scratch_free_gib=round(scratch_free_gib, 1))
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
`4eba80c7…`, the frozen split the V1 Base clean12 run uses. So both models are
evaluated on exactly the same patients.

Source order: `CS_SPLIT_FILE` (a local path or an `s3://` URI), then the
shared frozen copy at
`s3://cardiosentry/dataset/splits/clean12_patient_split_v1.csv` when it exists
(its sidecar, audit SHA-256, seed, and source manifest are all verified), then
computing it and publishing the shared copy once (conditional write, never
overwritten; `CS_SPLIT_PUBLISH=0` turns publishing off). `patient_split.csv`,
`record_split.csv`, and `split_audit.json` are uploaded with the run to S3. The split's SHA-256 is part of the
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

def parse_s3_uri(uri):
    if not uri.startswith("s3://") or "/" not in uri[len("s3://"):]:
        raise ValueError(f"Not an s3://bucket/key URI: {uri!r}")
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key

def load_frozen_split(location):
    if location.startswith("s3://"):
        bucket, key = parse_s3_uri(location)
        local = SCRATCH_TMP / "frozen_patient_split.csv"
        boto3.client("s3", region_name=AWS_REGION).download_file(bucket, key, str(local))
    else:
        local = Path(location).expanduser()
    return read_split_csv(local, location)

def shared_split_keys(uri):
    # <name>.csv (patient -> split, the authority), <name>.csv.sha256,
    # <name>_audit.json, and <name>_records.csv (record -> split, convenience).
    bucket, key = parse_s3_uri(uri)
    if not key.endswith(".csv"):
        raise ValueError(f"CS_SPLIT_S3_URI must end in .csv: {uri}")
    base = key[:-len(".csv")]
    return bucket, {"patients": key, "sha256": key + ".sha256",
                    "audit": base + "_audit.json", "records": base + "_records.csv"}

def s3_object_exists(client, bucket, key):
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise RuntimeError(
            f"S3 HEAD failed for s3://{bucket}/{key} (code={code}). The role needs "
            "s3:GetObject on that key and s3:ListBucket on the bucket, or set "
            "CS_SPLIT_S3_URI= to skip the shared split."
        ) from exc

def load_shared_split(client, uri):
    '''Download the shared split, check its SHA-256 sidecar, return (split, audit).'''
    bucket, keys = shared_split_keys(uri)
    local = SCRATCH_TMP / Path(keys["patients"]).name
    client.download_file(bucket, keys["patients"], str(local))
    expected = client.get_object(Bucket=bucket, Key=keys["sha256"])["Body"].read().decode().split()[0]
    if sha256_file(local) != expected:
        raise RuntimeError(f"{uri} does not match its .sha256 sidecar; refusing to train on it")
    audit = json.loads(client.get_object(Bucket=bucket, Key=keys["audit"])["Body"].read())
    return read_split_csv(local, uri), audit

def publish_shared_split(client, uri, files):
    '''Publish once, never overwrite: the patient CSV goes last, conditionally.'''
    bucket, keys = shared_split_keys(uri)
    for name in ("audit", "records", "sha256"):
        client.upload_file(str(files[name]), bucket, keys[name])
    try:
        with open(files["patients"], "rb") as handle:
            client.put_object(Bucket=bucket, Key=keys["patients"], Body=handle, IfNoneMatch="*")
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code", "") in ("PreconditionFailed", "412"):
            raise RuntimeError(
                f"Another run published {uri} while this one was computing. Re-run this "
                "cell: it will load and verify the published split."
            ) from exc
        raise
    remote = client.head_object(Bucket=bucket, Key=keys["patients"])
    if int(remote["ContentLength"]) != Path(files["patients"]).stat().st_size:
        raise RuntimeError(f"Size check failed after publishing {uri}")
    print("published shared split:", ", ".join(f"s3://{bucket}/{key}" for key in keys.values()))

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
# Source, in order: CS_SPLIT_FILE; else the shared frozen copy in S3; else
# compute it (and publish it below so every later run reuses it).
shared_split_client = boto3.client("s3", region_name=AWS_REGION) if SPLIT_S3_URI else None
shared_bucket, shared_keys = shared_split_keys(SPLIT_S3_URI) if SPLIT_S3_URI else (None, None)
publish_split = False
if SPLIT_FILE:
    patient_split = load_frozen_split(SPLIT_FILE)
    split_source = f"frozen file {SPLIT_FILE}"
elif SPLIT_S3_URI and s3_object_exists(shared_split_client, shared_bucket, shared_keys["patients"]):
    patient_split, shared_audit = load_shared_split(shared_split_client, SPLIT_S3_URI)
    split_source = f"shared copy {SPLIT_S3_URI} (sha256 sidecar verified)"
    if (shared_audit.get("split_version") != SPLIT_VERSION
            or int(shared_audit.get("split_seed", -1)) != SPLIT_SEED):
        raise RuntimeError(
            f"{SPLIT_S3_URI} was built as {shared_audit.get('split_version')} seed "
            f"{shared_audit.get('split_seed')}, but this run asks for {SPLIT_VERSION} seed "
            f"{SPLIT_SEED}. Point CS_SPLIT_S3_URI at a new name for a different split."
        )
    if shared_audit.get("record_manifest_sha256") != METADATA_SHA256:
        raise RuntimeError(
            f"{SPLIT_S3_URI} was built for a different record_manifest.csv "
            f"({shared_audit.get('record_manifest_sha256')}); this archive has {METADATA_SHA256}."
        )
    if split_sha256(patient_split) != shared_audit.get("split_sha256"):
        raise RuntimeError(f"{SPLIT_S3_URI} does not match the split_sha256 in its audit file")
    # Informational only: the frozen copy wins even if today's code would differ.
    if split_sha256(stratified_patient_split(record_df, LABELS, SPLIT_RATIOS, SPLIT_SEED)) != shared_audit["split_sha256"]:
        warnings.warn("The split code now produces a different assignment than the shared "
                      "frozen copy; using the frozen copy, as intended.")
else:
    patient_split = stratified_patient_split(record_df, LABELS, SPLIT_RATIOS, SPLIT_SEED)
    split_source = "computed from record_manifest.csv"
    publish_split = bool(SPLIT_S3_URI) and SPLIT_PUBLISH and S3_UPLOAD_ENABLED
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
# On a fresh machine, first pull this run's split back from S3 (if it exists)
# so a resumed run is compared against its real split, never overwritten.
patient_split_path = RUN_DIR / "patient_split.csv"
split_audit_path = RUN_DIR / "split_audit.json"
if not patient_split_path.is_file():
    download_file_from_s3(f"{S3_RUN_PREFIX}/{patient_split_path.name}", patient_split_path)
if patient_split_path.is_file():
    existing_split = pd.read_csv(patient_split_path, dtype={"patient_key": str, "split": str},
                                 keep_default_na=False).set_index("patient_key")["split"]
    if split_sha256(existing_split) != SPLIT_SHA256:
        raise RuntimeError(
            f"{patient_split_path} holds a different patient split. Choose a new "
            "CS_RUN_NAME, or pass that file as CS_SPLIT_FILE to reuse it."
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
sidecar_name = Path(shared_keys["patients"]).name if SPLIT_S3_URI else patient_split_path.name
patient_split_sidecar_path.write_text(f"{sha256_file(patient_split_path)}  {sidecar_name}\n")
if publish_split:
    publish_shared_split(shared_split_client, SPLIT_S3_URI, {
        "patients": patient_split_path, "sha256": patient_split_sidecar_path,
        "audit": split_audit_path, "records": record_split_path,
    })
elif SPLIT_S3_URI and split_source.startswith("computed"):
    print(f"shared split not published (CS_SPLIT_PUBLISH={int(SPLIT_PUBLISH)}, "
          f"CS_S3_UPLOAD={int(S3_UPLOAD_ENABLED)}); later runs will compute the same split")
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
if S3_UPLOAD_ENABLED:
    upload_run_files([patient_split_path, patient_split_sidecar_path, split_audit_path,
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
# save the sample grid to the run directory and S3 alongside benchmark.json.
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
if S3_UPLOAD_ENABLED:
    upload_file_to_s3(sample_grid_path, f"{S3_RUN_PREFIX}/{sample_grid_path.name}", required=True)
print("Inspect the sample grid and benchmark before starting the long training cell.")

"""),
    md(r"""## 6. Model, fusion neck, and loss

Twelve independent sigmoid outputs (NORM plus the 11 conditions) on the
supervised ImageNet-22K → ImageNet-1K (384 px) ConvNeXt-V1 Base checkpoint,
with the `CS_FUSION` neck from the table at the top. Train with
square-root-weighted BCE (`CONTRADICTION_LAMBDA = 0.0`): the noisy-OR
contradiction term saturates with many conditions and would mostly suppress
NORM instead of teaching exclusivity. Re-enable it only as an explicit ablation run.

The next cell is `fusion_necks.py` from this notebook's source folder, copied in
verbatim (the validator checks it is identical), so the notebook stays
self-contained. The backbone runs exactly as timm's: stem, then the four
stages; the neck reads the outputs of stages 2–4. In the `residual` head the
timm head still reads stage 4, and the fusion branch is added to its logits
through a zero-initialised Linear, so at step 0 the model *is* the plain V1
Base and the branch only contributes what the data supports. In the `replace`
head the timm head is removed and a new LayerNorm + Linear reads the pooled
fusion levels.

Checkpoint keys are the timm keys under `backbone.` plus `neck.*`, `fuse_norm.*`
and `fuse_fc.*`; `inference.pt` records the fusion settings needed to rebuild it.
"""),
    code(FUSION_SOURCE),
    code(r"""FUSION_NECK, FUSION_HEAD = resolve_fusion(FUSION, FUSION_HEAD_OVERRIDE)
FUSION_SPEC = fusion_description(FUSION_NECK, FUSION_HEAD, FUSION_CHANNELS, BIFPN_REPEATS)

def build_model(pretrained: bool) -> nn.Module:
    backbone = timm.create_model(
        MODEL_NAME,
        pretrained=pretrained,
        # Explicit cache_dir also covers libraries imported before HF_HOME was
        # set, so weight downloads really use the configured cache.
        cache_dir=str(HF_HOME) if pretrained else None,
        num_classes=len(LABELS),
        drop_path_rate=DROP_PATH_RATE,
        # Meta's ConvNeXt-V1 fine-tuning recipe uses a small classifier init so
        # large early gradients do not thrash the pretrained backbone.
        head_init_scale=HEAD_INIT_SCALE,
    )
    if FUSION_NECK == "none":
        return backbone
    return ConvNeXtMultiScaleFusion(
        backbone, FUSION_NECK, FUSION_HEAD, len(LABELS),
        channels=FUSION_CHANNELS, bifpn_repeats=BIFPN_REPEATS,
        head_init_scale=HEAD_INIT_SCALE,
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

model = build_model(pretrained=True)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
print(model.default_cfg)
print(f"parameters: {parameter_count / 1e6:.2f} M total; {trainable_count / 1e6:.2f} M trainable")
print("classifier outputs:", model.get_classifier().out_features)
assert model.get_classifier().out_features == len(LABELS)
fusion_parameter_count = (
    sum(p.numel() for m in model.new_modules() for p in m.parameters())
    if isinstance(model, ConvNeXtMultiScaleFusion) else 0
)
print(f"fusion: neck={FUSION_NECK} head={FUSION_HEAD}; "
      f"{fusion_parameter_count / 1e6:.2f} M new fusion parameters")
print(json.dumps(FUSION_SPEC, indent=2))

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
    # timm decays per block; LAYER_DECAY = 0.925 reproduces the LR profile of
    # Meta's 12-group layer decay 0.8. The new classifier gets the highest LR.
    # With a fusion neck the backbone keeps the plain model's per-block profile
    # and the neck + new classifier train at the full LR (fusion_necks.py).
    parameter_groups = fusion_param_groups(
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
        parameter_groups, lr=BASE_LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY,
        fused=USE_FUSED_ADAMW and next(active_model.parameters()).is_cuda,
    )
    return optimizer

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
_, pathology_only_parts = criterion_cpu(
    torch.tensor([[-20.0] + [20.0] * len(PATHOLOGY_LABELS)]),
    torch.tensor([[0.0] + [1.0] * len(PATHOLOGY_LABELS)]),
)
assert pathology_only_parts["contradiction"] < 1e-6
print("loss smoke test:", float(dummy_loss), {k: float(v) for k, v in dummy_parts.items()})
"""),
    md(r"""## 7. Tune training batches, evaluation batches, and PNG workers

The probe tests 32/16/8/... images per micro-batch against **90% of actual
CUDA VRAM** (48 GB on the L40S), including AdamW, gradients, and EMA. Each throughput measurement
uses complete effective-batch-64 updates. Profile the first three safe sizes
and select the fastest measured size, rather than assuming the largest is best.
A separate forward-only probe selects a larger evaluation batch while the
training model, optimizer state, and EMA are still resident.

Then time PNG loading with 4/6/8 workers (g6e.2xlarge has 8 vCPUs), using
uint8 batches and prefetch 4. An explicit `CS_NUM_WORKERS` bypasses that tuning.
Global and train-shuffle RNGs are restored afterwards. No images are duplicated
into a raw-pixel cache: the 450 GB NVMe holds the corpus and leaves space for
checkpoints and the compiler cache.
"""),
    code(r"""def make_grad_scaler():
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

def probe_micro_batch(candidates):
    total_vram = torch.cuda.get_device_properties(0).total_memory
    base_state = model.state_dict()  # the real pretrained model stays on CPU
    rows = []
    safe_sizes = 0
    for batch_size in candidates:
        if EFFECTIVE_BATCH_SIZE % batch_size:
            continue
        probe = probe_ema = probe_criterion = probe_optimizer = probe_scaler = None
        pixels = inputs = targets = factors = logits = loss = None
        try:
            torch.cuda.empty_cache()
            probe = build_model(pretrained=False)
            probe.load_state_dict(base_state, strict=True)
            probe = probe.to(DEVICE, memory_format=torch.channels_last).train()
            probe_ema = build_model_ema(probe)
            probe_criterion = CardioSentryLoss(POS_WEIGHT, CONTRADICTION_LAMBDA).to(DEVICE)
            probe_optimizer = build_optimizer(probe)
            probe_scaler = make_grad_scaler()
            pixels = torch.randint(0, 256, (batch_size, 3, *INPUT_SIZE),
                                   dtype=torch.uint8, device=DEVICE)
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
            throughput = EFFECTIVE_BATCH_SIZE * BATCH_PROBE_TIMED_UPDATES / elapsed
            status = "fits" if fraction <= VRAM_SAFETY_FRACTION else "over_budget"
            rows.append({"micro_batch": batch_size, "peak_GiB": gib(peak),
                         "VRAM_fraction": fraction, "images_per_second": throughput,
                         "status": status})
            print(f"batch {batch_size:2d}: {throughput:.1f} images/s, "
                  f"{gib(peak):.1f} GiB ({fraction:.0%}), {status}")
            if status == "fits":
                safe_sizes += 1
                if safe_sizes >= 3:
                    break
        except torch.cuda.OutOfMemoryError:
            rows.append({"micro_batch": batch_size, "status": "OOM"})
            print(f"batch {batch_size:2d}: OOM")
        finally:
            if probe_optimizer is not None:
                probe_optimizer.zero_grad(set_to_none=True)
            del probe, probe_ema, probe_criterion, probe_optimizer, probe_scaler
            del pixels, inputs, targets, factors, logits, loss
            gc.collect()
            torch.cuda.empty_cache()
    fits = [row for row in rows if row["status"] == "fits"]
    if not fits:
        raise RuntimeError("No micro-batch passed the VRAM preflight")
    selected = max(fits, key=lambda row: row["images_per_second"])["micro_batch"]
    return selected, pd.DataFrame(rows)

def probe_evaluation_batch(candidates, training_batch):
    # A real optimizer update allocates the same resident AdamW + EMA state as
    # training. Probing an isolated inference model would overestimate free VRAM.
    active_model = build_model(pretrained=False).to(DEVICE, memory_format=torch.channels_last)
    active_ema = build_model_ema(active_model)
    active_optimizer = build_optimizer(active_model)
    active_scaler = make_grad_scaler()
    active_criterion = CardioSentryLoss(POS_WEIGHT, CONTRADICTION_LAMBDA).to(DEVICE)
    pixels = inputs = targets = logits = loss = None
    rows = []
    selected = None
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
        for batch_size in candidates:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                pixels = torch.randint(0, 256, (batch_size, 3, *INPUT_SIZE), dtype=torch.uint8, device=DEVICE)
                with torch.inference_mode(), autocast_context():
                    # Warm kernels, then measure two complete preprocessing +
                    # inference batches. Keep up to three safe evaluation sizes.
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
        gc.collect()
        torch.cuda.empty_cache()
    fits = [row for row in rows if row["status"] == "fits"]
    if not fits:
        raise RuntimeError("No evaluation batch passed the VRAM preflight")
    selected = max(fits, key=lambda row: row["images_per_second"])["eval_batch"]
    return selected, pd.DataFrame(rows)

MICRO_BATCH_SIZE, batch_probe = probe_micro_batch(AUTO_BATCH_CANDIDATES)
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE
EVAL_BATCH_SIZE, eval_batch_probe = probe_evaluation_batch(EVAL_BATCH_CANDIDATES, MICRO_BATCH_SIZE)
display(batch_probe)
display(eval_batch_probe)
print("selected training/evaluation batches:", MICRO_BATCH_SIZE, EVAL_BATCH_SIZE)
print("gradient accumulation steps:", ACCUMULATION_STEPS)
seed_global_rngs(SEED)

def seed_data_worker(worker_id):
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
            # Non-persistent iterator destruction joins its workers. No private
            # DataLoader APIs are needed, and each trial releases its shared queues.
            del batch, iterator, loader
            gc.collect()
    best = max(rows, key=lambda row: row["decoded_images_per_second"])
    return best["workers"], pd.DataFrame(rows)

if AUTO_TUNE_WORKERS:
    NUM_WORKERS, worker_benchmark = tune_decoder_workers(train_dataset, MICRO_BATCH_SIZE, WORKER_CANDIDATES)
else:
    worker_benchmark = pd.DataFrame([{"workers": NUM_WORKERS, "selection": "explicit_or_fixed"}])
display(worker_benchmark)

# Ensure the chosen pinned uint8 queues fit /dev/shm before persistent workers
# start. Both training and validation workers can remain alive between passes.
queue_bytes = NUM_WORKERS * PREFETCH_FACTOR * (MICRO_BATCH_SIZE + EVAL_BATCH_SIZE) * 3 * math.prod(INPUT_SIZE)
if NUM_WORKERS > 0 and Path("/dev/shm").is_dir():
    shm_free = shutil.disk_usage("/dev/shm").free
    if queue_bytes * 1.25 > shm_free:
        raise RuntimeError("Insufficient /dev/shm for the selected queues; reduce CS_PREFETCH_FACTOR or CS_NUM_WORKERS.")
print(f"estimated train + validation uint8 queues: {gib(queue_bytes):.2f} GiB")

seed_global_rngs(SEED)
TRAIN_GENERATOR = torch.Generator().manual_seed(SEED)
# The repeat-factor sampler fixes the epoch length and order per (seed, epoch).
train_loader = DataLoader(train_dataset, sampler=train_sampler, drop_last=True,
                          **make_loader_options(MICRO_BATCH_SIZE, NUM_WORKERS))
val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False,
                        **make_loader_options(EVAL_BATCH_SIZE, NUM_WORKERS))
test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                         **make_loader_options(EVAL_BATCH_SIZE, NUM_WORKERS))
usable_train_batches = len(train_loader) - (len(train_loader) % ACCUMULATION_STEPS)
if usable_train_batches <= 0:
    raise RuntimeError("Training loader is too short for the configured effective batch")
OPTIMIZER_STEPS_PER_EPOCH = usable_train_batches // ACCUMULATION_STEPS
PIPELINE_TUNING = {
    "hardware_profile": HARDWARE_PROFILE, "micro_batch_size": MICRO_BATCH_SIZE,
    "eval_batch_size": EVAL_BATCH_SIZE, "num_workers": NUM_WORKERS,
    "prefetch_factor": PREFETCH_FACTOR, "estimated_queue_gib": gib(queue_bytes),
    "training_batch_probe": batch_probe.to_dict("records"),
    "eval_batch_probe": eval_batch_probe.to_dict("records"),
    "worker_benchmark": worker_benchmark.to_dict("records"),
}
(RUN_DIR / "input_pipeline_benchmark.json").write_text(json.dumps(PIPELINE_TUNING, indent=2))
print("loader batches (train/val/test):", len(train_loader), len(val_loader), len(test_loader))
print("selected decoder workers:", NUM_WORKERS)
print("optimizer steps per epoch:", OPTIMIZER_STEPS_PER_EPOCH)
"""),
    md(r"""## 8. Benchmark eager versus compiled execution on the L40S

Both modes use the selected loaders, uint8 → GPU preprocessing, fused AdamW,
EMA updates, and full accumulation windows. Compilation and warm-up are timed
separately and excluded from steady-state throughput. Select compilation only
if it fits the memory ceiling and improves projected epoch time by at least 2%.
Unsupported compiler builds, compiler OOMs, and slower compiled execution retain
the working eager mode. `CS_TORCH_COMPILE=0` skips the comparison.

The projection includes two validation passes per epoch (raw plus EMA).
The benchmark uses disposable copies; weights, global RNGs, and train-shuffle
state are restored before real training. The final model is compiled in place
so saved weights remain ordinary timm state dictionaries without wrapper keys.
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

def select_runtime_benchmark(rows, minimum_compile_gain=0.02):
    eager = next(row for row in rows if row["mode"] == "eager" and row["status"] == "fits")
    compiled = next((row for row in rows if row["mode"] == "compiled" and row["status"] == "fits"), None)
    if compiled is not None and compiled["estimated_sec_per_epoch"] < eager["estimated_sec_per_epoch"] * (1 - minimum_compile_gain):
        return compiled
    return eager

train_benchmark_batches = min(max(BENCHMARK_TRAIN_BATCHES, ACCUMULATION_STEPS), usable_train_batches)
train_benchmark_batches -= train_benchmark_batches % ACCUMULATION_STEPS
if train_benchmark_batches < ACCUMULATION_STEPS:
    raise RuntimeError("Benchmark must cover at least one complete effective-batch update")
train_generator_state = TRAIN_GENERATOR.get_state()
base_state = model.state_dict()
benchmark_rows = []

def benchmark_execution_mode(compiled):
    active_model = active_ema = active_criterion = active_optimizer = active_scaler = None
    benchmark_train_loader = benchmark_val_loader = None
    train_iterator = val_iterator = images = pixels = labels = factors = logits = loss = parts = None
    try:
        seed_global_rngs(SEED)
        TRAIN_GENERATOR.set_state(train_generator_state)
        active_model = build_model(pretrained=False)
        active_model.load_state_dict(base_state, strict=True)
        active_model = active_model.to(DEVICE, memory_format=torch.channels_last).train()
        active_ema = build_model_ema(active_model)
        active_criterion = CardioSentryLoss(POS_WEIGHT, CONTRADICTION_LAMBDA).to(DEVICE)
        active_optimizer = build_optimizer(active_model)
        active_scaler = make_grad_scaler()
        configure_model_compilation(active_model, compiled)
        # Separate disposable loaders give both modes exactly the same sample
        # order and never advance the real persistent training iterator.
        benchmark_train_loader = DataLoader(
            train_dataset, shuffle=True, drop_last=True,
            generator=torch.Generator().manual_seed(SEED),
            **make_loader_options(MICRO_BATCH_SIZE, NUM_WORKERS, persistent=False),
        )
        benchmark_val_loader = DataLoader(
            val_dataset, shuffle=False, drop_last=False,
            **make_loader_options(EVAL_BATCH_SIZE, NUM_WORKERS, persistent=False),
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
                active_scaler.scale(loss / ACCUMULATION_STEPS).backward()
                if (batch_index + 1) % ACCUMULATION_STEPS == 0:
                    optimizer_update(active_model, active_optimizer, active_scaler, active_ema)

        # At least one full update initializes AdamW state; the compiled path
        # also compiles forward AND backward before either timing begins.
        warmup_batches = ACCUMULATION_STEPS * max(1, math.ceil(BENCHMARK_WARMUP_BATCHES / ACCUMULATION_STEPS))
        startup_started = time.perf_counter()
        run_train_batches(warmup_batches)
        torch.cuda.synchronize()
        startup_seconds = time.perf_counter() - startup_started
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        run_train_batches(train_benchmark_batches)
        torch.cuda.synchronize()
        train_elapsed = time.perf_counter() - started
        train_peak = torch.cuda.max_memory_allocated()
        reserved_peak = torch.cuda.max_memory_reserved()

        active_model.eval()
        val_iterator = iter(benchmark_val_loader)
        val_warmup = min(BENCHMARK_WARMUP_BATCHES, max(0, len(val_loader) - 1))
        val_batches = min(BENCHMARK_VAL_BATCHES, len(val_loader) - val_warmup)
        if val_batches < 1:
            raise RuntimeError("Validation loader is too short for the benchmark")
        with torch.inference_mode():
            for _ in range(val_warmup):
                pixels, _, _ = next(val_iterator)
                images = prepare_image_batch(pixels)
                with autocast_context():
                    active_model(images)
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
        reserved_peak = max(reserved_peak, torch.cuda.max_memory_reserved())
        total_vram = torch.cuda.get_device_properties(0).total_memory
        sec_per_train_batch = train_elapsed / train_benchmark_batches
        sec_per_val_batch = val_elapsed / val_batches
        train_seconds = sec_per_train_batch * usable_train_batches
        val_seconds = sec_per_val_batch * len(val_loader)
        return {
            "mode": "compiled" if compiled else "eager",
            "status": "fits" if peak / total_vram <= VRAM_SAFETY_FRACTION else "over_budget",
            "startup_train_warmup_seconds": startup_seconds,
            "timed_train_batches": train_benchmark_batches, "timed_val_batches": val_batches,
            "sec_per_train_batch": sec_per_train_batch, "sec_per_val_batch": sec_per_val_batch,
            "train_images_per_second": MICRO_BATCH_SIZE * train_benchmark_batches / train_elapsed,
            "eval_images_per_second": val_images / val_elapsed,
            "train_sec_per_epoch": train_seconds, "val_sec_per_epoch": val_seconds,
            "estimated_sec_per_epoch": (train_seconds + 2 * val_seconds) * 1.05,
            "train_peak_allocated_gib": gib(train_peak), "peak_allocated_gib": gib(peak),
            "train_peak_reserved_gib": gib(reserved_peak),
            "vram_total_gib": gib(total_vram), "vram_peak_fraction": peak / total_vram,
        }
    finally:
        TRAIN_GENERATOR.set_state(train_generator_state)
        del train_iterator, val_iterator, images, pixels, labels, factors, logits, loss, parts
        del benchmark_train_loader, benchmark_val_loader
        del active_model, active_ema, active_criterion, active_optimizer, active_scaler
        gc.collect()
        torch.cuda.empty_cache()

for compiled in ([False, True] if TORCH_COMPILE_ENABLED else [False]):
    try:
        row = benchmark_execution_mode(compiled)
        benchmark_rows.append(row)
        print(f"{row['mode']}: {row['train_images_per_second']:.1f} train images/s, "
              f"{row['eval_images_per_second']:.1f} eval images/s, "
              f"{row['peak_allocated_gib']:.1f} GiB, {row['status']}")
        if not compiled and row["status"] != "fits":
            raise RuntimeError("Actual eager benchmark exceeds 90% VRAM; reduce batch candidates and rerun from a fresh kernel.")
    except Exception as exc:
        if not compiled:
            raise
        benchmark_rows.append({"mode": "compiled", "status": "failed",
                               "error": f"{type(exc).__name__}: {exc}"})
        warnings.warn("Compiler benchmark failed; keeping measured eager execution: " + str(exc))

chosen_benchmark = select_runtime_benchmark(benchmark_rows)
COMPILE_SELECTED = chosen_benchmark["mode"] == "compiled"
BENCHMARK = {
    **chosen_benchmark,
    "hardware_profile": HARDWARE_PROFILE, "dataset_mode": DATASET_MODE,
    "micro_batch_size": MICRO_BATCH_SIZE, "eval_batch_size": EVAL_BATCH_SIZE,
    "accumulation_steps": ACCUMULATION_STEPS, "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    "num_workers": NUM_WORKERS, "prefetch_factor": PREFETCH_FACTOR,
    "host_image_dtype": "uint8", "gpu_preprocessing": True,
    "fused_adamw": USE_FUSED_ADAMW, "compile_selected": COMPILE_SELECTED,
    "compile_mode": TORCH_COMPILE_MODE if COMPILE_SELECTED else None,
    "mode_comparison": benchmark_rows, "pipeline_tuning": PIPELINE_TUNING,
    "train_images": len(train_dataset), "val_images": len(val_dataset),
    "usable_train_batches": usable_train_batches, "val_batches": len(val_loader),
    "validation_passes_per_epoch": 2, "ema_decay": EMA_DECAY, "amp_dtype": str(AMP_DTYPE),
    "max_epochs": MAX_EPOCHS, "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "estimated_total_max_epochs": chosen_benchmark["estimated_sec_per_epoch"] * MAX_EPOCHS,
}
benchmark_path = RUN_DIR / "benchmark.json"
temporary = benchmark_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(BENCHMARK, indent=2, sort_keys=True))
os.replace(temporary, benchmark_path)
seed_global_rngs(SEED)
print("selected execution:", chosen_benchmark["mode"])
print("estimated epoch time:", format_duration(BENCHMARK["estimated_sec_per_epoch"]))
print(f"estimated maximum {MAX_EPOCHS}-epoch duration:", format_duration(BENCHMARK["estimated_total_max_epochs"]))
print("First-run compilation/warm-up is additional; it is excluded from steady-state projection.")
print("GPU allocation ceiling: 90% of measured CUDA memory.")
if S3_UPLOAD_ENABLED:
    upload_run_files([benchmark_path, RUN_DIR / "input_pipeline_benchmark.json"], required=True)
"""),
    md(r"""## 9. Metrics

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
    md(r"""## 10. Train with checkpoint resume

The harness moves the model to the GPU, builds the optimizer/scheduler/scaler,
freezes the run configuration, and validates any existing run directory before
writing anything: a reused `CS_RUN_NAME` with a different training identity
fails loudly instead of mixing results (benchmark timings and hardware-derived
micro-batch/accumulation are recorded as runtime metadata, not identity, so a
run may resume across machines at the same effective batch). Missing run files
are first pulled back from S3, so after an interruption (or on a fresh machine)
re-running the notebook resumes from the newest epoch-boundary checkpoint — `last.pt` or `best.pt`, both of which carry full
training state including raw weights, EMA weights and update count, optimizer,
scheduler, and RNG states — instead of starting over. The sampler order is a
function of the seed and epoch, so the resumed epoch draws exactly what it
would have. Mid-epoch `last_intra.pt` snapshots (every
`CS_INTRA_EPOCH_SYNC_MINUTES`, default 15) are uploaded for Spot visibility but
never drive resume, so the LR schedule stays exact. Both raw and EMA weights are evaluated
on validation every epoch; the better validation pathology-only Macro AP
(the 11 non-NORM labels) selects checkpoints
and drives early stopping; the 12-label Macro AP is tracked as a secondary
metric.

**Collapse guard.** From the second epoch on, every 100 micro-batches the
rolling training BCE is compared with the loss of the best constant predictor
(computed in section 4 from training prevalence and `pos_weight`). Five
consecutive windows at ≥ 97% of it, or a raw-weight validation macro AUROC
below 0.60 at any epoch end, stop training with `TrainingCollapseError` and
write `collapse_report.json`. The check runs before `last.pt` or `best.pt` is
replaced, so both still hold the last healthy state. Recovery: start a new
`CS_RUN_NAME` with a lower `CS_BASE_LR` (the error message suggests half).
Rerunning the same run name resumes the healthy epoch at the same LR, which
may collapse again.
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

def export_inference_checkpoint(active_model, weight_source, best_epoch, thresholds, destination):
    # Include only tensors and primitive metadata so weights_only=True can load it.
    atomic_torch_save({
        "model": {key: tensor.detach().cpu() for key, tensor in active_model.state_dict().items()},
        "model_name": MODEL_NAME,
        # Rebuild: timm.create_model(model_name, num_classes=len(label_order)),
        # then ConvNeXtMultiScaleFusion(backbone, fusion_neck, fusion_head, ...)
        # from fusion_necks.py unless fusion_neck is "none".
        "fusion_neck": FUSION_NECK,
        "fusion_head": FUSION_HEAD,
        "fusion_channels": int(FUSION_CHANNELS),
        "bifpn_repeats": int(BIFPN_REPEATS),
        "fusion_stage_indices": list(FUSION_STAGE_INDICES),
        "label_order": list(LABELS),
        "input_size_hw": list(INPUT_SIZE),
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "input_color_mode": IMAGE_COLOR_MODE,
        "resize_interpolation": RESIZE_INTERPOLATION.value,
        "resize_antialias": RESIZE_ANTIALIAS,
        "thresholds": list(map(float, thresholds)),
        "selected_weight_source": str(weight_source),
        "best_epoch": int(best_epoch),
        "config_fingerprint": CONFIG_FINGERPRINT,
    }, destination)

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
    for pixels, labels, batch_metadata in tqdm(loader, leave=False):
        images = prepare_image_batch(pixels)
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
    scores = {}
    macro_aurocs = {}
    for weight_source, candidate_model in (("raw", active_model), ("ema", active_ema.module)):
        output = predict_loader(candidate_model, loader, active_criterion)
        mask = primary_mask(output)
        if not mask.any():
            raise RuntimeError("Validation output has no primary-metric rows")
        metrics = evaluate_predictions(
            output["y_true"][mask], output["y_probability"][mask], [0.5] * len(LABELS)
        )
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

class TrainingCollapseError(RuntimeError):
    pass

def report_training_collapse(epoch, stage, details):
    # Called before last.pt is replaced, so the newest epoch-boundary
    # checkpoint is still the last healthy state.
    report = {
        "detected_at_utc": datetime.now(timezone.utc).isoformat(),
        "epoch": int(epoch), "stage": stage, "base_lr": BASE_LR,
        "constant_predictor_loss": CONSTANT_PREDICTOR_LOSS, **details,
    }
    report_path = RUN_DIR / "collapse_report.json"
    atomic_json_save(report, report_path)
    log_event("training_collapse", **report)
    upload_run_files([report_path, RUN_LOG_PATH], required=False)
    raise TrainingCollapseError(
        f"Training collapsed during epoch {epoch} ({stage}): {json.dumps(details)}. "
        "last.pt still holds the previous healthy epoch. Start a new run with a lower LR, "
        f"e.g. CS_BASE_LR={BASE_LR / 2:.1e} and a new CS_RUN_NAME."
    )

def train_one_epoch(active_model, loader, active_criterion, optimizer, scaler,
                     scheduler, active_ema=None, heartbeat_fn=None):
    active_model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_totals = torch.zeros(len(LOSS_COMPONENT_KEYS), device=DEVICE, dtype=torch.float32)
    sample_count = 0
    optimizer_updates = 0
    amp_skipped_optimizer_steps = 0
    scale_before = scaler.get_scale()
    # Rolling-window collapse check; reads the GPU only once per window.
    window_bce = torch.zeros((), device=DEVICE, dtype=torch.float32)
    window_samples = 0
    collapsed_windows = 0
    progress = tqdm(enumerate(loader), total=usable_train_batches, leave=False)
    for batch_index, (pixels, labels, factors) in progress:
        if batch_index >= usable_train_batches:
            break
        images = prepare_image_batch(pixels, factors)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast_context():
            logits = active_model(images)
            loss, parts = active_criterion(logits, labels)
            scaled_loss = loss / ACCUMULATION_STEPS
        scaler.scale(scaled_loss).backward()
        if (batch_index + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                active_model.parameters(), GRAD_CLIP_NORM,
                error_if_nonfinite=not scaler.is_enabled()
            )
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer.zero_grad(set_to_none=True)
            # A scale decrease means non-finite gradients caused GradScaler to
            # skip optimizer.step(). Keep LambdaLR aligned with real updates.
            if scale_after >= scale_before:
                scheduler.step()
                optimizer_updates += 1
                if active_ema is not None:
                    update_model_ema(active_ema, active_model)
            else:
                amp_skipped_optimizer_steps += 1
            scale_before = scale_after
        if heartbeat_fn is not None:
            heartbeat_fn()
        batch_samples = len(images)
        batch_loss_values = torch.stack((
            loss.detach(), parts["bce"], parts["contradiction"],
            parts["weighted_contradiction"],
        )).float()
        loss_totals.add_(batch_loss_values, alpha=batch_samples)
        sample_count += batch_samples
        window_bce.add_(parts["bce"].float(), alpha=batch_samples)
        window_samples += batch_samples
        if (batch_index + 1) % COLLAPSE_CHECK_EVERY_N_BATCHES == 0:
            # Active once a full epoch of updates has happened: before that a
            # loss near the constant predictor is ordinary early training.
            if COLLAPSE_GUARD_ENABLED and scheduler.last_epoch >= OPTIMIZER_STEPS_PER_EPOCH:
                window_mean = (window_bce / window_samples).item()
                if window_mean >= COLLAPSE_LOSS_RATIO * CONSTANT_PREDICTOR_LOSS:
                    collapsed_windows += 1
                else:
                    collapsed_windows = 0
                if collapsed_windows >= COLLAPSE_PATIENCE_WINDOWS:
                    report_training_collapse(scheduler.last_epoch // OPTIMIZER_STEPS_PER_EPOCH + 1, "training_loss", {
                        "window_train_bce": window_mean,
                        "consecutive_windows": collapsed_windows,
                        "batches_per_window": COLLAPSE_CHECK_EVERY_N_BATCHES,
                        "optimizer_step": int(scheduler.last_epoch),
                    })
            window_bce.zero_()
            window_samples = 0
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
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

model = model.to(DEVICE, memory_format=torch.channels_last)
model_ema = build_model_ema(model)
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

# Training identity (fingerprinted: any change refuses old checkpoints) vs
# runtime metadata (recorded but resumable: hardware, timings, S3 layout).
TRAINING_IDENTITY = {
    "run_name": RUN_NAME,
    "model": MODEL_NAME,
    "head_init_scale": HEAD_INIT_SCALE,
    "pretraining": "supervised ImageNet-22K -> supervised ImageNet-1K at 384px",
    "fusion": FUSION,
    "fusion_head_override": FUSION_HEAD_OVERRIDE or None,
    "fusion_spec": FUSION_SPEC,
    "label_order": LABELS,
    "dataset": "CardioSentry HEEDB clean12 ECG images (1024x768 PNG, 500 Hz records, 12 labels)",
    "dataset_archive": DATASET_TAR_NAME,
    "dataset_archive_sha256": EXPECTED_TAR_SHA256,
    "dataset_metadata_sha256": METADATA_SHA256,
    "dataset_manifest_sha256": MANIFEST_SHA256,
    "dataset_mode": DATASET_MODE,
    "split": {
        "version": SPLIT_VERSION,
        "unit": "patient_key",
        "seed": SPLIT_SEED,
        "ratios": SPLIT_RATIOS,
        "sha256": SPLIT_SHA256,
    },
    "train_variants": TRAIN_VARIANTS,
    "rare_label_sampling": {
        "enabled": RARE_SAMPLING,
        "method": ("repeat-factor sampling: label r = max(1, sqrt(threshold / train frequency)); "
                   "image r = max over its labels; floor(r) draws plus extra draws for "
                   "floor(sum(frac)) rows chosen without replacement with p ~ frac; "
                   "order keyed by (seed, epoch)") if RARE_SAMPLING else "one draw per image per epoch",
        "threshold": RARE_SAMPLING_THRESHOLD if RARE_SAMPLING else None,
        "label_repeat_factors": {label: float(value) for label, value in label_repeat_factor.items()},
        "draws_per_epoch": len(train_sampler),
        "pos_weight_counts": "expected per-epoch counts under the sampler",
    },
    "input_size_hw": list(INPUT_SIZE),
    "input_color_mode": IMAGE_COLOR_MODE,
    "resize_interpolation": RESIZE_INTERPOLATION.value,
    "resize_antialias": RESIZE_ANTIALIAS,
    "resize_preserves_aspect_ratio": False,
    "tensor_conversion": "PILToTensor uint8 on CPU -> float32 / 255 on GPU",
    "normalization_mean": list(IMAGENET_MEAN),
    "normalization_std": list(IMAGENET_STD),
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "early_stopping_min_epochs": EARLY_STOPPING_MIN_EPOCHS,
    "monitor": "validation_primary_pathology_only_macro_average_precision",
    "monitor_secondary": "validation_primary_macro_average_precision",
    "monitor_mode": "max",
    "reported_threshold_policies": ["fixed_0_5", "validation_tuned_per_label_f1"],
    "combination_low_support_threshold": COMBINATION_LOW_SUPPORT_THRESHOLD,
    "base_lr": BASE_LR,
    "min_lr_ratio": MIN_LR_RATIO,
    "ema": {"decay": EMA_DECAY, "use_warmup": True,
            "update_policy": "successful_optimizer_steps_only", "select_raw_or_ema": True},
    "optimizer": "AdamW",
    "weight_decay": WEIGHT_DECAY,
    "layer_decay": LAYER_DECAY,
    "layer_lr_scale_min": min(lr_scales),
    "layer_lr_scale_max": max(lr_scales),
    "drop_path_rate": DROP_PATH_RATE,
    "warmup_epochs": WARMUP_EPOCHS,
    "schedule": "linear_warmup_then_cosine",
    "scheduler_step_policy": "advance only after a successful optimizer update",
    "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    "loss": (
        "BCEWithLogitsLoss(pos_weight=sampled train negatives/positives)"
        if POS_WEIGHT_MODE == "full" else
        "BCEWithLogitsLoss(pos_weight=sqrt(sampled train negatives/positives))"
        if POS_WEIGHT_MODE == "sqrt" else
        "BCEWithLogitsLoss(unweighted)"
    ) + (" + contradiction" if CONTRADICTION_LAMBDA else ""),
    "pos_weight_mode": POS_WEIGHT_MODE,
    "train_pos_weight": dict(zip(LABELS, map(float, POS_WEIGHT.tolist()))),
    "contradiction_lambda": CONTRADICTION_LAMBDA,
    "contradiction_formula": "p(NORM) * (1 - product(1 - p(pathology)))",
    "pathology_pathology_penalty": False,
    "online_augmentation": {
        "brightness": TRAIN_BRIGHTNESS_JITTER, "contrast": TRAIN_CONTRAST_JITTER,
        "rng": "numpy.default_rng; (seed + (occurrence+1)*1000003 + epoch*1000033) mod 2**32; occurrence = repeat*N + row",
        "implementation": "GPU float32 RGB brightness clamp, luminance-mean contrast clamp, then normalize",
        "geometry": "complete native sheet; no spatial augmentation",
    },
    "amp": "BF16 if supported, otherwise FP16 with dynamic scaling",
    "tf32": not DETERMINISTIC,
    "deterministic_algorithms": DETERMINISTIC,
    "channels_last": True,
    "seed": SEED,
}
RUNTIME_METADATA = {
    # Hardware-dependent or measured values: recorded for provenance, excluded
    # from the fingerprint so a run can resume across machines (e.g. an A10G
    # micro-batch 4x16 checkpoint resuming on an L40S at 8x8 — same effective
    # batch, same LayerNorm/GroupNorm-only math, valid continuation).
    "reproducibility_mode": "deterministic opt-in" if DETERMINISTIC else "seeded performance mode; not bitwise deterministic",
    "collapse_guard": {
        "enabled": COLLAPSE_GUARD_ENABLED,
        "constant_predictor_loss": CONSTANT_PREDICTOR_LOSS,
        "loss_ratio": COLLAPSE_LOSS_RATIO,
        "check_every_n_batches": COLLAPSE_CHECK_EVERY_N_BATCHES,
        "patience_windows": COLLAPSE_PATIENCE_WINDOWS,
        "min_raw_val_macro_auroc": COLLAPSE_MIN_VAL_MACRO_AUROC,
    },
    "package_versions": PACKAGE_VERSIONS,
    "hardware_profile": HARDWARE_PROFILE,
    "fused_adamw": USE_FUSED_ADAMW,
    "compile_selected": COMPILE_SELECTED,
    "compile_mode": TORCH_COMPILE_MODE if COMPILE_SELECTED else None,
    "gpu_preprocessing": True,
    "host_image_dtype": "uint8",
    "amp_dtype": str(AMP_DTYPE),
    "optimizer_parameter_groups": len(optimizer.param_groups),
    "num_workers": NUM_WORKERS,
    "prefetch_factor": PREFETCH_FACTOR,
    "dataloader_start_method": DATALOADER_START_METHOD if NUM_WORKERS > 0 else "single-process",
    "progress_update_every_n_batches": PROGRESS_UPDATE_EVERY_N_BATCHES,
    "micro_batch_size": MICRO_BATCH_SIZE,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "accumulation_steps": ACCUMULATION_STEPS,
    "benchmark_estimated_sec_per_epoch": BENCHMARK["estimated_sec_per_epoch"] if "BENCHMARK" in globals() else None,
    "dataset_source": s3_uri(S3_DATASET_KEY),
    "s3_upload_enabled": S3_UPLOAD_ENABLED,
    "s3_bucket": S3_BUCKET if S3_UPLOAD_ENABLED else None,
    "s3_run_prefix": S3_RUN_PREFIX if S3_UPLOAD_ENABLED else None,
    "s3_upload_every_n_epochs": S3_UPLOAD_EVERY_N_EPOCHS if S3_UPLOAD_ENABLED else None,
    "scratch_device": SCRATCH_DEVICE if not SCRATCH_FALLBACK else "ebs-fallback",
    "split_image_counts": {name: len(frame) for name, frame in selected_frames.items()},
    "split_record_counts": {name: int(frame["record_id"].nunique()) for name, frame in selected_frames.items()},
}
CONFIG_FINGERPRINT = hashlib.sha256(
    json.dumps(TRAINING_IDENTITY, sort_keys=True).encode()
).hexdigest()[:16]
CONFIG = {
    "identity": TRAINING_IDENTITY,
    "runtime": RUNTIME_METADATA,
    "fingerprint": CONFIG_FINGERPRINT,
}

config_snapshot_path = RUN_DIR / "config_snapshot.json"
best_path = RUN_DIR / "best.pt"
last_path = RUN_DIR / "last.pt"
intra_path = RUN_DIR / "last_intra.pt"
history_path = RUN_DIR / "training_history.csv"
completion_path = RUN_DIR / "training_complete.json"

# Interrupted run on a fresh machine (or a wiped run directory): pull the
# missing run files back from S3 first, so the checks below validate the
# restored state and training resumes instead of restarting.
# last_intra.pt is restored for inspection only; resume trusts only
# epoch-boundary checkpoints (last.pt, best.pt).
restore_missing_run_files_from_s3([
    "last.pt", "best.pt", "last_intra.pt", "training_history.csv", "run.log",
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
    stored_identity = existing_config.get("identity")
    if not isinstance(stored_identity, dict):
        raise RuntimeError(
            "Existing config_snapshot.json predates the identity/runtime split "
            "and will not be trusted. Choose a new CS_RUN_NAME or repair it."
        )
    recomputed_fingerprint = hashlib.sha256(
        json.dumps(stored_identity, sort_keys=True).encode()
    ).hexdigest()[:16]
    if stored_fingerprint != recomputed_fingerprint:
        raise RuntimeError(
            "Existing config_snapshot.json has an invalid fingerprint and will not be overwritten. "
            "Choose a new CS_RUN_NAME or repair the existing snapshot."
        )
    require_matching_config_fingerprint("config_snapshot.json", stored_fingerprint)
    # Spot-replacement guard: versions are provenance (not identity), so a
    # mismatch warns loudly instead of blocking resume — the timm pin makes a
    # fresh-install mismatch nearly impossible, and any remaining drift (e.g. a
    # different DLAMI torch) must be visible, never silent.
    stored_versions = existing_config.get("runtime", {}).get("package_versions", {})
    if isinstance(stored_versions, dict) and stored_versions:
        drifted = {
            name: (stored_versions.get(name), PACKAGE_VERSIONS.get(name))
            for name in ("timm", "torch", "torchvision")
            if stored_versions.get(name) != PACKAGE_VERSIONS.get(name)
        }
        if drifted:
            warnings.warn(
                "Package versions differ from the stored run config: "
                + ", ".join(f"{name} {old!r} -> {new!r}" for name, (old, new) in drifted.items())
                + ". Resuming anyway; confirm this drift is acceptable."
            )

# Resume from the newest epoch-boundary checkpoint. Optional mid-epoch
# snapshots (last_intra.pt, epoch_complete=False) never drive resume:
# reloading a partially stepped optimizer/scheduler and then repeating the full
# epoch would double-count that epoch's early batches and shift the LR
# schedule, so only epoch-boundary states (last.pt at epoch end, always
# best.pt) advance. A Spot interruption therefore redoes at most one epoch from
# a clean start-of-epoch state, with history append-only and the schedule exact.
checkpoint = None
resume_source = None
resume_complete = True
for candidate_path in (last_path, best_path):
    if not candidate_path.is_file():
        continue
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
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
    if not bool(candidate.get("epoch_complete", True)):
        # Legacy mid-epoch last.pt (pre-boundary-only resume): never resume
        # from a partially stepped optimizer/scheduler. The previous
        # epoch-boundary checkpoint (or best.pt) is the resume point.
        warnings.warn(
            f"{candidate_path.name} is a mid-epoch snapshot and cannot resume "
            "training; it will be overwritten by this run."
        )
        del candidate
        continue
    if checkpoint is None or int(candidate.get("epoch", 0)) > int(checkpoint.get("epoch", 0)):
        if checkpoint is not None:
            del checkpoint
        checkpoint, resume_source = candidate, candidate_path.name
        resume_complete = True
    else:
        del candidate

start_epoch, best_pathology_only_macro_ap, bad_epochs, history = 1, -float("inf"), 0, []
if checkpoint is not None:
    model.load_state_dict(checkpoint["model"])
    model_ema.module.load_state_dict(checkpoint["model_ema"])
    model_ema.num_updates = int(checkpoint["ema_num_updates"])
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
    best_pathology_only_macro_ap = float(checkpoint["best_pathology_only_macro_average_precision"])
    bad_epochs = int(checkpoint["bad_epochs"])
    history = list(checkpoint["history"])
    if resume_complete:
        start_epoch = int(checkpoint["epoch"]) + 1
        print(
            f"resuming from {resume_source} after epoch {start_epoch - 1}; "
            f"best validation primary pathology-only Macro AP={best_pathology_only_macro_ap:.6f}"
        )
    del checkpoint
    checkpoint = None

if completion_path.is_file():
    completion_state = json.loads(completion_path.read_text())
    require_matching_config_fingerprint(
        "training_complete.json", completion_state.get("config_fingerprint")
    )

# Compile both copies in place only after EMA creation and checkpoint restore.
# The benchmarking decision was made before allocating the real training state.
configure_model_compilation(model, COMPILE_SELECTED)
configure_model_compilation(model_ema.module, COMPILE_SELECTED)
# Only a fully validated compatible run may create or refresh this file.
atomic_json_save(CONFIG, config_snapshot_path)
print("run configuration frozen:", config_snapshot_path)
"""),
    code(r"""# Long-running training cell. After interruption, restart the kernel and
# rerun the notebook with the same run name to restore epoch-boundary state.
if completion_path.is_file():
    print("Training already marked complete:")
    print(completion_path.read_text())
else:
    def build_full_checkpoint_state(current_epoch, epoch_complete):
        return {
            "epoch": current_epoch,
            "epoch_complete": bool(epoch_complete),
            "model": model.state_dict(),
            "model_ema": model_ema.module.state_dict(),
            "ema_num_updates": model_ema.num_updates,
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

    log_event("training_start", start_epoch=start_epoch, max_epochs=MAX_EPOCHS,
              intra_epoch_sync_minutes=INTRA_EPOCH_SYNC_MINUTES)
    stopped_early = (start_epoch - 1 >= EARLY_STOPPING_MIN_EPOCHS
                     and bad_epochs >= EARLY_STOPPING_PATIENCE)
    epochs_to_run = [] if stopped_early else range(start_epoch, MAX_EPOCHS + 1)
    for epoch in epochs_to_run:
        epoch_started = time.time()
        train_dataset.set_epoch(epoch)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats()
        _intra_epoch_last_sync = [epoch_started]
        def request_intra_epoch_sync():
            # Time-throttled mid-epoch snapshot for Spot visibility
            # (CS_INTRA_EPOCH_SYNC_MINUTES, default 15; 0 turns it off):
            # torch.save is local-NVMe fast and the S3 upload runs on a daemon
            # thread, so the hot loop never waits. Written to last_intra.pt
            # (never last.pt) so a partial optimizer/scheduler state can never
            # clobber the epoch-boundary resume point; resume trusts only
            # epoch-boundary checkpoints. Always saved locally even with
            # CS_S3_UPLOAD=0; only S3 is gated.
            if INTRA_EPOCH_SYNC_MINUTES <= 0:
                return
            now = time.time()
            if now - _intra_epoch_last_sync[0] < INTRA_EPOCH_SYNC_MINUTES * 60:
                return
            if S3_UPLOAD_ENABLED:
                # Do not replace the local snapshot while its previous upload
                # is still reading it: skip this heartbeat (the timestamp is
                # left alone so a later batch retries) rather than racing the
                # reader. Same-key serialization alone cannot protect the file
                # under an in-flight upload.
                with _upload_key_lock:
                    pending = _upload_key_threads.get(f"{S3_RUN_PREFIX}/{intra_path.name}")
                    if pending is not None and pending.is_alive():
                        return
            _intra_epoch_last_sync[0] = now
            atomic_torch_save(build_full_checkpoint_state(epoch, False), intra_path)
            pd.DataFrame(history).to_csv(history_path, index=False)
            log_event("intra_epoch_sync", epoch=epoch,
                      elapsed_min=round((now - epoch_started) / 60, 1))
            if not S3_UPLOAD_ENABLED:
                return
            spawn_s3_upload(intra_path, f"{S3_RUN_PREFIX}/{intra_path.name}")
            spawn_s3_upload(history_path, f"{S3_RUN_PREFIX}/{history_path.name}")
            spawn_s3_upload(RUN_LOG_PATH, f"{S3_RUN_PREFIX}/{RUN_LOG_PATH.name}")
        train_values = train_one_epoch(model, train_loader, criterion, optimizer,
                                       scaler, scheduler,
                                       active_ema=model_ema,
                                       heartbeat_fn=request_intra_epoch_sync)
        validation_choice, candidate_scores = select_validation_weights(
            model, model_ema, val_loader, criterion
        )
        validation_output = validation_choice["output"]
        validation_metrics = validation_choice["metrics"]
        mask = validation_choice["mask"]
        selected_weight_source = validation_choice["weight_source"]
        raw_validation_macro_auroc = validation_choice["candidate_macro_auroc"]["raw"]
        if COLLAPSE_GUARD_ENABLED and not raw_validation_macro_auroc >= COLLAPSE_MIN_VAL_MACRO_AUROC:
            report_training_collapse(epoch, "validation", {
                "raw_val_macro_auroc": raw_validation_macro_auroc,
                "ema_val_macro_auroc": validation_choice["candidate_macro_auroc"]["ema"],
                "train_loss": train_values["loss"],
            })
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
            "selected_weight_source": selected_weight_source,
            "val_raw_pathology_only_macro_average_precision": candidate_scores["raw"],
            "val_ema_pathology_only_macro_average_precision": candidate_scores["ema"],
            "val_raw_macro_auroc": raw_validation_macro_auroc,
            "val_ema_macro_auroc": validation_choice["candidate_macro_auroc"]["ema"],
            "constant_predictor_loss": CONSTANT_PREDICTOR_LOSS,
            "ema_num_updates": model_ema.num_updates,
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
            "learning_rate_head": max(group["lr"] for group in optimizer.param_groups),
            "epoch_seconds": time.time() - epoch_started,
            "peak_allocated_gib": gib(torch.cuda.max_memory_allocated()),
            "improved": improved,
            "bad_epochs": bad_epochs,
        }
        history.append(epoch_row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        # Full training state shared by both checkpoints, so either one can
        # resume training after an interruption.
        full_checkpoint_state = build_full_checkpoint_state(epoch, True)
        if improved:
            atomic_torch_save({
                **full_checkpoint_state,
                "selected_weight_source": selected_weight_source,
                "validation_macro_average_precision": validation_macro_average_precision,
                "validation_pathology_only_macro_average_precision": validation_pathology_only_macro_average_precision,
                "validation_average_precision_per_condition": {
                    label: validation_metrics["per_label"][label]["average_precision"]
                    for label in LABELS
                },
            }, best_path)
            # A new best must survive even an immediate Spot interruption.
            upload_file_to_s3(best_path, f"{S3_RUN_PREFIX}/{best_path.name}",
                              required=True)
            log_event("best_checkpoint", epoch=epoch,
                      val_pathology_only_macro_ap=validation_pathology_only_macro_average_precision)
        atomic_torch_save(dict(full_checkpoint_state), last_path)

        print(
            f"epoch {epoch:2d}/{MAX_EPOCHS} "
            f"train_loss={train_values['loss']:.4f} "
            f"val_primary_mAP={validation_macro_average_precision:.4f} "
            f"val_all_clean_mAP={validation_all['macro_average_precision']:.4f} "
            f"val_path_mAP={validation_pathology_only_macro_average_precision:.4f} "
            f"weights={selected_weight_source} "
            f"bad={bad_epochs} "
            f"({epoch_row['epoch_seconds'] / 60:.1f} min)",
            flush=True,
        )
        log_event("epoch_end", epoch=epoch,
                      train_loss=train_values["loss"],
                      val_pathology_only_macro_ap=validation_pathology_only_macro_average_precision,
                      improved=improved, bad_epochs=bad_epochs,
                      epoch_seconds=round(time.time() - epoch_started, 1))
        join_s3_uploads()
        if epoch % S3_UPLOAD_EVERY_N_EPOCHS == 0:
            upload_run_files(
                [last_path, history_path, config_snapshot_path, RUN_LOG_PATH]
                + ([best_path] if best_path.is_file() else []),
                required=True,
            )
        if epoch >= EARLY_STOPPING_MIN_EPOCHS and bad_epochs >= EARLY_STOPPING_PATIENCE:
            stopped_early = True
            print(f"early stopping after epoch {epoch} (patience {EARLY_STOPPING_PATIENCE})")
            break

    join_s3_uploads()
    upload_run_files(
        [last_path, best_path, history_path, config_snapshot_path, RUN_LOG_PATH],
        required=True,
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
            completion_path, f"{S3_RUN_PREFIX}/{completion_path.name}", required=True
        )
    log_event("training_finished", **{k: v for k, v in completion.items()})
    print("Training finished:", json.dumps(completion, indent=2))
"""),
    md(r"""## 11. Final validation, test, and per-source evaluation

The best validation checkpoint is fixed first; the test split is evaluated
exactly once afterwards. Thresholds are tuned per label on the validation rows
only, then frozen for test. The raw/EMA choice recorded in `best.pt` is also
fixed before test. Every row is primary in clean12 (no held-out
duplicates); all rows are predicted and saved so nothing is evaluated silently.
"""),
    code(r"""if not completion_path.is_file():
    raise RuntimeError("Finish training before final test evaluation.")
require_matching_config_fingerprint(
    "training_complete.json", json.loads(completion_path.read_text()).get("config_fingerprint")
)
if not best_path.is_file():
    raise FileNotFoundError(
        "No best.pt checkpoint found. Run the training cell first; this cell "
        "must never evaluate an unfinished run."
    )
best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
require_matching_config_fingerprint(
    "best.pt", best_checkpoint.get("config_fingerprint")
)
selected_weight_source = best_checkpoint["selected_weight_source"]
if selected_weight_source not in ("raw", "ema"):
    raise RuntimeError(f"Invalid selected weight source: {selected_weight_source!r}")
selected_state = best_checkpoint["model_ema" if selected_weight_source == "ema" else "model"]
model.load_state_dict(selected_state)
model.eval()
print("validation-selected weights:", selected_weight_source)
print(f"evaluating best checkpoint from epoch {best_checkpoint['epoch']} "
      f"(val primary pathology-only Macro AP="
      f"{best_checkpoint['validation_pathology_only_macro_average_precision']:.6f}; "
      f"12-label Macro AP={best_checkpoint['validation_macro_average_precision']:.6f})")

validation_output = predict_loader(model, val_loader, criterion)

val_primary = primary_mask(validation_output)
fixed_thresholds = [0.5] * len(LABELS)
tuned_thresholds = tune_thresholds_on_validation(
    validation_output["y_true"][val_primary],
    validation_output["y_probability"][val_primary],
)
print("validation-tuned thresholds:",
      {label: round(value, 4) for label, value in zip(LABELS, tuned_thresholds)})

# The checkpoint, weight source, and thresholds are fixed before test inference.
test_output = predict_loader(model, test_loader, criterion)
test_primary = primary_mask(test_output)

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
# Deployment export contains the validation-selected weights, never the raw
# optimizer state. The checkpoint is loadable with torch.load(weights_only=True).
inference_path = RUN_DIR / "inference.pt"
export_inference_checkpoint(
    model, selected_weight_source, best_checkpoint["epoch"], tuned_thresholds, inference_path
)

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

# Patient-bootstrap 95% confidence intervals for the test metrics. Patients,
# not images, are resampled (a patient's ECGs are correlated). Each round
# reweights rows by how often their patient was drawn; AP with sample weights
# equals AP on the resampled set. Undefined rounds (a label with no positive
# drawn) are skipped and counted.
def patient_bootstrap_average_precision(y_true, y_probability, patient_ids, rounds, seed):
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
    return draws

test_patient_ids = pd.Series(
    test_df["patient_id"].to_numpy(), index=test_df["record_id"].astype(str)
).reindex(np.asarray(test_output["metadata"]["record_id"], dtype=str)[test_primary])
if test_patient_ids.isna().any():
    raise RuntimeError("Some test predictions have no patient_id; cannot bootstrap by patient")
bootstrap_started = time.time()
bootstrap_draws = patient_bootstrap_average_precision(
    test_output["y_true"][test_primary], test_output["y_probability"][test_primary],
    test_patient_ids.to_numpy(), BOOTSTRAP_ROUNDS, SEED,
)
pathology_indices = [LABELS.index(label) for label in PATHOLOGY_LABELS]
bootstrap_macro = np.nanmean(bootstrap_draws[:, pathology_indices], axis=1)
lower_q, upper_q = 50 * (1 - BOOTSTRAP_CONFIDENCE), 100 - 50 * (1 - BOOTSTRAP_CONFIDENCE)
bootstrap_rows = []
for label_index, label in enumerate(LABELS):
    column = bootstrap_draws[:, label_index]
    valid = column[np.isfinite(column)]
    bootstrap_rows.append({
        "label": label,
        "test_average_precision": test_metrics["per_label"][label]["average_precision"],
        "ci_low": float(np.percentile(valid, lower_q)) if len(valid) else float("nan"),
        "ci_high": float(np.percentile(valid, upper_q)) if len(valid) else float("nan"),
        "valid_rounds": int(len(valid)),
        "support": test_metrics["per_label"][label]["support"],
    })
bootstrap_rows.append({
    "label": "PATHOLOGY_MACRO",
    "test_average_precision": test_metrics["pathology_only_macro_average_precision"],
    "ci_low": float(np.percentile(bootstrap_macro, lower_q)),
    "ci_high": float(np.percentile(bootstrap_macro, upper_q)),
    "valid_rounds": int(np.isfinite(bootstrap_macro).sum()),
    "support": int(test_primary.sum()),
})
bootstrap_ci = pd.DataFrame(bootstrap_rows)
bootstrap_ci.to_csv(RUN_DIR / "test_bootstrap_ci.csv", index=False)
print(f"patient bootstrap: {BOOTSTRAP_ROUNDS} rounds over {test_patient_ids.nunique():,} test patients "
      f"in {time.time() - bootstrap_started:.0f}s")

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
    "selected_weight_source": selected_weight_source,
    "inference_artifact": inference_path.name,
    "best_epoch": int(best_checkpoint["epoch"]),
    "selection_metric": "validation_primary_pathology_only_macro_average_precision",
    "best_checkpoint_validation_pathology_only_macro_average_precision": float(
        best_checkpoint["validation_pathology_only_macro_average_precision"]
    ),
    "best_checkpoint_validation_macro_average_precision": float(
        best_checkpoint["validation_macro_average_precision"]
    ),
    "threshold_selection": "per-label maximum F1 on primary validation rows only",
    "test_bootstrap_ci": {
        "method": "patient-level bootstrap, percentile interval",
        "rounds": BOOTSTRAP_ROUNDS,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "rows": bootstrap_rows,
    },
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
        "fusion": FUSION,
        "fusion_spec": FUSION_SPEC,
        "fusion_parameters": int(fusion_parameter_count),
        "best_checkpoint_mib": model_size_mib,
        "inference_weights_mib": inference_path.stat().st_size / 2**20,
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
print(f"test Average Precision with {BOOTSTRAP_CONFIDENCE:.0%} patient-bootstrap intervals")
display(bootstrap_ci.round(4))

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
    f"# CardioSentry ConvNeXt-V1 Base + {FUSION} fusion clean12 report (EC2 g6e.2xlarge)",
    "",
    f"- Dataset mode: {DATASET_MODE}",
    f"- Patient split: {SPLIT_VERSION}, seed {SPLIT_SEED}, sha256 {SPLIT_SHA256}",
    f"- Labels: " + ", ".join(f"{label} ({LABEL_NAMES[label]})" for label in LABELS),
    f"- Best epoch: {best_checkpoint['epoch']}",
    f"- Validation-selected weights: {selected_weight_source}",
    "- Inference export: inference.pt (selected weights, preprocessing, label order, frozen thresholds)",
    "- Selection rule: maximum validation pathology-only Macro Average Precision (never accuracy; 12-label Macro AP tracked as secondary)",
    f"- Validation primary Macro Average Precision: {validation_metrics['macro_average_precision']:.4f}",
    f"- Validation primary pathology-only Macro Average Precision: {validation_metrics['pathology_only_macro_average_precision']:.4f}",
    f"- Test primary Macro Average Precision: {test_metrics['macro_average_precision']:.4f}",
    f"- Test primary pathology-only Macro Average Precision: {test_metrics['pathology_only_macro_average_precision']:.4f} "
    f"({BOOTSTRAP_CONFIDENCE:.0%} patient-bootstrap CI {bootstrap_rows[-1]['ci_low']:.4f}–{bootstrap_rows[-1]['ci_high']:.4f})",
    f"- Rare-label sampling: {'repeat-factor, threshold ' + str(RARE_SAMPLING_THRESHOLD) if RARE_SAMPLING else 'off'}",
    f"- Fusion: CS_FUSION={FUSION} (neck {FUSION_NECK}, head {FUSION_HEAD}, "
    f"{fusion_parameter_count / 1e6:.2f} M new parameters)",
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
    "", f"## Test Average Precision with {BOOTSTRAP_CONFIDENCE:.0%} patient-bootstrap intervals", "",
    f"{BOOTSTRAP_ROUNDS} resamples of test patients (all of a patient's ECGs move together). "
    "Read rare labels with their interval and support.",
    "",
    "| label | AP | CI low | CI high | support |",
    "|---|---:|---:|---:|---:|",
] + [
    f"| {row['label']} | {value(row['test_average_precision'])} | {value(row['ci_low'])} | "
    f"{value(row['ci_high'])} | {row['support']} |"
    for row in bootstrap_rows
] + [
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

Everything below is written to the local NVMe run directory first and uploaded
to S3:

```text
<CS_RUN_ROOT>/<CS_RUN_NAME>/          -> s3://cardiosentry/runs/<CS_RUN_NAME>/
├── best.pt                              best validation pathology-Macro-AP checkpoint (full training state)
├── inference.pt                         selected raw/EMA weights + frozen preprocessing/thresholds
├── last.pt                              resume checkpoint (end of the newest finished epoch)
├── config_snapshot.json                 full training, sampling, data, and split identity
├── benchmark.json                       mode-aware time and VRAM estimates
├── input_pipeline_benchmark.json         training/eval batch probes and decoder-worker tuning
├── training_history.csv                 losses, raw/EMA AP and AUROC, per-label AP by epoch
├── last_intra.pt                        mid-epoch Spot-visibility snapshot (never used for resume)
├── run.log                              JSONL event log (setup, syncs, epochs, best checkpoints)
├── training_complete.json               terminal state
├── patient_split.csv (+ .sha256)        frozen patient → split assignment (+ site, records)
├── record_split.csv                     record_path → split (same split, record level)
├── split_audit.json                     split method, seed, SHA-256, per-split label support
├── source_label_support_by_split.csv    unique-ECG site/label audit
├── sample_grid_per_label.png            one training sheet per label
├── validation_tuned_thresholds.json     validation-only F1 thresholds
├── validation_predictions.csv           labels + twelve probabilities
├── test_predictions.csv                 labels + twelve probabilities
├── test_bootstrap_ci.csv                test AP per label and pathology Macro AP with 95% patient-bootstrap CI
├── metrics.csv                          fixed-0.5 + tuned aggregate/per-label metrics
├── per_source_metrics.csv               both threshold policies by site
├── source_condition_metrics.csv         site × label metrics with undefined-status reason
├── top_combination_metrics.csv          exact-match performance on common patterns
├── combination_summary.csv              overall exact/invalid-pattern summaries
├── result.json                          complete machine-readable report
├── report.md                            human-readable settings and results
├── learning_curves.png
├── test_precision_recall_curves.png
└── s3_upload_manifest.json              S3 URIs, sizes, and local SHA-256 of every uploaded file
```

Only the validation pathology-only Macro Average Precision (11 non-NORM labels)
controls checkpointing and early stopping. The test split is evaluated once,
after the best validation checkpoint and thresholds are fixed.

The NVMe instance store is wiped when the instance stops, but nothing above
lives only there. Stop the instance when the run is finished: the notebook does
not stop it for you.
"""),

]


for index, cell in enumerate(cells):
    cell["id"] = f"cs-v1-msff-c12-ec2-{index:02d}"

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
