"""Build the CardioSentry CAM-method comparison notebook (Grad-CAM vs Grad-CAM++ vs Score-CAM).

The notebook explains the clean12 ConvNeXt-V1 Base run the app serves, on
true-positive test pairs, and measures Deletion AUC, Insertion AUC and
Explanation Time. The method code is `xai_cam.py`, embedded verbatim.
"""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "CardioSentry_XAI_CAM_Comparison_Clean12_Colab.ipynb"
XAI_CAM_SOURCE = (HERE / "xai_cam.py").read_text()


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
    md(r"""# CardioSentry — choosing the explanation method: Grad-CAM vs Grad-CAM++ vs Score-CAM (Google Colab, L4 / A100 / T4)

This notebook compares three class-activation-map methods on **the model the app
serves**: the clean12 ConvNeXt-V1 Base run `convnextv1_base_clean12_a100_seed42`
(`best.pt`, EMA weights, epoch 6, SHA-256 `7e2583a6…`). All three methods explain
the same layer (`stages.3`, 1024 × 24 × 32), the same images and the same labels.
Only the explanation method changes.

| Metric | What it measures | Better |
|---|---|---|
| **Deletion AUC** | Pixels are removed (set to blank paper) in the map's order, most important first. Area under p(label) vs fraction removed. | lower |
| **Insertion AUC** | Start from blank paper and put pixels back in the map's order. Area under p(label) vs fraction inserted. | higher |
| **Explanation Time** | Wall-clock seconds to produce one map for one (image, label), timed separately on the GPU and on CPU. | lower |

**What is explained.** For each label except NORM, `CS_PAIRS_PER_LABEL` (default
20) **true positives** are drawn from the run's own **test** split
(`test_predictions.csv`): the label is present and the model scored it above its
validation-tuned threshold. That gives about 220 (image, label) pairs. A
**random map** at the same 24 × 32 resolution is scored as well, as a floor every
method has to beat.

**Fairness rules.**
- **One cut and one upsampling.** Every map is ReLU(Σ w_c A_c) at 24 × 32, then
  bilinearly upsampled to 768 × 1024; only the weights w_c differ.
- **Same tie-break.** Equal pixels are removed in the same seeded random order
  for every method of a pair.
- **Same precision.** Every curve uses the same precision.
- **Library check.** Section 6 checks Grad-CAM and Grad-CAM++ against
  `pytorch-grad-cam`.

**Cost.** Score-CAM needs one masked forward pass per channel (1,024 per image),
and each curve needs `CS_CURVE_STEPS + 1` forwards. Section 5 measures this GPU's
throughput and prints the projected run time before the main loop starts.

**Resume.** Per-image results are copied to Drive as they finish. After a
disconnect, choose **Run all** again with the same `CS_RUN_NAME`: finished images
are skipped. The ~220 test images are pulled from the 104.5 GiB clean12 archive
once, then cached on Drive as `eval_images.tar` (~0.2 GB), so later runtimes skip
the archive stream.

**Outputs** go to `MyDrive/CardioSentry/new_runs/xai_cam_comparison_clean12_v1/`:
- results: `summary.md`, `summary.csv`, `paired_vs_gradcam.csv`, `per_label.csv`,
  `per_pair_metrics.csv`, `timing_summary.csv`
- `figures/`
- raw maps and curves: `maps/`, `curves/`
- provenance: `config.json`, `pairs.csv`, `parity.csv`, `library_check.csv`

**Cheapest A100 workflow (two runtimes).**
1. On a **CPU** runtime, uncomment the `CS_STAGE_IMAGES_ONLY` line in cell 1b
   and choose Run all. It streams the archive, caches the ~210 test images on
   Drive, and stops with an intended `SystemExit`.
2. Comment that line out again, switch to the **A100** runtime, and choose Run
   all. It finds the cache, so the GPU never waits on the 104.5 GiB stream.

On the GPU, section 5c picks the largest batch that fits. It then times
`torch.compile` against eager on that batch and keeps the faster one.

Runtime: choose **L4** (recommended), A100, or T4 under Runtime → Change runtime
type. T4 runs in fp16; L4 and A100 run in bf16. The precision is part of the run
fingerprint, so a T4 run and an L4 run need different `CS_RUN_NAME`s.
"""),
    code(r"""# 1. Install only what Colab lacks. Colab's own PyTorch/CUDA build is kept.
# grad-cam (pytorch-grad-cam) is used only for the cross-check in section 6.
%pip install -q "timm==1.0.28" "grad-cam==1.5.7" "pandas>=2.1" "pillow>=10" "tqdm>=4.66" "scipy>=1.11"
"""),
    code(r"""# 1b. Run settings. Uncomment a line to change it (README lists every CS_* setting).
import os
# os.environ["CS_STAGE_IMAGES_ONLY"] = "1"   # on a CPU runtime: cache the test images on Drive, then stop
# os.environ["CS_RUN_NAME"] = "xai_cam_comparison_clean12_v1"
# os.environ["CS_PAIRS_PER_LABEL"] = "20"
# os.environ["CS_CURVE_STEPS"] = "50"
# os.environ["CS_DATASET_READ_MODE"] = "mount"  # if the Drive API sign-in fails
# os.environ["CS_COMPILE"] = "0"             # force eager
"""),
    code(r"""# 2. Configuration, Google Drive, and device.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import copy, gc, hashlib, itertools, json, math, platform, re, shutil, subprocess, sys
import tarfile, threading, time, warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from tqdm.auto import tqdm

def env_flag(name, default):
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off", ""}

def env_str(name, default):
    return os.environ.get(name, default).strip()

def gib(n_bytes):
    return n_bytes / 2**30

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ----------------------------------------------------------------------------
# Google Drive: the model run, the dataset archive, and every output.
# ----------------------------------------------------------------------------
DRIVE_MOUNT_POINT = Path("/content/drive")
if IN_COLAB and not (DRIVE_MOUNT_POINT / "MyDrive").is_dir():
    from google.colab import drive
    drive.mount(str(DRIVE_MOUNT_POINT))
CARDIOSENTRY_DRIVE_ROOT = Path(env_str(
    "CS_DRIVE_PROJECT_ROOT", str(DRIVE_MOUNT_POINT / "MyDrive" / "CardioSentry"))).expanduser()
if not CARDIOSENTRY_DRIVE_ROOT.is_dir():
    raise RuntimeError(
        f"{CARDIOSENTRY_DRIVE_ROOT} is not available. On Colab this cell mounts Google "
        "Drive; elsewhere set CS_DRIVE_PROJECT_ROOT to the CardioSentry folder.")
DRIVE_RUNS_ROOT = Path(env_str("CS_DRIVE_RUNS_DIR", str(CARDIOSENTRY_DRIVE_ROOT / "new_runs"))).expanduser()
DRIVE_SYNC_ENABLED = env_flag("CS_DRIVE_SYNC", "1")
# After the last cell succeeds, flush Drive and release the runtime (stops compute units).
RELEASE_RUNTIME_WHEN_DONE = env_flag("CS_RELEASE_RUNTIME", "1")

# ----------------------------------------------------------------------------
# The model being explained: the clean12 ConvNeXt-V1 Base run the app serves.
# ----------------------------------------------------------------------------
REFERENCE_RUN = env_str("CS_REFERENCE_RUN", "convnextv1_base_clean12_a100_seed42")
REFERENCE_RUN_DIR = DRIVE_RUNS_ROOT / REFERENCE_RUN
CHECKPOINT_PATH = Path(env_str("CS_CHECKPOINT", str(REFERENCE_RUN_DIR / "best.pt")))
# The app's models/active.yaml pins this hash; "" skips the check.
CHECKPOINT_SHA256 = env_str(
    "CS_CHECKPOINT_SHA256", "7e2583a6a9705eae27501a512c8a3c05929e9709713ff2b525c4ae4f4391078b").lower()
TEST_PREDICTIONS_PATH = Path(env_str("CS_TEST_PREDICTIONS", str(REFERENCE_RUN_DIR / "test_predictions.csv")))
# The frozen patient split's test set (seed 20261002, sha 4eba80c7...). 0 skips the check.
EXPECTED_TEST_ROWS = int(env_str("CS_EXPECTED_TEST_ROWS", "11323"))
MODEL_NAME = env_str("CS_MODEL_NAME", "convnext_base.fb_in22k_ft_in1k_384")
INPUT_HW = tuple(int(v) for v in env_str("CS_INPUT_HW", "768,1024").split(","))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ----------------------------------------------------------------------------
# Dataset archive: the same Drive copy the clean12 training notebooks read.
# ----------------------------------------------------------------------------
DATASET_TAR_NAME = env_str("CS_DATASET_TAR_NAME", "clean_ecg_signal_images_clean12.tar")
DATASET_DRIVE_DIR = Path(env_str("CS_DATASET_DRIVE_DIR", str(CARDIOSENTRY_DRIVE_ROOT / "new_dataset"))).expanduser()
DATASET_DRIVE_PATH = DATASET_DRIVE_DIR / DATASET_TAR_NAME
DATASET_SIDECAR_PATH = DATASET_DRIVE_DIR / f"{DATASET_TAR_NAME}.sha256"
REQUIRE_TAR_SHA256 = env_flag("CS_REQUIRE_TAR_SHA256", "1")
# "api": parallel ranged downloads through the Drive API (one extra sign-in).
# "mount": parallel reads through /content/drive (no extra sign-in).
DATASET_READ_MODE = env_str("CS_DATASET_READ_MODE", "api").lower()
if DATASET_READ_MODE not in {"api", "mount"}:
    raise ValueError("CS_DATASET_READ_MODE must be api or mount")
DATASET_DRIVE_FILE_ID = env_str("CS_DATASET_DRIVE_FILE_ID", "")
DATASET_DOWNLOAD_WORKERS = int(env_str(
    "CS_DATASET_DOWNLOAD_WORKERS", "16" if DATASET_READ_MODE == "api" else "8"))
DATASET_DOWNLOAD_CHUNK_BYTES = int(float(env_str("CS_DATASET_DOWNLOAD_CHUNK_MIB", "32")) * 2**20)
KIT_TOPDIR = "clean_ecg_signal_images_clean12"

# ----------------------------------------------------------------------------
# Experiment design. Every value here is in the run fingerprint: changing one
# needs a new CS_RUN_NAME, so one results folder never mixes two designs.
# ----------------------------------------------------------------------------
PAIRS_PER_LABEL = int(env_str("CS_PAIRS_PER_LABEL", "20"))
EXCLUDE_LABELS = [v for v in env_str("CS_EXCLUDE_LABELS", "NORM").split(",") if v]
PAIR_SEED = int(env_str("CS_PAIR_SEED", "42"))
# Curve points = CURVE_STEPS + 1; 50 steps = 2% of the pixels per step.
CURVE_STEPS = int(env_str("CS_CURVE_STEPS", "50"))
# What a deleted pixel becomes, and what insertion starts from:
#   paper (default): the sheet's own blank-paper colour, from its 15-px side pad
#   blur:  a Gaussian-blurred copy (Petsiuk et al.'s insertion canvas)
#   zero:  black
BASELINE = env_str("CS_BASELINE", "paper").lower()
PAPER_PAD_PX = int(env_str("CS_PAPER_PAD_PX", "15"))
BLUR_SIGMA_PX = float(env_str("CS_BLUR_SIGMA_PX", "10"))
# auto: bf16 where the GPU supports it (L4, A100), else fp16 (T4); fp32 on CPU.
AMP_DTYPE_NAME = env_str("CS_AMP_DTYPE", "auto").lower()
# Our probabilities vs the run's stored test probabilities. Both are bf16 (the
# run evaluated under bf16 autocast), so single values drift by ~1e-2 where the
# sigmoid is steepest; a wrong model or wrong pixels moves the MEDIAN to ~1e-1.
# The median is the identity test; the max only catches a gross outlier.
PARITY_MEDIAN_TOLERANCE = float(env_str("CS_PARITY_MEDIAN_TOLERANCE", "0.003"))
PARITY_MAX_TOLERANCE = float(env_str("CS_PARITY_MAX_TOLERANCE", "0.05"))
# torch.compile output vs eager on the same batch, max |difference|.
COMPILE_TOLERANCE = float(env_str("CS_COMPILE_TOLERANCE", "0.01"))

# Not in the fingerprint: they change speed or reporting, never a result.
# Stage only: pull the test images into the Drive cache (eval_images.tar) and
# stop. Run this on a cheap CPU runtime first; the archive stream is pure I/O,
# and an A100 would sit idle (and bill) for it.
STAGE_IMAGES_ONLY = env_flag("CS_STAGE_IMAGES_ONLY", "0")
# torch.compile for the Score-CAM and curve forwards (fixed batch shape).
# auto: compile on CUDA, time compiled vs eager on the real batch, keep the faster.
COMPILE = env_str("CS_COMPILE", "auto").lower()
if COMPILE not in {"auto", "1", "0"}:
    raise ValueError("CS_COMPILE must be auto, 1 or 0")
COMPILE_MODE = env_str("CS_COMPILE_MODE", "default")
BATCH_SIZE_OVERRIDE = int(env_str("CS_BATCH_SIZE", "0"))
BATCH_CANDIDATES = [int(v) for v in env_str("CS_BATCH_CANDIDATES", "64,48,32,24,16,12,8,4,2,1").split(",")]
LIBRARY_CHECK_PAIRS = int(env_str("CS_LIBRARY_CHECK_PAIRS", "3"))
TIMING_PAIRS = int(env_str("CS_TIMING_PAIRS", "10"))
TIMING_WARMUP = int(env_str("CS_TIMING_WARMUP", "2"))
CPU_TIMING = env_flag("CS_CPU_TIMING", "1")
CPU_TIMING_PAIRS = int(env_str("CS_CPU_TIMING_PAIRS", "3"))
CPU_THREADS = int(env_str("CS_CPU_THREADS", "0"))          # 0 = every core
# Score-CAM on a CPU is hours per map; time this many masked forwards and
# extrapolate, unless CS_CPU_SCORECAM_FULL=1 runs one map for real.
CPU_SCORECAM_PROBE_CHANNELS = int(env_str("CS_CPU_SCORECAM_PROBE_CHANNELS", "16"))
CPU_SCORECAM_BATCH = int(env_str("CS_CPU_SCORECAM_BATCH", "4"))
CPU_SCORECAM_FULL = env_flag("CS_CPU_SCORECAM_FULL", "0")
BOOTSTRAP_ROUNDS = int(env_str("CS_BOOTSTRAP_ROUNDS", "2000"))
BOOTSTRAP_SEED = int(env_str("CS_BOOTSTRAP_SEED", "42"))
EXAMPLE_LABELS = int(env_str("CS_EXAMPLE_LABELS", "6"))

# ----------------------------------------------------------------------------
# Output folders. Files are written locally first, then copied to Drive.
# ----------------------------------------------------------------------------
RUN_NAME = env_str("CS_RUN_NAME", "xai_cam_comparison_clean12_v1")
DRIVE_RUN_DIR = DRIVE_RUNS_ROOT / RUN_NAME
LOCAL_ROOT = Path(env_str("CS_LOCAL_ROOT", "/content/xai_scratch")).expanduser()
RUN_DIR = LOCAL_ROOT / RUN_NAME
IMAGES_ROOT = LOCAL_ROOT / "images"
for folder in (RUN_DIR, IMAGES_ROOT, RUN_DIR / "maps", RUN_DIR / "curves", RUN_DIR / "figures"):
    folder.mkdir(parents=True, exist_ok=True)
if DRIVE_SYNC_ENABLED:
    DRIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Device and precision.
# ----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    DEVICE_NAME = torch.cuda.get_device_name(DEVICE)
else:
    DEVICE_NAME = next((line.split(":", 1)[1].strip() for line in
                        Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
                       platform.machine()) if Path("/proc/cpuinfo").is_file() else platform.machine()
    warnings.warn("No GPU: the evaluation will run on CPU and take days at full size.")
_amp_choices = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
if AMP_DTYPE_NAME == "auto":
    if DEVICE.type != "cuda":
        AMP_DTYPE_NAME = "fp32"
    else:
        AMP_DTYPE_NAME = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
if AMP_DTYPE_NAME not in _amp_choices:
    raise ValueError("CS_AMP_DTYPE must be auto, bf16, fp16 or fp32")
AMP_DTYPE = _amp_choices[AMP_DTYPE_NAME]
DEVICE_SLUG = re.sub(r"[^a-z0-9]+", "_", DEVICE_NAME.lower()).strip("_") or DEVICE.type

print(f"device: {DEVICE} ({DEVICE_NAME}), precision {AMP_DTYPE_NAME}")
print(f"torch {torch.__version__}, timm {timm.__version__}, python {platform.python_version()}")
print("model run:", REFERENCE_RUN_DIR)
print("outputs:  ", DRIVE_RUN_DIR if DRIVE_SYNC_ENABLED else RUN_DIR)
"""),
    code(r"""# 2b. Drive persistence and the run fingerprint.
def drive_copy(local_path):
    '''Copy a file under RUN_DIR to the same relative path under DRIVE_RUN_DIR (atomic rename).'''
    if not DRIVE_SYNC_ENABLED:
        return
    local_path = Path(local_path)
    destination = DRIVE_RUN_DIR / local_path.relative_to(RUN_DIR)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    for attempt in range(5):
        try:
            shutil.copyfile(local_path, partial)
            os.replace(partial, destination)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

def restore_from_drive(relative):
    '''Copy DRIVE_RUN_DIR/relative to RUN_DIR/relative if Drive has it; True if it did.'''
    source = DRIVE_RUN_DIR / relative
    if not (DRIVE_SYNC_ENABLED and source.is_file()):
        return False
    target = RUN_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return True

def run_file(relative):
    '''RUN_DIR/relative, restored from Drive first if this runtime does not have it.'''
    path = RUN_DIR / relative
    if not path.is_file() and not restore_from_drive(relative):
        raise FileNotFoundError(f"{relative} is neither on this runtime nor in {DRIVE_RUN_DIR}")
    return path

def save_json(payload, relative):
    path = RUN_DIR / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    drive_copy(path)
    return path

def save_csv(frame, relative):
    path = RUN_DIR / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    drive_copy(path)
    return path

def sha256_file(path, chunk_size=16 * 2**20):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()

def stable_seed(*parts):
    '''A 31-bit seed from text, identical on every machine and Python run.'''
    text = ":".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF

DESIGN = {
    "reference_run": REFERENCE_RUN,
    "checkpoint_sha256": CHECKPOINT_SHA256 or "unchecked",
    "model_name": MODEL_NAME,
    "input_hw": list(INPUT_HW),
    "target_layer": "stages.3",
    "methods": ["gradcam", "gradcampp", "scorecam", "random"],
    "pairs_per_label": PAIRS_PER_LABEL,
    "exclude_labels": EXCLUDE_LABELS,
    "pair_seed": PAIR_SEED,
    "pair_rule": "true positive at the validation-tuned threshold, test split",
    "curve_steps": CURVE_STEPS,
    "baseline": BASELINE,
    "paper_pad_px": PAPER_PAD_PX,
    "blur_sigma_px": BLUR_SIGMA_PX if BASELINE == "blur" else None,
    "precision": AMP_DTYPE_NAME,
    "map": "relu(sum_c w_c A_c) at 24x32, bilinear (align_corners=False) to the input",
    "tie_break": "seeded random permutation per pair, shared by all methods",
}
FINGERPRINT = hashlib.sha256(json.dumps(DESIGN, sort_keys=True).encode()).hexdigest()[:16]

if STAGE_IMAGES_ONLY:
    print("CS_STAGE_IMAGES_ONLY=1: staging the test images only; the run design is "
          "recorded by the evaluation runtime")
elif restore_from_drive("config.json"):
    stored = json.loads((RUN_DIR / "config.json").read_text())
    if stored.get("fingerprint") != FINGERPRINT:
        changed = sorted(k for k in DESIGN if stored.get("design", {}).get(k) != DESIGN[k])
        raise RuntimeError(
            f"{DRIVE_RUN_DIR} holds a run with a different design (changed: {changed}). "
            "Set a new CS_RUN_NAME, or restore the original settings to resume.")
    print("resuming run", RUN_NAME, "fingerprint", FINGERPRINT)
else:
    save_json({"fingerprint": FINGERPRINT, "design": DESIGN,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "device": DEVICE_NAME, "torch": torch.__version__, "timm": timm.__version__},
              "config.json")
    print("new run", RUN_NAME, "fingerprint", FINGERPRINT)
"""),
    code(r"""# 2c. Preflight: every input exists, and the checkpoint is the app's, before anything long.
missing_inputs = [str(p) for p in (CHECKPOINT_PATH, TEST_PREDICTIONS_PATH) if not p.is_file()]
if not (DRIVE_RUN_DIR / "eval_images.tar").is_file() and not DATASET_DRIVE_PATH.is_file():
    missing_inputs.append(f"{DATASET_DRIVE_PATH} (and no eval_images.tar cache in {DRIVE_RUN_DIR})")
if missing_inputs:
    raise FileNotFoundError("Missing inputs on Drive:\n- " + "\n- ".join(missing_inputs)
                            + "\nCheck CS_REFERENCE_RUN / CS_CHECKPOINT / CS_DATASET_DRIVE_DIR.")

local_checkpoint = LOCAL_ROOT / "checkpoint" / CHECKPOINT_PATH.name
local_checkpoint.parent.mkdir(parents=True, exist_ok=True)
if STAGE_IMAGES_ONLY:
    checkpoint_sha = "not checked (stage-only)"
elif not (local_checkpoint.is_file() and local_checkpoint.stat().st_size == CHECKPOINT_PATH.stat().st_size):
    print(f"copying {CHECKPOINT_PATH} ({CHECKPOINT_PATH.stat().st_size / 2**30:.2f} GiB) ...")
    shutil.copyfile(CHECKPOINT_PATH, local_checkpoint.with_suffix(".partial"))
    os.replace(local_checkpoint.with_suffix(".partial"), local_checkpoint)
if not STAGE_IMAGES_ONLY:
    checkpoint_sha = sha256_file(local_checkpoint)
    if CHECKPOINT_SHA256 and checkpoint_sha != CHECKPOINT_SHA256:
        raise RuntimeError(f"{CHECKPOINT_PATH} has SHA-256 {checkpoint_sha}, expected {CHECKPOINT_SHA256} "
                           "(the app's active model). Set CS_CHECKPOINT_SHA256 to explain another checkpoint.")
print(f"inputs present; checkpoint sha256 {checkpoint_sha[:12]}...")
"""),
    md(r"""## 3. The method code (`xai_cam.py`)

This cell is `xai_cam.py` from the notebook's folder, copied in unchanged.
`cpu_timing.py` imports the same file to time the methods on the deployment laptop.
"""),
    code("# 3. Grad-CAM, Grad-CAM++, Score-CAM, deletion/insertion (xai_cam.py, verbatim).\n"
         + XAI_CAM_SOURCE),
    md(r"""## 4. Choose the (image, label) pairs from the test split

The pairs come from the reference run's `test_predictions.csv`: its test split is
the frozen patient split every clean12 model uses (11,323 records). A pair is a
**true positive**: `true_<label> = 1` and the model scored it above that label's
validation-tuned threshold (`pred_validation_tuned_<label> = 1`).

Explaining a label the model did not detect would measure a map of nothing.

Per label, the pairs are the first `CS_PAIRS_PER_LABEL` true positives in
SHA-256(`seed:label:record`) order. That is a seeded random sample that does not
depend on row order. An image can carry several labels, so it can appear in
several pairs; the statistics in section 9 resample by image to account for that.
"""),
    code(r"""# 4. Pairs: seeded true-positive sample per label from the test split.
predictions = pd.read_csv(TEST_PREDICTIONS_PATH, low_memory=False)
LABELS = [c[len("prob_"):] for c in predictions.columns if c.startswith("prob_")]
for label in LABELS:
    for prefix in ("true_", "pred_validation_tuned_"):
        if f"{prefix}{label}" not in predictions.columns:
            raise KeyError(f"{TEST_PREDICTIONS_PATH.name} has no {prefix}{label} column")
if "record_id" not in predictions.columns or predictions["record_id"].duplicated().any():
    raise ValueError("test_predictions.csv needs one row per record_id")
if EXPECTED_TEST_ROWS and len(predictions) != EXPECTED_TEST_ROWS:
    raise ValueError(
        f"{TEST_PREDICTIONS_PATH} has {len(predictions):,} rows, expected the frozen test split's "
        f"{EXPECTED_TEST_ROWS:,}. Is CS_REFERENCE_RUN a clean12 run? (CS_EXPECTED_TEST_ROWS=0 skips this.)")
if "split" in predictions.columns and set(predictions["split"].astype(str)) != {"test"}:
    raise ValueError(f"test_predictions.csv contains splits {sorted(set(predictions['split']))}")
unknown = sorted(set(EXCLUDE_LABELS) - set(LABELS))
if unknown:
    raise ValueError(f"CS_EXCLUDE_LABELS names unknown labels {unknown}; labels are {LABELS}")
EXPLAINED_LABELS = [label for label in LABELS if label not in EXCLUDE_LABELS]

pair_rows, availability = [], []
for label in EXPLAINED_LABELS:
    positives = predictions[predictions[f"true_{label}"] == 1]
    tp = positives[positives[f"pred_validation_tuned_{label}"] == 1].copy()
    tp["order"] = [hashlib.sha256(f"{PAIR_SEED}:{label}:{rid}".encode()).hexdigest()
                   for rid in tp["record_id"]]
    chosen = tp.sort_values("order").head(PAIRS_PER_LABEL)
    availability.append({"label": label, "test_positives": len(positives),
                         "true_positives": len(tp), "chosen": len(chosen)})
    for _, row in chosen.iterrows():
        pair_rows.append({"pair_id": f"{label}|{row['record_id']}", "record_id": row["record_id"],
                          "label": label, "label_index": LABELS.index(label),
                          "prob_reference": float(row[f"prob_{label}"])})
pairs = pd.DataFrame(pair_rows)
availability = pd.DataFrame(availability)
print(availability.to_string(index=False))
short = availability[availability["chosen"] < PAIRS_PER_LABEL]
if len(short):
    print(f"\nfewer than {PAIRS_PER_LABEL} true positives for: {', '.join(short['label'])} (all were taken)")

if restore_from_drive("pairs.csv"):
    stored_pairs = pd.read_csv(RUN_DIR / "pairs.csv")
    if (list(stored_pairs["pair_id"]) != list(pairs["pair_id"])):
        raise RuntimeError("pairs.csv on Drive differs from the pairs drawn now: the reference "
                           "predictions changed. Use a new CS_RUN_NAME.")
save_csv(pairs, "pairs.csv")
save_csv(availability, "pair_availability.csv")

IMAGE_IDS = sorted(pairs["record_id"].unique())

def image_member(record_id):
    return f"{KIT_TOPDIR}/images/{Path(record_id).with_suffix('.png').as_posix()}"

def image_path(record_id):
    return IMAGES_ROOT / image_member(record_id)

def image_slug(record_id):
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", Path(record_id).with_suffix("").as_posix())

print(f"\n{len(pairs)} pairs over {len(IMAGE_IDS)} images and {len(EXPLAINED_LABELS)} labels")
"""),
    md(r"""## 4b. Get the test images

There are two sources:

1. **`eval_images.tar` in this run's Drive folder (~0.2 GB).** It is written the
   first time and reused on every later runtime.
2. **The clean12 archive (104.5 GiB, the first time only).** Only the needed PNGs
   are extracted (`tar -T`), but the whole archive has to stream past, because a
   TAR has no index. Every byte is SHA-256 checked against the archive's sidecar.
   This takes roughly 20–60 minutes, depending on Drive throughput.

`CS_DATASET_READ_MODE=api` (default) asks for one extra Google sign-in; use the
account that owns `CardioSentry`. `mount` needs no sign-in.
"""),
    code(r"""# 4b. Stream helpers (same code as the clean12 training notebooks' section 3).
def expected_tar_sha256():
    if not DATASET_SIDECAR_PATH.is_file():
        if REQUIRE_TAR_SHA256:
            raise RuntimeError(f"No checksum sidecar at {DATASET_SIDECAR_PATH} and CS_REQUIRE_TAR_SHA256=1.")
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
    global _drive_credentials
    import google.auth
    from google.auth.transport.requests import Request as GoogleAuthRequest
    with _drive_credentials_lock:
        if _drive_credentials is None:
            if IN_COLAB:
                from google.colab import auth as colab_auth
                colab_auth.authenticate_user()
            _drive_credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive"])
        if force_refresh or not _drive_credentials.valid:
            _drive_credentials.refresh(GoogleAuthRequest())
        return _drive_credentials.token

def drive_api_get(url, params=None, headers=None, timeout=(30, 300)):
    for force_refresh in (False, True):
        response = requests.get(url, params=params, timeout=timeout, headers={
            **(headers or {}), "Authorization": f"Bearer {drive_api_token(force_refresh)}"})
        if response.status_code != 401:
            break
    return response

def drive_file_id_for(path):
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
            raise RuntimeError(f"Expected exactly one {part!r} on Drive while resolving {path}, "
                               f"found {len(found)}. Set CS_DATASET_DRIVE_FILE_ID.")
        parent = found[0]["id"]
    return parent

def iter_ordered_chunks(fetch, total, chunk_bytes, workers):
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
                problem = f"HTTP {code} ({len(response.content):,} bytes)"
            except requests.RequestException as exc:
                problem = repr(exc)
            time.sleep(min(60, 2 ** attempt))
        raise IOError(f"Drive read at byte {offset:,} kept failing: {problem}")
    return fetch

def mounted_file_fetcher(path, total, chunk_bytes):
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

def open_archive_chunks(total, expected_sha):
    if DATASET_READ_MODE == "mount":
        fetch = mounted_file_fetcher(DATASET_DRIVE_PATH, total, DATASET_DOWNLOAD_CHUNK_BYTES)
        closer = fetch.close
    else:
        file_id = drive_file_id_for(DATASET_DRIVE_PATH)
        response = drive_api_get(f"{DRIVE_API_FILES}/{file_id}", params={
            "fields": "id,name,size,sha256Checksum", "supportsAllDrives": "true"})
        response.raise_for_status()
        meta = response.json()
        if int(meta["size"]) != total:
            raise RuntimeError(f"Drive API size {int(meta['size']):,} != mounted size {total:,}")
        if expected_sha and meta.get("sha256Checksum") not in (None, expected_sha):
            raise RuntimeError(f"Drive's SHA-256 of {DATASET_DRIVE_PATH} is {meta['sha256Checksum']}, "
                               f"the sidecar says {expected_sha}: the Drive copy is damaged.")
        fetch = drive_api_fetcher(meta["id"], total, DATASET_DOWNLOAD_CHUNK_BYTES)
        closer = lambda: None
    return iter_ordered_chunks(fetch, total, DATASET_DOWNLOAD_CHUNK_BYTES, DATASET_DOWNLOAD_WORKERS), closer

def stream_extract_members(members):
    '''Stream the whole archive into `tar -x -T list`, hashing every byte; extract only `members`.'''
    expected_sha = expected_tar_sha256()
    total = int(DATASET_DRIVE_PATH.stat().st_size)
    list_path = LOCAL_ROOT / "wanted_members.txt"
    list_path.write_text("".join(f"{m}\n" for m in members))
    log_path = LOCAL_ROOT / "tar_extract.log"
    command = ["tar", "-x", "-f", "-", "-C", str(IMAGES_ROOT), "--no-same-owner",
               "--no-same-permissions", "-T", str(list_path)]
    digest, received, started = hashlib.sha256(), 0, time.time()
    chunks, closer = open_archive_chunks(total, expected_sha)
    progress = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                    desc=f"clean12 archive ({DATASET_READ_MODE}) -> {len(members)} PNGs")
    try:
        with log_path.open("wb") as tar_log:
            extractor = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=tar_log)
            try:
                for data in chunks:
                    digest.update(data)
                    extractor.stdin.write(data)
                    received += len(data)
                    progress.update(len(data))
                extractor.stdin.close()
                returncode = extractor.wait()
            except BaseException:
                extractor.kill()
                extractor.wait()
                raise
    finally:
        progress.close()
        chunks.close()
        closer()
    tail = log_path.read_text(errors="replace")[-2000:]
    problems = []
    if returncode != 0:
        problems.append(f"tar exited with {returncode}: {tail}")
    if received != total:
        problems.append(f"received {received:,} bytes of {total:,}")
    if expected_sha and digest.hexdigest() != expected_sha:
        problems.append(f"archive SHA-256 {digest.hexdigest()} != sidecar {expected_sha}")
    if problems:
        raise RuntimeError("Archive stream failed:\n- " + "\n- ".join(problems))
    seconds = time.time() - started
    print(f"archive verified ({expected_sha or 'size only'}); "
          f"{gib(total) * 1024 / max(seconds, 1e-9):.0f} MiB/s, {seconds / 60:.1f} min")
    return expected_sha
"""),
    code(r"""# 4c. Stage the test images: Drive cache first, the full archive only if needed.
EVAL_TAR = RUN_DIR / "eval_images.tar"
EVAL_TAR_INDEX = RUN_DIR / "eval_images.json"
needed_members = [image_member(rid) for rid in IMAGE_IDS]

def images_present():
    return all(image_path(rid).is_file() for rid in IMAGE_IDS)

def extract_eval_cache():
    '''Extract eval_images.tar from Drive if it holds every needed image. True on success.'''
    if not (restore_from_drive("eval_images.json") and restore_from_drive("eval_images.tar.sha256")):
        return False
    index = json.loads(EVAL_TAR_INDEX.read_text())
    if not set(needed_members) <= set(index["members"]):
        print("eval_images.tar on Drive lacks some of these pairs' images; rebuilding it")
        return False
    restore_from_drive("eval_images.tar")
    expected = (RUN_DIR / "eval_images.tar.sha256").read_text().split()[0]
    if sha256_file(EVAL_TAR) != expected:
        raise RuntimeError(f"{DRIVE_RUN_DIR / 'eval_images.tar'} fails its SHA-256; delete it and re-run")
    with tarfile.open(EVAL_TAR) as archive:
        archive.extractall(IMAGES_ROOT, filter="data")
    return images_present()

if images_present():
    print(f"all {len(IMAGE_IDS)} images already on this runtime")
elif extract_eval_cache():
    print(f"extracted {len(IMAGE_IDS)} images from the Drive cache eval_images.tar")
else:
    archive_sha = stream_extract_members(needed_members)
    missing = [rid for rid in IMAGE_IDS if not image_path(rid).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images were not in the archive, e.g. {missing[:3]}")
    with tarfile.open(EVAL_TAR, "w") as archive:
        for member in needed_members:
            archive.add(IMAGES_ROOT / member, arcname=member)
    tar_sha = sha256_file(EVAL_TAR)
    (RUN_DIR / "eval_images.tar.sha256").write_text(f"{tar_sha}  eval_images.tar\n")
    EVAL_TAR_INDEX.write_text(json.dumps({
        "members": needed_members, "source_archive": str(DATASET_DRIVE_PATH),
        "source_sha256": archive_sha, "created_utc": datetime.now(timezone.utc).isoformat()}, indent=1))
    for name in ("eval_images.tar", "eval_images.tar.sha256", "eval_images.json"):
        drive_copy(RUN_DIR / name)
    print(f"cached {len(needed_members)} images on Drive as eval_images.tar "
          f"({EVAL_TAR.stat().st_size / 2**20:.0f} MiB)")

if STAGE_IMAGES_ONLY:
    print("\nStage-only run finished: the test images are cached in "
          f"{DRIVE_RUN_DIR / 'eval_images.tar'}.\nSwitch to the A100/L4 runtime, set "
          "CS_STAGE_IMAGES_ONLY=0 (the default), keep the same CS_RUN_NAME, and choose Run all.")
    if IN_COLAB and DRIVE_SYNC_ENABLED:
        from google.colab import drive
        drive.flush_and_unmount()
    raise SystemExit("stage-only run complete (this stop is intended)")
"""),
    md(r"""## 5. Load the model, check parity, and size the batches

- **Checkpoint.** `best.pt` is copied to the local disk and checked against the
  SHA-256 that the app's `models/active.yaml` pins. The weights loaded are those
  the run selected on validation (`selected_weight_source`, EMA), exactly as the
  app does.
- **Parity.** The notebook rescores every chosen image against the run's stored
  test probabilities. The **median** difference must be ≤ 0.003, and no single
  value may differ by more than 0.05. That proves this is the same model on the
  same pixels.
  - Both copies are bf16, so a few values near p = 0.5 drift by ~0.01. That is
    rounding.
  - A wrong model or wrong preprocessing moves the median to ~0.1.
- **Batch size.** Section 5c picks the largest batch that runs the real Score-CAM
  and curve code without running out of memory. It then measures throughput and
  prints the projected run time.
"""),
    code(r"""# 5. Load the reference checkpoint (selected weights, strict).
blob = torch.load(local_checkpoint, map_location="cpu", weights_only=False, mmap=True)
if isinstance(blob, dict) and "model" in blob:
    weight_source = blob.get("selected_weight_source") or "raw"
    state = blob["model_ema" if weight_source == "ema" else "model"]
else:
    weight_source, state = "state_dict", blob
if state and all(k.startswith("_orig_mod.") for k in state):
    state = {k[len("_orig_mod."):]: v for k, v in state.items()}
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(LABELS))
model.load_state_dict(state, strict=True)
best_epoch = blob.get("epoch") if isinstance(blob, dict) else None
del blob, state
gc.collect()
model.eval().to(DEVICE, memory_format=torch.channels_last)
for parameter in model.parameters():
    parameter.requires_grad_(False)
split = ModelSplit(model)
MEAN = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD, device=DEVICE).view(3, 1, 1)
print(f"loaded {MODEL_NAME}: {weight_source} weights, epoch {best_epoch}, sha256 {checkpoint_sha[:12]}...")

def load_image01(record_id):
    '''The sheet as (3, H, W) float in [0, 1], decoded exactly as in training.'''
    with Image.open(image_path(record_id)) as image:
        image = image.convert("RGB")
        if image.size != (INPUT_HW[1], INPUT_HW[0]):
            image = image.resize((INPUT_HW[1], INPUT_HW[0]), Image.BICUBIC)
        pixels = np.asarray(image).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255.0)

def normalize(images01):
    '''(B, 3, H, W) in [0, 1] -> model input (channels_last).'''
    return ((images01 - MEAN) / STD).contiguous(memory_format=torch.channels_last)
"""),
    code(r"""# 5b. Parity: our probabilities vs the run's stored test probabilities.
reference = predictions.set_index("record_id").loc[IMAGE_IDS, [f"prob_{l}" for l in LABELS]].to_numpy()
parity_rows, parity_diffs = [], []
with torch.no_grad():
    for start in tqdm(range(0, len(IMAGE_IDS), 8), desc="parity"):
        ids = IMAGE_IDS[start:start + 8]
        batch = normalize(torch.stack([load_image01(rid) for rid in ids]).to(DEVICE))
        with autocast_context(DEVICE, AMP_DTYPE):
            ours = torch.sigmoid(model(batch).float()).cpu().numpy()
        fp32 = torch.sigmoid(model(batch).float()).cpu().numpy() if start == 0 else None
        parity_diffs.append((ours - reference[start:start + len(ids)]).ravel())
        for i, rid in enumerate(ids):
            row = {"record_id": rid,
                   "max_abs_diff_vs_reference": float(np.abs(ours[i] - reference[start + i]).max())}
            if fp32 is not None:
                row["max_abs_diff_vs_fp32"] = float(np.abs(ours[i] - fp32[i]).max())
            parity_rows.append(row)
parity = pd.DataFrame(parity_rows)
save_csv(parity, "parity.csv")
all_diffs = np.abs(np.concatenate(parity_diffs))
worst, median = float(all_diffs.max()), float(np.median(all_diffs))
print(f"|p_ours - p_reference| over {all_diffs.size:,} (image, label) values: median {median:.2e}, "
      f"99th pct {np.percentile(all_diffs, 99):.2e}, max {worst:.2e}; "
      f"{AMP_DTYPE_NAME} vs fp32 on the first batch: max {parity['max_abs_diff_vs_fp32'].max():.2e}")
if median > PARITY_MEDIAN_TOLERANCE or worst > PARITY_MAX_TOLERANCE:
    raise RuntimeError(
        f"Probabilities differ from the reference run: median {median:.3g} (limit "
        f"{PARITY_MEDIAN_TOLERANCE}), max {worst:.3g} (limit {PARITY_MAX_TOLERANCE}). "
        "This is not the same model or not the same pixels; see parity.csv.")
"""),
    code(r"""# 5c. Batch size, throughput, and the projected run time.
probe_image = load_image01(IMAGE_IDS[0]).to(DEVICE)
probe_x = normalize(probe_image[None])
probe_a = activations(split, probe_x, AMP_DTYPE)
CHANNELS = probe_a.shape[1]
probe_rank = pixel_rank(upsample(probe_a[0].mean(0), INPUT_HW), 0)
probe_baseline = make_baseline(probe_image, BASELINE, PAPER_PAD_PX, BLUR_SIGMA_PX)

def try_batch(size):
    '''Run the real Score-CAM and curve code at this batch size once.'''
    scorecam_channel_scores(split, probe_x, probe_a[:, :size], size, AMP_DTYPE)
    deletion_insertion(model, probe_image, probe_baseline, probe_rank, 0, size - 1, size,
                       MEAN, STD, AMP_DTYPE)
    synchronize(DEVICE)

if BATCH_SIZE_OVERRIDE:
    BATCH_SIZE = BATCH_SIZE_OVERRIDE
    try_batch(BATCH_SIZE)
else:
    BATCH_SIZE = None
    for size in sorted(BATCH_CANDIDATES, reverse=True):
        try:
            try_batch(size)
            BATCH_SIZE = size
            break
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    if BATCH_SIZE is None:
        raise RuntimeError("Even batch size 1 does not fit on this device")

with torch.no_grad():
    batch = probe_x.expand(BATCH_SIZE, -1, -1, -1).contiguous(memory_format=torch.channels_last)
    for _ in range(2):
        with autocast_context(DEVICE, AMP_DTYPE):
            model(batch)
    repeats = 3
    synchronize(DEVICE)
    started = time.perf_counter()
    for _ in range(repeats):
        with autocast_context(DEVICE, AMP_DTYPE):
            model(batch)
    synchronize(DEVICE)
    IMAGES_PER_SECOND = BATCH_SIZE * repeats / (time.perf_counter() - started)

# torch.compile with one fixed batch shape (FixedBatch pads the last chunk), timed
# against eager on the same batch; the faster path runs every Score-CAM and
# curve forward. The gradient path and the timing in section 8 stay eager.
EAGER_IMAGES_PER_SECOND = IMAGES_PER_SECOND
FAST_FORWARD, COMPILED_IMAGES_PER_SECOND, COMPILE_SECONDS = model, None, None
if COMPILE == "1" or (COMPILE == "auto" and DEVICE.type == "cuda"):
    try:
        compiled_forward = FixedBatch(torch.compile(model, mode=COMPILE_MODE, dynamic=False), BATCH_SIZE)
        started = time.perf_counter()
        with torch.no_grad(), autocast_context(DEVICE, AMP_DTYPE):
            for _ in range(2):
                compiled_forward(batch)
            compiled_forward(batch[:1])          # the padded path, same shape
        synchronize(DEVICE)
        COMPILE_SECONDS = time.perf_counter() - started
        with torch.no_grad(), autocast_context(DEVICE, AMP_DTYPE):
            check_eager = torch.sigmoid(model(batch[:2]).float())
            check_compiled = torch.sigmoid(compiled_forward(batch[:2]).float())
            compile_gap = float((check_eager - check_compiled).abs().max())
            synchronize(DEVICE)
            started = time.perf_counter()
            for _ in range(repeats):
                compiled_forward(batch)
            synchronize(DEVICE)
        COMPILED_IMAGES_PER_SECOND = BATCH_SIZE * repeats / (time.perf_counter() - started)
        print(f"torch.compile ({COMPILE_MODE}): {COMPILE_SECONDS:.0f} s to compile, "
              f"{COMPILED_IMAGES_PER_SECOND:.1f} vs eager {EAGER_IMAGES_PER_SECOND:.1f} images/s, "
              f"max |p_compiled - p_eager| {compile_gap:.1e}")
        if compile_gap > COMPILE_TOLERANCE:
            print("compiled outputs differ from eager beyond CS_COMPILE_TOLERANCE: staying eager")
        elif COMPILED_IMAGES_PER_SECOND > 1.03 * EAGER_IMAGES_PER_SECOND:
            FAST_FORWARD, IMAGES_PER_SECOND = compiled_forward, COMPILED_IMAGES_PER_SECOND
    except Exception as exc:                     # a compile failure must not stop the run
        print(f"torch.compile unavailable here ({type(exc).__name__}: {exc}); staying eager")
print("forward path for Score-CAM and curves:",
      "compiled" if FAST_FORWARD is not model else "eager")

n_curve_forwards = len(pairs) * 4 * 2 * (CURVE_STEPS + 1)
n_scorecam_forwards = len(IMAGE_IDS) * (CHANNELS + 1)
n_forwards = n_curve_forwards + n_scorecam_forwards
projected_hours = n_forwards / IMAGES_PER_SECOND / 3600
benchmark = {"device": DEVICE_NAME, "precision": AMP_DTYPE_NAME, "batch_size": BATCH_SIZE,
             "images_per_second": round(IMAGES_PER_SECOND, 2),
             "eager_images_per_second": round(EAGER_IMAGES_PER_SECOND, 2),
             "compiled_images_per_second": COMPILED_IMAGES_PER_SECOND and round(COMPILED_IMAGES_PER_SECOND, 2),
             "compile_seconds": COMPILE_SECONDS and round(COMPILE_SECONDS, 1),
             "forward_path": "compiled" if FAST_FORWARD is not model else "eager",
             "channels": CHANNELS,
             "scorecam_forwards": n_scorecam_forwards, "curve_forwards": n_curve_forwards,
             "projected_hours_full_run": round(projected_hours, 2)}
save_json(benchmark, f"benchmark_{DEVICE_SLUG}.json")
print(f"batch {BATCH_SIZE}: {IMAGES_PER_SECOND:.1f} images/s at {INPUT_HW[0]}x{INPUT_HW[1]} ({AMP_DTYPE_NAME})")
print(f"forwards: Score-CAM {n_scorecam_forwards:,} + curves {n_curve_forwards:,} = {n_forwards:,}")
print(f"projected time for the whole evaluation on this GPU: {projected_hours:.1f} h "
      "(finished images are skipped on resume)")
del probe_image, probe_x, probe_a, probe_rank, probe_baseline, batch
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
"""),
    md(r"""## 6. Cross-check against `pytorch-grad-cam`

On the first `CS_LIBRARY_CHECK_PAIRS` pairs, our Grad-CAM and Grad-CAM++ maps are
computed in fp32 and compared with `pytorch_grad_cam.GradCAM` and
`GradCAMPlusPlus` (v1.5.7) on the same layer. Both are min-max scaled and compared
at input resolution. The expectation is Pearson r ≈ 1.

Score-CAM is not compared. The library weights channels by a softmax over
channels; this notebook follows the authors' reference code and weights by the
class score. The difference is documented in `xai_cam.py`.
"""),
    code(r"""# 6. Library cross-check (Grad-CAM, Grad-CAM++).
library_rows = []
if LIBRARY_CHECK_PAIRS > 0:
    try:
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as exc:
        GradCAM = None
        warnings.warn(f"pytorch-grad-cam is not importable ({exc}); skipping the cross-check")
    if GradCAM is not None:
        library_classes = {"gradcam": GradCAM, "gradcampp": GradCAMPlusPlus}
        for _, pair in pairs.head(LIBRARY_CHECK_PAIRS).iterrows():
            k = int(pair["label_index"])
            x = normalize(load_image01(pair["record_id"]).to(DEVICE)[None])
            ours, _ = gradient_cams(split, activations(split, x, None), [k])
            for method, library_class in library_classes.items():
                with library_class(model=model, target_layers=[model.stages[-1]]) as library_cam:
                    theirs = library_cam(input_tensor=x.clone().requires_grad_(True),
                                         targets=[ClassifierOutputTarget(k)])[0]
                mine = upsample(ours[method][k], INPUT_HW).cpu().numpy()
                mine = (mine - mine.min()) / (mine.max() - mine.min() + 1e-7)
                library_rows.append({
                    "pair_id": pair["pair_id"], "method": method,
                    "pearson_r": float(np.corrcoef(mine.ravel(), theirs.ravel())[0, 1]),
                    "max_abs_diff": float(np.abs(mine - theirs).max())})
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        library_check = pd.DataFrame(library_rows)
        save_csv(library_check, "library_check.csv")
        print(library_check.to_string(index=False))
        if library_check["pearson_r"].min() < 0.999:
            warnings.warn("A map differs from pytorch-grad-cam (r < 0.999): investigate before reporting.")
"""),
    md(r"""## 7. Deletion and insertion curves for every pair

Per image, one fp32-head backward per label gives Grad-CAM and Grad-CAM++. One
Score-CAM pass (1,024 masked forwards) scores every label of the image at once.

Each method's map, and a seeded random 24 × 32 map, is then scored with a
deletion curve and an insertion curve of `CS_CURVE_STEPS + 1` points.

- Rows are appended to `per_pair_metrics.csv`.
- Maps (24 × 32) and curves go to `maps/` and `curves/`.
- After each image, everything is copied to Drive.
- On resume, an image with all its rows already on Drive is skipped. A
  half-finished image is recomputed.
"""),
    code(r"""# 7. Main loop.
EVAL_METHODS = ("gradcam", "gradcampp", "scorecam", "random")
METRICS_CSV = RUN_DIR / "per_pair_metrics.csv"
restore_from_drive("per_pair_metrics.csv")
metrics = pd.read_csv(METRICS_CSV) if METRICS_CSV.is_file() else pd.DataFrame()
expected_rows = {(p, m) for p in pairs["pair_id"] for m in EVAL_METHODS}
done_rows = set(zip(metrics["pair_id"], metrics["method"])) if len(metrics) else set()
pairs_by_image = {rid: group for rid, group in pairs.groupby("record_id", sort=False)}

def image_done(record_id):
    return all((p, m) in done_rows for p in pairs_by_image[record_id]["pair_id"] for m in EVAL_METHODS)

todo = [rid for rid in IMAGE_IDS if not image_done(rid)]
if len(metrics):
    # A half-finished image is recomputed from scratch: drop its partial rows.
    metrics = metrics[~metrics["record_id"].isin(todo)]
    metrics.to_csv(METRICS_CSV, index=False)
print(f"{len(IMAGE_IDS) - len(todo)} of {len(IMAGE_IDS)} images already done; {len(todo)} to go")

pad_spread_rows = []
for record_id in tqdm(todo, desc="images"):
    image_pairs = pairs_by_image[record_id]
    image01 = load_image01(record_id).to(DEVICE)
    baseline01 = make_baseline(image01, BASELINE, PAPER_PAD_PX, BLUR_SIGMA_PX)
    _, pad_spread = paper_colour(image01, PAPER_PAD_PX)
    x = normalize(image01[None])
    label_indices = [int(k) for k in image_pairs["label_index"]]

    a = activations(split, x, AMP_DTYPE)
    gradient_maps, _ = gradient_cams(split, a, label_indices)
    channel_scores, scorecam_forwards = scorecam_channel_scores(split, x, a, BATCH_SIZE, AMP_DTYPE,
                                                                forward=FAST_FORWARD)
    score_maps = scorecam_maps(a, channel_scores, label_indices)

    rows, map_arrays, curve_arrays = [], {}, {}
    for _, pair in image_pairs.iterrows():
        k, label, pair_seed = int(pair["label_index"]), pair["label"], stable_seed(PAIR_SEED, pair["pair_id"])
        method_maps = {
            "gradcam": gradient_maps["gradcam"][k],
            "gradcampp": gradient_maps["gradcampp"][k],
            "scorecam": score_maps[k],
            "random": random_map(tuple(a.shape[-2:]), pair_seed, DEVICE),
        }
        for method, cam in method_maps.items():
            rank = pixel_rank(upsample(cam, INPUT_HW), pair_seed)
            fractions, deletion, insertion = deletion_insertion(
                FAST_FORWARD, image01, baseline01, rank, k, CURVE_STEPS, BATCH_SIZE, MEAN, STD, AMP_DTYPE)
            rows.append({
                "pair_id": pair["pair_id"], "record_id": record_id, "label": label, "method": method,
                "deletion_auc": curve_auc(fractions, deletion),
                "insertion_auc": curve_auc(fractions, insertion),
                "p_full": float(deletion[0]), "p_baseline": float(insertion[0]),
                "p_reference": float(pair["prob_reference"]),
                "map_all_zero": bool(float(cam.max()) <= 0.0),
                "scorecam_forwards": scorecam_forwards if method == "scorecam" else 0,
                "paper_pad_spread": pad_spread,
            })
            map_arrays[f"{label}|{method}"] = cam.float().cpu().numpy()
            curve_arrays[f"{label}|{method}|deletion"] = deletion
            curve_arrays[f"{label}|{method}|insertion"] = insertion
    curve_arrays["fractions"] = fractions

    slug = image_slug(record_id)
    np.savez_compressed(RUN_DIR / "maps" / f"{slug}.npz", **map_arrays)
    np.savez_compressed(RUN_DIR / "curves" / f"{slug}.npz", **curve_arrays)
    image_rows = pd.DataFrame(rows)
    image_rows.to_csv(METRICS_CSV, mode="a", header=not METRICS_CSV.is_file(), index=False)
    for path in (RUN_DIR / "maps" / f"{slug}.npz", RUN_DIR / "curves" / f"{slug}.npz", METRICS_CSV):
        drive_copy(path)
    done_rows.update(zip(image_rows["pair_id"], image_rows["method"]))
    pad_spread_rows.append(pad_spread)

metrics = pd.read_csv(METRICS_CSV)
missing = expected_rows - set(zip(metrics["pair_id"], metrics["method"]))
if missing:
    raise RuntimeError(f"{len(missing)} (pair, method) rows are missing after the loop")
if pad_spread_rows and max(pad_spread_rows) > 0.05:
    warnings.warn(f"Some sheets' {PAPER_PAD_PX}-px side pad is not plain paper (std up to "
                  f"{max(pad_spread_rows):.3f}): check CS_PAPER_PAD_PX against the render.")
print(f"{len(metrics)} rows ({len(pairs)} pairs x {len(EVAL_METHODS)} methods) in {METRICS_CSV.name}")
"""),
    md(r"""## 8. Explanation Time

- **What is timed.** One map for one (image, label), computed from scratch the
  way a deployed system would: the input tensor already on the device, through
  to the map upsampled to 768 × 1024.
- **What is not timed.** Image decoding, model loading, and the evaluation
  curves.
- **Order.** Each method is warmed up first (`CS_TIMING_WARMUP` calls), then run
  on the same `CS_TIMING_PAIRS` pairs, one call at a time, synchronized on CUDA.
- **Forward passes.** The count is reported with each time because, unlike the
  seconds, it does not depend on the hardware: about 2 for the gradient methods,
  1 + 1,024 for Score-CAM.

Two devices are timed:

- **8a. This GPU,** with Score-CAM at the batch size from section 5c.
- **8b. This runtime's CPU,** in fp32 like the app. Grad-CAM and Grad-CAM++ are
  measured directly. Score-CAM is extrapolated by default: the activation pass
  plus (number of channels × measured seconds per masked forward). A full CPU
  Score-CAM map takes hours; `CS_CPU_SCORECAM_FULL=1` runs one anyway.

The deployment laptop is not a Colab CPU. Time it there with `cpu_timing.py`
(see the README).
"""),
    code(r"""# 8a. Explanation Time on this GPU (or CPU, if no GPU).
def timing_pairs(n):
    '''n pairs spread round-robin over the labels, in the seeded pair order.'''
    by_label = [group for _, group in pairs.groupby("label", sort=True)]
    chosen = []
    for position in range(max(len(g) for g in by_label)):
        for group in by_label:
            if position < len(group) and len(chosen) < n:
                chosen.append(group.iloc[position])
    return pd.DataFrame(chosen)

def time_methods(device_model, device, amp_dtype, timing_set, warmup, score_batch, note):
    device_split = ModelSplit(device_model)
    inputs = {rid: ((load_image01(rid).to(device)[None] - MEAN.to(device)) / STD.to(device))
              .contiguous(memory_format=torch.channels_last) for rid in timing_set["record_id"].unique()}
    rows = []
    for method in METHODS:
        first = timing_set.iloc[0]
        for _ in range(warmup):
            explain(device_split, inputs[first["record_id"]], int(first["label_index"]), method,
                    amp_dtype, score_batch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _, pair in timing_set.iterrows():
            seconds, (_, forwards) = timed(
                lambda: explain(device_split, inputs[pair["record_id"]], int(pair["label_index"]),
                                method, amp_dtype, score_batch), device)
            rows.append({"device": note, "method": method, "pair_id": pair["pair_id"],
                         "seconds": seconds, "forwards": forwards, "measured": "measured",
                         "batch_size": score_batch if method == "scorecam" else 1})
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            for row in rows[-len(timing_set):]:
                row["peak_gib"] = round(peak, 2)
    return pd.DataFrame(rows)

TIMING_SET = timing_pairs(TIMING_PAIRS)
gpu_timing = time_methods(model, DEVICE, AMP_DTYPE, TIMING_SET, TIMING_WARMUP, BATCH_SIZE,
                          f"{DEVICE_NAME} ({AMP_DTYPE_NAME})")
save_csv(gpu_timing, f"timing/timing_{DEVICE_SLUG}.csv")
print(gpu_timing.groupby("method")[["seconds", "forwards"]].median().round(4).to_string())
"""),
    code(r"""# 8b. Explanation Time on this runtime's CPU (fp32, like the app).
def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()

cpu_timing = pd.DataFrame()
if CPU_TIMING:
    cpu = torch.device("cpu")
    threads = CPU_THREADS or os.cpu_count()
    torch.set_num_threads(threads)
    cpu_note = f"{cpu_name()} ({threads} threads, fp32)"
    cpu_model = copy.deepcopy(model).to(cpu).float().eval()
    cpu_set = TIMING_SET.head(CPU_TIMING_PAIRS)
    cpu_split = ModelSplit(cpu_model)
    cpu_mean, cpu_std = MEAN.cpu(), STD.cpu()
    rows = []
    for method in GRADIENT_METHODS:
        first = cpu_set.iloc[0]
        x0 = ((load_image01(first["record_id"])[None] - cpu_mean) / cpu_std).contiguous(memory_format=torch.channels_last)
        explain(cpu_split, x0, int(first["label_index"]), method, None)        # warm-up
        for _, pair in tqdm(list(cpu_set.iterrows()), desc=f"CPU {method}"):
            x_cpu = ((load_image01(pair["record_id"])[None] - cpu_mean) / cpu_std).contiguous(
                memory_format=torch.channels_last)
            seconds, (_, forwards) = timed(
                lambda: explain(cpu_split, x_cpu, int(pair["label_index"]), method, None), cpu)
            rows.append({"device": cpu_note, "method": method, "pair_id": pair["pair_id"],
                         "seconds": seconds, "forwards": forwards, "measured": "measured", "batch_size": 1})
    for _, pair in cpu_set.head(1 if CPU_SCORECAM_FULL else len(cpu_set)).iterrows():
        x_cpu = ((load_image01(pair["record_id"])[None] - cpu_mean) / cpu_std).contiguous(
            memory_format=torch.channels_last)
        if CPU_SCORECAM_FULL:
            seconds, (_, forwards) = timed(
                lambda: explain(cpu_split, x_cpu, int(pair["label_index"]), "scorecam", None,
                                CPU_SCORECAM_BATCH), cpu)
            rows.append({"device": cpu_note, "method": "scorecam", "pair_id": pair["pair_id"],
                         "seconds": seconds, "forwards": forwards, "measured": "measured",
                         "batch_size": CPU_SCORECAM_BATCH})
            continue
        t_activations, a_cpu = timed(lambda: activations(cpu_split, x_cpu, None), cpu)
        probe = min(CPU_SCORECAM_PROBE_CHANNELS, a_cpu.shape[1])
        t_probe, (_, probe_forwards) = timed(
            lambda: scorecam_channel_scores(cpu_split, x_cpu, a_cpu[:, :probe], CPU_SCORECAM_BATCH, None), cpu)
        flat = a_cpu[0].flatten(1)
        valid_channels = int(((flat.max(dim=1).values - flat.min(dim=1).values) > 0).sum())
        per_forward = t_probe / max(probe_forwards, 1)
        rows.append({"device": cpu_note, "method": "scorecam", "pair_id": pair["pair_id"],
                     "seconds": t_activations + valid_channels * per_forward,
                     "forwards": 1 + valid_channels,
                     "measured": f"extrapolated from {probe_forwards} masked forwards",
                     "batch_size": CPU_SCORECAM_BATCH})
    cpu_timing = pd.DataFrame(rows)
    save_csv(cpu_timing, f"timing/timing_cpu_{re.sub(r'[^a-z0-9]+', '_', cpu_name().lower()).strip('_')}.csv")
    print(cpu_timing.groupby("method")[["seconds", "forwards"]].median().round(3).to_string())
    del cpu_model, cpu_split
    gc.collect()
else:
    print("CS_CPU_TIMING=0: skipped")
"""),
    md(r"""## 9. Results

- **Mean AUC with 95% CIs.** Each method's mean Deletion AUC and Insertion AUC
  over all pairs comes with a 95% bootstrap CI. Each of the `CS_BOOTSTRAP_ROUNDS`
  rounds resamples **images**, not rows: one image can be several pairs, and
  those pairs are not independent.
- **Paired differences.** For each method vs Grad-CAM, the difference is computed
  on the same pairs. Each row has:
  - the mean difference and its image-bootstrap CI
  - the share of pairs where the method beats Grad-CAM
  - a Wilcoxon signed-rank test over per-image mean differences
- **Per label.** `per_label.csv` splits the means by label, because rhythm
  labels (AF, SB, STACH) are whole-sheet findings while STEMI/STE/LBBB are local.
- **Explanation Time.** Reported as median and IQR, with the forward-pass count.
"""),
    code(r"""# 9. Statistics and tables.
from scipy.stats import wilcoxon

metrics = pd.read_csv(METRICS_CSV)
metrics["ins_minus_del"] = metrics["insertion_auc"] - metrics["deletion_auc"]
SCORES = {"deletion_auc": "lower", "insertion_auc": "higher", "ins_minus_del": "higher"}
image_ids = np.array(sorted(metrics["record_id"].unique()))
image_position = {rid: i for i, rid in enumerate(image_ids)}
rng = np.random.default_rng(BOOTSTRAP_SEED)
draws = rng.integers(0, len(image_ids), size=(BOOTSTRAP_ROUNDS, len(image_ids)))
weights = np.stack([np.bincount(d, minlength=len(image_ids)) for d in draws])   # (rounds, images)

def image_bootstrap(frame, column):
    '''Mean of `column` over rows, and its 95% CI resampling images.'''
    positions = frame["record_id"].map(image_position).to_numpy()
    sums = np.bincount(positions, weights=frame[column].to_numpy(), minlength=len(image_ids))
    counts = np.bincount(positions, minlength=len(image_ids)).astype(float)
    means = (weights @ sums) / np.maximum(weights @ counts, 1e-12)
    return float(frame[column].mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

summary_rows = []
for method in EVAL_METHODS:
    frame = metrics[metrics["method"] == method]
    row = {"method": METHOD_LABELS[method], "pairs": len(frame), "images": frame["record_id"].nunique()}
    for column in SCORES:
        mean, low, high = image_bootstrap(frame, column)
        row.update({column: mean, f"{column}_ci_low": low, f"{column}_ci_high": high})
    row["maps_all_zero"] = int(frame["map_all_zero"].sum())
    summary_rows.append(row)
summary = pd.DataFrame(summary_rows)

wide = metrics.pivot_table(index=["pair_id", "record_id", "label"], columns="method",
                           values=list(SCORES)).reset_index()
paired_rows = []
for method in ("gradcampp", "scorecam", "random"):
    for column, better in SCORES.items():
        diff = pd.DataFrame({"record_id": wide["record_id"],
                             "diff": wide[(column, method)] - wide[(column, "gradcam")]})
        mean, low, high = image_bootstrap(diff, "diff")
        per_image = diff.groupby("record_id")["diff"].mean()
        nonzero = per_image[per_image != 0]
        p_value = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 10 else float("nan")
        wins = (diff["diff"] < 0) if better == "lower" else (diff["diff"] > 0)
        paired_rows.append({
            "comparison": f"{METHOD_LABELS[method]} - Grad-CAM", "metric": column, "better": better,
            "mean_diff": mean, "ci_low": low, "ci_high": high,
            "share_of_pairs_where_method_beats_gradcam": float(wins.mean()),
            "wilcoxon_p_per_image": p_value, "images": len(per_image)})
paired = pd.DataFrame(paired_rows)

per_label = (metrics.assign(method=metrics["method"].map(METHOD_LABELS))
             .pivot_table(index="label", columns="method", values=["deletion_auc", "insertion_auc"],
                          aggfunc="mean").round(4))
per_label.columns = [f"{metric}|{method}" for metric, method in per_label.columns]
per_label = per_label.reset_index()

timing_frames = [f for f in (gpu_timing, cpu_timing) if len(f)]
timing_all = pd.concat(timing_frames, ignore_index=True)
timing_summary = (timing_all.groupby(["device", "method", "measured"], sort=False)
                  .agg(n=("seconds", "size"), median_s=("seconds", "median"),
                       q1_s=("seconds", lambda s: s.quantile(0.25)),
                       q3_s=("seconds", lambda s: s.quantile(0.75)),
                       forwards=("forwards", "median"))
                  .reset_index())
timing_summary["method"] = timing_summary["method"].map(METHOD_LABELS)
gradcam_median = timing_summary[timing_summary["method"] == "Grad-CAM"].set_index("device")["median_s"]
timing_summary["x_gradcam"] = [row.median_s / gradcam_median.get(row.device, np.nan)
                               for row in timing_summary.itertuples()]

save_csv(summary, "summary.csv")
save_csv(paired, "paired_vs_gradcam.csv")
save_csv(per_label, "per_label.csv")
save_csv(timing_summary, "timing_summary.csv")

def fmt_ci(row, column, digits=3):
    return (f"{row[column]:.{digits}f} [{row[f'{column}_ci_low']:.{digits}f}, "
            f"{row[f'{column}_ci_high']:.{digits}f}]")

lines = [f"# CAM method comparison — {REFERENCE_RUN}", "",
         f"{len(pairs)} true-positive test pairs over {len(image_ids)} images and "
         f"{len(EXPLAINED_LABELS)} labels; baseline `{BASELINE}`, {CURVE_STEPS} steps, "
         f"{AMP_DTYPE_NAME}; 95% CIs from {BOOTSTRAP_ROUNDS} image-bootstrap rounds.", "",
         "| Method | Deletion AUC ↓ | Insertion AUC ↑ | Insertion − Deletion ↑ | all-zero maps |",
         "|---|---|---|---|---|"]
for _, row in summary.iterrows():
    lines.append(f"| {row['method']} | {fmt_ci(row, 'deletion_auc')} | {fmt_ci(row, 'insertion_auc')} | "
                 f"{fmt_ci(row, 'ins_minus_del')} | {row['maps_all_zero']} |")
lines += ["", "## Paired differences vs Grad-CAM (same pairs)", "",
          "| Comparison | Metric | Mean diff [95% CI] | Beats Grad-CAM | Wilcoxon p (per image) |",
          "|---|---|---|---|---|"]
for _, row in paired.iterrows():
    lines.append(f"| {row['comparison']} | {row['metric']} ({row['better']} better) | "
                 f"{row['mean_diff']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
                 f"{row['share_of_pairs_where_method_beats_gradcam']:.0%} | {row['wilcoxon_p_per_image']:.3g} |")
lines += ["", "## Explanation Time (one map, one label)", "",
          "| Device | Method | n | Median s [IQR] | × Grad-CAM | Forward passes | Measured |",
          "|---|---|---|---|---|---|---|"]
for _, row in timing_summary.iterrows():
    lines.append(f"| {row['device']} | {row['method']} | {row['n']} | {row['median_s']:.3f} "
                 f"[{row['q1_s']:.3f}–{row['q3_s']:.3f}] | {row['x_gradcam']:.1f}× | "
                 f"{row['forwards']:.0f} | {row['measured']} |")
(RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n")
drive_copy(RUN_DIR / "summary.md")
print("\n".join(lines))
"""),
    code(r"""# 9b. Figures: mean curves, and example maps.
import matplotlib
import matplotlib.pyplot as plt

# Categorical slots 1-3 (validated palette) for the three methods; the random
# floor is a neutral reference line, not a fourth category.
COLOURS = {"gradcam": "#2a78d6", "gradcampp": "#eb6834", "scorecam": "#1baf7a", "random": "#8a8984"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "font.size": 10})

curves = {(m, mode): [] for m in EVAL_METHODS for mode in ("deletion", "insertion")}
for record_id in image_ids:
    with np.load(run_file(f"curves/{image_slug(record_id)}.npz")) as data:
        fractions = data["fractions"]
        for label in pairs_by_image[record_id]["label"]:
            for m in EVAL_METHODS:
                for mode in ("deletion", "insertion"):
                    curves[(m, mode)].append(data[f"{label}|{m}|{mode}"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for axis, mode in zip(axes, ("deletion", "insertion")):
    for m in EVAL_METHODS:
        mean_curve = np.mean(curves[(m, mode)], axis=0)
        auc = summary.set_index("method").loc[METHOD_LABELS[m], f"{mode}_auc"]
        axis.plot(fractions, mean_curve, color=COLOURS[m], linewidth=2,
                  linestyle="--" if m == "random" else "-", label=f"{METHOD_LABELS[m]} (AUC {auc:.3f})")
    axis.set_title(f"{mode.capitalize()} ({'lower' if mode == 'deletion' else 'higher'} AUC is better)",
                   loc="left", fontsize=11)
    axis.set_xlabel(f"fraction of pixels {'removed' if mode == 'deletion' else 'inserted'}")
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(frameon=False, fontsize=9)
axes[0].set_ylabel("mean p(label)")
fig.suptitle(f"Mean deletion and insertion curves, {len(pairs)} test pairs", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(RUN_DIR / "figures" / "mean_curves.png", dpi=160)
drive_copy(RUN_DIR / "figures" / "mean_curves.png")
plt.show()

example_pairs = pairs.groupby("label", sort=False).head(1).head(EXAMPLE_LABELS)
fig, axes = plt.subplots(len(example_pairs), 4, figsize=(16, 3.1 * len(example_pairs)))
axes = np.atleast_2d(axes)
for row_axes, (_, pair) in zip(axes, example_pairs.iterrows()):
    image = load_image01(pair["record_id"]).permute(1, 2, 0).numpy()
    with np.load(run_file(f"maps/{image_slug(pair['record_id'])}.npz")) as data:
        maps = {m: data[f"{pair['label']}|{m}"] for m in METHODS}
    row_axes[0].imshow(image)
    row_axes[0].set_title(f"{pair['label']}  p={pair['prob_reference']:.2f}", loc="left", fontsize=10)
    for axis, m in zip(row_axes[1:], METHODS):
        saliency = upsample(torch.from_numpy(maps[m]), INPUT_HW).numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-12)
        axis.imshow(image)
        axis.imshow(saliency, cmap="inferno", alpha=0.5)
        row = metrics[(metrics["pair_id"] == pair["pair_id"]) & (metrics["method"] == m)].iloc[0]
        axis.set_title(f"{METHOD_LABELS[m]}  del {row['deletion_auc']:.3f} / ins {row['insertion_auc']:.3f}",
                       loc="left", fontsize=10)
    for axis in row_axes:
        axis.set_axis_off()
fig.tight_layout()
fig.savefig(RUN_DIR / "figures" / "example_maps.png", dpi=110)
drive_copy(RUN_DIR / "figures" / "example_maps.png")
plt.show()
"""),
    md(r"""## 10. Flush Drive and release the runtime

The results are already on Drive. This cell flushes the mount and then **releases
the Colab runtime**, so it stops using compute units. Run all reaches it only if
every cell above succeeded. Set `CS_RELEASE_RUNTIME=0` to keep the runtime.
"""),
    code(r"""# 10. Flush Google Drive, then release the Colab runtime.
if IN_COLAB and DRIVE_SYNC_ENABLED:
    from google.colab import drive
    drive.flush_and_unmount()
    print("Google Drive flushed; results are in", DRIVE_RUN_DIR)
    if RELEASE_RUNTIME_WHEN_DONE:
        from google.colab import runtime
        print("releasing the Colab runtime ...")
        runtime.unassign()
else:
    print("Not on Colab (or Drive sync off): nothing to flush or release. Results:", RUN_DIR)
"""),
]


for index, cell in enumerate(cells):
    cell["id"] = f"cs-xai-cam-c12-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "machine_shape": "hm"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1))
print(f"wrote {OUTPUT} ({len(cells)} cells)")
