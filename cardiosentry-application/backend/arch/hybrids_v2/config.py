"""
CardioSentry baselines v2 — controlled experimental conditions.

EVERYTHING IN THIS FILE IS HELD CONSTANT ACROSS ALL ARCHITECTURES.

Same contract as cardiosentry_runs_v1: if two runs differ, the difference must
be attributable to the architecture and nothing else. `fingerprint()` hashes the
controlled settings so `compare.py` can PROVE two runs shared conditions rather
than assuming it.

WHAT CHANGED FROM v1 / v1_768
-----------------------------
1. NINE architectures instead of five (see models.py).
2. The input is a RECTANGLE, selected by name from dimensions.DIMENSIONS and
   set through the CS_DIM environment variable, because the aspect ratio is now
   an experimental variable rather than a fixed 384 or 768 square.
3. Two fingerprints, not one:
       fingerprint()           - everything, including the geometry
       protocol_fingerprint()  - everything EXCEPT the geometry
   compare.py needs both. Ranking nine architectures requires identical
   fingerprints. Ranking one architecture across seven geometries requires
   identical PROTOCOL fingerprints and deliberately different ones. Without the
   split you cannot do the second comparison without disabling the check that
   makes the first one trustworthy.

Everything else - the metrics, the augmentation policy, the optimiser, the
threshold rule, the per-source reporting - is byte-identical to v1 so the two
result sets sit in the same table.
"""
from __future__ import annotations
import hashlib, json, os

from . import dimensions as DIM   # vendored: was `import dimensions as DIM` (arch/README.md)

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- task definition
# Four INDEPENDENT binary heads. Not mutually exclusive: AF+LVH co-occurs on 256
# records. This is why the loss is BCEWithLogits and never CrossEntropy.
LABELS = ["STEMI", "AF", "LVH", "NORMAL"]
NUM_CLASSES = len(LABELS)

# ---------------------------------------------------------------- input geometry
#
# Selected by NAME, not by number, so a run records which geometry it used and
# compare.py can group by it:
#
#     CS_DIM=crop608 python train.py --model resnet50 --data ...
#     CS_DIM=sq768   python train.py --model resnet50 --data ...
#
# Read dimensions.py for what each name means and why these seven exist. The
# short version: v1 (384 square) and v1_768 (768 square) both squeezed the TIME
# axis by 22% and spent ~27% of the vertical budget on blank paper, and the
# printed record ID at the top of every sheet is a hard STEMI leak.
# VENDORED DEVIATION (arch/README.md): the research copy read this from
# the CS_DIM environment variable. Serving must not inherit whoever's
# shell it was started from — the geometry a checkpoint was trained at is
# a property of the checkpoint, and the descriptor's preprocess.input_size
# is what actually resizes the sheet. Pinned to the geometry the vendored
# checkpoints were trained at; a run at a different one needs its own
# vendored package, not an env var.
DIM_NAME = "nat768"
_DIM = DIM.resolve(DIM_NAME)

IMG_SIZE = tuple(_DIM["size"])        # (H, W) - torch order, NOT PIL order
IMG_H, IMG_W = IMG_SIZE
CROP_HEADER = bool(_DIM["crop"])      # cut above the printed ID before resizing
CROP_BOX_FRAC = DIM.CONTENT_CROP if CROP_HEADER else None

# Paint out the printed ID / Age / Sex block. ON by default and for every
# full-sheet geometry, because the record ID separates STEMI EXACTLY in this
# corpus (see dimensions.py). A crop* geometry has already thrown that band away,
# so redaction is skipped there rather than applied twice.
REDACT_HEADER = bool(_DIM.get("redact", True)) and not CROP_HEADER
REDACT_BOX_FRAC = DIM.REDACT_BOX if REDACT_HEADER else None
SOURCE_ASPECT = (DIM.SOURCE_W, DIM.SOURCE_H)

# ImageNet statistics: every backbone here ships ImageNet-pretrained weights.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------- augmentation
#
# DELIBERATELY MINIMAL. Unchanged from v1. Read this before adding anything.
#
# The corpus is ALREADY augmented. Stage 4 of the image pipeline baked in, per
# training variant: grid palette (red/orange/pink/grey), paper tilt, mild
# perspective, wrinkles and creases, lighting gradients, edge/corner shadow,
# defocus blur, sensor noise, JPEG artifacts and white-balance drift. Re-applying
# geometric or photometric augmentation here double-counts it and pushes images
# off the realistic manifold the corpus was built to sit on.
#
# THREE TRANSFORMS ARE FORBIDDEN ON ECG PAPER, and none of them are obvious:
#
#   * HORIZONTAL FLIP - the x axis is TIME. Flipping runs the heartbeat backwards:
#     the T wave precedes the QRS, and the lead columns (I/aVR/V1/V4 ...) land in
#     the wrong order. Label becomes meaningless.
#
#   * VERTICAL FLIP - the y axis is VOLTAGE, and its sign is diagnostic. Flipping
#     turns ST ELEVATION into ST DEPRESSION. It converts a STEMI into its clinical
#     opposite while keeping the STEMI label attached. This single line, present
#     by default in most image-classification recipes, would silently poison the
#     class you care most about.
#
#   * RANDOM RESIZED CROP - leads occupy fixed positions on the sheet. Cropping
#     amputates whole leads; a crop that removes V1-V3 removes the only place
#     anterior ST elevation is visible, again without touching the label.
#
# The header crop above is NOT one of these. It is a fixed, deterministic window
# applied identically to train, val and test, measured to clear every trace on
# every sheet in the corpus. It removes printed text, never waveform.
AUG = {
    "hflip": 0.0,               # FORBIDDEN - reverses time
    "vflip": 0.0,               # FORBIDDEN - inverts ST polarity
    "random_resized_crop": False,  # FORBIDDEN - amputates leads
    "rotation_deg": 0.0,        # already applied by corpus stage 4
    "brightness": 0.10,
    "contrast": 0.10,
    "saturation": 0.0,
    "hue": 0.0,
}

# ---------------------------------------------------------------- optimisation
# Identical for every architecture. Micro-batch varies with VRAM, but gradient
# accumulation keeps the EFFECTIVE batch (and therefore the optimisation
# trajectory) the same everywhere - see models.MICRO_BATCH.
EFFECTIVE_BATCH = 32
EPOCHS = 20
OPTIMIZER = "adamw"
LR = 1e-4
WEIGHT_DECAY = 0.05
BETAS = (0.9, 0.999)
SCHEDULE = "cosine"
WARMUP_EPOCHS = 2
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.0       # not meaningful for multi-label BCE
AMP = True                  # mixed precision; identical for all models

# Optional equal-budget LR search (--sweep). Every architecture gets the SAME
# three candidates and the SAME selection rule, which is the fair way to let each
# model reach its own optimum without hand-tuning one of them into the lead.
LR_GRID = [3e-5, 1e-4, 3e-4]

# ---------------------------------------------------------------- imbalance
# pos_weight is computed from the TRAIN split at runtime (manifest.pos_weight)
# rather than hard-coded, so it stays correct across prototype_500 / _800 / _1152.
USE_POS_WEIGHT = True

# ---------------------------------------------------------------- model selection
# Early stopping and "best checkpoint" both use this. Macro AUPRC is the right
# monitor for imbalanced multi-label: it is threshold-free (so it does not reward
# a lucky operating point) and, unlike AUROC, it does not flatter a model on the
# minority head.
MONITOR = "macro_auprc"
MONITOR_MODE = "max"
EARLY_STOP_PATIENCE = 5

# ---------------------------------------------------------------- evaluation
# Thresholds are tuned on VAL ONLY and then frozen and applied to TEST. Tuning
# on test would leak and inflate every number.
THRESHOLD_TUNING = "val_f1"
TEST_TIME_AUGMENTATION = False   # val/test hold exactly 1 canonical image per ECG

# Mandatory. The corpus carries a known source confound: all 1,442 STEMI records
# come from a single database, and stage 5's leakage probe identified source at
# 51.5% vs 37.2% chance. Headline metrics alone cannot distinguish "recognises
# STEMI" from "recognises the Chongqing scanner".
#
# v2 adds a second, sharper reason to read these numbers: the printed record ID
# separates STEMI from everything else exactly (see dimensions.py). On a
# full-sheet geometry the ID is IN THE IMAGE. On a crop* geometry it is not.
REPORT_PER_SOURCE = True

# ---------------------------------------------------------------- fingerprints
# Settings that must match for an ARCHITECTURE comparison to be fair.
_PROTOCOL = [
    "SEED", "LABELS", "NORM_MEAN", "NORM_STD", "AUG",
    "EFFECTIVE_BATCH", "EPOCHS", "OPTIMIZER", "LR", "WEIGHT_DECAY", "BETAS",
    "SCHEDULE", "WARMUP_EPOCHS", "GRAD_CLIP", "AMP", "USE_POS_WEIGHT",
    "MONITOR", "MONITOR_MODE", "EARLY_STOP_PATIENCE", "THRESHOLD_TUNING",
    "TEST_TIME_AUGMENTATION",
]
# ...plus the geometry, which must ALSO match for an architecture comparison but
# must deliberately DIFFER for a geometry comparison. Hence two hashes.
_GEOMETRY = ["DIM_NAME", "IMG_SIZE", "CROP_HEADER", "CROP_BOX_FRAC",
             "REDACT_HEADER", "REDACT_BOX_FRAC"]

_CONTROLLED = _PROTOCOL + _GEOMETRY


def controlled_settings() -> dict:
    g = globals()
    return {k: g[k] for k in _CONTROLLED}


def protocol_settings() -> dict:
    g = globals()
    return {k: g[k] for k in _PROTOCOL}


def _hash(d: dict) -> str:
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def fingerprint() -> str:
    """Stable hash of protocol AND geometry. Two runs that disagree here are not
    comparable as an architecture ranking, and compare.py will say so instead of
    quietly tabulating them."""
    return _hash(controlled_settings())


def protocol_fingerprint() -> str:
    """Stable hash of the protocol ALONE. Two runs that agree here but differ in
    `fingerprint` differ only in input geometry - which is exactly the resolution
    study, and `compare.py --across-dims` will tabulate those."""
    return _hash(protocol_settings())


def geometry_str() -> str:
    return (f"{DIM_NAME}  {IMG_H}x{IMG_W}"
            f"{'  (header cropped)' if CROP_HEADER else '  (full sheet)'}"
            f"{'  + ID redacted' if REDACT_HEADER else ''}")


if __name__ == "__main__":
    print(json.dumps(controlled_settings(), indent=2, default=str))
    print("\ngeometry:             ", geometry_str())
    print("fingerprint:          ", fingerprint())
    print("protocol_fingerprint: ", protocol_fingerprint())
