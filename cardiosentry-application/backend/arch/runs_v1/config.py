"""
CardioSentry — controlled experimental conditions.

EVERYTHING IN THIS FILE IS HELD CONSTANT ACROSS ALL ARCHITECTURES.

That is the entire point of the harness: if two runs differ, the difference must
be attributable to the architecture and nothing else. `fingerprint()` hashes the
controlled settings so `compare.py` can PROVE two runs shared conditions rather
than assuming it.

Change something here and every previously-finished run becomes non-comparable.
The fingerprint will change and compare.py will refuse to tabulate them together.
"""
from __future__ import annotations
import hashlib, json

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- task definition
# Four INDEPENDENT binary heads. Not mutually exclusive: AF+LVH co-occurs on 256
# records. This is why the loss is BCEWithLogits and never CrossEntropy.
LABELS = ["STEMI", "AF", "LVH", "NORMAL"]
NUM_CLASSES = len(LABELS)

# ---------------------------------------------------------------- input pipeline
# 384 square. Chosen because ViT-B/16-384 has fixed 384 position embeddings, and
# holding resolution constant matters more for a controlled comparison than
# giving each backbone its favourite input size.
#
# KNOWN LIMITATION, state it in the writeup: the source sheets are 1686x1311
# (~1.29:1). Squaring them compresses the time axis more than the voltage axis.
# The distortion is identical for every image and every model, so it does not
# bias the comparison - but it does mean absolute scores are not the ceiling.
# For the final full-corpus run, consider a non-square input.
IMG_SIZE = 384
SOURCE_ASPECT = (1686, 1311)

# ImageNet statistics: all five backbones ship ImageNet-pretrained weights.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------- augmentation
#
# DELIBERATELY MINIMAL. Read this before adding anything.
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
# What is left is a small brightness/contrast jitter, which is safe because it
# does not move any pixel and does not change any sign.
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
# pos_weight is computed from the TRAIN split at runtime (data.pos_weight) rather
# than hard-coded, so it stays correct across prototype_500 / _800 / _1152.
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
REPORT_PER_SOURCE = True

# ---------------------------------------------------------------- fingerprint
# Only settings that must match for a comparison to be fair.
_CONTROLLED = [
    "SEED", "LABELS", "IMG_SIZE", "NORM_MEAN", "NORM_STD", "AUG",
    "EFFECTIVE_BATCH", "EPOCHS", "OPTIMIZER", "LR", "WEIGHT_DECAY", "BETAS",
    "SCHEDULE", "WARMUP_EPOCHS", "GRAD_CLIP", "AMP", "USE_POS_WEIGHT",
    "MONITOR", "MONITOR_MODE", "EARLY_STOP_PATIENCE", "THRESHOLD_TUNING",
    "TEST_TIME_AUGMENTATION",
]


def controlled_settings() -> dict:
    g = globals()
    return {k: g[k] for k in _CONTROLLED}


def fingerprint() -> str:
    """Stable hash of the controlled conditions. Runs that disagree here are not
    comparable, and compare.py will say so instead of quietly tabulating them."""
    blob = json.dumps(controlled_settings(), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


if __name__ == "__main__":
    print(json.dumps(controlled_settings(), indent=2, default=str))
    print("\nfingerprint:", fingerprint())
