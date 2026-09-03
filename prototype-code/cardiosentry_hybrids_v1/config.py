"""
CardioSentry hybrids — controlled experimental conditions, round 2.

WHAT CHANGED FROM cardiosentry_runs_v1, AND WHY THE FINGERPRINT IS NOW TWO-TIER

Round 1 held EVERYTHING constant, including input resolution, and hashed the lot
into one fingerprint. That was correct when the question was "which backbone".

The question now is different: **is LVH limited by input resolution?** Resolution
has therefore been PROMOTED from a controlled constant to the experimental
variable. A single fingerprint can no longer express that — it would refuse to
compare a 384 run against a 768 run, which is the entire comparison.

So there are two hashes:

  PROTOCOL fingerprint  — seed, labels, normalisation, augmentation, optimiser,
                          effective batch, epochs, monitor, early stopping,
                          threshold rule. Every run MUST match this or the
                          comparison is invalid, and compare.py enforces it.

  INPUT fingerprint     — the per-run view geometry (which resolutions the model
                          sees). Runs are EXPECTED to differ here; compare.py
                          prints it as a column instead of refusing.

Everything else is byte-identical to round 1, deliberately: same seed, same
AdamW at 1e-4, same cosine schedule, same 20 epochs, same macro-AUPRC monitor,
same val-tuned frozen thresholds. Round-1 numbers therefore remain a legitimate
reference point for anything trained at 384.
"""
from __future__ import annotations
import hashlib, json

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- task definition
LABELS = ["STEMI", "AF", "LVH", "NORMAL"]
NUM_CLASSES = len(LABELS)

# The head this round exists to improve. Used for the LVH-focused tables in
# compare.py and for the per-head deltas in the report.
FOCUS_LABEL = "LVH"

# ---------------------------------------------------------------- input pipeline
#
# THE ARITHMETIC BEHIND THIS ROUND. Read it before changing any number here.
#
# Sheets render at 150 DPI = 5.91 px/mm. Standard calibration is 10 mm/mV and
# 25 mm/s, so on the SOURCE sheet 1 mV is 59 px tall and 1 s is 148 px wide.
#
# Source is 1686x1311. Round 1 squashed that to 384x384:
#
#     vertical   384/1311 = 0.293  ->  1.73 px/mm  ->  1 mV = 17.3 px
#     horizontal 384/1686 = 0.228  ->  1.35 px/mm  ->  1 mm = 1.35 px
#
# Sokolow-Lyon asks whether S(V1) + R(V5) exceeds 35 mm, and Cornell whether
# R(aVL) + S(V3) exceeds 20/28 mm. Both are decided by differences of a few
# millimetres. At 1.73 px/mm a 3 mm difference is 5 px, and it has additionally
# been through JPEG twice. That is the hypothesis this round tests.
#
# At 768: vertical 3.46 px/mm (1 mV = 34.6 px), horizontal 2.71 px/mm — exactly
# 2x the round-1 measurement precision on both axes.
#
# SQUARE IS KEPT ON PURPOSE. The 1.29:1 sheet squashed to square compresses time
# more than voltage, which is a real distortion — but it is the SAME distortion
# round 1 had. Fixing aspect ratio at the same time as raising resolution would
# change two things at once and neither result would be attributable. Aspect
# ratio is a separate experiment; do it after this one.
BASE_SIZE = 384          # the round-1 view. Every model still receives this.
HI_SIZE = 768            # the detail view. 2x linear, 4x pixels, 4x compute.

# Resolution the on-disk prototype copy is stored at. The dataset is pre-resized
# ONCE to this, and every view is derived from it, so no run ever pays to decode
# a 1686x1311 JPEG. Must be >= max(BASE_SIZE, HI_SIZE).
LOAD_SIZE = 768

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------- augmentation
#
# UNCHANGED FROM ROUND 1, AND THE THREE PROHIBITIONS STILL APPLY.
#
#   * HORIZONTAL FLIP - the x axis is TIME. Reverses the heartbeat and reorders
#     the lead columns (I/aVR/V1/V4 ...).
#   * VERTICAL FLIP - the y axis is VOLTAGE and its sign is diagnostic. Turns ST
#     ELEVATION into ST DEPRESSION, i.e. a STEMI into its clinical opposite,
#     with the STEMI label still attached.
#   * RANDOM RESIZED CROP - leads sit at fixed sheet positions. A crop removing
#     V1-V3 removes the only place anterior ST elevation is visible.
#
# There is a new temptation this round specifically: now that a 768 view exists,
# cropping to "just the precordial leads" looks attractive for LVH. Do not. The
# corpus has baked-in tilt and perspective from stage 4, so a fixed fractional
# crop does not land on the same leads in every image, and the .json lead boxes
# are not shipped with the prototype subset. A crop that silently amputates V5
# on tilted sheets would make LVH worse while looking like an improvement on the
# ones it happened to get right.
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
# Identical to round 1, down to the value. Micro-batch varies with VRAM but
# gradient accumulation restores the same EFFECTIVE batch everywhere.
EFFECTIVE_BATCH = 32
EPOCHS = 20
OPTIMIZER = "adamw"
LR = 1e-4
WEIGHT_DECAY = 0.05
BETAS = (0.9, 0.999)
SCHEDULE = "cosine"
WARMUP_EPOCHS = 2
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.0
AMP = True

LR_GRID = [3e-5, 1e-4, 3e-4]

# ---------------------------------------------------------------- performance
#
# NHWC memory layout for convolutional backbones. This is a LAYOUT change, not a
# maths change: the same convolutions run on the same numbers, laid out the way
# Ada/Ampere tensor cores want to read them. Worth 10-30% on ResNet/EfficientNet
# under AMP, and nothing at all on a pre-Turing card.
#
# It is deliberately NOT in the protocol fingerprint. Two runs that differ only
# in memory layout are the same experiment - float accumulation order can shift
# results by ~1e-6, which is far below the run-to-run noise the bootstrap CIs
# already report. Per-run values are recorded in result.json regardless.
#
# Skipped for resnet50_vit: a ViT is almost all LayerNorm and attention, where
# NHWC buys nothing and can insert extra transposes around the patch embedding.
CHANNELS_LAST = True

# ---------------------------------------------------------------- imbalance
USE_POS_WEIGHT = True

# ---------------------------------------------------------------- model selection
MONITOR = "macro_auprc"
MONITOR_MODE = "max"
EARLY_STOP_PATIENCE = 5

# ---------------------------------------------------------------- evaluation
THRESHOLD_TUNING = "val_f1"
TEST_TIME_AUGMENTATION = False
REPORT_PER_SOURCE = True

# Two-stream models are evaluated three times on test: both streams, then each
# stream's features zeroed in turn. This is the measurement the round exists for
# — it separates "the extra resolution carried the signal" from "two backbones
# ensembled". Costs one extra forward pass over 938 images.
STREAM_ABLATION = True

# ---------------------------------------------------------------- fingerprints
# Tier 1: must match across every compared run. compare.py refuses otherwise.
_PROTOCOL = [
    "SEED", "LABELS", "NORM_MEAN", "NORM_STD", "AUG",
    "EFFECTIVE_BATCH", "EPOCHS", "OPTIMIZER", "LR", "WEIGHT_DECAY", "BETAS",
    "SCHEDULE", "WARMUP_EPOCHS", "GRAD_CLIP", "AMP", "USE_POS_WEIGHT",
    "MONITOR", "MONITOR_MODE", "EARLY_STOP_PATIENCE", "THRESHOLD_TUNING",
    "TEST_TIME_AUGMENTATION",
]

# Tier 2: expected to differ. Reported, never enforced.
_INPUT = ["BASE_SIZE", "HI_SIZE", "LOAD_SIZE"]


def protocol_settings() -> dict:
    g = globals()
    return {k: g[k] for k in _PROTOCOL}


def input_settings() -> dict:
    g = globals()
    return {k: g[k] for k in _INPUT}


def controlled_settings() -> dict:
    """Everything, for the run snapshot."""
    return {**protocol_settings(), **input_settings()}


def _hash(d: dict) -> str:
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def protocol_fingerprint() -> str:
    """Training protocol. Runs that disagree here are not comparable."""
    return _hash(protocol_settings())


def input_fingerprint(views: tuple[int, ...] | None = None) -> str:
    """View geometry. Runs are EXPECTED to disagree here — that is the variable."""
    return _hash({"views": list(views)} if views else input_settings())


# Kept so checkpoints and artifacts.py keep working unchanged. Resume safety
# only needs the protocol to match; the views are recorded separately.
def fingerprint() -> str:
    return protocol_fingerprint()


if __name__ == "__main__":
    print(json.dumps(controlled_settings(), indent=2, default=str))
    print("\nprotocol fingerprint:", protocol_fingerprint())
    print("input    fingerprint:", input_fingerprint())
