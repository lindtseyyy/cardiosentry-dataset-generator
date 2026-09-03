"""
LVH-binary negative-transfer test — controlled experimental conditions.

WHAT THIS ROUND IS

Round 2 (cardiosentry_hybrids_v1) trained a four-head ResNet-50@768
(`resnet50_hires`) and reported the best LVH numbers so far: PTB-XL LVH AUPRC
0.9315. This round asks a different question: is the shared multi-label
objective (STEMI, AF, LVH, NORMAL) itself hurting LVH? The test is a ResNet-50
at the SAME 768x768 resolution trained on LVH ONLY, as a binary task, with
everything else held as close to the four-head run as possible. If the binary
model beats the four-head model on PTB-XL LVH, that is evidence of negative
transfer from the multi-label setup.

THE FAIRNESS CONTRACT

Every value below is byte-identical to cardiosentry_hybrids_v1's config: same
seed, same normalisation, same augmentation, same optimiser, same schedule,
same effective batch 32, same 20 epochs, same monitor NAME (`macro_auprc` -
which over a single head is exactly the LVH head's val AUPRC), same early
stopping, same val-F1 threshold rule, same micro-batch 8 that the reference
run used. The dataset is the SAME `prototype_1152_768` copy, so the model sees
the same bytes through the same loader.

THE ONE DECLARED DIFFERENCE - the experimental variable

The task: four sigmoid heads -> one. Everything downstream of that is the
forced consequence, declared rather than hidden:

  * LABELS = ["LVH"] instead of ["STEMI", "AF", "LVH", "NORMAL"]
  * the shared 2-layer MLP head keeps its exact structure (LayerNorm -> D(0.1)
    -> 2048x512 -> GELU -> D(0.1)) and only its final width changes 512x4 ->
    512x1
  * the loss sees one pos_weighted BCE term instead of four; the LVH
    pos_weight is the SAME number the four-head run used (2.5524)
  * monitor `macro_auprc` reduces to the single head's AUPRC by construction

FINGERPRINTS ARE TWO-TIER, LIKE ROUND 2's

  PROTOCOL fingerprint   - everything shared with the four-head run. compare.py
                           verifies the SHARED protocol settings against the
                           reference's recorded protocol_settings field by
                           field (task fields removed), and refuses otherwise.

  TASK fingerprint       - LABELS + head width. Expected to differ from the
                           reference; that is the experiment.

There is one more hash - the SHARED fingerprint - computed over the protocol
with the task fields removed. It must equal the same quantity computed from
the frozen reference settings in baseline.py. If it does not, some shared
value was changed by accident and the comparison is invalid.
"""
from __future__ import annotations
import hashlib, json

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- task definition
# THE EXPERIMENTAL VARIABLE. Everything else in this file must stay byte-
# identical to cardiosentry_hybrids_v1/config.py.
LABELS = ["LVH"]
NUM_CLASSES = len(LABELS)

FOCUS_LABEL = "LVH"

# ---------------------------------------------------------------- input pipeline
#
# The on-disk copy is the SAME prototype_1152_768 the four-head resnet50_hires
# run read: 768px sheets, one JPEG generation, quality 95. The model's only
# view is the stored size itself - no resampling at all - so the binary model
# and the reference saw identical bytes through identical transforms.
VIEW_SIZE = 768
LOAD_SIZE = 768

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------- augmentation
#
# UNCHANGED FROM THE FOUR-HEAD ROUND, AND THE THREE PROHIBITIONS STILL APPLY:
#   * HORIZONTAL FLIP - the x axis is TIME.
#   * VERTICAL FLIP - the y axis is VOLTAGE and its sign is diagnostic.
#   * RANDOM RESIZED CROP - amputates leads that sit at fixed sheet positions.
AUG = {
    "hflip": 0.0,
    "vflip": 0.0,
    "random_resized_crop": False,
    "rotation_deg": 0.0,
    "brightness": 0.10,
    "contrast": 0.10,
    "saturation": 0.0,
    "hue": 0.0,
}

# ---------------------------------------------------------------- optimisation
# Identical to the four-head round, down to the values.
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
# NHWC for the convolutional backbone - the same layout the four-head run used.
# Kept out of the fingerprints for the same reason as round 2: two runs that
# differ only in memory layout are the same experiment; the per-run value is
# recorded in result.json regardless.
CHANNELS_LAST = True

# ---------------------------------------------------------------- imbalance
USE_POS_WEIGHT = True

# ---------------------------------------------------------------- model selection
#
# "macro_auprc" is DELIBERATELY kept as the monitor name, not renamed to
# "auprc". Over a single head the macro average IS the head's AUPRC, so the
# binary run selects its checkpoint by exactly the quantity the four-head run
# would call val LVH AUPRC - while the recorded setting stays byte-identical.
MONITOR = "macro_auprc"
MONITOR_MODE = "max"
EARLY_STOP_PATIENCE = 5

# ---------------------------------------------------------------- evaluation
THRESHOLD_TUNING = "val_f1"
TEST_TIME_AUGMENTATION = False
REPORT_PER_SOURCE = True

# ---------------------------------------------------------------- fingerprints
# Tier 1: must match the four-head reference run field by field (task fields
# removed). compare.py refuses otherwise.
_PROTOCOL = [
    "SEED", "LABELS", "NORM_MEAN", "NORM_STD", "AUG",
    "EFFECTIVE_BATCH", "EPOCHS", "OPTIMIZER", "LR", "WEIGHT_DECAY", "BETAS",
    "SCHEDULE", "WARMUP_EPOCHS", "GRAD_CLIP", "AMP", "USE_POS_WEIGHT",
    "MONITOR", "MONITOR_MODE", "EARLY_STOP_PATIENCE", "THRESHOLD_TUNING",
    "TEST_TIME_AUGMENTATION",
]

# The task is the experimental variable: reported, never enforced against the
# reference.
_TASK = ["LABELS", "NUM_CLASSES"]


def protocol_settings() -> dict:
    g = globals()
    return {k: g[k] for k in _PROTOCOL}


def task_settings() -> dict:
    g = globals()
    return {k: g[k] for k in _TASK}


def shared_settings() -> dict:
    """Protocol with the task fields removed - the part that MUST equal the
    four-head reference's recorded settings."""
    return {k: v for k, v in protocol_settings().items() if k not in _TASK}


def controlled_settings() -> dict:
    return {**protocol_settings(), **task_settings(),
            "VIEW_SIZE": VIEW_SIZE, "LOAD_SIZE": LOAD_SIZE}


def _hash(d: dict) -> str:
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def protocol_fingerprint() -> str:
    """Full protocol hash, task included. Differs from the reference's because
    the task differs - that is the point."""
    return _hash(protocol_settings())


def shared_fingerprint() -> str:
    """Hash of the protocol with task fields removed. MUST equal the same
    quantity computed from the frozen reference in baseline.py."""
    return _hash(shared_settings())


def task_fingerprint() -> str:
    return _hash(task_settings())


def input_fingerprint(views: tuple[int, ...] | None = None) -> str:
    """Same function, same format as round 2, so the hash matches the
    reference's input fingerprint (6b9b1dd6f951bad0) when views == [768]."""
    return _hash({"views": list(views)} if views else {"views": [VIEW_SIZE]})


def fingerprint() -> str:
    return protocol_fingerprint()


if __name__ == "__main__":
    print(json.dumps(controlled_settings(), indent=2, default=str))
    print("\nprotocol fingerprint:", protocol_fingerprint())
    print("task     fingerprint:", task_fingerprint())
    print("shared   fingerprint:", shared_fingerprint())
    print("input    fingerprint:", input_fingerprint())
