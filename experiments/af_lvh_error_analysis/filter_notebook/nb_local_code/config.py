"""
Filter-bandwidth inference experiment - shared constants.

The experiment: the 10 AF+LVH test ECGs that the frozen four-head
ResNet-50@768 (`resnet50_hires`) scored as LVH false negatives were
regenerated through the SAME synthetic-ECG pipeline with three band-pass
settings (0.05-40 Hz control, 0.05-100 Hz, 0.05-150 Hz). This round loads
the FROZEN checkpoint and runs the LVH head over all three regenerations
plus the original released images.

FROZEN DISCIPLINE (same as previous rounds):

  * nothing is retrained, re-thresholded or re-split;
  * the checkpoint, the frozen test_predictions.csv and the synthetic
    generation config are read-only;
  * the LVH decision threshold is the frozen val-tuned 0.47;
  * the live 40 Hz probabilities must reproduce the frozen file before any
    comparison number is printed - a moving reference cannot anchor a
    verdict.

The only things that may differ between the three bands are the pixels the
filter produced; the generation config was restored to 0.05-40 Hz after the
experiment build.
"""
from __future__ import annotations

# ------------------------------------------------------------------ the task
# Order matters: the reference run used this order and the LVH column of the
# saved threshold vector is index 2. Do not reorder.
LABELS = ["STEMI", "AF", "LVH", "NORMAL"]
FOCUS_LABEL = "LVH"
LVH_INDEX = LABELS.index("LVH")            # == 2

# ------------------------------------------------------------------ the model
# Same architecture trimmed from round 2 (hybrids.py), rebuilt from timm.
BACKBONE = "resnet50.tv_in1k"
HEAD_HIDDEN = 512
HEAD_DROPOUT = 0.1

# Frozen val-tuned LVH decision threshold of the reference run. Every
# threshold-dependent number below is computed AT this value, never re-tuned.
LVH_THRESHOLD = 0.47

# The frozen per-label decision thresholds of the reference run (order =
# LABELS). Used only by the decision-level cross-check of the 40 Hz control.
FROZEN_THRESHOLDS = [0.44, 0.81, 0.47, 0.39]

# Cross-check drift bands (see analyze.py):
#   < 1e-6   identical        same math to float precision
#   < 1e-3   within_drift     torch/cuDNN float drift (reference used fp16)
#   < 1e-2   encoding_drift   JPEG-encoder/Pillow differences in the 768px
#                             views; acceptable ONLY if all 40 Hz decisions
#                             still match the frozen decisions exactly
#   >= 1e-2  MISMATCH         stop - the pipeline does not reproduce the
#                             frozen evaluation
DRIFT_BANDS = [(1e-6, "identical"), (1e-3, "within_drift"),
               (1e-2, "encoding_drift")]
# A flip at the frozen threshold is flagged "marginal" when its probability
# sits closer to 0.47 than this - i.e. the flip could be an artefact of the
# observed drift rather than the filter change.
MARGIN_FLAG = 0.02

# The reference run's recorded identity, cross-checked against the saved
# checkpoint and (for probabilities) against the frozen predictions file.
REFERENCE_MODEL = "resnet50_hires"
REFERENCE_PROTOCOL_FP = "36988c5e0ba5d72d"
REFERENCE_VIEWS = [768]
REFERENCE_TIMM = "1.0.28"

# ------------------------------------------------------------------ input pipeline
# Must match the reference run byte for byte (transcribed from
# cardiosentry_hybrids_v1/config.py): stored 768px views, NO resampling at
# load time, ImageNet normalisation. The 768px views of the regenerated
# sheets are produced by prepare_views.py with the EXACT transform
# prepare_dataset.py used for the corpus (RGB, BICUBIC to 768x768, JPEG q95).
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
LOAD_SIZE = 768
CHANNELS_LAST = True
AMP = True
VIEW_QUALITY = 95

# ------------------------------------------------------------------ the probe
# The 10 AF+LVH test cases the frozen model scored as LVH false negatives,
# in the frozen file's own probability order (highest first).
CASES = ["ECG003330", "ECG006595", "ECG006463", "ECG005876", "ECG005955",
         "ECG005573", "ECG004242", "ECG004025", "ECG005462", "ECG003966"]

# Band settings of the regeneration experiment. The 40 Hz entry is the
# CONTROL: it reproduces the released images pixel-identically (verified
# when the experiment was built), so it double-checks the whole pipeline.
BANDS = ["0.05-40Hz", "0.05-100Hz", "0.05-150Hz"]

# Variant keys used throughout: the original released sheet, the regenerated
# control, and the two experimental bands.
VARIANTS = ["orig_40", "regen_40", "regen_100", "regen_150"]
VARIANT_BAND = {"orig_40": "0.05-40Hz", "regen_40": "0.05-40Hz",
                "regen_100": "0.05-100Hz", "regen_150": "0.05-150Hz"}
VARIANT_LABEL = {"orig_40": "original 40Hz (released image)",
                 "regen_40": "regenerated 40Hz (control)",
                 "regen_100": "regenerated 100Hz",
                 "regen_150": "regenerated 150Hz"}

# Frozen test-split geometry, cross-checked against the frozen file.
TEST_N = 938
TEST_POSITIVES_LVH = 255
