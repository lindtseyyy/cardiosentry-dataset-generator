"""Shared configuration for the synthetic paper-ECG image corpus.

Extracted from ECG-diagnosis-prototype/pipeline/config.py. Everything that
resolved labels from the three raw catalogues has been removed: this pipeline
starts from a CSV that the dataset pipeline already produced (by default
`data/balanced/balanced_ecg_metadata.csv`), so label decisions arrive as data
rather than being re-derived here.

Every path, render parameter and augmentation knob lives here so the stages
stay declarative and a build is reproducible from one file.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent      # ecg_dataset_pipeline/

# The record CSV this pipeline consumes. Any CSV with the columns named in
# CSV_COLUMNS below works; --input-csv on stage1 overrides it.
# Defaults to the RELEASE copy, not data/balanced/, because only the release
# carries the frozen `split` column written by scripts/freeze_splits.py. Point
# this at data/balanced/ and stage 1 would invent a fresh split instead of
# reading the frozen one, which silently breaks comparability between runs.
RELEASE = os.environ.get("ECG_RELEASE", "v2")
INPUT_CSV = ROOT / "data" / "releases" / RELEASE / "balanced" / "balanced_ecg_metadata.csv"
FALLBACK_INPUT_CSV = ROOT / "data" / "releases" / RELEASE / "harmonized" / "harmonized_ecg_metadata.csv"

# ecg-image-kit checkout. Not vendored here - clone it yourself (see README)
# or point ECGKIT_DIR at an existing copy.
KIT = Path(os.environ.get(
    "ECGKIT_DIR", ROOT / "ecg-image-kit")) / "codes" / "ecg-image-generator"

# Stage 3 shells out to the kit, so it needs the same interpreter run_all.sh
# uses. Keep the ECGKIT_PY override working in both places.
VENV_PY = Path(os.environ.get(
    "ECGKIT_PY", Path.home() / ".cache" / "ecgkit-venv" / "bin" / "python"))

# The image corpus belongs to the release it was built from, so it lives
# alongside it by default. Override with ECG_IMAGE_BUILD.
BUILD = Path(os.environ.get(
    "ECG_IMAGE_BUILD", ROOT / "data" / "releases" / RELEASE / "synthetic_data"))
MANIFEST = BUILD / "image_manifest.csv"   # stage1 output: one row per record
STAGED_MAP = BUILD / "staged_map.csv"     # stage2 output: renamed -> manifest
STAGED = BUILD / "staged"                 # harmonised WFDB, one subdir per chunk
RENDERED = BUILD / "rendered"             # clean PNG straight from the kit
IMAGES = BUILD / "images"                 # final augmented JPEG corpus
INDEX_CSV = BUILD / "index.csv"           # the corpus manifest to train from

# ---------------------------------------------------------------- input CSV
# Column names in the input CSV. Change these (not the stage code) to accept a
# differently-shaped CSV.
CSV_COLUMNS = {
    "record_id": "global_record_id",     # unique, provenance-preserving id
    "source": "source_dataset",          # PTBXL / CHAPMAN / STEMI
    "original_id": "original_record_id",
    "patient": "split_group_id",         # group key - never split across folds
    "label": "cardiosentry_label",       # "AF", "AF+LVH", ...
    "signal_path": "signal_path",        # WFDB base path, no extension
    "header_path": "header_path",
    "age": "age",
    "sex": "sex",
    "split": "split",                    # optional; assigned here when absent
    "validation": "validation_status",   # optional; see ACCEPT_VALIDATION
}

# Binary label columns carried through to index.csv untouched. These stay the
# source of truth for training; `cls` below is only a directory/reporting name.
LABEL_COLUMNS = ["STEMI", "AF", "LVH", "NORMAL"]

# Rows whose validation_status is not in this set are skipped. Set to None to
# accept everything the CSV contains.
ACCEPT_VALIDATION = {"VALID", "VALID_WITH_WARNINGS"}

CLASSES = ["STEMI", "LVH", "AF", "NORMAL"]

# ---------------------------------------------------------------- metadata
# The kit prints Age and Sex on every sheet and raises KeyError if either
# header comment is missing, so stage1 drops records it cannot decode.
#
# Per-source sex encodings, because the three catalogues disagree:
#   PTB-XL   0 = male, 1 = female
#   STEMI    1 = male, 0 = female   (the Chongqing set's `gender` column)
#   Chapman  the literal words, from the WFDB header
SEX_DECODERS = {
    "PTBXL": {"0": "M", "1": "F"},
    "STEMI": {"1": "M", "0": "F"},
    "CHAPMAN": {"male": "M", "female": "F"},
}
SEX_WORD = {"M": "Male", "F": "Female"}          # as printed on the sheet

PTBXL_AGE_SENTINEL = 300     # PTB-XL masks age > 89 as 300
AGE_MASKED_AS = 90           # ...printed as this instead

# ---------------------------------------------------------------- signal
FS = 500                 # Hz, native to all three sources
N_SAMPLES = 5000         # 10 s
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
ADC_GAIN = 1000.0        # ADU per mV
BANDPASS = (0.05, 40.0)  # Hz - see README for why the low-pass is 40, not 150

# ---------------------------------------------------------------- corpus
# How many render variants each record contributes. The input CSV is already
# balanced, so this defaults to 1 everywhere; raise a class here to inflate it
# without re-balancing upstream. A multi-label record takes the largest value
# among its positive classes.
RENDERS_PER_RECORD = {"STEMI": 1, "LVH": 1, "AF": 1, "NORMAL": 1}

# ---------------------------------------------------------------- split-aware
# How many augmented variants each record contributes, BY SPLIT.
#
# TRAIN may be expanded: more variants of one ECG is honest data augmentation,
# because the model never sees those records at evaluation time.
#
# VAL and TEST are pinned to exactly 1 and must stay there. Two variants of one
# ECG in a test set silently doubles that patient's weight in every metric, so
# the score starts depending on which ECGs happened to get more variants. This
# is an evaluation-integrity constraint, not a tunable.
VARIANTS_PER_SPLIT = {"train": 4, "val": 1, "test": 1}
EVAL_SPLITS = ("val", "test")          # hard-pinned to 1 variant; see above

# Staged records are written in fixed-size chunks so stage 3 can hand the kit
# many small batches and keep every core busy. Each kit invocation pays a ~7 s
# TensorFlow import, so chunks much smaller than this waste real time.
CHUNK_SIZE = 400
# Both heavy stages are CPU-bound, so oversubscribing cores only adds context
# switching. Default to the machine's core count rather than a fixed 8.
WORKERS = int(os.environ.get("ECG_WORKERS", 0)) or (os.cpu_count() or 4)

# Split assignment (stage 1). Skipped entirely when the input CSV already
# carries a `split` column. Grouped on CSV_COLUMNS["patient"], never on the
# record id: the same patient must not straddle a fold boundary.
SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}
USE_PTBXL_FOLDS = True       # honour PTB-XL's own strat_fold, read from
PTBXL_TEST_FOLDS = {10}      # source_extra_json when the column is present
PTBXL_VAL_FOLDS = {9}
SEED = 20260818

# Every staged record is renamed to this scheme. The kit prints the record name
# on the sheet as "ID: <name>", and the native names are source-identifying
# (00001_hr / JS00001 / 00101), which would hand the model a free source label.
RECORD_NAME_FMT = "ECG{:06d}"

# ---------------------------------------------------------------- render
# 150 DPI -> 5.91 px/mm -> 147.6 px/s at 25 mm/s -> image Nyquist 73.8 Hz, a
# 1.8x margin over the 40 Hz signal band, and the kit's own recommended floor
# for digitisation. Measured 739 KB/image vs 1.2 MB at 200 DPI.
DPI = 150
# The kit fixes 25 mm/s and 10 mm/mV and prints them on the sheet; these are
# recorded here for the DPI/Nyquist arithmetic above, not to configure it.
PAPER_SPEED = 25         # mm/s
PAPER_GAIN = 10          # mm/mV
MARGIN_PX = 18           # thin margin, applied by stage 4 (kit only does int inches)
# The 1 mm grid, not the added noise, is what resists compression: a clean sheet
# is 2.5 MB at q92/4:4:4 even with zero noise. q85 with 4:2:0 chroma costs
# nothing diagnostically (colour carries no signal here) and halves the corpus.
JPEG_QUALITY = 85
JPEG_SUBSAMPLING = 2     # 4:2:0

# ---------------------------------------------------------------- grid colour
# Real ECG paper is printed with a coloured grid and a black trace. The kit can
# only choose a grid palette per INVOCATION, and each invocation pays a ~7 s
# TensorFlow import, so per-image colour through the kit would mean one process
# per image. Instead every sheet is rendered once in the kit's grey "bw" style
# and stage 4 recolours the grid per image with a 256-entry lookup table.
#
# That works because the rendered PNG is pure greyscale with the grid on two
# exact levels -- major 102, minor 191 -- and the trace below 60. Verified on
# real renders: 23.3% of pixels at 191, 5.1% at 102, 1.4% below 60.
#
# Colours are (major, minor, paper) as 0-255 RGB, sampled from real printouts.
# WEIGHTS must never correlate with class: stage 4 seeds its palette choice on
# the output identity only, exactly like every other effect.
# NOTE ON TUNING: the MINOR gridline is 23.3% of the sheet's pixels and the
# major only 5.1%, so the minor colour is what the eye reads as "the colour of
# the paper". A first pass used a washed-out minor for red and orange and every
# sheet came out looking pink or sepia. The minors below are therefore kept
# saturated and clearly separated in hue.
# Muted deliberately. A first pass used fully saturated inks and the sheets
# read as vivid pink/orange poster paper rather than a clinical printout: the
# minor line is 23% of all pixels, so its chroma sets the whole page's cast.
# Real ECG paper grids are desaturated and low-contrast against the stock --
# the grid is meant to be read through, not looked at.
GRID_PALETTES = {
    # major gridline        minor gridline        paper
    "red":    ((178,  86,  84), (228, 180, 176), (253, 249, 247)),
    "pink":   ((188, 124, 148), (236, 200, 212), (254, 251, 252)),
    "orange": ((186, 126,  86), (236, 202, 176), (254, 250, 245)),
    "grey":   ((124, 124, 124), (198, 198, 198), (252, 252, 252)),   # neutral
}
GRID_PALETTE_WEIGHTS = {"red": 0.38, "orange": 0.22, "pink": 0.20, "grey": 0.20}

# Per-image ink-strength jitter, biased below 1.0 so faded printouts are the
# common case and a crisp fresh one the exception.
GRID_SATURATION = (0.72, 1.02)

# Greyscale control points the LUT interpolates between. Below TRACE_MAX the
# pixel is ink and is left essentially black, so recolouring never tints the
# trace the model has to read.
GRID_TRACE_MAX = 55      # <= this is trace/text, kept neutral
GRID_MAJOR_LEVEL = 102   # kit's major gridline grey
GRID_MINOR_LEVEL = 191   # kit's minor gridline grey
GRID_PAPER_LEVEL = 250   # >= this is bare paper

# ---------------------------------------------------------------- lead labels
# Where the lead name (I, aVR, V1, ...) sits relative to its trace baseline.
#
# The kit prints it BELOW the baseline (y_offset - lead_name_offset - 0.2).
# Real printouts from the Philippines -- and most clinical carts generally --
# put it ABOVE and to the left, clear of the trace. Applied by patch_kit.py.
#
# Units are plot units: 1.0 = 1 mV = 10 mm at the standard 10 mm/mV gain, and
# rows are row_height = 8 units apart, so there is ample headroom. Positive
# moves the label up the page.
LEAD_NAME_DY = 1.15

# ---------------------------------------------------------------- optics
# Mild defocus only. At 150 DPI the ST segment is a few pixels tall, and a
# blur wide enough to be obvious is a blur wide enough to erase it.
DEFOCUS_PROB = 0.55
DEFOCUS_RADIUS = (0.3, 0.9)      # px, Gaussian

# ---------------------------------------------------------------- paper wear
# How hard the wrinkle photograph is pressed into the sheet. The first pass ran
# 0.65-1.05 and produced sheets that read as "crumpled and thrown away" rather
# than "folded once in a pocket"; at the low end here the texture is present
# but never competes with the trace.
TEXTURE_STRENGTH = (0.30, 0.62)
TEXTURE_CLIP = (0.86, 1.10)      # multiplicative field bounds
CREASE_PROB = 0.62
CREASE_AMP = (0.06, 0.13)
CREASE_CLIP = (0.88, 1.14)

# Lighting effects multiply, and independently-sampled ramp x falloff x edge
# shadow x exposure can compound to ~0.5x, which produced sheets that read as
# dark grey card rather than paper. After all shading, the sheet's bright paper
# level is renormalised into this window, which bounds the compounding without
# flattening any individual effect.
PAPER_LEVEL_TARGET = (216, 248)   # 95th-percentile luminance of the finished sheet

# Flags passed to gen_ecg_images_from_data_batch.py. The kit's own distortion
# layer (--augment / --wrinkles) is deliberately NOT used; see stage4.
KIT_RENDER_FLAGS = [
    "--num_columns", "4",        # 3x4 layout
    "--full_mode", "II",         # continuous lead II rhythm strip
    "-r", str(DPI),
    "--pad_inches", "0",         # int-only in the kit; margin added in stage 4
    "--print_header",            # ID / age / sex from header comments
    "--calibration_pulse", "1",
    # Grey grid, matching a real BTL CardioPoint printout: major (0.4,0.4,0.4),
    # minor (0.75,0.75,0.75), black trace. In the kit this palette is only
    # reachable through its "bw" style, which --random_bw 1 selects for every
    # image. It is a grid-colour switch here, not a greyscale conversion.
    "--random_bw", "1",
    "--store_config", "1",
    "--lead_bbox",
    "--lead_name_bbox",
]


# ---------------------------------------------------------------- helpers
def slug(name: str) -> str:
    """Filesystem-safe form of a class name.

    Multi-label records arrive as "AF+LVH", which is legal on every platform
    this runs on but reads badly in shell globs, so it becomes "AF_LVH". Used
    for directory names only - index.csv keeps the original string.
    """
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in name) or "UNLABELLED"
