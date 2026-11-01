"""Central configuration (PLAN.md v1.11: sections 3-4, 6-11, 13-14, 16-17).

The notebook's Section 1 builds a Config, overrides paths for EC2 vs local,
freezes it to generation_config.json, and every stage reads from that object.
No other module carries tunable constants except their own pure-math helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


TARGET_LABELS = ["NORM", "AF", "IAVB", "LBBB", "RBBB", "PAC", "PVC",
                 "LAFB", "LAE", "TInv", "LQT", "PRWP"]

SEED = 20260918
SPLIT_RATIO = {"train": 0.80, "val": 0.10, "test": 0.10}
EXPECTED_ACCEPTED = 31109

# Render geometry (PLAN 6.5, 7.x)
DPI = 200
MASTER_W, MASTER_H = 2200, 1700
FINAL_W, FINAL_H = 1024, 768
RESIZE_H = 768            # LANCZOS fit height; width follows aspect (994)
PAD_EACH_SIDE = 15        # (1024 - 994) / 2
EFFECTIVE_DPI_FINAL = DPI * (RESIZE_H / MASTER_H)   # ~90.35
NOMINAL_MAJOR_MASTER_PX = 5.0 / 25.4 * DPI          # 39.37
NOMINAL_MAJOR_FINAL_PX = NOMINAL_MAJOR_MASTER_PX * (RESIZE_H / MASTER_H)  # 17.79
# Minor-grid nominals: the spectral validator measures the MINOR grid
# (1 mm) because the major peak sits in cardiac-harmonic clutter while the
# minor peak (r=56) is isolated and 18x stronger. Same printed transform,
# so distortion affects both identically; major-equivalent = 5x minor.
NOMINAL_MINOR_MASTER_PX = NOMINAL_MAJOR_MASTER_PX / 5.0   # 7.874
NOMINAL_MINOR_FINAL_PX = NOMINAL_MAJOR_FINAL_PX / 5.0     # 3.558

# Kit invocation (PLAN 6.x)
KIT_COLUMNS = 4
KIT_FULL_MODE = "II"
KIT_PAD_INCHES = 0
FIXED_FONT = "Arial.ttf"          # clean images, every split
KIT_ENV = {                       # patched-layout geometry (plot units)
    "ECG_ROW_GAP": "4.4",
    "ECG_RHYTHM_GAP": "5.5",
    "ECG_BOTTOM_RESERVE": "1.9",
}
KIT_RENDER_FLAGS = [
    "--num_columns", "4",
    "--full_mode", "II",
    "-r", str(DPI),
    "--pad_inches", "0",
    "--calibration_pulse", "1",
    "--random_bw", "1",           # grey master; recoloured per variant
]

# Grid palettes: (major, minor, paper) RGB, sampled from real printouts.
GRID_PALETTES = {
    "red":    ((178, 86, 84), (228, 180, 176), (253, 249, 247)),
    "pink":   ((188, 124, 148), (236, 200, 212), (254, 251, 252)),
    "orange": ((186, 126, 86), (236, 202, 176), (254, 250, 245)),
    "grey":   ((124, 124, 124), (198, 198, 198), (252, 252, 252)),
}
GRID_PALETTE_WEIGHTS = {"red": 0.38, "orange": 0.22, "pink": 0.20, "grey": 0.20}
GRID_SATURATION = (0.72, 1.02)
GRID_TRACE_MAX = 55
GRID_MAJOR_LEVEL = 102
GRID_MINOR_LEVEL = 191
GRID_PAPER_LEVEL = 250

# Augmentation bounds (PLAN 10.3)
ROT_DEG = (-3.0, 3.0)
PERSPECTIVE_PROB = 0.7
PERSPECTIVE_MAG = (0.004, 0.018)
TEXTURE_STRENGTH = (0.30, 0.62)
TEXTURE_CLIP = (0.86, 1.10)
CREASE_PROB = 0.62
CREASE_AMP = (0.06, 0.13)
CREASE_CLIP = (0.88, 1.14)
RAMP_RANGE = (0.10, 0.22)
FALLOFF_RANGE = (0.06, 0.18)
EDGE_SHADOW_PROB = 0.65
EDGE_SHADOW_DEPTH = (0.10, 0.26)
WB_SIGMA = 0.012
EXPOSURE = (0.92, 1.06)
CONTRAST = (0.94, 1.08)
PAPER_LEVEL_TARGET = (216, 248)
SENSOR_NOISE_SIGMA = (1.5, 4.5)
DEFOCUS_PROB = 0.55
DEFOCUS_RADIUS = (0.3, 0.9)

# Deterministic stress recipe (PLAN 10.6) — identical for every record.
STRESS_RECIPE = {
    "palette": "orange",
    "saturation": 0.9,
    # -2.0 deg (right side down): keeps the left-positioned "25mm/s" footer
    # legend inside the frame (verified intact); +2.5 deg clipped it.
    "rot_deg": -2.0,
    "perspective_mag": 0.010,
    "texture_strength": 0.5,
    "texture_index": 3,          # wrinkle photo pick (sorted list)
    "creases": True,
    "crease_amp": 0.09,
    "illum_angle": 0.7,          # radians
    "illum_ramp": 0.16,
    "illum_falloff": 0.12,
    "edge_shadow": 0,            # edge index 0..3
    "edge_depth": 0.18,
    "exposure": 0.96,
    "contrast": 1.03,
    "paper_target": 232.0,
    "noise_sigma": 3.0,
    "defocus_radius": 0.6,
}

# Variants per split (PLAN 10.1)
TRAIN_VARIANTS = ["clean", "v01", "v02", "v03"]
EVAL_VARIANTS = ["clean", "stress"]

# Image subsets (PLAN v1.2 §11, §14): eval pairs are separated into
# first-class subsets — separate directories + manifests, never pooled.
EVAL_SUBSETS = ("val_clean", "val_stress", "test_clean", "test_stress")
SUBSETS = ("train",) + EVAL_SUBSETS


def subset_for(split: str, variant_id: str) -> str:
    """Derive the image subset from (split, variant_id).

    Train rows (any train variant) -> "train"; eval rows ->
    "<split>_<variant_id>". Anything else is a caller bug: raise loudly.
    Single source of truth shared by the chunk worker and the gates.
    """
    if split == "train":
        if variant_id not in TRAIN_VARIANTS:
            raise ValueError(f"train cannot carry variant {variant_id!r}")
        return "train"
    if split in ("val", "test"):
        if variant_id not in EVAL_VARIANTS:
            raise ValueError(f"{split} cannot carry variant {variant_id!r}")
        return f"{split}_{variant_id}"
    raise ValueError(f"unknown split {split!r}")

# Split gates (PLAN 4.3-4.4)
LABEL_MIN_ABSOLUTE = 30
LABEL_MIN_FRACTION = 0.05
LABEL_RATIO_TOL = 0.25           # relative tolerance on 80/10/10
SOURCE_PP_TOL = 3.0              # percentage points per source per split
SOURCE_MIN_RECORDS_GATE = 50     # smaller sources (ptb: 15) exempt, reported

# Grid-distortion gates (PLAN 10.7)
GRID_WINDOWS = 9
GRID_WINDOW_SIZE = 200
GRID_SEARCH_BAND = (0.8, 1.25)   # of nominal period
MAX_NOMINAL_DEV = 0.05
# Mismatch gate calibrated to 4% post-preview (was 3% pre-measurement):
# 84-image preview measured mismatch p50 0.89 / p90 2.03 / max 2.77% while
# the instrument noise floor on clean images is <=0.26%, so ~2.8% is genuine
# keystone effect at the approved 1.8% bound. 4% keeps 40%+ headroom over
# production extremes yet still catches any accidental stretch >=4%.
MAX_CLEAN_DERIVED_MISMATCH = 0.04
MAX_FAILED_WINDOWS = 1
GRIDCHECK_SAMPLE_N = 500

# Accepted subset actions (PLAN 3)
ACCEPT_ACTIONS = {"copied_as_is", "cropped_centered", "cropped_af_verified",
                  "cropped_pac_localized", "cropped_pvc_localized",
                  "cropped_af_flagged"}

CHUNK_SIZE = 200
CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                   "V1", "V2", "V3", "V4", "V5", "V6"]
ADC_GAIN = 1000.0

TRAIN_TAR = "cardiosentry_v1_10s_768x1024_multivariant.tar"
MASTERS_TAR = "cardiosentry_v1_10s_masters_200dpi.tar"
TAR_MTIME = 0                    # fixed mtime for reproducible archives

S3_BUCKET = "cardiosentry-fcmae-758821198776-us-east-1-an"
S3_PREFIX = "datasets/V1"
S3_MULTIPART_THRESHOLD = 64 * 1024 * 1024
S3_MULTIPART_CHUNKSIZE = 8 * 1024 * 1024
S3_CHECKSUM_ALGORITHM = "CRC64NVME"

# Input subset tarball (PLAN v1.3 §3): fetched from S3 when no local copy
# exists. The sidecar object (<uri>.sha256) is the SHA-256 authority.
INPUT_S3_URI = (f"s3://{S3_BUCKET}/datasets/raw_signals_data/V1/"
                "V1_10S_STANDARDIZED_SUBSET.tar.gz")


@dataclass
class Config:
    """Run configuration. Paths have local defaults; Section 1 overrides
    them for EC2, then freezes the resolved object to generation_config.json.
    """
    subset_dir: str = ("/home/lindtseyxvii/Codes/cardiosentry/dataset-files"
                       "/MAIN_DATASET/V1_10S_STANDARDIZED_SUBSET")
    input_s3_uri: str = INPUT_S3_URI  # fetched iff subset_dir is missing
    work_dir: str = ""            # outputs/ root (masters, dataset, TAR, …)
    kit_dir: str = ""             # extracted bundle: ecg-image-generator/
    bundle_path: str = ""         # ecg_image_v1_code.tar.gz
    kit_python: str = ""          # venv python for kit subprocesses
    seed: int = SEED
    approve_preview: bool = False
    verify_s3_bytes: bool = False
    upload_masters_tar: bool = True
    workers: int = 0              # 0 = os.cpu_count()
    s3_bucket: str = S3_BUCKET
    s3_prefix: str = S3_PREFIX

    # resolved paths (filled by resolve())
    dataset_dir: str = field(default="", init=False)
    master_dir: str = field(default="", init=False)
    scratch_dir: str = field(default="", init=False)
    index_parts_dir: str = field(default="", init=False)
    preview_dir: str = field(default="", init=False)
    input_dir: str = field(default="", init=False)

    def resolve(self) -> "Config":
        if not self.work_dir:
            raise ValueError("work_dir must be set (Section 1)")
        w = Path(self.work_dir)
        self.dataset_dir = str(w / "dataset")
        self.master_dir = str(w / "master")
        self.scratch_dir = str(w / "staged_scratch")
        self.index_parts_dir = str(w / "index_parts")
        self.preview_dir = str(w / "preview")
        self.input_dir = str(w / "input")
        if not self.workers:
            import os as _os
            self.workers = _os.cpu_count() or 4
        return self

    def to_json(self) -> str:
        d = asdict(self)
        d.update({
            "target_labels": TARGET_LABELS,
            "split_ratio": SPLIT_RATIO,
            "dpi": DPI,
            "train_variants": TRAIN_VARIANTS,
            "eval_variants": EVAL_VARIANTS,
            "subsets": list(SUBSETS),
            "filtering": "no additional signal-domain filtering",
            "plan_version": "v1.11",
        })
        return json.dumps(d, indent=2, sort_keys=True) + "\n"


def default_config() -> Config:
    return Config()
