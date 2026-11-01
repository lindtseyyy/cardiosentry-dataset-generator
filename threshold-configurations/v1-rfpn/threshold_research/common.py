"""Shared constants, input validation and cached-prediction loading.

Scores are read as float32 because the model emits float32 sigmoid outputs and the
prediction CSVs store their shortest float32 repr; parsing to float32 recovers the
exact model value, so `score >= threshold` reproduces the notebook's decisions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("NORM", "SB", "STACH", "LBBB", "LQT", "AF", "AFL",
          "3AVB", "2AVB", "1AVB", "STEMI", "STE")
NORMAL_LABEL = "NORM"
PATHOLOGY_LABELS = tuple(label for label in LABELS if label != NORMAL_LABEL)
LABEL_NAMES = {
    "NORM": "Normal ECG", "SB": "Sinus bradycardia", "STACH": "Sinus tachycardia",
    "LBBB": "Left bundle branch block", "LQT": "QT/QTc prolongation",
    "AF": "Atrial fibrillation", "AFL": "Atrial flutter",
    "3AVB": "Third-degree AV block", "2AVB": "Second-degree AV block",
    "1AVB": "First-degree AV block", "STEMI": "STEMI pattern", "STE": "ST elevation",
}

RESEARCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = RESEARCH_DIR.parents[2]
RUN_DIR = REPO_ROOT / "prototype-results/new_runs/convnext_tiny_v1_rfpn_clean12_colab_a100_seed42"
MODEL_DIR = RUN_DIR / "convnext_tiny_rfpn"
OUTPUT_DIR = RESEARCH_DIR / "outputs"
FROZEN_DIR = OUTPUT_DIR / "frozen"
FROZEN_MANIFEST = FROZEN_DIR / "freeze_manifest.json"
SEED = 20261015


class LabelOrderError(ValueError):
    """Raised when score/label columns do not follow the checkpoint label order."""


class HeldOutSplitAccessError(RuntimeError):
    """Raised when test data is offered to a threshold-fitting routine."""


@dataclass(frozen=True)
class SplitData:
    name: str
    record_ids: np.ndarray
    patient_keys: np.ndarray
    sources: np.ndarray
    y_true: np.ndarray      # (n, 12) int8, 0/1
    scores: np.ndarray      # (n, 12) float32 in [0, 1]
    label_order: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_label_order(label_order) -> tuple[str, ...]:
    label_order = tuple(label_order)
    if label_order != LABELS:
        raise LabelOrderError(f"label order {label_order} != checkpoint order {LABELS}")
    return label_order


def checkpoint_label_order() -> tuple[str, ...]:
    """Label order recorded by the training run (config snapshot, fingerprinted)."""
    config = json.loads((MODEL_DIR / "config_snapshot.json").read_text())
    return validate_label_order(config["identity"]["label_order"])


def validate_scores_and_labels(y_true: np.ndarray, scores: np.ndarray) -> None:
    if y_true.shape != scores.shape or y_true.ndim != 2 or y_true.shape[1] != len(LABELS):
        raise ValueError(f"expected (n, {len(LABELS)}) arrays, got {y_true.shape} and {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or inf")
    if scores.min() < 0 or scores.max() > 1:
        raise ValueError("scores outside [0, 1]; logits passed instead of sigmoid probabilities?")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("labels must be 0/1; uncertain/missing encodings need explicit masking")


def load_predictions(path: Path, expected_split: str, label_order=None) -> SplitData:
    """Load a cached prediction CSV; columns are resolved by name in checkpoint order."""
    label_order = validate_label_order(label_order or checkpoint_label_order())
    frame = pd.read_csv(path, dtype={"record_id": str, "source": str, "split": str})
    splits = set(frame["split"].unique())
    if splits != {expected_split}:
        raise ValueError(f"{path.name}: expected split {expected_split!r}, found {sorted(splits)}")
    if frame["record_id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate record_id rows")
    missing = [c for label in label_order for c in (f"true_{label}", f"prob_{label}") if c not in frame]
    if missing:
        raise LabelOrderError(f"{path.name}: missing columns {missing}")
    y_true = frame[[f"true_{label}" for label in label_order]].to_numpy(dtype=np.int8)
    scores = frame[[f"prob_{label}" for label in label_order]].to_numpy(dtype=np.float32)
    validate_scores_and_labels(y_true, scores)
    record_split = pd.read_csv(RUN_DIR / "record_split.csv", dtype=str)
    keys = frame[["record_id"]].merge(
        record_split.rename(columns={"record_path": "record_id"}), on="record_id", how="left")
    if keys["patient_key"].isna().any():
        raise ValueError(f"{path.name}: records without patient_key in record_split.csv")
    if not (keys["split"] == expected_split).all():
        raise ValueError(f"{path.name}: record_split.csv disagrees with the CSV split column")
    return SplitData(
        name=expected_split,
        record_ids=frame["record_id"].to_numpy(),
        patient_keys=(keys["site"] + ":" + keys["patient_key"]).to_numpy(),
        sources=frame["source"].to_numpy(),
        y_true=y_true,
        scores=scores,
        label_order=label_order,
    )


def require_development_split(split: SplitData) -> SplitData:
    """Gate for every threshold-fitting/selection entry point."""
    if split.name != "validation":
        raise HeldOutSplitAccessError(f"threshold fitting accepts only the validation split, got {split.name!r}")
    return split


def as_threshold(value: float) -> np.float32:
    """Thresholds are compared in float32, the dtype of the model's outputs."""
    return np.float32(value)


def predict(scores: np.ndarray, thresholds) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if scores.ndim != 2 or thresholds.shape != (scores.shape[1],):
        raise ValueError(f"thresholds shape {thresholds.shape} does not match scores {scores.shape}")
    return (scores.astype(np.float32) >= thresholds[None, :]).astype(np.int8)
