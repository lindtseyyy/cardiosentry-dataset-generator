"""Pure helpers for the kit-v1 -> clean12 cross-evaluation (no I/O except writing the summary files)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

CLEAN12_LABELS = ["NORM", "SB", "STACH", "LBBB", "LQT", "AF", "AFL", "3AVB", "2AVB", "1AVB", "STEMI", "STE"]
NORMAL_LABEL = "NORM"
PATHOLOGY_LABELS = [label for label in CLEAN12_LABELS if label != NORMAL_LABEL]

# kit-v1 output name -> clean12 label (the 12 shared conditions; the other 9 kit-v1 outputs are unused).
KITV1_TO_CLEAN12 = {
    "Atrial Fibrillation": "AF",
    "Atrial Flutter": "AFL",
    "Strict STEMI": "STEMI",
    "ST elevation": "STE",
    "LBBB": "LBBB",
    "First-degree AV block": "1AVB",
    "2AVB": "2AVB",
    "Third-degree / complete AV block": "3AVB",
    "QT/QTc prolongation": "LQT",
    "Sinus bradycardia": "SB",
    "Sinus tachycardia": "STACH",
    "Normal ECG": "NORM",
}
CLEAN12_TO_KITV1 = {clean: kit for kit, clean in KITV1_TO_CLEAN12.items()}
MACRO_ROW = "MACRO_11_PATHOLOGIES"


def split_sha256(patient_split: pd.Series) -> str:
    """Canonical identity of a patient -> split assignment (as the clean12 training notebook)."""
    lines = [f"{key},{value}" for key, value in sorted(patient_split.items())]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def build_clean12_frame(manifest_df: pd.DataFrame, record_df: pd.DataFrame,
                        labels=CLEAN12_LABELS) -> pd.DataFrame:
    """Join the render manifest to the record manifest exactly as the clean12 training notebook does."""
    manifest_df = manifest_df.copy()
    record_df = record_df.copy()
    for column in ("source_header", "image_path", "status"):
        if column not in manifest_df.columns:
            raise ValueError(f"manifest.csv is missing {column}")
    for column in ("record_path", "site", "patient_key", "labels", *labels):
        if column not in record_df.columns:
            raise ValueError(f"record_manifest.csv is missing {column}")
    record_df["patient_key"] = record_df["patient_key"].astype(str).str.strip()
    if record_df["patient_key"].eq("").any():
        raise ValueError("record_manifest.csv has records without a patient_key")
    for label in labels:
        values = pd.to_numeric(record_df[label], errors="raise")
        if not values.isin([0, 1]).all():
            raise ValueError(f"record_manifest.csv column {label} is not 0/1")
        record_df[label] = values.astype(np.int8)
    manifest_df["status"] = manifest_df["status"].astype(str).str.strip().str.lower()
    usable = manifest_df[manifest_df["status"].isin(["success", "skipped"])]
    if usable["source_header"].duplicated().any():
        raise ValueError("manifest.csv contains duplicate source_header values")
    joined = usable.merge(record_df, left_on="source_header", right_on="record_path",
                          how="inner", suffixes=("", "_record"))
    joined = joined[joined[list(labels)].sum(axis=1) > 0].reset_index(drop=True)
    joined["image_path"] = joined["image_path"].astype(str)
    return joined


def assign_split(frame: pd.DataFrame, patient_split: pd.Series) -> pd.DataFrame:
    missing = set(frame["patient_key"]) - set(patient_split.index)
    if missing:
        raise ValueError(f"{len(missing):,} patients have no split, e.g. {sorted(missing)[:3]}")
    return frame.assign(split=frame["patient_key"].map(patient_split))


def exclude_kitv1_patients(validation_df: pd.DataFrame, kitv1_records: pd.DataFrame,
                           kitv1_splits=("train",)) -> tuple[pd.DataFrame, dict]:
    """Drop validation records whose patient_key is in the given kit-v1 split(s)."""
    for column in ("patient_key", "split"):
        if column not in kitv1_records.columns:
            raise ValueError(f"kit-v1 record_manifest.csv is missing {column}")
    keys = kitv1_records["patient_key"].astype(str).str.strip()
    excluded_keys = set(keys[kitv1_records["split"].astype(str).isin(kitv1_splits)])
    mask = validation_df["patient_key"].astype(str).isin(excluded_keys)
    info = {
        "validation_records": int(len(validation_df)),
        "validation_patients": int(validation_df["patient_key"].nunique()),
        "excluded_records": int(mask.sum()),
        "excluded_patients": int(validation_df.loc[mask, "patient_key"].nunique()),
        "kept_records": int((~mask).sum()),
        "kept_patients": int(validation_df.loc[~mask, "patient_key"].nunique()),
        "excluded_kitv1_splits": list(kitv1_splits),
    }
    # Informational: overlap with every kit-v1 split, and the site-qualified count.
    for split in ("train", "validation", "test"):
        split_keys = set(keys[kitv1_records["split"].astype(str).eq(split)])
        info[f"records_with_patient_in_kitv1_{split}"] = int(validation_df["patient_key"].isin(split_keys).sum())
    if "site" in kitv1_records.columns and "site" in validation_df.columns:
        pairs = set(zip(kitv1_records["site"].astype(str),
                        keys.where(kitv1_records["split"].astype(str).isin(kitv1_splits))))
        info["excluded_records_site_qualified"] = int(sum(
            (site, key) in pairs for site, key in zip(validation_df["site"].astype(str),
                                                      validation_df["patient_key"].astype(str))))
    return validation_df.loc[~mask].reset_index(drop=True), info


def kitv1_columns_for_clean12(kitv1_label_order) -> list[int]:
    """Index of each clean12 label's kit-v1 output, in CLEAN12_LABELS order."""
    order = list(kitv1_label_order)
    missing = [name for name in KITV1_TO_CLEAN12 if name not in order]
    if missing:
        raise ValueError(f"kit-v1 label order lacks {missing}")
    return [order.index(CLEAN12_TO_KITV1[label]) for label in CLEAN12_LABELS]


def ap_summary(targets: np.ndarray, kitv1_probability: np.ndarray, clean12_probability: np.ndarray,
               labels=CLEAN12_LABELS) -> pd.DataFrame:
    """Per-label AP for both models (columns in `labels` order) plus the 11-pathology macro row."""
    targets = np.asarray(targets)
    if not (targets.shape == kitv1_probability.shape == clean12_probability.shape):
        raise ValueError("targets and probabilities must have the same shape")
    rows = []
    for column, label in enumerate(labels):
        y = targets[:, column].astype(int)
        positives = int(y.sum())
        if positives == 0 or positives == len(y):
            raise ValueError(f"{label}: AP is undefined with {positives} positives in {len(y)} records")
        kit_ap = float(average_precision_score(y, kitv1_probability[:, column]))
        clean_ap = float(average_precision_score(y, clean12_probability[:, column]))
        rows.append({"label": label, "kitv1_output": CLEAN12_TO_KITV1[label], "records": int(len(y)),
                     "positives": positives, "prevalence": positives / len(y),
                     "kitv1_ap": kit_ap, "clean12_ap": clean_ap, "ap_difference": clean_ap - kit_ap})
    table = pd.DataFrame(rows)
    pathology = table[table["label"].ne(NORMAL_LABEL)]
    if len(pathology) != 11:
        raise ValueError(f"expected 11 pathology labels, found {len(pathology)}")
    macro = {"label": MACRO_ROW, "kitv1_output": "", "records": int(len(targets)), "positives": np.nan,
             "prevalence": np.nan, "kitv1_ap": pathology["kitv1_ap"].mean(),
             "clean12_ap": pathology["clean12_ap"].mean()}
    macro["ap_difference"] = macro["clean12_ap"] - macro["kitv1_ap"]
    summary = pd.concat([table, pd.DataFrame([macro])], ignore_index=True)
    summary["positives"] = summary["positives"].astype("Int64")      # blank, not 2.0, in the CSV
    return summary


def select_checkpoint_weights(checkpoint: dict) -> tuple[dict, dict]:
    """The state dict that training selected on validation (EMA or raw), plus provenance."""
    source = checkpoint.get("selected_weight_source")
    if source not in ("raw", "ema"):
        raise ValueError(f"checkpoint has no valid selected_weight_source: {source!r}")
    state = checkpoint["model_ema" if source == "ema" else "model"]
    prefix = "_orig_mod."
    if state and all(key.startswith(prefix) for key in state):
        state = {key[len(prefix):]: value for key, value in state.items()}
    identity = (checkpoint.get("config") or {}).get("identity", {})
    info = {
        "selected_weight_source": source,
        "epoch": checkpoint.get("epoch"),
        "validation_pathology_only_macro_average_precision":
            checkpoint.get("validation_pathology_only_macro_average_precision"),
        "config_fingerprint": checkpoint.get("config_fingerprint"),
        "model": identity.get("model"),
        "label_order": identity.get("label_order"),
    }
    return state, info


def save_summary_file(data: bytes, destination, overwrite: bool) -> Path:
    """Write one summary file (e.g. to the mounted Google Drive) without ever replacing an
    existing result unless `overwrite`. Written to *.partial, size-checked, then renamed."""
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; set CS_OVERWRITE_OUTPUT=1 to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.write_bytes(data)
    if partial.stat().st_size != len(data):
        raise IOError(f"{partial}: wrote {partial.stat().st_size} of {len(data)} bytes")
    os.replace(partial, destination)
    return destination
