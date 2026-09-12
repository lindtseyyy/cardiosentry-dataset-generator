"""Generate one synthetic paper ECG image per FCMAE manifest record.

The clinical layout is rendered by the locally patched ecg-image-kit. Every
rendering parameter is fixed across the corpus; the only image-level visual
choice is the ECG grid colour. Grid palettes are assigned deterministically
from a seeded hash of the anonymous FCMAE record identifier, without consulting
source, diagnosis, or morphology fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import wfdb
from PIL import Image
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from scipy.stats import chi2_contingency


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset-files" / "fcmae-pretraining-v1-standalone"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "experiments" / "fcmae_synthetic_images_diagnostic_v1" / "full"
)
DEFAULT_KIT_ROOT = (
    PROJECT_ROOT
    / "dataset-combination-code"
    / "ecg_dataset_pipeline"
    / "ecg-image-kit"
)
DEFAULT_KIT_PY = Path.home() / ".cache" / "ecgkit-venv" / "bin" / "python"

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
WINDOW_SECONDS = 10.0
DIAGNOSTIC_BAND_HZ = (0.05, 150.0)
PTB_FILTER_ORDER = 2
ADC_GAIN = 1000.0
DPI = 600
PAPER_SPEED_MM_PER_SECOND = 25.0
IMAGE_HORIZONTAL_SAMPLE_RATE_HZ = DPI * PAPER_SPEED_MM_PER_SECOND / 25.4
IMAGE_HORIZONTAL_NYQUIST_HZ = IMAGE_HORIZONTAL_SAMPLE_RATE_HZ / 2.0
MAX_ABS_MV = 20.0
# Frozen after testing the eligible 24,942-record corpus.  Assignment still
# uses only the anonymous record ID, but this seed also passes the post-hoc
# source and morphology independence checks recorded in grid_palette_audit.json.
GRID_SEED = 20260924
SEGMENT_SEED = 20260908
FIXED_FONT = "Times_New_Roman.ttf"
ROW_GAP = 4.4
RHYTHM_GAP = 5.5
BOTTOM_RESERVE = 1.9

EXPECTED_NATIVE_FS = {
    "cpsc_2018": 500.0,
    "cpsc_2018_extra": 500.0,
    "st_petersburg_incart": 257.0,
    "ptb": 1000.0,
    "ptb-xl": 500.0,
    "georgia": 500.0,
    "chapman_shaoxing_ningbo": 500.0,
    "sphdb": 500.0,
}

# "Source filter" means filtering already embodied in the distributed bytes.
# A missing cutoff is deliberately null: sampling frequency is not evidence of
# an acquisition filter, and a filter used in a later research paper is not
# automatically a property of the released waveform.
SOURCE_SIGNAL_POLICY = {
    "cpsc_2018": {
        "source_signal_variant": "PhysioNet Challenge 2020 native 500 Hz MAT",
        "source_filter_status": "not_documented_in_distribution",
        "source_filter_low_hz": None,
        "source_filter_high_hz": None,
        "source_filter_notch_hz": None,
        "source_filter_description": "No acquisition cutoff is declared in the supplied WFDB header or Challenge source description.",
        "source_filter_reference": "https://physionet.org/content/challenge-2020/1.0.2/",
        "highest_fidelity_assessment": "highest sampling-rate waveform in the specified local Challenge distribution",
    },
    "cpsc_2018_extra": {
        "source_signal_variant": "PhysioNet Challenge 2020 native 500 Hz MAT",
        "source_filter_status": "not_documented_in_distribution",
        "source_filter_low_hz": None,
        "source_filter_high_hz": None,
        "source_filter_notch_hz": None,
        "source_filter_description": "No acquisition cutoff is declared in the supplied WFDB header or Challenge source description.",
        "source_filter_reference": "https://physionet.org/content/challenge-2020/1.0.2/",
        "highest_fidelity_assessment": "highest sampling-rate waveform in the specified local Challenge distribution",
    },
    "st_petersburg_incart": {
        "source_signal_variant": "PhysioNet Challenge 2020 native 257 Hz MAT",
        "source_filter_status": "not_documented_in_distribution",
        "source_filter_low_hz": None,
        "source_filter_high_hz": None,
        "source_filter_notch_hz": None,
        "source_filter_description": "No acquisition cutoff is declared in the supplied WFDB header or PhysioNet source description; native Nyquist is 128.5 Hz.",
        "source_filter_reference": "https://physionet.org/content/incartdb/1.0.0/",
        "highest_fidelity_assessment": "native 257 Hz source retained; 150 Hz cannot be reconstructed above its 128.5 Hz Nyquist limit",
    },
    "ptb": {
        "source_signal_variant": "PhysioNet Challenge 2020 12-lead 1000 Hz MAT",
        "source_filter_status": "documented_source_hardware_bandwidth",
        "source_filter_low_hz": 0.0,
        "source_filter_high_hz": 1000.0,
        "source_filter_notch_hz": None,
        "source_filter_description": "Upstream PTB recorder bandwidth is documented as 0-1 kHz; the specified Challenge copy retains 1000 Hz sampling and 1 uV/LSB calibration for the 12 conventional leads.",
        "source_filter_reference": "https://physionet.org/content/ptbdb/1.0.0/",
        "highest_fidelity_assessment": "highest temporal-resolution 12-lead waveform in the specified local Challenge distribution; upstream PTBDB documents 0.5 uV/LSB and three additional Frank leads",
    },
    "ptb-xl": {
        "source_signal_variant": "PTB-XL high-resolution records500 waveform via PhysioNet Challenge 2020",
        "source_filter_status": "not_documented_in_distribution",
        "source_filter_low_hz": None,
        "source_filter_high_hz": None,
        "source_filter_notch_hz": None,
        "source_filter_description": "The 500 Hz, 1 uV/LSB high-resolution version is used; no acquisition cutoff is declared in its WFDB header or PhysioNet description.",
        "source_filter_reference": "https://physionet.org/content/ptb-xl/1.0.3/",
        "highest_fidelity_assessment": "500 Hz records500 variant retained, not the 100 Hz convenience copy",
    },
    "georgia": {
        "source_signal_variant": "PhysioNet Challenge 2020 native 500 Hz MAT",
        "source_filter_status": "not_documented_in_distribution",
        "source_filter_low_hz": None,
        "source_filter_high_hz": None,
        "source_filter_notch_hz": None,
        "source_filter_description": "No acquisition cutoff is declared in the supplied WFDB header or Challenge source description.",
        "source_filter_reference": "https://physionet.org/content/challenge-2020/1.0.2/",
        "highest_fidelity_assessment": "highest sampling-rate waveform in the specified local Challenge distribution",
    },
    "chapman_shaoxing_ningbo": {
        "source_signal_variant": "PhysioNet ecg-arrhythmia native 500 Hz WFDB waveform",
        "source_filter_status": "not_documented_in_distribution",
        "source_filter_low_hz": None,
        "source_filter_high_hz": None,
        "source_filter_notch_hz": None,
        "source_filter_description": "The release documents 500 Hz acquisition but does not declare the distributed waveform's acquisition cutoffs.",
        "source_filter_reference": "https://physionet.org/content/ecg-arrhythmia/1.0.0/",
        "highest_fidelity_assessment": "native 500 Hz WFDB waveform retained without research-pipeline denoising",
    },
    "sphdb": {
        "source_signal_variant": "SPHDB native 500 Hz float16 HDF5 waveform",
        "source_filter_status": "documented_machine_filtered_cutoffs_not_reported",
        "source_filter_low_hz": None,
        "source_filter_high_hz": None,
        "source_filter_notch_hz": None,
        "source_filter_description": "The MedEx MECG-200 removed power-line interference, baseline wander, and muscle-contraction noise before export; numerical cutoffs were not reported.",
        "source_filter_reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9174207/",
        "highest_fidelity_assessment": "native distributed 500 Hz HDF5 waveform retained; no second filtering pass",
    },
}

assert IMAGE_HORIZONTAL_NYQUIST_HZ > DIAGNOSTIC_BAND_HZ[1]

# Only major/minor grid ink changes. Paper and trace endpoints are identical
# for every ECG, and there is no saturation, exposure, noise, blur, rotation,
# texture, crease, crop, perspective, or JPEG variation.
GRID_PALETTES = {
    "red": ((178, 86, 84), (228, 180, 176)),
    "pink": ((188, 124, 148), (236, 200, 212)),
    "orange": ((186, 126, 86), (236, 202, 176)),
    "grey": ((124, 124, 124), (198, 198, 198)),
}
PALETTE_NAMES = tuple(GRID_PALETTES)
TRACE_RGB = (18, 18, 18)
PAPER_RGB = (252, 252, 252)

MORPHOLOGY_COLUMNS = [
    "morphology_normal_baseline",
    "morphology_ischemic_stt",
    "morphology_atrial_rhythm",
    "morphology_structural_voltage",
    "morphology_conduction_other",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _digest(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()


def load_generation_manifest(
    dataset_root: Path,
    grid_seed: int = GRID_SEED,
    eligible_record_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load the curated release and assign anonymous IDs plus balanced palettes.

    Palette assignment uses only ``fcmae_record_id`` and ``grid_seed``. Records
    are sorted by their cryptographic digest and assigned round-robin, making
    global palette counts differ by at most one without using source or
    morphology metadata.
    """
    manifest_path = dataset_root / "manifest.csv"
    df = pd.read_csv(manifest_path, low_memory=False)
    required = {
        "fcmae_record_id",
        "source_dataset",
        "signal_format",
        "signal_ref_path",
        "data_ref_path",
        "sampling_frequency",
        "lead_names",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")
    if df["fcmae_record_id"].duplicated().any():
        dup = df.loc[df["fcmae_record_id"].duplicated(), "fcmae_record_id"].iloc[0]
        raise ValueError(f"duplicate fcmae_record_id: {dup}")

    df = df.copy()
    if eligible_record_ids is not None:
        eligible = {str(value) for value in eligible_record_ids}
        df = df[df["fcmae_record_id"].astype(str).isin(eligible)].copy()
        missing_eligible = sorted(eligible - set(df["fcmae_record_id"].astype(str)))
        if missing_eligible:
            raise KeyError(f"eligible record IDs absent from manifest: {missing_eligible[:5]}")
    if df.empty:
        raise ValueError("no fidelity-qualified records remain")
    df.insert(0, "image_id", [f"ECG{i:06d}" for i in range(1, len(df) + 1)])
    ranked = sorted(
        range(len(df)),
        key=lambda i: _digest(grid_seed, str(df.iloc[i]["fcmae_record_id"])),
    )
    assigned = np.empty(len(df), dtype=object)
    for rank, row_index in enumerate(ranked):
        assigned[row_index] = PALETTE_NAMES[rank % len(PALETTE_NAMES)]
    df["grid_palette"] = assigned
    return df


def _cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    values = table.to_numpy()
    if values.size == 0 or min(values.shape) < 2 or values.sum() == 0:
        return 0.0, 1.0
    chi2, p_value, _, _ = chi2_contingency(values)
    denom = values.sum() * min(values.shape[0] - 1, values.shape[1] - 1)
    return float(math.sqrt(chi2 / denom)) if denom else 0.0, float(p_value)


def palette_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return count tables and association statistics for palette auditing."""
    source_table = pd.crosstab(df["source_dataset"], df["grid_palette"]).reindex(
        columns=PALETTE_NAMES, fill_value=0
    )
    source_v, source_p = _cramers_v(source_table)
    tests = [{
        "field": "source_dataset",
        "cramers_v": source_v,
        "chi_square_p": source_p,
    }]
    for column in MORPHOLOGY_COLUMNS:
        if column not in df:
            continue
        table = pd.crosstab(df[column].fillna(0).astype(int), df["grid_palette"]).reindex(
            columns=PALETTE_NAMES, fill_value=0
        )
        value, p_value = _cramers_v(table)
        tests.append({"field": column, "cramers_v": value, "chi_square_p": p_value})

    audit = {
        "assignment_inputs": ["fcmae_record_id", "grid_seed"],
        "grid_seed": int(GRID_SEED),
        "palette_counts": {k: int(v) for k, v in df["grid_palette"].value_counts().items()},
        "association_tests": tests,
        "interpretation": (
            "Palette is assigned without reading source or morphology. "
            "Cramer's V is reported only as a finite-sample audit, not as an assignment rule."
        ),
    }
    return source_table, audit


def _canonical_indices(names: Iterable[str]) -> list[int]:
    lookup = {str(name).strip().upper(): i for i, name in enumerate(names)}
    missing = [lead for lead in LEADS if lead.upper() not in lookup]
    if missing:
        raise ValueError(f"missing canonical leads: {missing}")
    return [lookup[lead.upper()] for lead in LEADS]


def _segment_start(record_id: str, n_samples: int, wanted: int) -> int:
    if n_samples <= wanted:
        return 0
    value = int.from_bytes(_digest(SEGMENT_SEED, record_id)[:8], "big")
    return value % (n_samples - wanted + 1)


def source_fidelity_policy_table() -> pd.DataFrame:
    rows = []
    for source, policy in SOURCE_SIGNAL_POLICY.items():
        row = {"source_dataset": source, "expected_native_sampling_frequency_hz": EXPECTED_NATIVE_FS[source]}
        row.update(policy)
        if source == "ptb":
            row.update({
                "applied_filter_type": "Butterworth band-pass",
                "applied_filter_low_hz": DIAGNOSTIC_BAND_HZ[0],
                "applied_filter_high_hz": DIAGNOSTIC_BAND_HZ[1],
                "applied_filter_order_per_pass": PTB_FILTER_ORDER,
                "applied_filter_phase": "zero-phase forward-backward",
                "applied_filter_reason": "limit the documented 0-1 kHz PTB source to the adult diagnostic image band",
            })
        else:
            row.update({
                "applied_filter_type": "none",
                "applied_filter_low_hz": None,
                "applied_filter_high_hz": None,
                "applied_filter_order_per_pass": None,
                "applied_filter_phase": "not_applicable",
                "applied_filter_reason": "preserve the distributed samples; do not invent or repeat undocumented source preprocessing",
            })
        rows.append(row)
    return pd.DataFrame(rows)


_GAIN_RE = re.compile(
    r"^(?P<gain>[0-9.+-]+)(?:\((?P<baseline>-?[0-9]+)\))?/(?P<unit>\S+)$"
)


def _read_wfdb_mat_for_audit(row: dict, dataset_root: Path) -> tuple[np.ndarray, dict]:
    """Decode the selected native MAT samples without passing through WFDB.

    Direct decoding makes the full preflight audit substantially faster while
    retaining the calibration and missing-value rules declared by each header.
    """
    header_path = dataset_root / str(row["header_ref_path"])
    lines = header_path.read_text(errors="replace").splitlines()
    if not lines:
        raise ValueError("empty WFDB header")
    first = lines[0].split()
    n_sig = int(first[1])
    fs = float(first[2].split("/")[0])
    n_samples = int(first[3])
    signal_lines = lines[1:1 + n_sig]
    if len(signal_lines) != 12:
        raise ValueError(f"expected 12 signal lines, found {len(signal_lines)}")

    names, gains, baselines, formats, units = [], [], [], [], []
    for line in signal_lines:
        parts = line.split()
        formats.append(parts[1].split("x")[0].split(":")[0].split("+")[0])
        match = _GAIN_RE.match(parts[2])
        if not match:
            raise ValueError(f"cannot parse gain token {parts[2]!r}")
        gains.append(float(match.group("gain")))
        baselines.append(float(match.group("baseline") or 0))
        units.append(match.group("unit"))
        names.append(parts[-1])

    indices = _canonical_indices(names)
    digital = np.asarray(
        loadmat(dataset_root / str(row["data_ref_path"]), variable_names=["val"])["val"]
    )
    if digital.shape[0] != 12 and digital.shape[1] == 12:
        digital = digital.T
    if digital.shape[0] != 12 or digital.shape[1] != n_samples:
        raise ValueError(f"MAT/header shape mismatch: {digital.shape} vs 12x{n_samples}")
    digital = digital[indices]
    gains_a = np.asarray(gains, dtype=np.float64)[indices, None]
    baselines_a = np.asarray(baselines, dtype=np.float64)[indices, None]

    wanted = int(round(WINDOW_SECONDS * fs))
    start = _segment_start(str(row["fcmae_record_id"]), n_samples, wanted)
    stop = min(n_samples, start + wanted)
    selected_digital = digital[:, start:stop].T
    missing = selected_digital == -32768
    positive_rail = selected_digital == 32767
    physical = (selected_digital.astype(np.float64) - baselines_a.T) / gains_a.T
    physical[missing] = np.nan
    details = {
        "native_container_dtype": str(digital.dtype),
        "native_wfdb_formats": "|".join(sorted(set(formats))),
        "native_units": "|".join(sorted(set(units))),
        "native_adc_gain_min": float(np.min(gains_a)),
        "native_adc_gain_max": float(np.max(gains_a)),
        "missing_sentinel_samples": int(missing.sum()),
        "positive_rail_samples": int(positive_rail.sum()),
    }
    return physical, details


def _read_hdf5_for_audit(row: dict, dataset_root: Path) -> tuple[np.ndarray, dict]:
    path = dataset_root / str(row["data_ref_path"])
    fs = float(row["sampling_frequency"])
    wanted = int(round(WINDOW_SECONDS * fs))
    with h5py.File(path, "r") as fh:
        if "ecg" not in fh:
            raise KeyError("HDF5 dataset 'ecg' missing")
        ds = fh["ecg"]
        dtype = str(ds.dtype)
        if ds.ndim != 2 or 12 not in ds.shape:
            raise ValueError(f"unexpected HDF5 shape {ds.shape}")
        n_samples = int(ds.shape[1] if ds.shape[0] == 12 else ds.shape[0])
        start = _segment_start(str(row["fcmae_record_id"]), n_samples, wanted)
        stop = min(n_samples, start + wanted)
        if ds.shape[0] == 12:
            signal = np.asarray(ds[:, start:stop], dtype=np.float64).T
        else:
            signal = np.asarray(ds[start:stop, :], dtype=np.float64)
    names = str(row.get("lead_names") or "").split("|")
    signal = signal[:, _canonical_indices(names)]
    return signal, {
        "native_container_dtype": dtype,
        "native_wfdb_formats": "not_applicable",
        "native_units": "mV",
        "native_adc_gain_min": None,
        "native_adc_gain_max": None,
        "missing_sentinel_samples": 0,
        "positive_rail_samples": 0,
    }


def _audit_one_record(row: dict, dataset_root: Path) -> dict:
    base = {
        "fcmae_record_id": str(row["fcmae_record_id"]),
        "source_dataset": str(row["source_dataset"]),
        "signal_format": str(row["signal_format"]),
        "native_sampling_frequency_hz": float(row["sampling_frequency"]),
        "native_num_samples": int(row["num_samples"]),
        "native_duration_seconds": float(row["duration_seconds"]),
    }
    try:
        if str(row["signal_format"]) == "wfdb_mat":
            signal, details = _read_wfdb_mat_for_audit(row, dataset_root)
        elif str(row["signal_format"]) == "hdf5":
            signal, details = _read_hdf5_for_audit(row, dataset_root)
        else:
            raise ValueError(f"unsupported signal format {row['signal_format']!r}")
        base.update(details)
        base["audited_segment_samples"] = int(signal.shape[0])
        base["blank_padding_samples"] = max(
            0,
            int(round(WINDOW_SECONDS * float(row["sampling_frequency"]))) - signal.shape[0],
        )
        finite = np.isfinite(signal)
        base["nonfinite_samples"] = int(signal.size - finite.sum())

        flat_leads = 0
        max_abs = 0.0
        min_p2p = math.inf
        for lead in range(12):
            values = signal[:, lead]
            values = values[np.isfinite(values)]
            if values.size < 2:
                flat_leads += 1
                continue
            p2p = float(np.ptp(values))
            flat_leads += int(p2p == 0.0)
            min_p2p = min(min_p2p, p2p)
            max_abs = max(max_abs, float(np.max(np.abs(values))))
        base["flat_leads"] = int(flat_leads)
        base["minimum_lead_peak_to_peak_mv"] = None if math.isinf(min_p2p) else min_p2p
        base["maximum_absolute_mv"] = max_abs

        reasons = []
        if base["nonfinite_samples"]:
            reasons.append("NONFINITE_OR_MISSING_SAMPLES")
        if base["positive_rail_samples"]:
            reasons.append("POSITIVE_ADC_RAIL")
        if flat_leads:
            reasons.append("FLAT_OR_EMPTY_LEAD")
        if max_abs > MAX_ABS_MV:
            reasons.append("EXTREME_ABSOLUTE_VOLTAGE_OVER_20_MV")
        expected_fs = EXPECTED_NATIVE_FS.get(base["source_dataset"])
        if expected_fs is None or not math.isclose(base["native_sampling_frequency_hz"], expected_fs):
            reasons.append("UNEXPECTED_NATIVE_SAMPLING_FREQUENCY")
        base["fidelity_eligible"] = int(not reasons)
        base["fidelity_exclusion_reasons"] = "|".join(reasons)
        base["audit_error"] = ""
    except Exception as exc:
        base.update({
            "fidelity_eligible": 0,
            "fidelity_exclusion_reasons": "AUDIT_DECODE_ERROR",
            "audit_error": f"{type(exc).__name__}: {exc}",
        })
    return base


def audit_waveform_fidelity(
    dataset_root: Path,
    *,
    workers: int = 12,
) -> tuple[pd.DataFrame, dict]:
    """Audit the exact deterministic render segment of every manifest record."""
    dataset_root = Path(dataset_root).resolve()
    raw = pd.read_csv(dataset_root / "manifest.csv", low_memory=False)
    records = raw.to_dict("records")
    worker_count = max(1, min(int(workers), len(records)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        audited = list(pool.map(lambda row: _audit_one_record(row, dataset_root), records))
    audit = pd.DataFrame(audited)
    reason_counts: dict[str, int] = {}
    for value in audit["fidelity_exclusion_reasons"].fillna(""):
        for reason in filter(None, str(value).split("|")):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    source_rows = []
    for source, group in audit.groupby("source_dataset", sort=False):
        policy = SOURCE_SIGNAL_POLICY[source]
        finite_gains = [
            float(value)
            for value in group["native_adc_gain_min"].dropna().unique()
        ]
        source_rows.append({
            "source_dataset": source,
            "records_audited": int(len(group)),
            "records_eligible": int(group["fidelity_eligible"].sum()),
            "records_excluded": int((group["fidelity_eligible"] == 0).sum()),
            "native_sampling_frequencies_hz": sorted(
                float(value) for value in group["native_sampling_frequency_hz"].unique()
            ),
            "native_container_dtypes": sorted(
                str(value) for value in group["native_container_dtype"].dropna().unique()
            ),
            "native_wfdb_formats": sorted(
                str(value) for value in group["native_wfdb_formats"].dropna().unique()
            ),
            "native_adc_gains_adu_per_mv": sorted(finite_gains),
            "eligible_maximum_absolute_mv": float(
                group.loc[group["fidelity_eligible"] == 1, "maximum_absolute_mv"].max()
            ),
            "records_requiring_blank_padding": int((group["blank_padding_samples"].fillna(0) > 0).sum()),
            "source_filter_status": policy["source_filter_status"],
            "source_filter_low_hz": policy["source_filter_low_hz"],
            "source_filter_high_hz": policy["source_filter_high_hz"],
            "source_filter_notch_hz": policy["source_filter_notch_hz"],
            "applied_filter_type": "Butterworth band-pass" if source == "ptb" else "none",
            "applied_filter_low_hz": DIAGNOSTIC_BAND_HZ[0] if source == "ptb" else None,
            "applied_filter_high_hz": DIAGNOSTIC_BAND_HZ[1] if source == "ptb" else None,
        })
    summary = {
        "policy": "diagnostic_fidelity_v1",
        "raw_corpus_construction": {
            "waveform_decode_or_reencode": False,
            "resampling": False,
            "digital_filtering": False,
            "normalization": False,
            "amplitude_clipping": False,
            "standalone_materialization": "physical copies of the selected native source/header bytes",
            "integrity": "manifest data/header SHA-256 plus release checksums.sha256",
        },
        "records_audited": int(len(audit)),
        "records_eligible": int(audit["fidelity_eligible"].sum()),
        "records_excluded": int((audit["fidelity_eligible"] == 0).sum()),
        "exclusion_reason_counts_nonexclusive": reason_counts,
        "quality_thresholds": {
            "maximum_absolute_mv": MAX_ABS_MV,
            "flat_leads_allowed": 0,
            "missing_or_nonfinite_samples_allowed": 0,
            "positive_adc_rail_samples_allowed": 0,
        },
        "segment_policy": {
            "seconds": WINDOW_SECONDS,
            "selection": "seeded hash offset for records longer than 10 seconds",
            "short_record_handling": "preserve available samples and render unavailable tail as a blank gap",
            "segment_seed": SEGMENT_SEED,
        },
        "rendering_policy": {
            "native_sampling_frequencies_preserved": True,
            "default_applied_filter": "none",
            "ptb_applied_filter_hz": list(DIAGNOSTIC_BAND_HZ),
            "median_or_baseline_centering": False,
            "amplitude_normalization": False,
            "amplitude_clipping": False,
            "dpi": DPI,
            "paper_speed_mm_per_second": PAPER_SPEED_MM_PER_SECOND,
            "image_horizontal_sample_rate_hz": IMAGE_HORIZONTAL_SAMPLE_RATE_HZ,
            "image_horizontal_nyquist_hz": IMAGE_HORIZONTAL_NYQUIST_HZ,
        },
        "source_summary": source_rows,
    }
    return audit, summary


def _read_wfdb_segment(row: dict, dataset_root: Path) -> tuple[np.ndarray, float, int]:
    base = dataset_root / str(row["signal_ref_path"])
    header = wfdb.rdheader(str(base))
    fs = float(header.fs)
    wanted = int(round(WINDOW_SECONDS * fs))
    start = _segment_start(str(row["fcmae_record_id"]), int(header.sig_len), wanted)
    stop = min(int(header.sig_len), start + wanted)
    indices = _canonical_indices(header.sig_name)
    rec = wfdb.rdrecord(str(base), sampfrom=start, sampto=stop, channels=indices)
    return np.asarray(rec.p_signal, dtype=np.float64), fs, start


def _read_wfdb_full(row: dict, dataset_root: Path) -> tuple[np.ndarray, float, int]:
    base = dataset_root / str(row["signal_ref_path"])
    header = wfdb.rdheader(str(base))
    fs = float(header.fs)
    wanted = int(round(WINDOW_SECONDS * fs))
    start = _segment_start(str(row["fcmae_record_id"]), int(header.sig_len), wanted)
    indices = _canonical_indices(header.sig_name)
    rec = wfdb.rdrecord(str(base), channels=indices)
    return np.asarray(rec.p_signal, dtype=np.float64), fs, start


def _read_hdf5_segment(row: dict, dataset_root: Path) -> tuple[np.ndarray, float, int]:
    path = dataset_root / str(row["data_ref_path"])
    fs = float(row["sampling_frequency"])
    wanted = int(round(WINDOW_SECONDS * fs))
    with h5py.File(path, "r") as fh:
        if "ecg" not in fh:
            raise KeyError(f"{path}: expected HDF5 dataset 'ecg'")
        ds = fh["ecg"]
        if ds.ndim != 2:
            raise ValueError(f"{path}: expected 2-D ECG array, got {ds.shape}")
        if ds.shape[0] == 12:
            n_samples = int(ds.shape[1])
            start = _segment_start(str(row["fcmae_record_id"]), n_samples, wanted)
            stop = min(n_samples, start + wanted)
            signal = np.asarray(ds[:, start:stop], dtype=np.float64).T
        elif ds.shape[1] == 12:
            n_samples = int(ds.shape[0])
            start = _segment_start(str(row["fcmae_record_id"]), n_samples, wanted)
            stop = min(n_samples, start + wanted)
            signal = np.asarray(ds[start:stop, :], dtype=np.float64)
        else:
            raise ValueError(f"{path}: no 12-lead axis in {ds.shape}")

    names = str(row.get("lead_names") or "").split("|")
    signal = signal[:, _canonical_indices(names)]
    return signal, fs, start


def read_preserved_signal(row: dict, dataset_root: Path) -> tuple[np.ndarray, dict]:
    """Read a deterministic window without resampling, centering, or clipping.

    The sole digital filter introduced here is the documented 0.05-150 Hz PTB
    conditioning step. PTB's full record is filtered before window selection so
    the 0.05 Hz edge is not estimated from an isolated ten-second crop.
    """
    signal_format = str(row["signal_format"])
    source = str(row["source_dataset"])
    if signal_format == "wfdb_mat" and source == "ptb":
        full_signal, native_fs, start = _read_wfdb_full(row, dataset_root)
        if not np.isfinite(full_signal).all():
            raise ValueError("native PTB waveform contains missing or non-finite samples")
        nyquist = native_fs / 2.0
        sos = butter(
            PTB_FILTER_ORDER,
            [DIAGNOSTIC_BAND_HZ[0] / nyquist, DIAGNOSTIC_BAND_HZ[1] / nyquist],
            btype="band",
            output="sos",
        )
        full_signal = sosfiltfilt(sos, full_signal, axis=0)
        wanted = int(round(WINDOW_SECONDS * native_fs))
        signal = full_signal[start:start + wanted]
        applied_filter = {
            "applied_filter_type": "Butterworth band-pass",
            "applied_filter_low_hz": DIAGNOSTIC_BAND_HZ[0],
            "applied_filter_high_hz": DIAGNOSTIC_BAND_HZ[1],
            "applied_filter_order_per_pass": PTB_FILTER_ORDER,
            "applied_filter_phase": "zero-phase forward-backward",
            "applied_filter_implementation": "scipy.signal.sosfiltfilt",
        }
    elif signal_format == "wfdb_mat":
        signal, native_fs, start = _read_wfdb_segment(row, dataset_root)
        applied_filter = {
            "applied_filter_type": "none",
            "applied_filter_low_hz": None,
            "applied_filter_high_hz": None,
            "applied_filter_order_per_pass": None,
            "applied_filter_phase": "not_applicable",
            "applied_filter_implementation": "none",
        }
    elif signal_format == "hdf5":
        signal, native_fs, start = _read_hdf5_segment(row, dataset_root)
        applied_filter = {
            "applied_filter_type": "none",
            "applied_filter_low_hz": None,
            "applied_filter_high_hz": None,
            "applied_filter_order_per_pass": None,
            "applied_filter_phase": "not_applicable",
            "applied_filter_implementation": "none",
        }
    else:
        raise ValueError(f"unsupported signal_format={signal_format!r}")

    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError(f"expected samples x 12 leads, got {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError("native render segment contains missing or non-finite samples")
    max_abs = float(np.max(np.abs(signal)))
    if max_abs > MAX_ABS_MV:
        raise ValueError(f"native render segment exceeds fidelity QC limit: {max_abs:g} mV")

    wanted = int(round(WINDOW_SECONDS * native_fs))
    available = min(len(signal), wanted)
    signal = signal[:wanted]
    padding_samples = max(0, wanted - len(signal))
    if padding_samples:
        # A blank trace is honest about unavailable time. Repeating the last
        # voltage or synthesizing zeros would manufacture morphology.
        signal = np.pad(
            signal,
            ((0, padding_samples), (0, 0)),
            mode="constant",
            constant_values=np.nan,
        )

    source_policy = dict(SOURCE_SIGNAL_POLICY[source])
    source_high = source_policy.get("source_filter_high_hz")
    effective_high = min(
        native_fs / 2.0,
        float(source_high) if source_high is not None else native_fs / 2.0,
        float(applied_filter["applied_filter_high_hz"])
        if applied_filter["applied_filter_high_hz"] is not None
        else native_fs / 2.0,
    )
    meta = {
        "native_sampling_frequency_hz": float(native_fs),
        "native_nyquist_hz": float(native_fs / 2.0),
        "segment_start_sample": int(start),
        "segment_start_seconds": round(float(start / native_fs), 6),
        "available_segment_samples": int(available),
        "available_segment_seconds": round(float(available / native_fs), 6),
        "paper_window_seconds": WINDOW_SECONDS,
        "blank_padding_samples": int(padding_samples),
        "blank_padding_seconds": round(float(padding_samples / native_fs), 6),
        "output_sampling_frequency_hz": float(native_fs),
        "resampling_applied": 0,
        "baseline_centering_applied": 0,
        "amplitude_normalization_applied": 0,
        "amplitude_clipping_applied": 0,
        "native_segment_maximum_absolute_mv": max_abs,
        "effective_upper_frequency_bound_hz": effective_high,
    }
    meta.update(source_policy)
    meta.update(applied_filter)
    return signal, meta


def stage_record(row: dict, dataset_root: Path, staged_dir: Path) -> dict:
    signal, meta = read_preserved_signal(row, dataset_root)
    staged_dir.mkdir(parents=True, exist_ok=True)
    wfdb.wrsamp(
        record_name=str(row["image_id"]),
        fs=float(meta["output_sampling_frequency_hz"]),
        units=["mV"] * 12,
        sig_name=LEADS,
        p_signal=signal,
        fmt=["16"] * 12,
        adc_gain=[ADC_GAIN] * 12,
        baseline=[0] * 12,
        comments=[],
        write_dir=str(staged_dir),
    )
    staged = wfdb.rdrecord(str(staged_dir / str(row["image_id"]))).p_signal
    source_nan = np.isnan(signal)
    if not np.array_equal(source_nan, np.isnan(staged)):
        raise ValueError("staging changed the missing-sample mask")
    finite = ~source_nan
    max_error = float(np.max(np.abs(signal[finite] - staged[finite]))) if finite.any() else 0.0
    if max_error > (0.5 / ADC_GAIN + 1e-12):
        raise ValueError(f"staging quantization error {max_error:g} mV exceeds 0.5 uV")
    meta.update({
        "staging_wfdb_format": "16",
        "staging_adc_gain_adu_per_mv": ADC_GAIN,
        "staging_quantization_step_mv": 1.0 / ADC_GAIN,
        "staging_maximum_absolute_error_mv": max_error,
    })
    return meta


def _kit_generator_dir(kit_root: Path) -> Path:
    return kit_root / "codes" / "ecg-image-generator"


def verify_preflight(dataset_root: Path, kit_root: Path, kit_python: Path) -> dict:
    manifest = dataset_root / "manifest.csv"
    kit_dir = _kit_generator_dir(kit_root)
    required = [
        manifest,
        kit_python,
        kit_dir / "gen_ecg_images_from_data_batch.py",
        kit_dir / "Fonts" / FIXED_FONT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required paths:\n  " + "\n  ".join(missing))
    single = (kit_dir / "gen_ecg_image_from_data.py").read_text()
    if "PATCHED-FIXEDFONT" not in single:
        raise RuntimeError(
            "ecg-image-kit lacks the fixed-font patch; run image_pipeline/patch_kit.py"
        )
    manifest_df = pd.read_csv(manifest, low_memory=False)
    observed_sources = set(manifest_df["source_dataset"].astype(str))
    if observed_sources != set(EXPECTED_NATIVE_FS):
        raise ValueError(
            f"unexpected source set: {sorted(observed_sources)} != {sorted(EXPECTED_NATIVE_FS)}"
        )
    for source, expected_fs in EXPECTED_NATIVE_FS.items():
        observed_fs = sorted(
            float(value)
            for value in manifest_df.loc[
                manifest_df["source_dataset"] == source, "sampling_frequency"
            ].unique()
        )
        if observed_fs != [expected_fs]:
            raise ValueError(f"{source}: native frequency {observed_fs}, expected {expected_fs}")
    ptbxl_ids = manifest_df.loc[
        manifest_df["source_dataset"] == "ptb-xl", "original_record_id"
    ].astype(str)
    if not ptbxl_ids.str.startswith("HR").all():
        raise ValueError("PTB-XL manifest contains a non-HR record; 100 Hz copies are forbidden")
    return {
        "dataset_root": str(dataset_root.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "kit_root": str(kit_root.resolve()),
        "kit_python": str(kit_python.expanduser().absolute()),
        "header_printing": False,
        "fixed_font": FIXED_FONT,
        "raw_waveform_transcoding_in_dataset_build": False,
        "native_sampling_frequencies_preserved": True,
        "ptbxl_high_resolution_records500_only": True,
    }


def render_staged(staged_dir: Path, clean_dir: Path, kit_root: Path, kit_python: Path) -> None:
    kit_dir = _kit_generator_dir(kit_root)
    clean_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(kit_python),
        "gen_ecg_images_from_data_batch.py",
        "-i", str(staged_dir.resolve()),
        "-o", str(clean_dir.resolve()),
        "-se", str(GRID_SEED),
        "--num_columns", "4",
        "--full_mode", "II",
        "-r", str(DPI),
        "--pad_inches", "0",
        "--calibration_pulse", "1",
        "--random_grid_present", "1",
        "--random_bw", "1",
    ]
    env = dict(os.environ)
    env.update({
        "ECG_FIXED_FONT": FIXED_FONT,
        "ECG_ROW_GAP": str(ROW_GAP),
        "ECG_RHYTHM_GAP": str(RHYTHM_GAP),
        "ECG_BOTTOM_RESERVE": str(BOTTOM_RESERVE),
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "CUDA_VISIBLE_DEVICES": "",
        # One renderer per worker. Letting every subprocess also start a full
        # BLAS/OpenMP pool badly oversubscribes a 12-vCPU Colab runtime.
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    completed = subprocess.run(
        command,
        cwd=kit_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        message = (completed.stderr or completed.stdout or "unknown renderer error")[-3000:]
        raise RuntimeError(f"ecg-image-kit failed:\n{message}")


def _grid_lut(name: str) -> np.ndarray:
    major, minor = (np.asarray(c, dtype=np.float32) for c in GRID_PALETTES[name])
    trace = np.asarray(TRACE_RGB, dtype=np.float32)
    paper = np.asarray(PAPER_RGB, dtype=np.float32)
    xs = np.array([0, 55, 102, 191, 250, 255], dtype=np.float32)
    grid = np.arange(256, dtype=np.float32)
    lut = np.empty((256, 3), dtype=np.float32)
    for channel in range(3):
        ys = np.array(
            [0, trace[channel], major[channel], minor[channel], paper[channel], 255],
            dtype=np.float32,
        )
        lut[:, channel] = np.interp(grid, xs, ys)
    return np.clip(lut, 0, 255).astype(np.uint8)


def recolour_grid(clean_png: Path, output_png: Path, palette: str) -> None:
    grey = np.asarray(Image.open(clean_png).convert("L"))
    rgb = _grid_lut(palette)[grey]
    output_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, "RGB").save(output_png, "PNG", optimize=True)


def header_band_minimum(clean_png: Path) -> int:
    """Prototype-compatible smoke check for absence of the printed header block."""
    grey = np.asarray(Image.open(clean_png).convert("L"))
    pixels_per_mm = DPI / 25.4
    band = grey[
        int(14 * pixels_per_mm):int(30 * pixels_per_mm),
        int(15 * pixels_per_mm):int(95 * pixels_per_mm),
    ]
    return int(band.min())


def _process_chunk(
    chunk_number: int,
    rows: list[dict],
    dataset_root: Path,
    output_root: Path,
    kit_root: Path,
    kit_python: Path,
    keep_work: bool,
) -> tuple[pd.DataFrame, list[dict]]:
    part_dir = output_root / "index_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_csv = part_dir / f"chunk_{chunk_number:05d}.csv"
    images_dir = output_root / "images"

    if part_csv.exists():
        try:
            existing = pd.read_csv(part_csv, low_memory=False)
            paths_ok = len(existing) == len(rows) and all(
                (output_root / path).is_file() for path in existing["image_path"]
            )
            if paths_ok:
                return existing, []
        except (pd.errors.EmptyDataError, KeyError):
            pass

    work_dir = output_root / "work" / f"chunk_{chunk_number:05d}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    staged_dir = work_dir / "staged"
    clean_dir = work_dir / "clean"

    staged_meta: dict[str, dict] = {}
    failures: list[dict] = []
    staged_rows: list[dict] = []
    for row in rows:
        try:
            staged_meta[row["image_id"]] = stage_record(row, dataset_root, staged_dir)
            staged_rows.append(row)
        except Exception as exc:  # keep a complete audit before failing the build
            failures.append({
                "fcmae_record_id": row["fcmae_record_id"],
                "image_id": row["image_id"],
                "stage": "transcode",
                "error": f"{type(exc).__name__}: {exc}",
            })

    if staged_rows:
        try:
            render_staged(staged_dir, clean_dir, kit_root, kit_python)
        except Exception as exc:
            for row in staged_rows:
                failures.append({
                    "fcmae_record_id": row["fcmae_record_id"],
                    "image_id": row["image_id"],
                    "stage": "render",
                    "error": f"{type(exc).__name__}: {exc}",
                })
            staged_rows = []

    output_rows = []
    for row in staged_rows:
        clean_png = clean_dir / f"{row['image_id']}-0.png"
        output_png = images_dir / f"{row['image_id']}.png"
        try:
            if not clean_png.is_file():
                raise FileNotFoundError(clean_png)
            no_header_min = header_band_minimum(clean_png)
            # Keep this value as a visual audit metric only.  A high-amplitude
            # physiological trace can legitimately enter the nominal header
            # band, so treating a dark pixel as proof of a printed demographic
            # header creates false record failures.  Header absence is instead
            # enforced structurally: staged headers contain no comments and
            # render_staged never passes the kit's --print_header flag.
            recolour_grid(clean_png, output_png, row["grid_palette"])
            generated = dict(row)
            generated.update(staged_meta[row["image_id"]])
            generated.update({
                "image_path": str(output_png.relative_to(output_root)),
                "image_sha256": sha256_file(output_png),
                "header_printed": 0,
                "header_band_minimum": no_header_min,
                "fixed_font": FIXED_FONT,
                "dpi": DPI,
                "paper_speed_mm_per_second": PAPER_SPEED_MM_PER_SECOND,
                "image_horizontal_sample_rate_hz": IMAGE_HORIZONTAL_SAMPLE_RATE_HZ,
                "image_horizontal_nyquist_hz": IMAGE_HORIZONTAL_NYQUIST_HZ,
                "layout": "3x4_plus_lead_II_rhythm",
                "variation_policy": "grid_colour_only",
                "fidelity_policy": "diagnostic_fidelity_v1",
            })
            output_rows.append(generated)
        except Exception as exc:
            failures.append({
                "fcmae_record_id": row["fcmae_record_id"],
                "image_id": row["image_id"],
                "stage": "recolour_or_verify",
                "error": f"{type(exc).__name__}: {exc}",
            })

    result = pd.DataFrame(output_rows)
    result.to_csv(part_csv, index=False)
    if not keep_work and not failures:
        shutil.rmtree(work_dir, ignore_errors=True)
    return result, failures


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(value)
    os.replace(partial, path)


def _checkpoint_paths(checkpoint_dir: Path, chunk_number: int) -> tuple[Path, Path]:
    archive = checkpoint_dir / f"chunk_{chunk_number:05d}.tar"
    return archive, archive.with_suffix(".tar.sha256")


def _checkpoint_is_valid(archive: Path, checksum: Path) -> bool:
    if not archive.is_file() or not checksum.is_file():
        return False
    fields = checksum.read_text().split()
    if not fields or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
        return False
    return sha256_file(archive) == fields[0]


def _write_chunk_checkpoint(
    *,
    chunk_number: int,
    result: pd.DataFrame,
    expected_records: int,
    output_root: Path,
    checkpoint_dir: Path,
) -> Path:
    """Atomically persist one complete chunk to Drive as a restartable tar."""
    if len(result) != expected_records:
        raise RuntimeError(
            f"chunk {chunk_number}: refusing incomplete checkpoint "
            f"({len(result)}/{expected_records} records)"
        )
    archive, checksum = _checkpoint_paths(checkpoint_dir, chunk_number)
    restored_ledger = output_root / "restored_checkpoints.json"
    if restored_ledger.is_file() and archive.is_file() and checksum.is_file():
        restored_names = set(json.loads(restored_ledger.read_text()).get("archives", []))
        if archive.name in restored_names:
            return archive
    if _checkpoint_is_valid(archive, checksum):
        return archive

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    local_dir = output_root / "checkpoint_staging"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_tar = local_dir / archive.name
    local_partial = local_tar.with_name(local_tar.name + ".partial")
    drive_partial = archive.with_name(archive.name + ".partial")
    for path in (local_partial, drive_partial):
        if path.exists():
            path.unlink()

    part_csv = output_root / "index_parts" / f"chunk_{chunk_number:05d}.csv"
    image_paths = [output_root / str(value) for value in result["image_path"]]
    required = [part_csv, *image_paths]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"chunk {chunk_number}: checkpoint inputs missing: {missing[:3]}"
        )

    try:
        with tarfile.open(local_partial, "w") as tf:
            tf.add(part_csv, arcname=f"index_parts/{part_csv.name}", recursive=False)
            for image_path in image_paths:
                tf.add(
                    image_path,
                    arcname=f"images/{image_path.name}",
                    recursive=False,
                )
        os.replace(local_partial, local_tar)
        digest = sha256_file(local_tar)
        shutil.copyfile(local_tar, drive_partial)
        os.replace(drive_partial, archive)
        _atomic_write_text(checksum, f"{digest}  {archive.name}\n")
    finally:
        for path in (local_partial, local_tar, drive_partial):
            if path.exists():
                path.unlink()
    return archive


def restore_chunk_checkpoints(
    checkpoint_dir: Path,
    output_root: Path,
) -> dict:
    """Verify and restore every complete Drive chunk into ephemeral storage."""
    checkpoint_dir = Path(checkpoint_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    restored_archives = 0
    restored_records = 0
    ignored_archives = []
    verified_archives = []

    for archive in sorted(checkpoint_dir.glob("chunk_*.tar")):
        checksum = archive.with_suffix(".tar.sha256")
        if not _checkpoint_is_valid(archive, checksum):
            ignored_archives.append(archive.name)
            print(f"checkpoint ignored (missing/bad checksum): {archive.name}", flush=True)
            continue
        with tarfile.open(archive, "r") as tf:
            members = tf.getmembers()
            part_members = [
                member for member in members
                if re.fullmatch(r"index_parts/chunk_[0-9]{5}\.csv", member.name)
            ]
            image_members = [
                member for member in members
                if re.fullmatch(r"images/ECG[0-9]{6}\.png", member.name)
            ]
            if (
                len(part_members) != 1
                or len(members) != 1 + len(image_members)
                or any(not member.isfile() for member in members)
            ):
                raise RuntimeError(f"unsafe or malformed checkpoint: {archive}")
            tf.extractall(output_root, members=members, filter="data")
        part = output_root / part_members[0].name
        rows = pd.read_csv(part, low_memory=False)
        if len(rows) != len(image_members):
            raise RuntimeError(
                f"{archive}: {len(rows)} index rows for {len(image_members)} images"
            )
        restored_archives += 1
        restored_records += len(rows)
        verified_archives.append(archive.name)
        print(
            f"restored {restored_records:,} images from "
            f"{restored_archives:,} verified checkpoint shards",
            flush=True,
        )
    result = {
        "checkpoint_archives_restored": restored_archives,
        "records_restored": restored_records,
        "ignored_archives": ignored_archives,
        "archives": verified_archives,
    }
    _atomic_write_text(
        output_root / "restored_checkpoints.json",
        json.dumps(result, indent=2) + "\n",
    )
    return result


def run_generation(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    kit_root: Path = DEFAULT_KIT_ROOT,
    kit_python: Path = DEFAULT_KIT_PY,
    *,
    record_ids: Iterable[str] | None = None,
    limit: int | None = None,
    workers: int = 12,
    chunk_size: int = 200,
    keep_work: bool = False,
    fidelity_audit: pd.DataFrame | None = None,
    fidelity_summary: dict | None = None,
    checkpoint_dir: Path | None = None,
) -> pd.DataFrame:
    """Generate exactly one PNG for every selected manifest row.

    The build is resumable at chunk granularity. Supplying ``record_ids`` or
    ``limit`` is intended for smoke tests; omit both for the complete corpus.
    """
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    kit_root = Path(kit_root).resolve()
    # Do not call resolve() here. A venv's ``bin/python`` is normally a symlink
    # to the system executable; dereferencing it makes Python lose the venv and
    # therefore all renderer dependencies.
    kit_python = Path(kit_python).expanduser().absolute()
    preflight = verify_preflight(dataset_root, kit_root, kit_python)
    raw_manifest = pd.read_csv(dataset_root / "manifest.csv", low_memory=False)
    if fidelity_audit is None:
        fidelity_audit, fidelity_summary = audit_waveform_fidelity(
            dataset_root,
            workers=workers,
        )
    if fidelity_summary is None:
        raise ValueError("fidelity_summary is required when fidelity_audit is supplied")
    if fidelity_audit["fcmae_record_id"].duplicated().any():
        raise ValueError("fidelity audit contains duplicate record IDs")
    audited_ids = set(fidelity_audit["fcmae_record_id"].astype(str))
    raw_ids = set(raw_manifest["fcmae_record_id"].astype(str))
    if audited_ids != raw_ids:
        raise ValueError("fidelity audit record IDs do not exactly match the raw manifest")
    eligible_ids = fidelity_audit.loc[
        fidelity_audit["fidelity_eligible"].astype(int) == 1,
        "fcmae_record_id",
    ].astype(str)
    full = load_generation_manifest(dataset_root, eligible_record_ids=eligible_ids)
    audit_metadata_columns = [
        "fcmae_record_id",
        "native_container_dtype",
        "native_wfdb_formats",
        "native_units",
        "native_adc_gain_min",
        "native_adc_gain_max",
        "missing_sentinel_samples",
        "positive_rail_samples",
        "audited_segment_samples",
        "blank_padding_samples",
        "nonfinite_samples",
        "flat_leads",
        "minimum_lead_peak_to_peak_mv",
        "maximum_absolute_mv",
        "fidelity_eligible",
        "fidelity_exclusion_reasons",
    ]
    full = full.merge(
        fidelity_audit[audit_metadata_columns],
        on="fcmae_record_id",
        how="left",
        validate="one_to_one",
    )

    palette_table, palette_stats = palette_audit(full)
    if record_ids is not None:
        wanted = list(record_ids)
        selected = full[full["fcmae_record_id"].isin(wanted)].copy()
        absent = sorted(set(wanted) - set(selected["fcmae_record_id"]))
        if absent:
            raise KeyError(f"record IDs not found in manifest: {absent}")
    elif limit is not None:
        selected = full.head(int(limit)).copy()
    else:
        selected = full.copy()
    if selected.empty:
        raise ValueError("no manifest records selected")

    output_root.mkdir(parents=True, exist_ok=True)
    fidelity_audit.to_csv(output_root / "waveform_fidelity_audit.csv", index=False)
    fidelity_audit.loc[
        fidelity_audit["fidelity_eligible"].astype(int) == 0
    ].to_csv(output_root / "waveform_fidelity_exclusions.csv", index=False)
    (output_root / "waveform_fidelity_summary.json").write_text(
        json.dumps(fidelity_summary, indent=2) + "\n"
    )
    source_fidelity_policy_table().to_csv(
        output_root / "source_fidelity_policy.csv", index=False
    )
    palette_table.to_csv(output_root / "grid_palette_by_source.csv")
    (output_root / "grid_palette_audit.json").write_text(
        json.dumps(palette_stats, indent=2) + "\n"
    )
    full[["image_id", "fcmae_record_id", "grid_palette"]].to_csv(
        output_root / "grid_palette_assignment.csv", index=False
    )

    selected_ids_hash = hashlib.sha256(
        "\n".join(selected["fcmae_record_id"].astype(str)).encode("utf-8")
    ).hexdigest()
    build_config = {
        **preflight,
        "selected_records": int(len(selected)),
        "raw_manifest_records": int(len(raw_manifest)),
        "fidelity_qualified_records": int(len(full)),
        "fidelity_excluded_records": int(len(raw_manifest) - len(full)),
        "selected_record_ids_sha256": selected_ids_hash,
        "images_per_record": 1,
        "grid_seed": GRID_SEED,
        "segment_seed": SEGMENT_SEED,
        "palette_names": list(PALETTE_NAMES),
        "palette_assignment_uses_source_or_morphology": False,
        "native_sampling_frequencies_preserved": True,
        "native_sampling_frequency_by_source_hz": EXPECTED_NATIVE_FS,
        "source_signal_policy": SOURCE_SIGNAL_POLICY,
        "window_seconds": WINDOW_SECONDS,
        "diagnostic_reference_band_hz": list(DIAGNOSTIC_BAND_HZ),
        "default_applied_filter": "none",
        "source_specific_applied_filter": {
            "ptb": {
                "type": "Butterworth band-pass",
                "low_hz": DIAGNOSTIC_BAND_HZ[0],
                "high_hz": DIAGNOSTIC_BAND_HZ[1],
                "order_per_pass": PTB_FILTER_ORDER,
                "phase": "zero-phase forward-backward",
            }
        },
        "baseline_centering": False,
        "amplitude_normalization": False,
        "amplitude_clipping": False,
        "short_record_padding": "blank NaN gap",
        "dpi": DPI,
        "paper_speed_mm_per_second": PAPER_SPEED_MM_PER_SECOND,
        "image_horizontal_sample_rate_hz": IMAGE_HORIZONTAL_SAMPLE_RATE_HZ,
        "image_horizontal_nyquist_hz": IMAGE_HORIZONTAL_NYQUIST_HZ,
        "print_header": False,
        "header_validation": {
            "mode": "structural",
            "staged_wfdb_comments": [],
            "renderer_print_header_flag": False,
            "pixel_band_minimum_is_metadata_only": True,
        },
        "fixed_font": FIXED_FONT,
        "augmentation": {
            "grid_colour": True,
            "rotation": False,
            "perspective": False,
            "texture": False,
            "creases": False,
            "lighting": False,
            "colour_cast": False,
            "noise": False,
            "blur": False,
            "jpeg": False,
        },
    }
    policy_fields = {
        key: value
        for key, value in build_config.items()
        if key not in {"dataset_root", "kit_root", "kit_python"}
    }
    build_config["generation_policy_sha256"] = hashlib.sha256(
        json.dumps(policy_fields, sort_keys=True).encode("utf-8")
    ).hexdigest()
    config_path = output_root / "generation_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        keys = (
            "manifest_sha256",
            "selected_record_ids_sha256",
            "images_per_record",
            "generation_policy_sha256",
        )
        if any(previous.get(k) != build_config.get(k) for k in keys):
            raise RuntimeError(
                f"{output_root} contains a different build; choose a new output directory"
            )
    config_path.write_text(json.dumps(build_config, indent=2) + "\n")

    records = selected.to_dict("records")
    chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir).expanduser().absolute()
        checkpoint_config = {
            "checkpoint_format": "fcmae_chunk_tar_v1",
            "manifest_sha256": build_config["manifest_sha256"],
            "selected_record_ids_sha256": build_config["selected_record_ids_sha256"],
            "generation_policy_sha256": build_config["generation_policy_sha256"],
            "selected_records": int(len(selected)),
            "chunk_size": int(chunk_size),
            "expected_chunks": int(len(chunks)),
        }
        checkpoint_config_path = checkpoint_dir / "checkpoint_config.json"
        if checkpoint_config_path.exists():
            previous = json.loads(checkpoint_config_path.read_text())
            if previous != checkpoint_config:
                raise RuntimeError(
                    f"{checkpoint_dir} belongs to a different generation plan; "
                    "use a new checkpoint directory"
                )
        _atomic_write_text(
            checkpoint_config_path,
            json.dumps(checkpoint_config, indent=2) + "\n",
        )

    results: list[pd.DataFrame] = []
    failures: list[dict] = []
    worker_count = max(1, min(int(workers), len(chunks)))
    started = time.monotonic()
    processed = 0
    checkpointed = 0
    print(
        f"starting {len(selected):,} images in {len(chunks):,} chunks "
        f"with {worker_count} workers",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(
                _process_chunk,
                number,
                chunk,
                dataset_root,
                output_root,
                kit_root,
                kit_python,
                keep_work,
            ): number
            for number, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            chunk_number = futures[future]
            result, chunk_failures = future.result()
            results.append(result)
            failures.extend(chunk_failures)
            processed += len(chunks[chunk_number])
            generated = sum(len(frame) for frame in results)
            if (
                checkpoint_dir is not None
                and not chunk_failures
                and len(result) == len(chunks[chunk_number])
            ):
                print(
                    f"checkpointing completed chunk {chunk_number:05d} "
                    f"({len(result):,} images) to persistent storage",
                    flush=True,
                )
                archive = _write_chunk_checkpoint(
                    chunk_number=chunk_number,
                    result=result,
                    expected_records=len(chunks[chunk_number]),
                    output_root=output_root,
                    checkpoint_dir=checkpoint_dir,
                )
                checkpointed += len(result)
                _atomic_write_text(
                    checkpoint_dir / "checkpoint_status.json",
                    json.dumps(
                        {
                            "status": "rendering",
                            "processed_records_this_pass": processed,
                            "generated_records_this_pass": generated,
                            "checkpointed_records_this_pass": checkpointed,
                            "failures_this_pass": len(failures),
                            "last_checkpoint": archive.name,
                            "generation_policy_sha256": build_config[
                                "generation_policy_sha256"
                            ],
                        },
                        indent=2,
                    )
                    + "\n",
                )
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = processed / elapsed
            eta_seconds = (len(selected) - processed) / rate if rate else math.inf
            print(
                f"generated {generated:,}/{len(selected):,} | "
                f"processed {processed:,}/{len(selected):,} | "
                f"failures {len(failures):,} | "
                f"elapsed {elapsed / 3600:.2f} h | ETA {eta_seconds / 3600:.2f} h",
                flush=True,
            )

    # Avoid pandas' deprecated dtype inference for empty/all-NA chunk frames.
    nonempty_results = [frame for frame in results if not frame.empty]
    index = (
        pd.concat(nonempty_results, ignore_index=True)
        if nonempty_results
        else pd.DataFrame()
    )
    if not index.empty:
        order = {image_id: i for i, image_id in enumerate(selected["image_id"])}
        index["_order"] = index["image_id"].map(order)
        index = index.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    index.to_csv(output_root / "index.csv", index=False)
    pd.DataFrame(failures, columns=["fcmae_record_id", "image_id", "stage", "error"]).to_csv(
        output_root / "failures.csv", index=False
    )
    if checkpoint_dir is not None:
        shutil.copy2(
            output_root / "failures.csv",
            checkpoint_dir / "last_failures.csv",
        )

    image_count = int(index["image_path"].nunique()) if not index.empty else 0
    record_count = int(index["fcmae_record_id"].nunique()) if not index.empty else 0
    if failures or image_count != len(selected) or record_count != len(selected):
        if checkpoint_dir is not None:
            _atomic_write_text(
                checkpoint_dir / "checkpoint_status.json",
                json.dumps(
                    {
                        "status": "incomplete",
                        "images_generated": image_count,
                        "records_generated": record_count,
                        "records_expected": int(len(selected)),
                        "failures": int(len(failures)),
                        "generation_policy_sha256": build_config[
                            "generation_policy_sha256"
                        ],
                    },
                    indent=2,
                )
                + "\n",
            )
        raise RuntimeError(
            f"incomplete build: {image_count} images for {record_count}/{len(selected)} records; "
            f"{len(failures)} failures (see failures.csv)"
        )
    if checkpoint_dir is not None:
        _atomic_write_text(
            checkpoint_dir / "checkpoint_status.json",
            json.dumps(
                {
                    "status": "complete",
                    "images": image_count,
                    "records": record_count,
                    "generation_policy_sha256": build_config[
                        "generation_policy_sha256"
                    ],
                },
                indent=2,
            )
            + "\n",
        )
    print(f"complete: {image_count:,} images for {record_count:,} ECG records")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--kit-root", type=Path, default=DEFAULT_KIT_ROOT)
    parser.add_argument("--kit-python", type=Path, default=DEFAULT_KIT_PY)
    parser.add_argument("--record-id", action="append", dest="record_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    run_generation(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        kit_root=args.kit_root,
        kit_python=args.kit_python,
        record_ids=args.record_ids,
        limit=args.limit,
        workers=args.workers,
        chunk_size=args.chunk_size,
        checkpoint_dir=args.checkpoint_dir,
        keep_work=args.keep_work,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
