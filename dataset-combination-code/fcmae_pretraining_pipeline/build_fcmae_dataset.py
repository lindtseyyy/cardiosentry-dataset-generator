#!/usr/bin/env python3
"""Build a morphology-focused raw-signal corpus for ConvNeXt-V2 FCMAE.

The source databases are read-only.  The release contains relative symlinks,
record-level provenance, curation/exclusion ledgers, and integrity hashes.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import heapq
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FAMILIES = (
    "normal_baseline",
    "ischemic_stt",
    "atrial_rhythm",
    "structural_voltage",
    "conduction_other",
)

CHALLENGE_SOURCES = (
    "cpsc_2018",
    "cpsc_2018_extra",
    "st_petersburg_incart",
    "ptb",
    "ptb-xl",
    "georgia",
)

SOURCE_PREFIX = {
    "cpsc_2018": "CPSC2018",
    "cpsc_2018_extra": "CPSCEXTRA",
    "st_petersburg_incart": "INCART",
    "ptb": "PTB",
    "ptb-xl": "PTBXL",
    "georgia": "GEORGIA",
    "chapman_shaoxing_ningbo": "CHAPMAN",
    "sphdb": "SPHDB",
}

# Exact downstream conditions are reserved, not relabelled as morphology
# neighbours. Composite AF+flutter assertions are reserved conservatively.
EXACT_AF_CODES = {
    "164889003",
    "426749004",
    "282825002",
    "314208002",
    "195080001",
}
EXACT_LVH_CODES = {"164873001"}
CHAPMAN_UNDOCUMENTED_LVH_LIKE = "55827005"

SPH_EXACT_AF_CODES = {"50"}
SPH_EXACT_LVH_CODES = {"142"}
SPH_NORMAL_CODES = {"1", "21", "22"}
SPH_ATRIAL_CODES = {"23", "30", "31", "36", "37", "50", "51", "54"}
SPH_CONDUCTION_CODES = {
    "60", "80", "81", "82", "83", "84", "85", "86", "87", "88",
    "101", "102", "104", "105", "106", "108",
}
SPH_STRUCTURAL_CODES = {"120", "121", "125", "140", "142", "143", "153"}
SPH_ISCHEMIC_CODES = {
    "145", "146", "147", "148", "152", "153", "155",
    "160", "161", "165", "166", "362", "363", "367",
}


@dataclass
class Record:
    fcmae_record_id: str
    source_dataset: str
    source_collection: str
    original_record_id: str
    canonical_record_id: str
    patient_id: str
    patient_id_available: int
    patient_group_id: str
    patient_group_is_synthetic: int
    original_signal_path: str
    original_header_path: str
    original_data_path: str
    signal_format: str
    sampling_frequency: float
    duration_seconds: float
    num_leads: int
    num_samples: int
    lead_names: str
    original_diagnosis_codes: str
    original_diagnosis_statements: str
    normalized_diagnoses: str
    age: str
    sex: str
    source_extra_json: str
    morphology_normal_baseline: int = 0
    morphology_ischemic_stt: int = 0
    morphology_atrial_rhythm: int = 0
    morphology_structural_voltage: int = 0
    morphology_conduction_other: int = 0
    morphology_families: str = ""
    n_morphology_families: int = 0
    exact_target_flags: str = ""
    exact_target_decision: str = "NOT_EXACT_TARGET"
    v2_overlap: int = 0
    v2_global_record_id: str = ""
    validation_status: str = "VALID_METADATA"
    validation_issues: str = ""
    selection_status: str = "ELIGIBLE"
    selection_reason: str = ""
    inclusion_reason: str = ""
    selection_seed: int = 0
    data_sha256: str = ""
    header_sha256: str = ""
    signal_ref_path: str = ""
    header_ref_path: str = ""
    data_ref_path: str = ""

    def family_set(self) -> set[str]:
        return {f for f in FAMILIES if getattr(self, f"morphology_{f}") == 1}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_task(path: str) -> tuple[str, str, str]:
    try:
        return path, sha256_file(Path(path)), ""
    except Exception as exc:  # noqa: BLE001
        return path, "", f"{type(exc).__name__}: {exc}"


def _validate_signal_task(task: tuple[str, str, int, int]) -> tuple[str, str, str]:
    """Return path, validation status, issue for one native signal file."""
    path_s, fmt, expected_leads, expected_samples = task
    path = Path(path_s)
    try:
        if fmt == "hdf5":
            import h5py

            with h5py.File(path, "r") as h5:
                if "ecg" not in h5:
                    return path_s, "INVALID_SIGNAL", "HDF5 dataset 'ecg' missing"
                shape = tuple(int(x) for x in h5["ecg"].shape)
        else:
            from scipy.io import whosmat

            entries = {name: shape for name, shape, _dtype in whosmat(path)}
            if "val" not in entries:
                return path_s, "INVALID_SIGNAL", "MAT variable 'val' missing"
            shape = tuple(int(x) for x in entries["val"])

        if len(shape) != 2 or expected_leads not in shape:
            return path_s, "INVALID_SIGNAL", f"unexpected waveform shape {shape}"
        actual_samples = shape[1] if shape[0] == expected_leads else shape[0]
        if expected_samples and actual_samples != expected_samples:
            return (
                path_s,
                "INVALID_SIGNAL",
                f"sample mismatch metadata={expected_samples} actual={actual_samples}",
            )
        return path_s, "VALID_SIGNAL", ""
    except Exception as exc:  # noqa: BLE001
        return path_s, "INVALID_SIGNAL", f"{type(exc).__name__}: {exc}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")


def rel_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def is_missing_value(value: Any) -> bool:
    return str(value).strip().lower() in {"", "nan", "na", "n/a", "none", "unknown"}


def parse_wfdb_header(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError("empty WFDB header")
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError("truncated WFDB record line")
    num_leads = int(first[1])
    fs = float(first[2].split("/")[0])
    num_samples = int(first[3])
    signal_lines = lines[1 : 1 + num_leads]
    data_names: list[str] = []
    lead_names: list[str] = []
    for line in signal_lines:
        parts = line.split()
        if parts:
            data_names.append(parts[0])
            lead_names.append(parts[-1])
    comments: dict[str, str] = {}
    for line in lines:
        if line.startswith("#") and ":" in line:
            key, value = line[1:].split(":", 1)
            comments[key.strip()] = value.strip()
    unique_data = list(dict.fromkeys(data_names))
    if len(unique_data) != 1:
        raise ValueError(f"expected one multiplexed data file, found {unique_data}")
    data_path = path.parent / unique_data[0]
    return {
        "num_leads": num_leads,
        "fs": fs,
        "num_samples": num_samples,
        "duration_seconds": num_samples / fs if fs else 0.0,
        "lead_names": lead_names,
        "comments": comments,
        "data_path": data_path,
    }


def load_diagnosis_names(pipeline_dir: Path, chapman_root: Path) -> dict[str, str]:
    names = {
        row["code"].strip(): row["description"].strip()
        for row in read_csv_rows(pipeline_dir / "challenge_diagnoses.csv")
    }
    condition_path = chapman_root / "ConditionNames_SNOMED-CT.csv"
    for row in read_csv_rows(condition_path):
        code = row.get("Snomed_CT", "").strip()
        name = row.get("Full Name", "").strip()
        if code and name:
            names.setdefault(code, name)
    return names


def classify_families(
    source: str, codes: list[str], diagnosis_names: dict[str, str]
) -> set[str]:
    families: set[str] = set()
    if source == "sphdb":
        code_set = set(codes)
        if code_set & SPH_NORMAL_CODES:
            families.add("normal_baseline")
        if code_set & SPH_ISCHEMIC_CODES:
            families.add("ischemic_stt")
        if code_set & SPH_ATRIAL_CODES:
            families.add("atrial_rhythm")
        if code_set & SPH_STRUCTURAL_CODES:
            families.add("structural_voltage")
        if code_set & SPH_CONDUCTION_CODES:
            families.add("conduction_other")
        return families

    descriptions = [diagnosis_names.get(code, "") for code in codes]
    for description in descriptions:
        text = re.sub(r"\s+", " ", description.lower()).strip()
        if not text:
            continue
        if text in {"normal ecg", "sinus rhythm", "sinus bradycardia", "sinus tachycardia"}:
            families.add("normal_baseline")
        if any(
            token in text
            for token in (
                "ischemi", "infarction", "st-t", "st elevation", "st depression",
                "st interval", "st changes", "t-wave", "t wave", "q-wave", "q wave",
                "repolarization", "qt interval", "u-wave", "u wave", "ventricular strain",
            )
        ):
            families.add("ischemic_stt")
        if any(token in text for token in ("atrial", "supraventricular", "sinus arrhythmia", "junctional")):
            families.add("atrial_rhythm")
        if any(
            token in text
            for token in ("hypertrophy", "enlargement", "overload", "axis", "voltage", "strain", "cardiomyopathy")
        ):
            families.add("structural_voltage")
        if any(
            token in text
            for token in (
                "block", "conduction", "pacing", "pacemaker", "pre-excitation", "preexcitation",
                "wolff-parkinson", "ventricular premature", "ventricular ectopic",
                "ventricular tachycardia", "ventricular fibrillation", "ventricular flutter",
                "ventricular escape", "ventricular bigeminy", "ventricular trigeminy",
                "fusion beat", "abnormal qrs", "r-wave abnormal", "r wave abnormal",
                "cardiac dysrhythmia",
            )
        ):
            families.add("conduction_other")
    return families


def exact_target_flags(source: str, codes: list[str]) -> set[str]:
    code_set = set(codes)
    flags: set[str] = set()
    if source == "sphdb":
        if code_set & SPH_EXACT_AF_CODES:
            flags.add("AF")
        if code_set & SPH_EXACT_LVH_CODES:
            flags.add("LVH")
        return flags
    if code_set & EXACT_AF_CODES:
        flags.add("AF")
    if code_set & EXACT_LVH_CODES:
        flags.add("LVH")
    if source == "chapman_shaoxing_ningbo" and CHAPMAN_UNDOCUMENTED_LVH_LIKE in code_set:
        flags.add("POTENTIAL_LVH_UNDOCUMENTED")
    return flags


def has_unmapped_diagnosis(record: Record) -> bool:
    return any(
        "UNMAPPED_CODE" in item for item in record.normalized_diagnoses.split("|")
    )


def decorate_record(
    record: Record,
    codes: list[str],
    diagnosis_names: dict[str, str],
    raw_codes: list[str] | None = None,
) -> None:
    raw_codes = raw_codes if raw_codes is not None else codes
    names = [
        " + ".join(diagnosis_names.get(component, "UNMAPPED_CODE") for component in raw.split("+"))
        for raw in raw_codes
    ]
    record.original_diagnosis_codes = "|".join(raw_codes)
    record.original_diagnosis_statements = "; ".join(
        f"{code} ({name})" for code, name in zip(raw_codes, names)
    )
    record.normalized_diagnoses = "|".join(names)
    families = classify_families(record.source_dataset, codes, diagnosis_names)
    for family_name in families:
        setattr(record, f"morphology_{family_name}", 1)
    record.morphology_families = "|".join(f for f in FAMILIES if f in families)
    record.n_morphology_families = len(families)
    targets = exact_target_flags(record.source_dataset, codes)
    record.exact_target_flags = "|".join(sorted(targets))
    if targets:
        record.exact_target_decision = "RESERVED_FOR_SUPERVISED_USE"


def load_ptbxl_lookup(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(path):
        try:
            key = str(int(float(row["ecg_id"])))
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = row
    return out


def load_v2_overlap(metadata_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    chapman: dict[str, str] = {}
    ptbxl: dict[str, str] = {}
    for row in read_csv_rows(metadata_path):
        source = row.get("source_dataset", "")
        oid = row.get("original_record_id", "").strip()
        gid = row.get("global_record_id", "").strip()
        if source == "CHAPMAN":
            chapman[oid] = gid
        elif source == "PTBXL":
            try:
                ptbxl[str(int(float(oid)))] = gid
            except ValueError:
                continue
    return chapman, ptbxl


def make_challenge_record(
    header: Path,
    source: str,
    repo_root: Path,
    diagnosis_names: dict[str, str],
    ptbxl_lookup: dict[str, dict[str, str]],
    seed: int,
) -> Record:
    record_id = header.stem
    prefix = SOURCE_PREFIX[source]
    canonical_id = record_id
    patient_id = ""
    patient_available = 0
    source_extra: dict[str, Any] = {}
    age = ""
    sex = ""
    issues: list[str] = []
    try:
        parsed = parse_wfdb_header(header)
        data_path = parsed["data_path"]
        comments = parsed["comments"]
        age = comments.get("Age", "")
        sex = comments.get("Sex", "")
        codes = [x.strip() for x in comments.get("Dx", "").split(",") if x.strip()]
        if source == "ptb-xl":
            match = re.fullmatch(r"HR0*(\d+)", record_id, flags=re.IGNORECASE)
            if not match:
                issues.append("PTB-XL Challenge ID is not HR<digits>")
            else:
                canonical_id = str(int(match.group(1)))
                native = ptbxl_lookup.get(canonical_id)
                if native:
                    raw_pid = native.get("patient_id", "").strip()
                    if raw_pid:
                        patient_id = f"PTBXL_P{int(float(raw_pid))}"
                        patient_available = 1
                    source_extra = {
                        "ptbxl_ecg_id": canonical_id,
                        "ptbxl_scp_codes": native.get("scp_codes", ""),
                        "ptbxl_report": native.get("report", ""),
                        "ptbxl_recording_date": native.get("recording_date", ""),
                        "ptbxl_strat_fold": native.get("strat_fold", ""),
                        "ptbxl_metadata_joined": True,
                    }
                else:
                    source_extra = {
                        "ptbxl_ecg_id": canonical_id,
                        "ptbxl_metadata_joined": False,
                        "note": "Challenge-2020 PTB-XL record absent from local PTB-XL v1.0.3 metadata",
                    }
        if not data_path.exists():
            issues.append(f"data file missing: {data_path.name}")
        if parsed["num_leads"] != 12:
            issues.append(f"expected 12 leads, found {parsed['num_leads']}")
        if parsed["fs"] <= 0 or parsed["num_samples"] <= 0:
            issues.append("non-positive sampling frequency or sample count")
    except Exception as exc:  # noqa: BLE001
        parsed = {
            "data_path": header.with_suffix(".mat"),
            "num_leads": 0,
            "fs": 0.0,
            "num_samples": 0,
            "duration_seconds": 0.0,
            "lead_names": [],
            "comments": {},
        }
        data_path = parsed["data_path"]
        codes = []
        issues.append(f"{type(exc).__name__}: {exc}")

    synthetic_group = f"{source}::RECORD::{record_id}"
    patient_group_id = f"{source}::{patient_id}" if patient_available else synthetic_group
    record = Record(
        fcmae_record_id=f"{prefix}_{record_id}",
        source_dataset=source,
        source_collection="challenge_2020",
        original_record_id=record_id,
        canonical_record_id=canonical_id,
        patient_id=patient_id,
        patient_id_available=patient_available,
        patient_group_id=patient_group_id,
        patient_group_is_synthetic=0 if patient_available else 1,
        original_signal_path=rel_to_repo(header.with_suffix(""), repo_root),
        original_header_path=rel_to_repo(header, repo_root),
        original_data_path=rel_to_repo(data_path, repo_root),
        signal_format="wfdb_mat",
        sampling_frequency=float(parsed["fs"]),
        duration_seconds=float(parsed["duration_seconds"]),
        num_leads=int(parsed["num_leads"]),
        num_samples=int(parsed["num_samples"]),
        lead_names="|".join(parsed["lead_names"]),
        original_diagnosis_codes="",
        original_diagnosis_statements="",
        normalized_diagnoses="",
        age=age,
        sex=sex,
        source_extra_json=json.dumps(source_extra, ensure_ascii=False, sort_keys=True),
        selection_seed=seed,
    )
    decorate_record(record, codes, diagnosis_names)
    if issues:
        record.validation_status = "INVALID_METADATA"
        record.validation_issues = "; ".join(issues)
    return record


def inventory_challenge(
    root: Path,
    repo_root: Path,
    diagnosis_names: dict[str, str],
    ptbxl_lookup: dict[str, dict[str, str]],
    seed: int,
) -> list[Record]:
    records: list[Record] = []
    for source in CHALLENGE_SOURCES:
        source_root = root / source
        for header in sorted(source_root.glob("*/*.hea")):
            records.append(
                make_challenge_record(
                    header, source, repo_root, diagnosis_names, ptbxl_lookup, seed
                )
            )
    return records


def inventory_chapman(
    root: Path, repo_root: Path, diagnosis_names: dict[str, str], seed: int
) -> list[Record]:
    records: list[Record] = []
    for header in sorted((root / "WFDBRecords").glob("*/*/*.hea")):
        source = "chapman_shaoxing_ningbo"
        record_id = header.stem
        issues: list[str] = []
        try:
            parsed = parse_wfdb_header(header)
            data_path = parsed["data_path"]
            comments = parsed["comments"]
            codes = [x.strip() for x in comments.get("Dx", "").split(",") if x.strip()]
            if not data_path.exists():
                issues.append(f"data file missing: {data_path.name}")
            if parsed["num_leads"] != 12:
                issues.append(f"expected 12 leads, found {parsed['num_leads']}")
        except Exception as exc:  # noqa: BLE001
            parsed = {
                "data_path": header.with_suffix(".mat"),
                "num_leads": 0,
                "fs": 0.0,
                "num_samples": 0,
                "duration_seconds": 0.0,
                "lead_names": [],
                "comments": {},
            }
            data_path = parsed["data_path"]
            comments = {}
            codes = []
            issues.append(f"{type(exc).__name__}: {exc}")
        record = Record(
            fcmae_record_id=f"{SOURCE_PREFIX[source]}_{record_id}",
            source_dataset=source,
            source_collection="chapman_shaoxing_ningbo",
            original_record_id=record_id,
            canonical_record_id=record_id,
            patient_id="",
            patient_id_available=0,
            patient_group_id=f"{source}::RECORD::{record_id}",
            patient_group_is_synthetic=1,
            original_signal_path=rel_to_repo(header.with_suffix(""), repo_root),
            original_header_path=rel_to_repo(header, repo_root),
            original_data_path=rel_to_repo(data_path, repo_root),
            signal_format="wfdb_mat",
            sampling_frequency=float(parsed["fs"]),
            duration_seconds=float(parsed["duration_seconds"]),
            num_leads=int(parsed["num_leads"]),
            num_samples=int(parsed["num_samples"]),
            lead_names="|".join(parsed["lead_names"]),
            original_diagnosis_codes="",
            original_diagnosis_statements="",
            normalized_diagnoses="",
            age=comments.get("Age", ""),
            sex=comments.get("Sex", ""),
            source_extra_json=json.dumps(
                {
                    "Rx": comments.get("Rx", ""),
                    "Hx": comments.get("Hx", ""),
                    "Sx": comments.get("Sx", ""),
                    "patient_identifier_in_distribution": False,
                },
                sort_keys=True,
            ),
            selection_seed=seed,
        )
        decorate_record(record, codes, diagnosis_names)
        if issues:
            record.validation_status = "INVALID_METADATA"
            record.validation_issues = "; ".join(issues)
        records.append(record)
    return records


def inventory_sphdb(
    root: Path, repo_root: Path, seed: int
) -> tuple[list[Record], dict[str, str]]:
    code_names = {
        row["Code"].strip(): row["Description"].strip()
        for row in read_csv_rows(root / "code.csv")
    }
    records: list[Record] = []
    for row in read_csv_rows(root / "metadata.csv"):
        source = "sphdb"
        record_id = row.get("ECG_ID", "").strip()
        patient_id = row.get("Patient_ID", "").strip()
        raw_codes = [x.strip() for x in row.get("AHA_Code", "").split(";") if x.strip()]
        codes = [component for raw in raw_codes for component in raw.split("+") if component]
        data_path = root / "records" / f"{record_id}.h5"
        issues: list[str] = []
        if not record_id:
            issues.append("missing ECG_ID")
        if not data_path.exists():
            issues.append("HDF5 signal file missing")
        try:
            num_samples = int(float(row.get("N", "0")))
        except ValueError:
            num_samples = 0
            issues.append("invalid sample count N")
        record = Record(
            fcmae_record_id=f"{SOURCE_PREFIX[source]}_{record_id}",
            source_dataset=source,
            source_collection="sphdb",
            original_record_id=record_id,
            canonical_record_id=record_id,
            patient_id=patient_id,
            patient_id_available=1 if patient_id else 0,
            patient_group_id=(
                f"{source}::{patient_id}" if patient_id else f"{source}::RECORD::{record_id}"
            ),
            patient_group_is_synthetic=0 if patient_id else 1,
            original_signal_path=rel_to_repo(data_path, repo_root),
            original_header_path="",
            original_data_path=rel_to_repo(data_path, repo_root),
            signal_format="hdf5",
            sampling_frequency=500.0,
            duration_seconds=num_samples / 500.0 if num_samples else 0.0,
            num_leads=12,
            num_samples=num_samples,
            lead_names="I|II|III|aVR|aVL|aVF|V1|V2|V3|V4|V5|V6",
            original_diagnosis_codes="",
            original_diagnosis_statements="",
            normalized_diagnoses="",
            age=row.get("Age", "").strip(),
            sex=row.get("Sex", "").strip(),
            source_extra_json=json.dumps(
                {"date": row.get("Date", ""), "source_row": row},
                ensure_ascii=False,
                sort_keys=True,
            ),
            selection_seed=seed,
        )
        decorate_record(record, codes, code_names, raw_codes=raw_codes)
        if issues:
            record.validation_status = "INVALID_METADATA"
            record.validation_issues = "; ".join(issues)
        records.append(record)
    return records, code_names


def audit_meeti(root: Path, repo_root: Path, seed: int) -> dict[str, Any]:
    """Inventory modalities and inspect a deterministic sample of MAT keys."""
    counts: collections.Counter[str] = collections.Counter()
    mat_count = 0
    rng = random.Random(seed)
    reservoir: list[Path] = []
    largest: list[tuple[int, str]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            suffix = path.suffix.lower() or "[none]"
            counts[suffix] += 1
            if suffix != ".mat":
                continue
            mat_count += 1
            if len(reservoir) < 48:
                reservoir.append(path)
            else:
                index = rng.randrange(mat_count)
                if index < len(reservoir):
                    reservoir[index] = path
            try:
                item = (path.stat().st_size, str(path))
                if len(largest) < 16:
                    heapq.heappush(largest, item)
                elif item[0] > largest[0][0]:
                    heapq.heapreplace(largest, item)
            except OSError:
                pass

    inspected: list[dict[str, Any]] = []
    signal_key_files: list[str] = []
    inspect_paths = list(dict.fromkeys(reservoir + [Path(p) for _s, p in largest]))
    try:
        from scipy.io import whosmat

        for path in sorted(inspect_paths):
            try:
                entries = whosmat(path)
                keys = [item[0] for item in entries]
                inspected.append(
                    {
                        "path": rel_to_repo(path, repo_root),
                        "size_bytes": path.stat().st_size,
                        "keys": keys,
                    }
                )
                if set(keys) & {"signal", "val", "ecg"}:
                    signal_key_files.append(rel_to_repo(path, repo_root))
            except Exception as exc:  # noqa: BLE001
                inspected.append(
                    {
                        "path": rel_to_repo(path, repo_root),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    except ImportError as exc:
        inspected.append({"inspection_error": str(exc)})

    standalone_waveforms = sum(counts[x] for x in (".hea", ".dat", ".h5", ".npy"))
    usable = bool(standalone_waveforms or signal_key_files)
    return {
        "root": rel_to_repo(root, repo_root),
        "file_counts_by_extension": dict(sorted(counts.items())),
        "mat_files": mat_count,
        "standalone_waveform_files": standalone_waveforms,
        "sampled_mat_files_inspected": len(inspected),
        "sampled_files_with_waveform_key": signal_key_files,
        "sample_details": inspected,
        "raw_signal_availability": "AVAILABLE" if usable else "NOT_AVAILABLE_IN_LOCAL_DOWNLOAD",
        "selection_decision": (
            "REQUIRES_IMPLEMENTATION_REVIEW" if usable else "EXCLUDE_SOURCE_MISSING_RAW_SIGNAL"
        ),
        "note": (
            "The local tree was exhaustively inventoried by extension. MAT files were sampled "
            "with reservoir sampling plus the largest files; no signal/val/ecg key was found. "
            "The local download contains reports/interpretations and a partial PNG set, but no "
            "raw waveform files usable for this raw-signal corpus."
            if not usable
            else "At least one possible local waveform component was found; inspect before use."
        ),
    }


def apply_initial_exclusions(
    records: list[Record],
    v2_chapman: dict[str, str],
    v2_ptbxl: dict[str, str],
    duplicate_findings: list[dict[str, Any]],
) -> None:
    seen: dict[tuple[str, str], Record] = {}
    for record in records:
        key = (record.source_dataset, record.canonical_record_id)
        if key in seen:
            record.selection_status = "EXCLUDED"
            record.selection_reason = "DUPLICATE_RECORD_ID_WITHIN_SOURCE"
            duplicate_findings.append(
                {
                    "finding_type": "DUPLICATE_RECORD_ID_WITHIN_SOURCE",
                    "scope": record.source_dataset,
                    "key": record.canonical_record_id,
                    "kept_record_id": seen[key].fcmae_record_id,
                    "excluded_record_id": record.fcmae_record_id,
                    "sha256": "",
                    "note": "Second local candidate with the same canonical source record ID",
                }
            )
            continue
        seen[key] = record

        if record.validation_status != "VALID_METADATA":
            record.selection_status = "EXCLUDED"
            record.selection_reason = "INVALID_OR_MISSING_SIGNAL_METADATA"
            continue

        v2_gid = ""
        if record.source_dataset == "chapman_shaoxing_ningbo":
            v2_gid = v2_chapman.get(record.canonical_record_id, "")
        elif record.source_dataset == "ptb-xl":
            v2_gid = v2_ptbxl.get(record.canonical_record_id, "")
        if v2_gid:
            record.v2_overlap = 1
            record.v2_global_record_id = v2_gid
            record.selection_status = "EXCLUDED"
            record.selection_reason = "EXISTING_V2_RECORD_OVERLAP"
            if record.exact_target_flags:
                record.exact_target_decision = "EXCLUDED_V2_OVERLAP"
            continue

        if record.exact_target_flags:
            record.selection_status = "RESERVED"
            record.selection_reason = "EXACT_OR_UNCERTAIN_DOWNSTREAM_TARGET_RESERVED"
            continue

        if record.n_morphology_families == 0:
            record.selection_status = "EXCLUDED"
            record.selection_reason = "NO_RELEVANT_MORPHOLOGY_FAMILY"


def soft_select_source(
    records: list[Record],
    quota: int,
    shares: dict[str, float],
    max_known_patient_records: int,
    seed: int,
) -> list[Record]:
    candidates = [record for record in records if record.selection_status == "ELIGIBLE"]
    if not candidates:
        return []
    target_n = min(quota, len(candidates))
    prevalence = collections.Counter(
        family for record in candidates for family in record.family_set()
    )

    def rank(record: Record) -> tuple[float, int, str]:
        rarity = sum(1.0 / prevalence[f] for f in record.family_set() if prevalence[f])
        return (-rarity, -record.n_morphology_families, stable_key(seed, record.fcmae_record_id))

    pools = {
        family: sorted(
            (record for record in candidates if family in record.family_set()), key=rank
        )
        for family in FAMILIES
    }
    pointers = collections.Counter()
    targets = {
        family: min(len(pools[family]), math.ceil(target_n * float(shares[family])))
        for family in FAMILIES
    }
    counts = collections.Counter()
    selected_ids: set[str] = set()
    selected: list[Record] = []
    patient_counts = collections.Counter()
    patient_families: dict[str, set[str]] = collections.defaultdict(set)

    def eligible(record: Record) -> bool:
        if record.fcmae_record_id in selected_ids:
            return False
        if not record.patient_id_available:
            return True
        pid = record.patient_group_id
        if patient_counts[pid] >= max_known_patient_records:
            return False
        if patient_counts[pid] == 0:
            return True
        return bool(record.family_set() - patient_families[pid])

    def add(record: Record) -> None:
        selected.append(record)
        selected_ids.add(record.fcmae_record_id)
        for family in record.family_set():
            counts[family] += 1
        if record.patient_id_available:
            pid = record.patient_group_id
            patient_counts[pid] += 1
            patient_families[pid].update(record.family_set())

    exhausted: set[str] = set()
    while len(selected) < target_n:
        deficits = {
            family: (targets[family] - counts[family]) / max(1, targets[family])
            for family in FAMILIES
            if counts[family] < targets[family] and family not in exhausted
        }
        if not deficits:
            break
        family = max(deficits, key=lambda f: (deficits[f], -FAMILIES.index(f)))
        pool = pools[family]
        chosen: Record | None = None
        while pointers[family] < len(pool):
            candidate = pool[pointers[family]]
            pointers[family] += 1
            if eligible(candidate):
                chosen = candidate
                break
        if chosen is None:
            exhausted.add(family)
        else:
            add(chosen)

    fill_order = sorted(candidates, key=rank)
    for record in fill_order:
        if len(selected) >= target_n:
            break
        if eligible(record):
            add(record)

    for record in selected:
        record.selection_status = "SELECTED"
        record.selection_reason = "SELECTED_SOFT_MORPHOLOGY_AND_SOURCE_BALANCE"
        record.inclusion_reason = (
            "eligible morphology neighbour; deterministic source quota and soft multi-family balance"
        )
    for record in candidates:
        if record.selection_status == "ELIGIBLE":
            record.selection_status = "EXCLUDED"
            if record.patient_id_available and patient_counts[record.patient_group_id] >= max_known_patient_records:
                record.selection_reason = "PATIENT_RECORDING_CAP"
            elif (
                record.patient_id_available
                and patient_counts[record.patient_group_id] > 0
                and not (record.family_set() - patient_families[record.patient_group_id])
            ):
                record.selection_reason = "REDUNDANT_PATIENT_MORPHOLOGY"
            else:
                record.selection_reason = "SOURCE_QUOTA_SOFT_BALANCE"
    return selected


def validate_selected_signals(records: list[Record], repo_root: Path, workers: int) -> None:
    tasks: list[tuple[str, str, int, int]] = []
    path_to_records: dict[str, list[Record]] = collections.defaultdict(list)
    for record in records:
        if record.selection_status != "SELECTED":
            continue
        native = str(repo_root / record.original_data_path)
        tasks.append((native, record.signal_format, record.num_leads, record.num_samples))
        path_to_records[native].append(record)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path, status, issue in pool.map(_validate_signal_task, tasks, chunksize=64):
            for record in path_to_records[path]:
                record.validation_status = status
                record.validation_issues = issue
                if status != "VALID_SIGNAL":
                    record.selection_status = "EXCLUDED"
                    record.selection_reason = "INVALID_WAVEFORM_CONTENT"
                    record.inclusion_reason = ""


def hash_and_deduplicate_selected(
    records: list[Record],
    repo_root: Path,
    v2_manifest_path: Path,
    workers: int,
    duplicate_findings: list[dict[str, Any]],
) -> None:
    selected = [record for record in records if record.selection_status == "SELECTED"]
    paths = sorted({str(repo_root / record.original_data_path) for record in selected})
    hashes: dict[str, str] = {}
    errors: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path, digest, error in pool.map(_hash_task, paths, chunksize=16):
            if digest:
                hashes[path] = digest
            else:
                errors[path] = error
    for record in selected:
        path = str(repo_root / record.original_data_path)
        record.data_sha256 = hashes.get(path, "")
        if path in errors:
            record.selection_status = "EXCLUDED"
            record.selection_reason = "SIGNAL_HASH_FAILED"
            record.validation_status = "INVALID_SIGNAL"
            record.validation_issues = errors[path]

    v2_hash_to_rows: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    if v2_manifest_path.exists():
        for row in read_csv_rows(v2_manifest_path):
            digest = row.get("data_sha256", "").strip()
            if digest:
                v2_hash_to_rows[digest].append(row)

    by_hash: dict[str, list[Record]] = collections.defaultdict(list)
    for record in records:
        if record.selection_status == "SELECTED" and record.data_sha256:
            by_hash[record.data_sha256].append(record)

    for digest, group in sorted(by_hash.items()):
        if digest in v2_hash_to_rows:
            for record in group:
                record.selection_status = "EXCLUDED"
                record.selection_reason = "EXISTING_V2_WAVEFORM_HASH_OVERLAP"
                record.inclusion_reason = ""
                record.v2_overlap = 1
                record.v2_global_record_id = "|".join(
                    sorted({row["global_record_id"] for row in v2_hash_to_rows[digest]})
                )
                duplicate_findings.append(
                    {
                        "finding_type": "EXISTING_V2_WAVEFORM_HASH_OVERLAP",
                        "scope": "cross-dataset",
                        "key": digest,
                        "kept_record_id": record.v2_global_record_id,
                        "excluded_record_id": record.fcmae_record_id,
                        "sha256": digest,
                        "note": "Selected raw data file is byte-identical to a V2 waveform",
                    }
                )
            continue
        if len(group) <= 1:
            continue
        keep = sorted(group, key=lambda r: (r.source_dataset, r.fcmae_record_id))[0]
        for record in sorted(group, key=lambda r: (r.source_dataset, r.fcmae_record_id))[1:]:
            record.selection_status = "EXCLUDED"
            record.selection_reason = "DUPLICATE_WAVEFORM_SHA256"
            record.inclusion_reason = ""
            duplicate_findings.append(
                {
                    "finding_type": "DUPLICATE_WAVEFORM_SHA256",
                    "scope": "cross-dataset" if record.source_dataset != keep.source_dataset else record.source_dataset,
                    "key": digest,
                    "kept_record_id": keep.fcmae_record_id,
                    "excluded_record_id": record.fcmae_record_id,
                    "sha256": digest,
                    "note": "Byte-identical native waveform file; one record retained",
                }
            )


def create_relative_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target.resolve(), start=link.parent.resolve())
    os.symlink(relative_target, link)


def materialize_references(records: list[Record], repo_root: Path, output_dir: Path) -> None:
    for record in records:
        if record.selection_status != "SELECTED":
            continue
        native_data = repo_root / record.original_data_path
        record_dir = output_dir / "records" / record.source_dataset / record.fcmae_record_id
        data_link = record_dir / native_data.name
        create_relative_symlink(native_data, data_link)
        record.data_ref_path = data_link.relative_to(output_dir).as_posix()
        if record.signal_format == "hdf5":
            record.signal_ref_path = record.data_ref_path
        else:
            native_header = repo_root / record.original_header_path
            header_link = record_dir / native_header.name
            create_relative_symlink(native_header, header_link)
            record.header_ref_path = header_link.relative_to(output_dir).as_posix()
            record.signal_ref_path = header_link.with_suffix("").relative_to(output_dir).as_posix()
            record.header_sha256 = sha256_file(native_header)


AUDIT_FIELDS = [field.name for field in Record.__dataclass_fields__.values()]
MANIFEST_FIELDS = [
    "fcmae_record_id",
    "source_dataset",
    "source_collection",
    "original_record_id",
    "canonical_record_id",
    "patient_id",
    "patient_id_available",
    "patient_group_id",
    "patient_group_is_synthetic",
    "original_signal_path",
    "original_header_path",
    "original_data_path",
    "signal_ref_path",
    "header_ref_path",
    "data_ref_path",
    "signal_format",
    "sampling_frequency",
    "duration_seconds",
    "num_leads",
    "num_samples",
    "lead_names",
    "original_diagnosis_codes",
    "original_diagnosis_statements",
    "normalized_diagnoses",
    "morphology_normal_baseline",
    "morphology_ischemic_stt",
    "morphology_atrial_rhythm",
    "morphology_structural_voltage",
    "morphology_conduction_other",
    "morphology_families",
    "n_morphology_families",
    "exact_target_flags",
    "exact_target_decision",
    "v2_overlap",
    "v2_global_record_id",
    "inclusion_reason",
    "selection_seed",
    "validation_status",
    "validation_issues",
    "age",
    "sex",
    "source_extra_json",
    "data_sha256",
    "header_sha256",
]


def source_summaries(
    records: list[Record], quotas: dict[str, int], meeti_audit: dict[str, Any]
) -> list[dict[str, Any]]:
    final_total = sum(record.selection_status == "SELECTED" for record in records)
    rows: list[dict[str, Any]] = []
    for source in list(CHALLENGE_SOURCES) + ["chapman_shaoxing_ningbo", "sphdb"]:
        source_records = [record for record in records if record.source_dataset == source]
        selected = [record for record in source_records if record.selection_status == "SELECTED"]
        known_patients = {record.patient_id for record in selected if record.patient_id_available}
        patient_counts = collections.Counter(
            record.patient_group_id for record in selected if record.patient_id_available
        )
        decisions = collections.Counter(record.selection_reason for record in source_records)
        rows.append(
            {
                "source_dataset": source,
                "records_discovered": len(source_records),
                "source_quota": quotas[source],
                "records_selected": len(selected),
                "selected_percent": round(100 * len(selected) / final_total, 4) if final_total else 0,
                "unique_patients_known": len(known_patients),
                "effective_patient_groups": len({record.patient_group_id for record in selected}),
                "selected_missing_patient_id": sum(not record.patient_id_available for record in selected),
                "known_patients_with_multiple_selected_ecgs": sum(v > 1 for v in patient_counts.values()),
                "max_selected_ecgs_per_known_patient": max(patient_counts.values(), default=0),
                "excluded_v2_record_overlap": decisions["EXISTING_V2_RECORD_OVERLAP"],
                "excluded_v2_hash_overlap": decisions["EXISTING_V2_WAVEFORM_HASH_OVERLAP"],
                "reserved_exact_or_uncertain_target": decisions[
                    "EXACT_OR_UNCERTAIN_DOWNSTREAM_TARGET_RESERVED"
                ],
                "excluded_duplicate": decisions["DUPLICATE_RECORD_ID_WITHIN_SOURCE"]
                + decisions["DUPLICATE_WAVEFORM_SHA256"],
                "excluded_invalid_signal_or_metadata": decisions[
                    "INVALID_OR_MISSING_SIGNAL_METADATA"
                ]
                + decisions["INVALID_WAVEFORM_CONTENT"]
                + decisions["SIGNAL_HASH_FAILED"],
                "excluded_no_relevant_morphology": decisions["NO_RELEVANT_MORPHOLOGY_FAMILY"],
                "excluded_source_quota": decisions["SOURCE_QUOTA_SOFT_BALANCE"],
                "excluded_patient_cap_or_redundancy": decisions["PATIENT_RECORDING_CAP"]
                + decisions["REDUNDANT_PATIENT_MORPHOLOGY"],
                "patient_id_note": (
                    "source-native patient IDs available for selected records"
                    if known_patients
                    else "patient IDs unavailable; effective groups are explicitly synthetic per record"
                ),
            }
        )
    meeti_count = int(meeti_audit.get("mat_files", 0))
    rows.append(
        {
            "source_dataset": "meeti",
            "records_discovered": meeti_count,
            "source_quota": 0,
            "records_selected": 0,
            "selected_percent": 0,
            "unique_patients_known": 0,
            "effective_patient_groups": 0,
            "selected_missing_patient_id": 0,
            "known_patients_with_multiple_selected_ecgs": 0,
            "max_selected_ecgs_per_known_patient": 0,
            "excluded_v2_record_overlap": 0,
            "excluded_v2_hash_overlap": 0,
            "reserved_exact_or_uncertain_target": 0,
            "excluded_duplicate": 0,
            "excluded_invalid_signal_or_metadata": meeti_count,
            "excluded_no_relevant_morphology": 0,
            "excluded_source_quota": 0,
            "excluded_patient_cap_or_redundancy": 0,
            "patient_id_note": "subject IDs exist in paths, but no locally usable raw signal was present",
        }
    )
    return rows


def morphology_summaries(records: list[Record]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [record for record in records if record.selection_status == "SELECTED"]
    total = len(selected)
    family_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        count = sum(family in record.family_set() for record in selected)
        family_rows.append(
            {
                "morphology_family": family,
                "records": count,
                "percent_of_selected_records": round(100 * count / total, 4) if total else 0,
                "note": "non-exclusive; percentages sum above 100% when records are multi-family",
            }
        )
    multi = sum(record.n_morphology_families > 1 for record in selected)
    family_rows.append(
        {
            "morphology_family": "multi_family_records",
            "records": multi,
            "percent_of_selected_records": round(100 * multi / total, 4) if total else 0,
            "note": "one physical ECG record with more than one morphology flag",
        }
    )
    combo_counts = collections.Counter(record.morphology_families for record in selected)
    combo_rows = [
        {
            "morphology_combination": combo,
            "records": count,
            "percent_of_selected_records": round(100 * count / total, 4) if total else 0,
        }
        for combo, count in sorted(combo_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return family_rows, combo_rows


def exclusion_summaries(records: list[Record], meeti_audit: dict[str, Any]) -> list[dict[str, Any]]:
    counts = collections.Counter(
        record.selection_reason for record in records if record.selection_status != "SELECTED"
    )
    if meeti_audit.get("selection_decision") == "EXCLUDE_SOURCE_MISSING_RAW_SIGNAL":
        counts["MEETI_LOCAL_RAW_SIGNAL_UNAVAILABLE"] += int(meeti_audit.get("mat_files", 0))
    return [
        {"exclusion_reason": reason, "records": count}
        for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if reason
    ]


def build_summary(
    records: list[Record],
    source_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    exclusion_rows: list[dict[str, Any]],
    duplicate_findings: list[dict[str, Any]],
    v2_expected: tuple[int, int],
    meeti_audit: dict[str, Any],
) -> dict[str, Any]:
    selected = [record for record in records if record.selection_status == "SELECTED"]
    known_patient_keys = {
        f"{record.source_dataset}::{record.patient_id}"
        for record in selected
        if record.patient_id_available
    }
    effective_groups = {
        f"{record.source_dataset}::{record.patient_group_id}" for record in selected
    }
    v2_record_overlap = sum(
        record.selection_reason == "EXISTING_V2_RECORD_OVERLAP" for record in records
    )
    v2_hash_overlap = sum(
        record.selection_reason == "EXISTING_V2_WAVEFORM_HASH_OVERLAP" for record in records
    )
    return {
        "total_selected_ecg_records": len(selected),
        "total_unique_patients": None,
        "total_unique_patients_note": (
            "Not identifiable because several source distributions omit patient IDs. "
            "Use total_unique_patients_known as a lower bound and total_effective_patient_groups "
            "as the leakage-control grouping count."
        ),
        "total_unique_patients_known": len(known_patient_keys),
        "total_effective_patient_groups": len(effective_groups),
        "selected_records_missing_patient_identifier": sum(
            not record.patient_id_available for record in selected
        ),
        "selected_records_with_incomplete_metadata": sum(
            not record.patient_id_available
            or is_missing_value(record.age)
            or is_missing_value(record.sex)
            or has_unmapped_diagnosis(record)
            for record in selected
        ),
        "selected_records_with_unmapped_diagnosis_codes": sum(
            has_unmapped_diagnosis(record) for record in selected
        ),
        "selected_multi_family_records": sum(
            record.n_morphology_families > 1 for record in selected
        ),
        "selected_multi_diagnosis_records": sum(
            len([code for code in record.original_diagnosis_codes.split("|") if code]) > 1
            for record in selected
        ),
        "exact_target_records_identified": sum(bool(record.exact_target_flags) for record in records),
        "exact_target_records_reserved_outside_v2": sum(
            record.selection_reason == "EXACT_OR_UNCERTAIN_DOWNSTREAM_TARGET_RESERVED"
            for record in records
        ),
        "selected_exact_target_records": sum(bool(record.exact_target_flags) for record in selected),
        "records_excluded_existing_v2_record_id": v2_record_overlap,
        "records_excluded_existing_v2_waveform_hash": v2_hash_overlap,
        "expected_v2_chapman_records": v2_expected[0],
        "expected_v2_ptbxl_records": v2_expected[1],
        "v2_overlap_reconciliation": (
            "PASS"
            if v2_record_overlap == sum(v2_expected)
            else "FAIL_MISMATCH_REVIEW_REQUIRED"
        ),
        "records_excluded_for_duplication": sum(
            finding["finding_type"]
            in {"DUPLICATE_RECORD_ID_WITHIN_SOURCE", "DUPLICATE_WAVEFORM_SHA256"}
            for finding in duplicate_findings
        ),
        "meeti_records_excluded_missing_raw_signal": (
            int(meeti_audit.get("mat_files", 0))
            if meeti_audit.get("selection_decision") == "EXCLUDE_SOURCE_MISSING_RAW_SIGNAL"
            else 0
        ),
        "records_per_source": {
            row["source_dataset"]: row["records_selected"] for row in source_rows
        },
        "unique_patients_known_per_source": {
            row["source_dataset"]: row["unique_patients_known"] for row in source_rows
        },
        "effective_patient_groups_per_source": {
            row["source_dataset"]: row["effective_patient_groups"] for row in source_rows
        },
        "records_per_morphology_family": {
            row["morphology_family"]: row["records"] for row in family_rows
        },
        "exclusions_by_reason": {
            row["exclusion_reason"]: row["records"] for row in exclusion_rows
        },
        "source_distribution_percent": {
            row["source_dataset"]: row["selected_percent"] for row in source_rows
        },
        "morphology_distribution_percent_nonexclusive": {
            row["morphology_family"]: row["percent_of_selected_records"]
            for row in family_rows
        },
    }


def render_release_readme(
    summary: dict[str, Any], source_rows: list[dict[str, Any]], family_rows: list[dict[str, Any]]
) -> str:
    source_table = "\n".join(
        f"| {row['source_dataset']} | {row['records_selected']:,} | "
        f"{row['unique_patients_known']:,} | {row['effective_patient_groups']:,} | "
        f"{row['selected_missing_patient_id']:,} | {row['selected_percent']:.2f}% |"
        for row in source_rows
    )
    family_table = "\n".join(
        f"| {row['morphology_family']} | {row['records']:,} | "
        f"{row['percent_of_selected_records']:.2f}% |"
        for row in family_rows
    )
    return f"""# FCMAE morphology-focused raw ECG corpus — v1

This release contains **{summary['total_selected_ecg_records']:,} unique raw-signal ECG
records** selected for self-supervised ConvNeXt-V2 FCMAE pretraining. It is not
a STEMI/AF/LVH/NORMAL classification dataset. Diagnostic information is retained
for provenance and morphology-aware sampling only.

No paper ECG images were generated. `records/` contains relative symbolic links
to the immutable native signal/header files under `dataset-files/`.

## Patient accounting

There are **{summary['total_unique_patients_known']:,} source-native unique patient IDs**
among selected records and **{summary['total_effective_patient_groups']:,} effective
patient groups** after adding explicitly synthetic one-record groups where a source
does not distribute patient identifiers. A truthful global unique-patient total is
therefore not identifiable. **{summary['selected_records_missing_patient_identifier']:,}
selected records lack a source-native patient ID.** Never reinterpret synthetic
group IDs as known people.

Known patients were capped at two recordings; a second was eligible only when it
added a morphology family for that patient.

## Source distribution

| source | records | known patients | effective groups | missing patient ID | corpus share |
| --- | ---: | ---: | ---: | ---: | ---: |
{source_table}

MEETI contributes zero records because the local download contains
{summary['meeti_records_excluded_missing_raw_signal']:,} MAT report/interpretation
records but no locally usable raw waveform component. See `meeti_audit.json`.

## Morphology distribution

Morphology families are non-exclusive, so percentages intentionally sum above
100%. **{summary['selected_multi_diagnosis_records']:,} records have multiple
source diagnoses and {summary['selected_multi_family_records']:,} have multiple
morphology families.** Every multi-label ECG still appears only once in
`manifest.csv`.

| morphology family/statistic | records | percent of corpus |
| --- | ---: | ---: |
{family_table}

## V2 exclusion and exact-target policy

The record-level V2 exclusion reconciled as **{summary['v2_overlap_reconciliation']}**:
{summary['records_excluded_existing_v2_record_id']:,} canonical record-ID overlaps
were excluded ({summary['expected_v2_chapman_records']:,} Chapman and
{summary['expected_v2_ptbxl_records']:,} PTB-XL expected). A further
{summary['records_excluded_existing_v2_waveform_hash']:,} byte-identical signal
files were excluded by hash. The full ledger is `v2_overlap.csv`.

`exact_target_audit.csv` identifies {summary['exact_target_records_identified']:,}
records with exact AF/LVH or the undocumented LVH-like Chapman code; of these,
{summary['exact_target_records_reserved_outside_v2']:,} non-V2 records were reserved.
The corpus contains **{summary['selected_exact_target_records']} selected exact-target records**.
No source provides an explicit STEMI label. MI, acute MI, ischemia, ST elevation,
ST depression, and repolarization abnormalities are retained under their original
names as morphology neighbours; none is relabelled STEMI. Atrial flutter and other
supraventricular rhythms are not relabelled AF. Voltage/axis/chamber findings are
not relabelled LVH.

## Integrity and limitations

`manifest.csv` preserves source paths, record IDs, available patient IDs, original
diagnostic codes/statements, normalized descriptions, non-exclusive morphology
flags, sampling details, inclusion reason, and SHA-256. `checksums.sha256` verifies
the release metadata and every referenced signal/header target.

The source vocabulary did not define every diagnosis code. There are
**{summary['selected_records_with_unmapped_diagnosis_codes']:,} selected records**
with at least one unmapped code; the original code is preserved and
`unmapped_diagnoses.csv` reports its frequency. Such a code never creates a
morphology family by itself.

Exact raw-file SHA-256 catches byte-identical records, but cannot prove that two
signals are unrelated after resampling, scaling, cropping, or re-encoding. Patient
linkage is unavailable for Chapman and most Challenge sources, including PTB and
INCART, so patient-level duplication in those sources cannot be fully ruled out.
Cross-institution patient linkage is also unavailable.

The curated source and morphology shares are sampling controls, not epidemiologic
prevalence estimates. Review `candidate_audit.csv`, `exclusions.csv`, and the
configuration before changing the downstream use of this corpus.

This symlink release does not relicense source data. Access, use, and
redistribution remain governed by each source distribution's license and data
use terms.
"""


def write_release_files(
    output_dir: Path,
    records: list[Record],
    source_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    combo_rows: list[dict[str, Any]],
    exclusion_rows: list[dict[str, Any]],
    duplicate_findings: list[dict[str, Any]],
    meeti_audit: dict[str, Any],
    summary: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    selected = [record for record in records if record.selection_status == "SELECTED"]
    exclusions = [record for record in records if record.selection_status != "SELECTED"]
    v2_overlap = [record for record in records if record.v2_overlap]
    exact_targets = [record for record in records if record.exact_target_flags]
    missing_metadata = [
        {
            **asdict(record),
            "missing_fields": "|".join(
                field_name
                for field_name, missing in (
                    ("patient_id", not record.patient_id_available),
                    ("age", is_missing_value(record.age)),
                    ("sex", is_missing_value(record.sex)),
                    (
                        "diagnosis_normalization",
                        has_unmapped_diagnosis(record),
                    ),
                )
                if missing
            ),
        }
        for record in selected
        if not record.patient_id_available
        or is_missing_value(record.age)
        or is_missing_value(record.sex)
        or has_unmapped_diagnosis(record)
    ]

    unmapped_counts: collections.Counter[tuple[str, str, str]] = collections.Counter()
    for record in records:
        codes = record.original_diagnosis_codes.split("|")
        names = record.normalized_diagnoses.split("|")
        for code, name in zip(codes, names):
            if "UNMAPPED_CODE" in name:
                unmapped_counts[(record.source_dataset, code, record.selection_status)] += 1
    unmapped_rows: list[dict[str, Any]] = []
    for source, code in sorted({(key[0], key[1]) for key in unmapped_counts}):
        by_status = {
            status: unmapped_counts[(source, code, status)]
            for status in ("SELECTED", "RESERVED", "EXCLUDED")
        }
        unmapped_rows.append(
            {
                "source_dataset": source,
                "diagnosis_code": code,
                "normalized_description": "UNMAPPED_CODE",
                "candidate_records": sum(by_status.values()),
                "selected_records": by_status["SELECTED"],
                "reserved_records": by_status["RESERVED"],
                "excluded_records": by_status["EXCLUDED"],
                "policy": "preserve original; never assign morphology from this code alone",
            }
        )

    write_csv(output_dir / "manifest.csv", (asdict(r) for r in selected), MANIFEST_FIELDS)
    write_csv(output_dir / "candidate_audit.csv", (asdict(r) for r in records), AUDIT_FIELDS)
    write_csv(output_dir / "exclusions.csv", (asdict(r) for r in exclusions), AUDIT_FIELDS)
    write_csv(
        output_dir / "v2_overlap.csv",
        (asdict(r) for r in v2_overlap),
        [
            "fcmae_record_id", "source_dataset", "original_record_id", "canonical_record_id",
            "v2_global_record_id", "original_diagnosis_codes", "original_diagnosis_statements",
            "exact_target_flags", "selection_reason", "original_data_path", "data_sha256",
        ],
    )
    write_csv(
        output_dir / "exact_target_audit.csv",
        (asdict(r) for r in exact_targets),
        [
            "fcmae_record_id", "source_dataset", "original_record_id", "canonical_record_id",
            "patient_id", "original_diagnosis_codes", "original_diagnosis_statements",
            "exact_target_flags", "exact_target_decision", "v2_overlap", "v2_global_record_id",
            "selection_status", "selection_reason", "original_data_path",
        ],
    )
    duplicate_fields = [
        "finding_type", "scope", "key", "kept_record_id", "excluded_record_id", "sha256", "note"
    ]
    write_csv(output_dir / "duplicates.csv", duplicate_findings, duplicate_fields)
    write_csv(output_dir / "source_summary.csv", source_rows, list(source_rows[0]))
    write_csv(output_dir / "morphology_summary.csv", family_rows, list(family_rows[0]))
    write_csv(output_dir / "morphology_combinations.csv", combo_rows, list(combo_rows[0]))
    write_csv(output_dir / "exclusion_summary.csv", exclusion_rows, list(exclusion_rows[0]))
    write_csv(
        output_dir / "missing_metadata.csv",
        missing_metadata,
        MANIFEST_FIELDS + ["missing_fields"],
    )
    write_csv(
        output_dir / "unmapped_diagnoses.csv",
        unmapped_rows,
        list(unmapped_rows[0]) if unmapped_rows else ["source_dataset", "diagnosis_code"],
    )
    write_json(output_dir / "meeti_audit.json", meeti_audit)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "BUILD_PROVENANCE.json", provenance)
    (output_dir / "README.md").write_text(
        render_release_readme(summary, source_rows, family_rows), encoding="utf-8"
    )


def write_checksums(output_dir: Path, records: list[Record]) -> None:
    entries: dict[str, str] = {}
    for record in records:
        if record.selection_status != "SELECTED":
            continue
        entries[record.data_ref_path] = record.data_sha256
        if record.header_ref_path:
            entries[record.header_ref_path] = record.header_sha256
    for path in output_dir.iterdir():
        if path.is_file() and not path.is_symlink() and path.name != "checksums.sha256":
            entries[path.relative_to(output_dir).as_posix()] = sha256_file(path)
    with (output_dir / "checksums.sha256").open("w", encoding="utf-8") as fh:
        for relative, digest in sorted(entries.items()):
            fh.write(f"{digest}  {relative}\n")


def git_state(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).splitlines()
        return {"commit": commit, "working_tree_dirty": bool(status), "status_lines": status}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def make_provenance(
    repo_root: Path,
    pipeline_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    paths = config["paths"]
    inputs = {
        "config": config_path,
        "builder": pipeline_dir / "build_fcmae_dataset.py",
        "requirements": pipeline_dir / "requirements.txt",
        "challenge_diagnosis_map": pipeline_dir / "challenge_diagnoses.csv",
        "chapman_condition_map": repo_root / paths["chapman"] / "ConditionNames_SNOMED-CT.csv",
        "sphdb_code_map": repo_root / paths["sphdb"] / "code.csv",
        "sphdb_metadata": repo_root / paths["sphdb"] / "metadata.csv",
        "ptbxl_metadata": repo_root / paths["ptbxl_metadata"],
        "v2_metadata": repo_root / paths["v2_metadata"],
        "v2_manifest": repo_root / paths["v2_manifest"],
    }
    return {
        "release_name": config["release_name"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "build_fcmae_dataset.py 1.0",
        "python": sys.version,
        "selection_seed": config["selection_seed"],
        "config": config,
        "git": git_state(repo_root),
        "input_sha256": {
            name: sha256_file(path) if path.exists() and path.is_file() else None
            for name, path in inputs.items()
        },
        "source_references": {
            "challenge_2020": "https://physionet.org/content/challenge-2020/1.0.2/",
            "chapman_shaoxing_ningbo": "https://physionet.org/content/ecg-arrhythmia/1.0.0/",
            "sphdb": "https://springernature.figshare.com/collections/5779802/1",
            "meeti": "https://doi.org/10.1038/s41597-026-06796-1",
            "challenge_diagnosis_map": "https://github.com/physionetchallenges/evaluation-2020",
        },
        "clinical_decisions": {
            "labels_are_training_targets": False,
            "multi_label_records_are_duplicated": False,
            "exact_af": "reserved",
            "exact_lvh": "reserved",
            "explicit_stemi_labels_found": False,
            "mi_ischemia_st_elevation": "retained under original diagnosis as ischemic/ST-T morphology",
            "chapman_55827005": "reserved as undocumented LVH-like cohort",
            "meeti": "excluded when local raw waveform component is unavailable",
        },
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.json",
    )
    parser.add_argument("--workers", type=int, default=min(8, max(1, os.cpu_count() or 1)))
    parser.add_argument(
        "--skip-meeti-scan",
        action="store_true",
        help="skip the expensive MEETI modality inventory (not allowed for a final release)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="inventory and select but do not validate/hash/materialize a release",
    )
    args = parser.parse_args(argv)

    pipeline_dir = Path(__file__).resolve().parent
    repo_root = pipeline_dir.parents[1]
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    seed = int(config["selection_seed"])
    quotas = {key: int(value) for key, value in config["source_quotas"].items()}
    shares = {
        key: float(value) for key, value in config["morphology_minimum_shares"].items()
    }
    if set(shares) != set(FAMILIES):
        raise ValueError(f"morphology_minimum_shares must define exactly {FAMILIES}")
    if set(quotas) != set(CHALLENGE_SOURCES) | {"chapman_shaoxing_ningbo", "sphdb"}:
        raise ValueError("source_quotas keys do not match the configured source set")

    output_parent = repo_root / paths["output_parent"]
    final_output = output_parent / config["release_name"]
    if final_output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing release: {final_output}; move it aside or change release_name"
        )
    build_output = output_parent / f".{config['release_name']}.building-{os.getpid()}"
    if build_output.exists():
        raise FileExistsError(f"stale build directory exists: {build_output}")

    challenge_root = repo_root / paths["challenge_2020"]
    chapman_root = repo_root / paths["chapman"]
    sphdb_root = repo_root / paths["sphdb"]
    meeti_root = repo_root / paths["meeti"]
    required = [
        challenge_root,
        chapman_root / "WFDBRecords",
        sphdb_root / "metadata.csv",
        meeti_root,
        repo_root / paths["ptbxl_metadata"],
        repo_root / paths["v2_metadata"],
        repo_root / paths["v2_manifest"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required inputs are missing:\n" + "\n".join(missing))

    print("[1/9] loading diagnosis maps and V2 overlap keys", flush=True)
    diagnosis_names = load_diagnosis_names(pipeline_dir, chapman_root)
    ptbxl_lookup = load_ptbxl_lookup(repo_root / paths["ptbxl_metadata"])
    v2_chapman, v2_ptbxl = load_v2_overlap(repo_root / paths["v2_metadata"])
    print(
        f"      V2 keys: Chapman={len(v2_chapman):,}, PTB-XL={len(v2_ptbxl):,}", flush=True
    )

    print("[2/9] inventorying Challenge-2020 WFDB records", flush=True)
    records = inventory_challenge(
        challenge_root, repo_root, diagnosis_names, ptbxl_lookup, seed
    )
    print(f"      Challenge records={len(records):,}", flush=True)

    print("[3/9] inventorying Chapman/Shaoxing/Ningbo WFDB records", flush=True)
    chapman_records = inventory_chapman(chapman_root, repo_root, diagnosis_names, seed)
    records.extend(chapman_records)
    print(f"      Chapman records={len(chapman_records):,}", flush=True)

    print("[4/9] inventorying SPHDB records", flush=True)
    sphdb_records, _sph_names = inventory_sphdb(sphdb_root, repo_root, seed)
    records.extend(sphdb_records)
    print(f"      SPHDB records={len(sphdb_records):,}", flush=True)

    print("[5/9] auditing local MEETI modalities", flush=True)
    if args.skip_meeti_scan:
        if not args.dry_run:
            raise ValueError("--skip-meeti-scan may only be used with --dry-run")
        meeti_audit = {
            "root": rel_to_repo(meeti_root, repo_root),
            "mat_files": 0,
            "raw_signal_availability": "NOT_AUDITED",
            "selection_decision": "NOT_AUDITED",
            "note": "MEETI scan skipped in dry-run mode",
        }
    else:
        meeti_audit = audit_meeti(meeti_root, repo_root, seed)
    print(
        f"      MEETI MAT={int(meeti_audit.get('mat_files', 0)):,}; "
        f"raw={meeti_audit.get('raw_signal_availability')}",
        flush=True,
    )
    if meeti_audit.get("raw_signal_availability") == "AVAILABLE":
        raise RuntimeError(
            "MEETI audit found a possible waveform component. The conservative builder stops "
            "instead of silently excluding or misreading it; inspect meeti_audit logic."
        )

    print("[6/9] applying exclusions and deterministic soft balancing", flush=True)
    duplicate_findings: list[dict[str, Any]] = []
    apply_initial_exclusions(records, v2_chapman, v2_ptbxl, duplicate_findings)
    for source in quotas:
        selected = soft_select_source(
            [record for record in records if record.source_dataset == source],
            quotas[source],
            shares,
            int(config["max_records_per_known_patient"]),
            seed,
        )
        print(f"      {source}: selected before signal validation={len(selected):,}", flush=True)

    if args.dry_run:
        counts = collections.Counter(
            record.source_dataset for record in records if record.selection_status == "SELECTED"
        )
        print("dry run selected:", dict(counts), flush=True)
        selected_records = [record for record in records if record.selection_status == "SELECTED"]
        family_counts = collections.Counter(
            family for record in selected_records for family in record.family_set()
        )
        reason_counts = collections.Counter(
            record.selection_reason for record in records if record.selection_status != "SELECTED"
        )
        print("dry run morphology (non-exclusive):", dict(family_counts), flush=True)
        print("dry run exclusions:", dict(reason_counts), flush=True)
        return 0

    print("[7/9] validating selected waveform containers", flush=True)
    validate_selected_signals(records, repo_root, args.workers)
    print("[8/9] hashing signals and checking exact duplicates/V2 hashes", flush=True)
    hash_and_deduplicate_selected(
        records,
        repo_root,
        repo_root / paths["v2_manifest"],
        args.workers,
        duplicate_findings,
    )

    # Safety invariants must hold before anything is materialized.
    final_selected = [record for record in records if record.selection_status == "SELECTED"]
    assert len({record.fcmae_record_id for record in final_selected}) == len(final_selected)
    assert not any(record.v2_overlap for record in final_selected)
    assert not any(record.exact_target_flags for record in final_selected)
    assert not any(record.n_morphology_families == 0 for record in final_selected)
    assert all(record.validation_status == "VALID_SIGNAL" for record in final_selected)
    assert all(record.data_sha256 for record in final_selected)
    actual_v2_overlap = sum(
        record.selection_reason == "EXISTING_V2_RECORD_OVERLAP" for record in records
    )
    expected_v2_overlap = len(v2_chapman) + len(v2_ptbxl)
    if actual_v2_overlap != expected_v2_overlap:
        raise RuntimeError(
            f"V2 overlap reconciliation failed: local matches={actual_v2_overlap}, "
            f"V2 keys={expected_v2_overlap}"
        )

    print("[9/9] materializing symlink references and audit files", flush=True)
    try:
        build_output.mkdir(parents=True)
        materialize_references(records, repo_root, build_output)
        source_rows = source_summaries(records, quotas, meeti_audit)
        family_rows, combo_rows = morphology_summaries(records)
        exclusion_rows = exclusion_summaries(records, meeti_audit)
        summary = build_summary(
            records,
            source_rows,
            family_rows,
            exclusion_rows,
            duplicate_findings,
            (len(v2_chapman), len(v2_ptbxl)),
            meeti_audit,
        )
        provenance = make_provenance(
            repo_root, pipeline_dir, config_path, config, summary
        )
        write_release_files(
            build_output,
            records,
            source_rows,
            family_rows,
            combo_rows,
            exclusion_rows,
            duplicate_findings,
            meeti_audit,
            summary,
            provenance,
        )
        write_checksums(build_output, records)
        os.replace(build_output, final_output)
    except Exception:
        if build_output.exists():
            shutil.rmtree(build_output)
        raise

    print(f"release created: {final_output}", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
