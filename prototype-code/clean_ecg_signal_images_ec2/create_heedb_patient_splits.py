#!/usr/bin/env python3
"""Make patient-disjoint 80/10/10 HEEDB ECG splits from the audited 500 Hz tree.

Default: plan and report counts without copying data. --execute makes an
independent train/validation/test tree in extracted/ECG_500Hz_patient_splits.
The original and combined ECG trees are read only. Future ECG images must
inherit the split of their source record from record_manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
EXTRACTED = REPO / "dataset-files/HEEDB/extracted"
SOURCE = EXTRACTED / "ECG_combined_500Hz"
CONVERTED = EXTRACTED / "ECG_500Hz"
INDEX = EXTRACTED / "ecg_combined_patient_index.csv"
DESTINATION = EXTRACTED / "ECG_500Hz_patient_splits"
STAGING = EXTRACTED / ".ECG_500Hz_patient_splits.building"
PLAN_REPORT = EXTRACTED / "ecg_patient_split_plan.json"
SPLITS = ("train", "validation", "test")
TARGET = np.array([0.8, 0.1, 0.1], dtype=np.float64)
TRIALS = 256


def read_index() -> list[dict[str, str]]:
    with INDEX.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"record_path", "patient_key", "site", "codes_physician", "labels", "label_count"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns in {INDEX}: {required - set(reader.fieldnames or [])}")
        rows = list(reader)
    if not rows or len({row["record_path"] for row in rows}) != len(rows):
        raise ValueError("Empty or duplicate record paths in patient index")
    if any(not row["patient_key"] for row in rows):
        raise ValueError("Patient index contains a missing patient ID")
    return rows


def plan(rows: list[dict[str, str]]) -> tuple[dict[str, str], dict]:
    patients = sorted({row["patient_key"] for row in rows})
    patient_number = {key: index for index, key in enumerate(patients)}
    named_labels = sorted({label for row in rows for label in row["labels"].split(";") if label})
    features = ["Records", *named_labels, "No target label", "Site I0001", "Site I0006",
                "Originally 250 Hz", "Multiple target labels"]
    feature_number = {name: index for index, name in enumerate(features)}
    matrix = np.zeros((len(patients), len(features)), dtype=np.int32)
    converted = {path.relative_to(CONVERTED).as_posix() for path in CONVERTED.rglob("*.hea")}
    if len(converted) != 5_428:
        raise ValueError(f"Expected 5,428 converted records, found {len(converted):,}")

    for row in rows:
        index = patient_number[row["patient_key"]]
        matrix[index, 0] += 1
        labels = [label for label in row["labels"].split(";") if label]
        for label in labels:
            matrix[index, feature_number[label]] += 1
        if not labels:
            matrix[index, feature_number["No target label"]] += 1
        matrix[index, feature_number[f"Site {row['site']}"]] += 1
        if row["record_path"] in converted:
            matrix[index, feature_number["Originally 250 Hz"]] += 1
        if len(labels) > 1:
            matrix[index, feature_number["Multiple target labels"]] += 1

    total = matrix.sum(axis=0, dtype=np.int64)
    if np.any(total == 0):
        raise ValueError(f"A balancing feature has zero records: {features[np.flatnonzero(total == 0)[0]]}")
    n_train = round(0.8 * len(patients))
    n_validation = round(0.1 * len(patients))
    n_test = len(patients) - n_train - n_validation
    best_score = None
    best_order = None
    best_counts = None
    best_seed = None
    for trial in range(TRIALS):
        seed = 42 + trial
        order = np.random.default_rng(seed).permutation(len(patients))
        train_counts = matrix[order[:n_train]].sum(axis=0, dtype=np.int64)
        validation_counts = matrix[order[n_train:n_train + n_validation]].sum(axis=0, dtype=np.int64)
        counts = np.stack((train_counts, validation_counts, total - train_counts - validation_counts))
        deviations = np.abs(counts / total[None, :] - TARGET[:, None]) * 100
        score = (float(deviations.max()), float(deviations.mean()), seed)
        if best_score is None or score < best_score:
            best_score, best_order, best_counts, best_seed = score, order.copy(), counts.copy(), seed

    assert best_order is not None and best_counts is not None and best_seed is not None
    assignment_codes = np.empty(len(patients), dtype=np.uint8)
    assignment_codes[best_order[:n_train]] = 0
    assignment_codes[best_order[n_train:n_train + n_validation]] = 1
    assignment_codes[best_order[n_train + n_validation:]] = 2
    assignment = {patient: SPLITS[int(assignment_codes[index])] for index, patient in enumerate(patients)}
    patient_counts = {SPLITS[index]: int(count) for index, count in enumerate((n_train, n_validation, n_test))}
    record_counts = {SPLITS[index]: int(best_counts[index, 0]) for index in range(3)}
    feature_report = {}
    for column, name in enumerate(features):
        feature_report[name] = {
            "total": int(total[column]),
            "counts": {SPLITS[index]: int(best_counts[index, column]) for index in range(3)},
            "percentages": {SPLITS[index]: round(100 * best_counts[index, column] / total[column], 4)
                            for index in range(3)},
        }
    report = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "planned",
        "source": str(SOURCE),
        "patient_index": str(INDEX),
        "target_percentages": {split: 100 * TARGET[index] for index, split in enumerate(SPLITS)},
        "selection": f"Best of {TRIALS} deterministic patient permutations; maximum feature percentage-point deviation minimized",
        "selected_seed": best_seed,
        "patients_total": len(patients),
        "records_total": len(rows),
        "patient_counts": patient_counts,
        "record_counts": record_counts,
        "patient_overlap_count": 0,
        "maximum_feature_deviation_percentage_points": round(best_score[0], 4),
        "mean_feature_deviation_percentage_points": round(best_score[1], 4),
        "features": feature_report,
    }
    return assignment, report


def signal_for_header(header: Path) -> Path:
    with header.open(encoding="utf-8") as handle:
        first = handle.readline().split()
        signal = handle.readline().split()
    if len(first) < 4 or first[0] != header.stem or first[1:4] != ["12", "500", "5000"]:
        raise ValueError(f"Unexpected WFDB record line: {header}")
    if not signal or Path(signal[0]).name != signal[0]:
        raise ValueError(f"Unexpected WFDB signal name: {header}")
    path = header.parent / signal[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def same_bytes(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    with a.open("rb") as left, b.open("rb") as right:
        while True:
            original = left.read(1024 * 1024)
            copied = right.read(1024 * 1024)
            if original != copied:
                return False
            if not original:
                return True


def write_manifests(rows: list[dict[str, str]], assignment: dict[str, str], report: dict) -> None:
    patient_records = Counter(row["patient_key"] for row in rows)
    patient_labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        patient_labels[row["patient_key"]].update(label for label in row["labels"].split(";") if label)
    with (STAGING / "patient_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("patient_key", "split", "record_count", "labels_union"))
        for patient in sorted(assignment):
            writer.writerow((patient, assignment[patient], patient_records[patient],
                             ";".join(sorted(patient_labels[patient]))))
    converted = {path.relative_to(CONVERTED).as_posix() for path in CONVERTED.rglob("*.hea")}
    with (STAGING / "record_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("record_path", "split_header_path", "patient_key", "split", "site",
                         "source_rate_hz", "codes_physician", "labels", "label_count"))
        for row in rows:
            split = assignment[row["patient_key"]]
            writer.writerow((row["record_path"], f"{split}/{row['record_path']}",
                             row["patient_key"], split, row["site"],
                             250 if row["record_path"] in converted else 500,
                             row["codes_physician"], row["labels"], row["label_count"]))
    (STAGING / "README.md").write_text(
        "# HEEDB 500 Hz patient-level splits\n\n"
        "All 193,924 WFDB records are assigned by BDSPPatientID to train, validation, "
        "or test. No patient appears in more than one split. The records contain "
        "12-lead signals (.hea with .mat or .dat companions), not rendered images. "
        "Any images generated from a record must inherit its `split` from "
        "`record_manifest.csv`; all variants of that image remain in the same split.\n\n"
        "The original 500 Hz records and the separately upsampled 250 Hz records "
        "were copied from the audited `ECG_combined_500Hz` tree. Files are independent "
        "copies. `patient_assignments.csv` is the frozen patient assignment, and "
        "`split_audit.json` contains counts and per-label proportions.\n",
        encoding="utf-8",
    )
    (STAGING / "split_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def materialize(rows: list[dict[str, str]], assignment: dict[str, str], report: dict) -> None:
    if DESTINATION.exists() or STAGING.exists():
        raise FileExistsError(f"Destination or staging directory exists: {DESTINATION}, {STAGING}")
    required_bytes = 0
    for row in rows:
        header = SOURCE / row["record_path"]
        signal = signal_for_header(header)
        required_bytes += header.stat().st_size + signal.stat().st_size
    free_bytes = shutil.disk_usage(EXTRACTED).free
    if free_bytes < required_bytes + 2 * 1024**3:
        raise OSError(f"Need {required_bytes:,} bytes plus 2 GiB margin; only {free_bytes:,} free")
    print(f"Copying {required_bytes / 1024**3:.2f} GiB to {DESTINATION}", flush=True)
    STAGING.mkdir()
    for split in SPLITS:
        (STAGING / split).mkdir()
    for index, row in enumerate(rows, 1):
        split = assignment[row["patient_key"]]
        source_header = SOURCE / row["record_path"]
        source_signal = signal_for_header(source_header)
        output_header = STAGING / split / row["record_path"]
        output_signal = output_header.parent / source_signal.name
        output_header.parent.mkdir(parents=True, exist_ok=True)
        if output_header.exists() or output_signal.exists():
            raise FileExistsError(f"Duplicate output: {output_header}")
        shutil.copy2(source_signal, output_signal)
        shutil.copy2(source_header, output_header)
        if index % 10_000 == 0 or index == len(rows):
            print(f"Copied {index:,}/{len(rows):,} records", flush=True)

    write_manifests(rows, assignment, report)
    print("Verifying copied files and patient separation...", flush=True)
    seen_records = set()
    seen_patients = {}
    for index, row in enumerate(rows, 1):
        split = assignment[row["patient_key"]]
        source_header = SOURCE / row["record_path"]
        source_signal = signal_for_header(source_header)
        output_header = STAGING / split / row["record_path"]
        output_signal = signal_for_header(output_header)
        if not same_bytes(source_header, output_header) or not same_bytes(source_signal, output_signal):
            raise OSError(f"Copied record differs from source: {row['record_path']}")
        if source_header.stat().st_ino == output_header.stat().st_ino or source_signal.stat().st_ino == output_signal.stat().st_ino:
            raise OSError(f"Source and output are hardlinked: {row['record_path']}")
        if row["record_path"] in seen_records:
            raise ValueError(f"Record appears twice: {row['record_path']}")
        seen_records.add(row["record_path"])
        prior = seen_patients.setdefault(row["patient_key"], split)
        if prior != split:
            raise ValueError(f"Patient appears across splits: {row['patient_key']}")
        if index % 10_000 == 0 or index == len(rows):
            print(f"Verified {index:,}/{len(rows):,} records", flush=True)
    if len(seen_records) != len(rows) or len(seen_patients) != report["patients_total"]:
        raise ValueError("Split coverage is incomplete")
    output_counts = {split: sum(1 for _ in (STAGING / split).rglob("*.hea")) for split in SPLITS}
    if output_counts != report["record_counts"]:
        raise ValueError(f"Output counts do not match plan: {output_counts}")
    signal_count = sum(1 for split in SPLITS for path in (STAGING / split).rglob("*")
                       if path.is_file() and path.suffix in {".mat", ".dat"})
    if signal_count != len(rows):
        raise ValueError(f"Expected {len(rows):,} signal files, found {signal_count:,}")
    with (STAGING / "patient_assignments.csv").open(newline="", encoding="utf-8") as handle:
        saved_patients = list(csv.DictReader(handle))
    if len(saved_patients) != len(assignment) or len({row["patient_key"] for row in saved_patients}) != len(assignment):
        raise ValueError("Saved patient assignment count or uniqueness is wrong")
    if any(assignment[row["patient_key"]] != row["split"] for row in saved_patients):
        raise ValueError("Saved patient assignment differs from validated assignment")
    with (STAGING / "record_manifest.csv").open(newline="", encoding="utf-8") as handle:
        saved_records = list(csv.DictReader(handle))
    if len(saved_records) != len(rows) or len({row["record_path"] for row in saved_records}) != len(rows):
        raise ValueError("Saved record manifest count or uniqueness is wrong")
    if any(assignment[row["patient_key"]] != row["split"] or
           row["split_header_path"] != f"{row['split']}/{row['record_path']}"
           for row in saved_records):
        raise ValueError("Saved record manifest contains a split mismatch")
    report["status"] = "validated"
    report["bytewise_verified_records"] = len(rows)
    report["output_record_counts"] = output_counts
    report["output_signal_count"] = signal_count
    (STAGING / "split_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.rename(STAGING, DESTINATION)
    print(f"Published {DESTINATION}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Copy and validate the split dataset")
    args = parser.parse_args()
    rows = read_index()
    assignment, report = plan(rows)
    PLAN_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Patient counts: {report['patient_counts']}")
    print(f"Record counts: {report['record_counts']}")
    print(f"Maximum feature deviation: {report['maximum_feature_deviation_percentage_points']:.4f} percentage points")
    print(f"Plan: {PLAN_REPORT}")
    if args.execute:
        materialize(rows, assignment, report)
    else:
        print("Dry run only. Add --execute to create split folders.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Split preparation stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
