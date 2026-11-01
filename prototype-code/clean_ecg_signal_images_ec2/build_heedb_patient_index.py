#!/usr/bin/env python3
"""Join every combined HEEDB ECG to its source patient and physician labels.

This reads the full source metadata and diagnosis CSVs, but writes only rows
for ECGs present in extracted/ECG_combined_500Hz. A complete, conflict-free
patient mapping is required before patient-level splitting.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
EXTRACTED = REPO / "dataset-files/HEEDB/extracted"
COMBINED = EXTRACTED / "ECG_combined_500Hz"
HEEDB = REPO / "dataset-files/HEEDB"
OUTPUT = EXTRACTED / "ecg_combined_patient_index.csv"
SITES = ("I0001", "I0006")


def load_classifier():
    path = EXTRACTED / "download_heedb_selected_ecgs_revised.py"
    spec = importlib.util.spec_from_file_location("heedb_selection_rules", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load label rules from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_name(raw: str) -> str:
    name = raw.strip().replace("\\", "/").lstrip("/")
    if name.endswith(".hea"):
        name = name[:-4]
    if name.startswith("WFDB/"):
        name = name[5:]
    if not name or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError(f"Invalid source FileName: {raw!r}")
    return name


def csv_rows(path: Path, required: set[str]):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        columns = next(reader)
        if not required.issubset(columns):
            raise ValueError(f"Missing {required - set(columns)} in {path}")
        positions = {column: columns.index(column) for column in required}
        for row_number, row in enumerate(reader, 2):
            yield row_number, {column: row[position] for column, position in positions.items()}


def main() -> int:
    classifier = load_classifier()
    records: dict[str, set[str]] = {site: set() for site in SITES}
    for path in COMBINED.rglob("*.hea"):
        relative = path.relative_to(COMBINED)
        if len(relative.parts) < 4 or relative.parts[0] not in records or relative.parts[1] != "WFDB":
            raise ValueError(f"Unexpected combined record path: {relative}")
        records[relative.parts[0]].add(relative.relative_to(Path(relative.parts[0]) / "WFDB").with_suffix("").as_posix())
    expected = sum(map(len, records.values()))
    print(f"Indexing {expected:,} ECG records", flush=True)

    patients: dict[tuple[str, str], str] = {}
    diagnoses: dict[tuple[str, str], str] = {}
    counters = Counter()
    conflicts: list[str] = []
    for site in SITES:
        path = HEEDB / site / "metadata/metadata.csv"
        print(f"Reading patient metadata: {path}", flush=True)
        for row_number, row in csv_rows(path, {"BDSPPatientID", "FileName", "FileID"}):
            counters[f"{site}_metadata_rows"] += 1
            if row_number % 1_000_000 == 0:
                print(f"  {site}: scanned {row_number:,} metadata rows", flush=True)
            try:
                name = source_name(row["FileName"])
            except ValueError:
                continue
            if name not in records[site]:
                continue
            key = (site, name)
            patient = classifier.patient_id(row["BDSPPatientID"])
            if not patient or row["FileID"].strip() != Path(name).name:
                counters["selected_metadata_invalid"] += 1
                continue
            prior = patients.setdefault(key, patient)
            if prior != patient:
                conflicts.append(f"{key}: conflicting patient IDs {prior!r}, {patient!r}")
            counters["selected_metadata_rows"] += 1
        print(f"  {site}: mapped {sum(key[0] == site for key in patients):,}/{len(records[site]):,} ECGs", flush=True)

    if len(patients) != expected or conflicts:
        missing = [(site, name) for site in SITES for name in records[site] if (site, name) not in patients]
        print(f"Missing patient mappings: {len(missing):,}; conflicts: {len(conflicts):,}", file=sys.stderr)
        for item in missing[:10]:
            print(f"  missing {item}", file=sys.stderr)
        for item in conflicts[:10]:
            print(f"  conflict {item}", file=sys.stderr)
        return 1

    for site in SITES:
        path = HEEDB / site / "12SL_diagnoses/diagnoses_acquisition.csv"
        print(f"Reading physician diagnoses: {path}", flush=True)
        for row_number, row in csv_rows(path, {"FileName", "codes_physician"}):
            counters[f"{site}_diagnosis_rows"] += 1
            if row_number % 1_000_000 == 0:
                print(f"  {site}: scanned {row_number:,} diagnosis rows", flush=True)
            try:
                name = source_name(row["FileName"])
            except ValueError:
                continue
            if name not in records[site]:
                continue
            key = (site, name)
            codes = row["codes_physician"].strip()
            prior = diagnoses.setdefault(key, codes)
            if prior != codes:
                conflicts.append(f"{key}: conflicting physician codes {prior!r}, {codes!r}")
            counters["selected_diagnosis_rows"] += 1
        print(f"  {site}: matched {sum(key[0] == site for key in diagnoses):,}/{len(records[site]):,} diagnoses", flush=True)

    if conflicts:
        print(f"Conflicting source rows: {len(conflicts):,}", file=sys.stderr)
        for item in conflicts[:10]:
            print(f"  {item}", file=sys.stderr)
        return 1

    output_rows = []
    label_counts = Counter()
    missing_diagnoses = 0
    for site in SITES:
        for name in sorted(records[site]):
            key = (site, name)
            codes = diagnoses.get(key, "")
            if key not in diagnoses:
                missing_diagnoses += 1
            labels, _, _ = classifier.classify_codes(codes)
            if "Normal ECG" in labels and not classifier.clean_normal_codes(codes):
                labels.remove("Normal ECG")
            label_counts.update(labels)
            output_rows.append({
                "record_path": f"{site}/WFDB/{name}.hea",
                "patient_key": patients[key],
                "site": site,
                "codes_physician": codes,
                "labels": ";".join(labels),
                "label_count": len(labels),
            })

    temporary = OUTPUT.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    os.replace(temporary, OUTPUT)
    print(f"Wrote {OUTPUT}")
    print(f"Records: {expected:,}; unique patient IDs: {len({row['patient_key'] for row in output_rows}):,}")
    print(f"Missing diagnosis rows: {missing_diagnoses:,}; records with no target label: {sum(not row['labels'] for row in output_rows):,}")
    print(f"Label counts: {dict(label_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
