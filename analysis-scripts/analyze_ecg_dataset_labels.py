#!/usr/bin/env python3
"""Audit locally stored ECG labels and build record-level target annotations.

The script intentionally derives labels only from local metadata/SCP/SNOMED
annotations.  It never reads waveform samples to infer a diagnosis.
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TARGETS = [
    "NORMAL",
    "AF",
    "LVH",
    "LBBB",
    "ISCHEMIC_STT",
    "NONSPECIFIC_STT",
    "LONG_QT",
]
ABNORMAL_TARGETS = TARGETS[1:]
DISPLAY = {
    "NORMAL": "NORMAL",
    "AF": "AF",
    "LVH": "LVH",
    "LBBB": "LBBB",
    "ISCHEMIC_STT": "Ischemic ST-T",
    "NONSPECIFIC_STT": "Non-specific ST-T",
    "LONG_QT": "Long QT",
}

SNOMED_AF = "164889003"
SNOMED_FLUTTER = "164890007"
SNOMED_LVH = "164873001"
SNOMED_LBBB = {"164909002", "733534002"}
SNOMED_LONG_QT = "111975006"
SNOMED_NONSPECIFIC_STT = "428750005"
SNOMED_SINUS_RHYTHM = "426783006"

PTBXL_LBBB = {"CLBBB", "ILBBB"}
PTBXL_ISCHEMIC_STT = {
    "ISC_",
    "ISCAL",
    "ISCIN",
    "ISCIL",
    "ISCAS",
    "ISCLA",
    "ISCAN",
}
PTBXL_NONSPECIFIC_STT = {"NST_", "NT_", "STD_", "STE_"}
PTBXL_NORMAL_NEUTRAL = {"NORM", "SR", "SBRAD", "STACH", "SARRH"}

SECONDARY_TERMINOLOGY = {
    "426783006": "ECG: sinus rhythm",
    "428750005": "Nonspecific ST-T abnormality",
    "111975006": "Prolonged QT interval",
    "733534002": "Complete left bundle branch block",
    "164909002": "Left bundle branch block",
    "164889003": "Atrial fibrillation",
    "164873001": "Left ventricular hypertrophy",
    "164890007": "Atrial flutter",
    "164930006": "ST elevation",
    "429622005": "ST depression",
    "164934002": "T wave abnormality",
    "59931005": "T wave inversion",
    "164865005": "Myocardial infarction",
    "164867002": "Old myocardial infarction",
    "57054005": "Acute myocardial infarction",
    "445118002": "Left anterior fascicular block",
    "1155004": "Myocardial hypertrophy determined by ECG",
    "55827005": "Undefined in the local codebook",
}

SECONDARY_URLS = {
    "426783006": "https://www.ncbi.nlm.nih.gov/medgen/742926",
    "428750005": "https://www.ncbi.nlm.nih.gov/medgen/744959",
    "111975006": "https://www.ncbi.nlm.nih.gov/medgen/57494",
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    directory: Path


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"

    def clean(value: Any) -> str:
        if value is None or value == "":
            return "n.a."
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---:" if key.endswith("_numeric") else "---" for key, _ in columns) + " |"
    body = ["| " + " | ".join(clean(row.get(key.removesuffix("_numeric"), "")) for key, _ in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def normalize_patient_id(value: str) -> str:
    value = (value or "").strip()
    if not value or value.lower() == "nan":
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return value


def parse_dx_header(path: Path) -> list[str]:
    dx_pattern = re.compile(r"^#\s*Dx:\s*(.*)$", re.IGNORECASE)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = dx_pattern.match(line.strip())
            if match:
                return [code.strip() for code in match.group(1).split(",") if code.strip()]
    return []


def load_snomed_codebook(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    descriptions: dict[str, str] = {}
    acronyms: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            code = (row.get("Snomed_CT") or "").strip()
            if not code:
                continue
            acronym = (row.get("Acronym Name") or "").strip()
            description = (row.get("Full Name") or "").strip()
            acronyms.setdefault(code, acronym)
            descriptions.setdefault(code, description)
    return descriptions, acronyms


def load_scp_statements(path: Path) -> dict[str, dict[str, str]]:
    statements: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            code = (row.get("") or "").strip()
            if code:
                statements[code] = row
    return statements


def target_combination(row: dict[str, Any]) -> str:
    labels = [DISPLAY[label] for label in TARGETS if int(row[label]) == 1]
    if not labels:
        return "NO_TARGET"
    if len(labels) == 1:
        return f"{labels[0]} only"
    return " + ".join(labels)


def abnormal_combination(row: dict[str, Any]) -> str:
    labels = [DISPLAY[label] for label in ABNORMAL_TARGETS if int(row[label]) == 1]
    if not labels:
        return "NO_ABNORMAL_TARGET"
    if len(labels) == 1:
        return f"{labels[0]} only"
    return " + ".join(labels)


def scan_waveform_pairs(directory: Path, header_suffix: str, waveform_suffix: str) -> list[Path]:
    return sorted(path for path in directory.rglob(f"*{header_suffix}") if path.with_suffix(waveform_suffix).is_file())


def ptbxl_records(dataset_root: Path, challenge_root: Path, scp: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata_path = dataset_root / "ptb-xl-dataset" / "ptbxl_database.csv"
    native_root = dataset_root / "ptb-xl-dataset"
    challenge_ptbxl = challenge_root / "ptb-xl"

    challenge_by_stem = {path.stem: path for path in scan_waveform_pairs(challenge_ptbxl, ".hea", ".mat")}
    records: list[dict[str, Any]] = []
    metadata_rows = 0
    raw_norm_count = 0
    conflict_count = 0
    native_pair_count = 0
    fallback_count = 0

    with metadata_path.open("r", newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            metadata_rows += 1
            ecg_id = int(source["ecg_id"])
            try:
                parsed = ast.literal_eval(source["scp_codes"])
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Invalid scp_codes for PTB-XL ecg_id={ecg_id}") from exc
            codes = {str(code): float(score) for code, score in parsed.items()}
            code_set = set(codes)

            native_rel = Path(source["filename_hr"])
            native_header = native_root / native_rel.with_suffix(".hea")
            native_waveform = native_root / native_rel.with_suffix(".dat")
            challenge_header = challenge_by_stem.get(f"HR{ecg_id:05d}")
            challenge_waveform = challenge_header.with_suffix(".mat") if challenge_header else None
            native_ok = native_header.is_file() and native_waveform.is_file()
            challenge_ok = bool(challenge_header and challenge_waveform and challenge_waveform.is_file())
            if not native_ok and not challenge_ok:
                continue
            if native_ok:
                native_pair_count += 1
                selected_header = native_header
                selected_waveform = native_waveform
            else:
                fallback_count += 1
                selected_header = challenge_header
                selected_waveform = challenge_waveform

            raw_normal = "NORM" in code_set
            conflict_codes = sorted(code_set - PTBXL_NORMAL_NEUTRAL) if raw_normal else []
            # PTB-XL also stores structured abnormal findings outside
            # scp_codes.  The request explicitly names axis deviation and
            # ectopy as reasons an ECG must not become a clean NORMAL record.
            # Preserve these exact source values in the conflict field.
            if raw_normal:
                heart_axis = (source.get("heart_axis") or "").strip()
                if heart_axis and heart_axis != "MID":
                    conflict_codes.append(f"heart_axis={heart_axis}")
                extra_beats = (source.get("extra_beats") or "").strip()
                if extra_beats:
                    conflict_codes.append(f"extra_beats={extra_beats}")
                pacemaker = (source.get("pacemaker") or "").strip()
                if pacemaker:
                    conflict_codes.append(f"pacemaker={pacemaker}")
                infarction_stage = (source.get("infarction_stadium1") or "").strip()
                if infarction_stage:
                    conflict_codes.append(f"infarction_stadium1={infarction_stage}")
            conflict_codes = sorted(set(conflict_codes))
            normal_conflict = bool(conflict_codes)
            raw_norm_count += int(raw_normal)
            conflict_count += int(normal_conflict)

            row: dict[str, Any] = {
                "dataset": "PTB-XL",
                "record_id": str(ecg_id),
                "patient_id": normalize_patient_id(source.get("patient_id", "")),
                "original_diagnosis_labels": ";".join(f"{code}:{codes[code]:g}" for code in sorted(codes)),
                "original_diagnosis_descriptions": "; ".join(
                    f"{code}={scp.get(code, {}).get('description', 'UNDEFINED')}" for code in sorted(codes)
                ),
                "NORMAL": int(raw_normal and not normal_conflict),
                "AF": int("AFIB" in code_set),
                "LVH": int("LVH" in code_set),
                "LBBB": int(bool(code_set & PTBXL_LBBB)),
                "ISCHEMIC_STT": int(bool(code_set & PTBXL_ISCHEMIC_STT)),
                "NONSPECIFIC_STT": int(bool(code_set & PTBXL_NONSPECIFIC_STT)),
                "LONG_QT": int("LNGQT" in code_set),
                "normal_asserted_raw": int(raw_normal),
                "normal_conflict": int(normal_conflict),
                "normal_conflict_labels": ";".join(conflict_codes),
                "sinus_rhythm_asserted": int("SR" in code_set),
                "metadata_path": str(metadata_path.resolve()),
                "header_path": str(selected_header.resolve()),
                "waveform_path": str(selected_waveform.resolve()),
                "alternate_header_path": str(challenge_header.resolve()) if native_ok and challenge_ok and challenge_header else "",
                "alternate_waveform_path": str(challenge_waveform.resolve()) if native_ok and challenge_ok and challenge_waveform else "",
            }
            row["target_label_count"] = sum(int(row[label]) for label in TARGETS)
            row["abnormal_target_count"] = sum(int(row[label]) for label in ABNORMAL_TARGETS)
            row["target_label_combination"] = target_combination(row)
            row["abnormal_label_combination"] = abnormal_combination(row)
            records.append(row)

    audit = {
        "dataset": "PTB-XL",
        "metadata_or_header_records": metadata_rows,
        "usable_records": len(records),
        "missing_waveform_or_header": metadata_rows - len(records),
        "native_high_resolution_pairs_used": native_pair_count,
        "challenge_ptbxl_fallback_pairs_used": fallback_count,
        "duplicate_alternate_ptbxl_copy_excluded": len(challenge_by_stem),
        "raw_normal_assertions": raw_norm_count,
        "normal_conflicts_excluded_from_normal": conflict_count,
        "notes": "The CinC 2020 PTB-XL copy is the same source and is used only as a waveform fallback, never as a second dataset.",
    }
    return records, audit


def snomed_records(
    config: DatasetConfig,
    code_descriptions: dict[str, str],
    code_acronyms: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_headers = sorted(config.directory.rglob("*.hea"))
    records: list[dict[str, Any]] = []
    headers_without_dx = 0
    for header in all_headers:
        waveform = header.with_suffix(".mat")
        if not waveform.is_file():
            continue
        codes = parse_dx_header(header)
        if not codes:
            headers_without_dx += 1
            continue
        code_set = set(codes)
        descriptions = []
        for code in codes:
            local = code_descriptions.get(code, "")
            secondary = SECONDARY_TERMINOLOGY.get(code, "")
            description = local or secondary or "UNDEFINED IN LOCAL CODEBOOK"
            acronym = code_acronyms.get(code, "")
            prefix = f"{acronym}: " if acronym else ""
            descriptions.append(f"{code}={prefix}{description}")

        row: dict[str, Any] = {
            "dataset": config.name,
            "record_id": header.stem,
            "patient_id": "",
            "original_diagnosis_labels": ";".join(codes),
            "original_diagnosis_descriptions": "; ".join(descriptions),
            "NORMAL": 0,
            "AF": int(SNOMED_AF in code_set),
            "LVH": int(SNOMED_LVH in code_set),
            "LBBB": int(bool(code_set & SNOMED_LBBB)),
            "ISCHEMIC_STT": 0,
            "NONSPECIFIC_STT": int(SNOMED_NONSPECIFIC_STT in code_set),
            "LONG_QT": int(SNOMED_LONG_QT in code_set),
            "normal_asserted_raw": 0,
            "normal_conflict": 0,
            "normal_conflict_labels": "",
            "sinus_rhythm_asserted": int(SNOMED_SINUS_RHYTHM in code_set),
            "metadata_path": str(header.resolve()),
            "header_path": str(header.resolve()),
            "waveform_path": str(waveform.resolve()),
            "alternate_header_path": "",
            "alternate_waveform_path": "",
        }
        row["target_label_count"] = sum(int(row[label]) for label in TARGETS)
        row["abnormal_target_count"] = sum(int(row[label]) for label in ABNORMAL_TARGETS)
        row["target_label_combination"] = target_combination(row)
        row["abnormal_label_combination"] = abnormal_combination(row)
        records.append(row)

    audit = {
        "dataset": config.name,
        "metadata_or_header_records": len(all_headers),
        "usable_records": len(records),
        "missing_waveform_or_header": len(all_headers) - len(records),
        "native_high_resolution_pairs_used": "n.a.",
        "challenge_ptbxl_fallback_pairs_used": "n.a.",
        "duplicate_alternate_ptbxl_copy_excluded": 0,
        "raw_normal_assertions": 0,
        "normal_conflicts_excluded_from_normal": 0,
        "notes": (
            "Usable requires a paired .hea/.mat and a parseable #Dx field. "
            f"Headers paired with waveforms but lacking #Dx: {headers_without_dx}."
        ),
    }
    return records, audit


def code_inventory(
    records: list[dict[str, Any]],
    scp: dict[str, dict[str, str]],
    code_descriptions: dict[str, str],
    code_acronyms: dict[str, str],
) -> list[dict[str, Any]]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        if row["dataset"] == "PTB-XL":
            labels = [item.split(":", 1)[0] for item in row["original_diagnosis_labels"].split(";") if item]
        else:
            labels = [item for item in row["original_diagnosis_labels"].split(";") if item]
        counters[row["dataset"]].update(set(labels))

    rows: list[dict[str, Any]] = []
    for dataset in sorted(counters):
        for code, count in sorted(counters[dataset].items(), key=lambda item: (-item[1], item[0])):
            if dataset == "PTB-XL":
                source = scp.get(code, {})
                description = source.get("description", "UNDEFINED")
                acronym = code
                representation = "SCP-ECG"
                locally_defined = int(code in scp)
                diagnostic_role = ",".join(
                    role for role in ("diagnostic", "form", "rhythm") if source.get(role, "").strip() in {"1", "1.0"}
                )
            else:
                description = code_descriptions.get(code) or SECONDARY_TERMINOLOGY.get(code) or "UNDEFINED IN LOCAL CODEBOOK"
                acronym = code_acronyms.get(code, "")
                representation = "SNOMED CT"
                locally_defined = int(code in code_descriptions)
                diagnostic_role = "diagnosis/finding"
            rows.append(
                {
                    "dataset": dataset,
                    "coding_system": representation,
                    "original_label_or_code": code,
                    "acronym": acronym,
                    "description": description,
                    "records_with_code": count,
                    "defined_in_local_codebook": locally_defined,
                    "statement_role": diagnostic_role,
                }
            )
    return rows


def dataset_code_counts(inventory: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for row in inventory:
        result[row["dataset"]][row["original_label_or_code"]] = int(row["records_with_code"])
    return result


def mapping_rows(
    inventory: list[dict[str, Any]],
    scp: dict[str, dict[str, str]],
    code_descriptions: dict[str, str],
) -> list[dict[str, Any]]:
    counts = dataset_code_counts(inventory)
    rows: list[dict[str, Any]] = []

    def add(dataset: str, target: str, code: str, include: str, reason: str, description: str | None = None) -> None:
        count = counts.get(dataset, {}).get(code, 0)
        if count == 0:
            return
        if description is None:
            if dataset == "PTB-XL":
                description = scp.get(code, {}).get("description", "UNDEFINED")
            else:
                description = code_descriptions.get(code) or SECONDARY_TERMINOLOGY.get(code) or "UNDEFINED IN LOCAL CODEBOOK"
        rows.append(
            {
                "Target Label": target,
                "Dataset": dataset,
                "Original Label/Code": code,
                "Original Description": description,
                "Include?": include,
                "Records carrying code": count,
                "Reason": reason,
            }
        )

    add("PTB-XL", "NORMAL", "NORM", "YES", "Explicit SCP diagnostic statement 'normal ECG'; main NORMAL count excludes NORM records carrying another non-neutral structured finding (SCP code, axis, ectopy, pacemaker, or infarction-stage annotation).")
    add("PTB-XL", "AF", "AFIB", "YES", "Direct atrial fibrillation rhythm statement; recorded likelihood 0 is retained because the code is present.")
    add("PTB-XL", "AF", "AFLT", "UNCERTAIN — DO NOT COUNT", "Atrial flutter is distinct from atrial fibrillation.")
    add("PTB-XL", "LVH", "LVH", "YES", "Direct diagnostic statement for left ventricular hypertrophy.")
    add("PTB-XL", "LVH", "VCLVH", "UNCERTAIN — DO NOT COUNT", "Voltage criteria are a form statement, not the PTB-XL diagnostic LVH statement.")
    add("PTB-XL", "LBBB", "CLBBB", "YES", "Explicit complete left bundle branch block; included under the broad LBBB target.")
    add("PTB-XL", "LBBB", "ILBBB", "YES", "Explicit incomplete left bundle branch block; included under the broad LBBB target and kept auditable as a separate source code.")
    add("PTB-XL", "LBBB", "LAFB", "NO", "Left anterior fascicular block is not left bundle branch block.")
    add("PTB-XL", "LBBB", "LPFB", "NO", "Left posterior fascicular block is not left bundle branch block.")

    for code in sorted(PTBXL_ISCHEMIC_STT):
        add("PTB-XL", "ISCHEMIC_STT", code, "YES", "PTB-XL diagnostic statement in subclass ISC_, ISCA, or ISCI; explicitly interpreted as ischemic ST-T change.")
    for code in ("INJAS", "INJAL", "INJIN", "INJIL", "INJLA"):
        add("PTB-XL", "ISCHEMIC_STT", code, "UNCERTAIN — DO NOT COUNT", "Subendocardial injury is stored under myocardial-infarction subclasses, not the requested ischemic ST-T diagnostic subclasses.")
    for code in ("STD_", "STE_", "NT_", "TAB_", "INVT", "NDT"):
        add("PTB-XL", "ISCHEMIC_STT", code, "NO", "Morphologic or non-specific ST/T statement without an explicit ischemic interpretation.")

    nonspecific_reasons = {
        "NST_": "Explicit non-specific ST changes diagnostic/form statement.",
        "NT_": "Explicit non-specific T-wave changes form statement.",
        "STD_": "Explicit non-specific ST depression form statement; not counted as ischemia.",
        "STE_": "Explicit non-specific ST elevation form statement; not counted as ischemia.",
    }
    for code, reason in nonspecific_reasons.items():
        add("PTB-XL", "NONSPECIFIC_STT", code, "YES", reason)
    add("PTB-XL", "NONSPECIFIC_STT", "NDT", "UNCERTAIN — DO NOT COUNT", "'Non-diagnostic T abnormalities' is not explicitly named non-specific.")
    add("PTB-XL", "NONSPECIFIC_STT", "TAB_", "UNCERTAIN — DO NOT COUNT", "Generic T-wave abnormality does not assert non-specificity.")
    add("PTB-XL", "NONSPECIFIC_STT", "INVT", "UNCERTAIN — DO NOT COUNT", "T-wave inversion is a morphology and not explicitly non-specific.")
    add("PTB-XL", "LONG_QT", "LNGQT", "YES", "Direct long QT-interval diagnostic statement.")
    add("PTB-XL", "LONG_QT", "LPR", "NO", "Prolonged PR is a different ECG interval.")

    snomed_datasets = ["CPSC 2018", "CPSC-Extra", "Georgia", "Chapman-Shaoxing-Ningbo"]
    for dataset in snomed_datasets:
        add(dataset, "NORMAL", SNOMED_SINUS_RHYTHM, "UNCERTAIN — DO NOT COUNT", "The standardized concept asserts sinus rhythm, not a normal ECG; source-specific coexistence with other diagnoses is audited separately.")
        add(dataset, "AF", SNOMED_AF, "YES", "Exact SNOMED CT concept for atrial fibrillation.")
        add(dataset, "AF", SNOMED_FLUTTER, "NO", "Atrial flutter is distinct from atrial fibrillation; the Chapman acronym 'AF' is a naming trap.")
        add(dataset, "LVH", SNOMED_LVH, "YES", "Exact left ventricular hypertrophy concept.")
        add(dataset, "LVH", "55827005", "UNCERTAIN — DO NOT COUNT", "High-frequency code is undefined in the shipped local codebook; waveform-based inference is prohibited.")
        add(dataset, "LVH", "1155004", "UNCERTAIN — DO NOT COUNT", "Myocardial hypertrophy by ECG is not explicitly left-ventricular hypertrophy.")
        add(dataset, "LBBB", "164909002", "YES", "Exact generic left bundle branch block concept.")
        add(dataset, "LBBB", "733534002", "YES", "Exact complete left bundle branch block concept; included under broad LBBB.")
        add(dataset, "LBBB", "445118002", "NO", "Left anterior fascicular block is not LBBB.")
        for code in ("429622005", "164930006", "164931005", "164934002", "59931005"):
            add(dataset, "ISCHEMIC_STT", code, "NO", "Morphologic ST/T finding without an explicit ischemic interpretation.")
        for code in ("164865005", "164867002", "57054005"):
            add(dataset, "ISCHEMIC_STT", code, "UNCERTAIN — DO NOT COUNT", "Myocardial infarction code does not specifically assert the requested ischemic ST-T change label.")
        add(dataset, "NONSPECIFIC_STT", SNOMED_NONSPECIFIC_STT, "YES", "Exact SNOMED concept 'Nonspecific ST-T abnormality'; local Chapman codebook abbreviates it as STTC / ST-T Change.")
        add(dataset, "NONSPECIFIC_STT", "164934002", "UNCERTAIN — DO NOT COUNT", "Generic T-wave abnormality is not explicitly non-specific.")
        add(dataset, "NONSPECIFIC_STT", "59931005", "UNCERTAIN — DO NOT COUNT", "T-wave inversion is not explicitly non-specific.")
        add(dataset, "NONSPECIFIC_STT", "429622005", "UNCERTAIN — DO NOT COUNT", "ST depression is not explicitly non-specific in the local source label.")
        add(dataset, "NONSPECIFIC_STT", "164930006", "UNCERTAIN — DO NOT COUNT", "ST elevation is not explicitly non-specific in the local source label.")
        add(dataset, "LONG_QT", SNOMED_LONG_QT, "YES", "Exact SNOMED CT concept for prolonged QT interval.")

    order = {target: index for index, target in enumerate(TARGETS)}
    return sorted(rows, key=lambda row: (order[row["Target Label"]], row["Dataset"], row["Include?"] != "YES", row["Original Label/Code"]))


def dataset_info_rows(dataset_root: Path, challenge_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "dataset": "PTB-XL",
            "metadata_files_used": f"{dataset_root / 'ptb-xl-dataset/ptbxl_database.csv'}; {dataset_root / 'ptb-xl-dataset/scp_statements.csv'}",
            "record_identifier": "ecg_id",
            "patient_identifier": "patient_id",
            "diagnosis_fields": "scp_codes; scp_statements diagnostic/form/rhythm roles",
            "coding_system": "SCP-ECG abbreviations with likelihood values",
            "multi_diagnosis": "Yes",
            "normal_annotation": "Explicit NORM = normal ECG; other non-neutral structured findings are flagged and excluded from the clean NORMAL count.",
        },
        {
            "dataset": "CPSC 2018",
            "metadata_files_used": f"{challenge_root / 'cpsc_2018/**/[record].hea'}",
            "record_identifier": "WFDB record name (A-prefix)",
            "patient_identifier": "Unavailable",
            "diagnosis_fields": "#Dx comment",
            "coding_system": "SNOMED CT concept IDs",
            "multi_diagnosis": "Yes",
            "normal_annotation": "No explicit normal-ECG concept found; 426783006 means sinus rhythm and is not counted as NORMAL.",
        },
        {
            "dataset": "CPSC-Extra",
            "metadata_files_used": f"{challenge_root / 'cpsc_2018_extra/**/[record].hea'}",
            "record_identifier": "WFDB record name (Q-prefix)",
            "patient_identifier": "Unavailable",
            "diagnosis_fields": "#Dx comment",
            "coding_system": "SNOMED CT concept IDs",
            "multi_diagnosis": "Yes",
            "normal_annotation": "No explicit normal-ECG concept found; 426783006 means sinus rhythm and is not counted as NORMAL.",
        },
        {
            "dataset": "Georgia",
            "metadata_files_used": f"{challenge_root / 'georgia/**/[record].hea'}",
            "record_identifier": "WFDB record name (E-prefix)",
            "patient_identifier": "Unavailable",
            "diagnosis_fields": "#Dx comment",
            "coding_system": "SNOMED CT concept IDs",
            "multi_diagnosis": "Yes",
            "normal_annotation": "No explicit normal-ECG concept found; 426783006 means sinus rhythm and is not counted as NORMAL.",
        },
        {
            "dataset": "Chapman-Shaoxing-Ningbo",
            "metadata_files_used": f"{dataset_root / 'chapman-dataset/WFDBRecords/**/[record].hea'}; {dataset_root / 'chapman-dataset/ConditionNames_SNOMED-CT.csv'}",
            "record_identifier": "WFDB record name (JS-prefix)",
            "patient_identifier": "Unavailable",
            "diagnosis_fields": "#Dx comment",
            "coding_system": "SNOMED CT concept IDs; local codebook supplies acronyms/text for a subset",
            "multi_diagnosis": "Yes",
            "normal_annotation": "No explicit normal-ECG concept; SR/426783006 is sinus rhythm only and is not counted as NORMAL.",
        },
    ]


def dataset_counts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    datasets = list(dict.fromkeys(row["dataset"] for row in records))
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        subset = [row for row in records if row["dataset"] == dataset]
        result: dict[str, Any] = {"Dataset": dataset}
        for target in TARGETS:
            result[target] = sum(int(row[target]) for row in subset)
        result["Total ECGs"] = len(subset)
        rows.append(result)
    total: dict[str, Any] = {"Dataset": "TOTAL UNIQUE POSITIVES"}
    for target in TARGETS:
        total[target] = sum(int(row[target]) for row in records)
    total["Total ECGs"] = len(records)
    rows.append(total)
    return rows


def patient_counts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    datasets = list(dict.fromkeys(row["dataset"] for row in records))
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        subset = [row for row in records if row["dataset"] == dataset]
        patient_available = all(bool(row["patient_id"]) for row in subset)
        for target in TARGETS:
            positive = [row for row in subset if int(row[target]) == 1]
            rows.append(
                {
                    "Dataset": dataset,
                    "Label": target,
                    "Unique ECG Records": len(positive),
                    "Unique Patients": len({row["patient_id"] for row in positive}) if patient_available else "Unavailable",
                }
            )
    return rows


def combination_rows(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    combined = Counter(row["target_label_combination"] for row in records if row["target_label_count"] > 0)
    all_rows = [
        {"Label Combination": combination, "Records": count}
        for combination, count in sorted(combined.items(), key=lambda item: (-item[1], item[0]))
    ]
    by_dataset: list[dict[str, Any]] = []
    for dataset in list(dict.fromkeys(row["dataset"] for row in records)):
        counter = Counter(
            row["target_label_combination"]
            for row in records
            if row["dataset"] == dataset and row["target_label_count"] > 0
        )
        for combination, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            by_dataset.append({"Dataset": dataset, "Label Combination": combination, "Records": count})
    return all_rows, by_dataset


def multilabel_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    abnormal = [row for row in records if row["abnormal_target_count"] > 0]
    distribution = Counter(int(row["abnormal_target_count"]) for row in abnormal)
    multi = sum(count for number, count in distribution.items() if number >= 2)
    return [
        {"Metric": "Single-label abnormal ECGs", "Records": distribution.get(1, 0), "Percentage": f"{distribution.get(1, 0) / len(abnormal):.1%}" if abnormal else "n.a."},
        {"Metric": "Exactly 2 target abnormalities", "Records": distribution.get(2, 0), "Percentage": f"{distribution.get(2, 0) / len(abnormal):.1%}" if abnormal else "n.a."},
        {"Metric": "Exactly 3 target abnormalities", "Records": distribution.get(3, 0), "Percentage": f"{distribution.get(3, 0) / len(abnormal):.1%}" if abnormal else "n.a."},
        {"Metric": "Exactly 4 target abnormalities", "Records": distribution.get(4, 0), "Percentage": f"{distribution.get(4, 0) / len(abnormal):.1%}" if abnormal else "n.a."},
        {"Metric": "5 or more target abnormalities", "Records": sum(count for number, count in distribution.items() if number >= 5), "Percentage": f"{sum(count for number, count in distribution.items() if number >= 5) / len(abnormal):.1%}" if abnormal else "n.a."},
        {"Metric": "Total multi-label abnormal ECGs", "Records": multi, "Percentage": f"{multi / len(abnormal):.1%}" if abnormal else "n.a."},
        {"Metric": "All abnormal target-positive ECGs", "Records": len(abnormal), "Percentage": "100.0%" if abnormal else "n.a."},
    ]


def pairwise_outputs(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    count_matrix: list[dict[str, Any]] = []
    percent_matrix: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for row_label in ABNORMAL_TARGETS:
        count_row: dict[str, Any] = {"Label": row_label}
        percent_row: dict[str, Any] = {"Label": row_label}
        row_total = sum(int(row[row_label]) for row in records)
        for col_label in ABNORMAL_TARGETS:
            if row_label == col_label:
                count_row[col_label] = "—"
                percent_row[col_label] = "—"
                continue
            overlap = sum(int(row[row_label]) and int(row[col_label]) for row in records)
            percentage = overlap / row_total if row_total else 0.0
            count_row[col_label] = overlap
            percent_row[col_label] = f"{percentage:.1%}"
            long_rows.append(
                {
                    "Row Label": row_label,
                    "Other Label": col_label,
                    "Overlap Records": overlap,
                    "Row Label Records": row_total,
                    "Percent of Row Label": percentage,
                }
            )
        count_matrix.append(count_row)
        percent_matrix.append(percent_row)
    return count_matrix, percent_matrix, long_rows


def source_bias(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals = {target: sum(int(row[target]) for row in records) for target in TARGETS}
    datasets = list(dict.fromkeys(row["dataset"] for row in records))
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        contributions = []
        for dataset in datasets:
            count = sum(int(row[target]) for row in records if row["dataset"] == dataset)
            if count:
                contributions.append((dataset, count))
        for dataset, count in sorted(contributions, key=lambda item: (-item[1], item[0])):
            percentage = count / totals[target] if totals[target] else 0.0
            rows.append(
                {
                    "Label": target,
                    "Dataset": dataset,
                    "Positive Records": count,
                    "% of All Positives": percentage,
                    "% of All Positives Display": f"{percentage:.1%}",
                    "Risk Flag": "HIGH DATASET-SOURCE BIAS RISK" if percentage > 0.70 or len(contributions) == 1 else "",
                    "No. of Source Datasets": len(contributions),
                }
            )
    return rows


def trainability(records: list[dict[str, Any]], bias_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in ABNORMAL_TARGETS:
        positives = [row for row in records if int(row[target]) == 1]
        datasets = sorted({row["dataset"] for row in positives})
        multi = sum(int(row["abnormal_target_count"] >= 2) for row in positives)
        ptbxl_patients = {row["patient_id"] for row in positives if row["dataset"] == "PTB-XL" and row["patient_id"]}
        largest = max((row for row in bias_rows if row["Label"] == target), key=lambda row: row["% of All Positives"], default=None)
        largest_text = f"{largest['Dataset']} ({largest['% of All Positives']:.1%})" if largest else "n.a."

        if target == "ISCHEMIC_STT":
            recommendation = "CAUTION"
            rationale = "Exact PTB-XL ischemic statements, but only one source dataset contributes positives; rendering/source leakage risk is extreme."
        elif target == "LONG_QT":
            recommendation = "CAUTION"
            rationale = "Exact annotations, but class size and source imbalance should be checked against the largest contributor before training."
        elif target == "NONSPECIFIC_STT":
            recommendation = "CAUTION"
            rationale = "Cross-source code is defensible, but PTB-XL mixes diagnostic and form statements and the visual phenotype is heterogeneous."
        elif target == "LBBB":
            recommendation = "KEEP"
            rationale = "Direct LBBB diagnoses from multiple sources; broad target combines complete, incomplete, and generic LBBB, so subtype sensitivity analysis is advised."
        else:
            recommendation = "KEEP"
            rationale = "Direct target-specific annotations from multiple independent source datasets and usable multi-label examples."

        rows.append(
            {
                "Target": target,
                "Total Records": len(positives),
                "Unique Patients": f"Unavailable overall; {len(ptbxl_patients)} identifiable PTB-XL patients",
                "Multi-label Records": multi,
                "No. of Source Datasets": len(datasets),
                "Largest Source Contribution": largest_text,
                "Recommendation": recommendation,
                "Rationale": rationale,
            }
        )
    return rows


def stt_comparison(records: list[dict[str, Any]], mapping: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in ("ISCHEMIC_STT", "NONSPECIFIC_STT"):
        positives = [row for row in records if int(row[target]) == 1]
        ptbxl_patients = {row["patient_id"] for row in positives if row["dataset"] == "PTB-XL" and row["patient_id"]}
        included = [row for row in mapping if row["Target Label"] == target and row["Include?"] == "YES"]
        exact_codes = "; ".join(f"{row['Dataset']}: {row['Original Label/Code']}" for row in included)
        other = "NONSPECIFIC_STT" if target == "ISCHEMIC_STT" else "ISCHEMIC_STT"
        rows.append(
            {
                "Target": target,
                "Unique Records": len(positives),
                "Unique Patients": f"Unavailable overall; {len(ptbxl_patients)} identifiable PTB-XL patients",
                "Datasets Supporting": ", ".join(sorted({row["dataset"] for row in positives})) or "None",
                "Exact Included Labels": exact_codes,
                "Overlap With Other ST-T Target": sum(int(row[other]) for row in positives),
                "Overlap With LVH": sum(int(row["LVH"]) for row in positives),
                "Overlap With LBBB": sum(int(row["LBBB"]) for row in positives),
                "Overlap With AF": sum(int(row["AF"]) for row in positives),
                "Isolated Cases": sum(int(row["abnormal_target_count"] == 1) for row in positives),
                "Multi-label Cases": sum(int(row["abnormal_target_count"] >= 2) for row in positives),
                "Annotation Consistency": (
                    "Single-source PTB-XL diagnostic subclasses explicitly assert ischemic ST-T changes."
                    if target == "ISCHEMIC_STT"
                    else "SNOMED 428750005 is consistent across header datasets; PTB-XL broadens the target with explicitly non-specific diagnostic/form statements."
                ),
            }
        )
    return rows


def normal_conflict_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row["dataset"],
            "record_id": row["record_id"],
            "patient_id": row["patient_id"],
            "original_diagnosis_labels": row["original_diagnosis_labels"],
            "conflicting_labels": row["normal_conflict_labels"],
            "retained_abnormal_target_combination": row["abnormal_label_combination"],
        }
        for row in records
        if int(row["normal_conflict"]) == 1
    ]


def duplicate_audit(records: list[dict[str, Any]], audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in list(dict.fromkeys(row["dataset"] for row in records)):
        ids = [row["record_id"] for row in records if row["dataset"] == dataset]
        rows.append(
            {
                "scope": dataset,
                "check": "Duplicate record identifiers within counted source",
                "result": len(ids) - len(set(ids)),
                "interpretation": "No duplicate record IDs" if len(ids) == len(set(ids)) else "Duplicates require investigation",
            }
        )
    ptb_audit = next(row for row in audits if row["dataset"] == "PTB-XL")
    rows.append(
        {
            "scope": "PTB-XL vs CinC 2020 PTB-XL copy",
            "check": "Known duplicate source copy",
            "result": ptb_audit["duplicate_alternate_ptbxl_copy_excluded"],
            "interpretation": "Excluded from dataset totals; used only as waveform fallback where native files were missing.",
        }
    )
    rows.append(
        {
            "scope": "Across named datasets",
            "check": "Patient-level duplicate verification",
            "result": "Unavailable",
            "interpretation": "Only PTB-XL provides patient IDs; cross-source patient deduplication cannot be proven from local annotations.",
        }
    )
    return rows


def normal_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ptb = [row for row in records if row["dataset"] == "PTB-XL"]
    sinus_rows = []
    for dataset in list(dict.fromkeys(row["dataset"] for row in records)):
        subset = [row for row in records if row["dataset"] == dataset]
        sinus_count = sum(int(row["sinus_rhythm_asserted"]) for row in subset)
        sinus_with_other = sum(
            int(row["sinus_rhythm_asserted"] and len(row["original_diagnosis_labels"].split(";")) > 1)
            for row in subset
        )
        sinus_rows.append(
            {
                "Dataset": dataset,
                "Raw Explicit Normal Assertions": sum(int(row["normal_asserted_raw"]) for row in subset),
                "Clean NORMAL Count": sum(int(row["NORMAL"]) for row in subset),
                "Normal Conflicts": sum(int(row["normal_conflict"]) for row in subset),
                "Sinus Rhythm Assertions (not NORMAL)": sinus_count if dataset != "PTB-XL" else "n.a.",
                "Sinus Rhythm With Other Diagnoses": sinus_with_other if dataset != "PTB-XL" else "n.a.",
                "Sinus Rhythm Only Proxy (not counted)": (sinus_count - sinus_with_other) if dataset != "PTB-XL" else "n.a.",
            }
        )
    assert len(ptb) > 0
    return sinus_rows


def report_text(
    dataset_info: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    mappings: list[dict[str, Any]],
    counts: list[dict[str, Any]],
    patients: list[dict[str, Any]],
    combinations: list[dict[str, Any]],
    multilabel: list[dict[str, Any]],
    pair_counts: list[dict[str, Any]],
    pair_percents: list[dict[str, Any]],
    bias: list[dict[str, Any]],
    trainability_rows: list[dict[str, Any]],
    stt_rows: list[dict[str, Any]],
    normal_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> str:
    totals = counts[-1]
    bias_labels = sorted({row["Label"] for row in bias if row["Risk Flag"]})
    conflict_count = sum(int(row["normal_conflict"]) for row in records)
    clean_norm = totals["NORMAL"]
    raw_norm = sum(int(row["normal_asserted_raw"]) for row in records)
    abnormal = sum(int(row["abnormal_target_count"] > 0) for row in records)
    multi = sum(int(row["abnormal_target_count"] >= 2) for row in records)

    sections = [
        "# ECG dataset label availability audit",
        "",
        "## Concise summary",
        "",
        f"The local source collections contain **{len(records):,} usable, de-duplicated ECG records** under the file-pair definition used here. "
        f"Usable means a record has parseable source annotations plus a paired header/waveform file; it does not assert signal quality. "
        f"Across the seven requested targets, the clean counts are NORMAL {totals['NORMAL']:,}, AF {totals['AF']:,}, LVH {totals['LVH']:,}, "
        f"LBBB {totals['LBBB']:,}, ischemic ST-T {totals['ISCHEMIC_STT']:,}, non-specific ST-T {totals['NONSPECIFIC_STT']:,}, "
        f"and long QT {totals['LONG_QT']:,}. There are {abnormal:,} ECGs with at least one abnormal target, of which {multi:,} ({multi / abnormal:.1%}) are multi-label.",
        "",
        f"NORMAL is the most restrictive label. Only PTB-XL carries an explicit `NORM`/normal-ECG statement. "
        f"It appears on {raw_norm:,} usable records; {conflict_count:,} carry another non-neutral structured finding (SCP code, axis, ectopy, pacemaker, or infarction-stage annotation) and are excluded from the clean NORMAL count, leaving {clean_norm:,}. "
        "SNOMED `426783006` is sinus rhythm, not a normal-ECG assertion, so it is not counted as NORMAL.",
        "",
        f"Source-bias flags apply to: {', '.join(bias_labels) if bias_labels else 'none'}. "
        "The CinC 2020 PTB-XL copy is explicitly de-duplicated and used only to fill missing local PTB-XL waveform pairs.",
        "",
        "From a dataset perspective, **non-specific ST-T is the stronger of the two ST-T candidates** because it has cross-dataset support. "
        "Ischemic ST-T has a clean PTB-XL mapping but a single-source rendering-bias risk. Neither target should be merged with generic ST depression, ST elevation, T-wave abnormality, or myocardial-infarction codes.",
        "",
        "## Counting scope and method",
        "",
        "Counts come from local `scp_codes`/`scp_statements.csv` and WFDB `#Dx` header annotations. No waveform-derived diagnoses were created. "
        "PTB-XL counts use its native metadata, with the local CinC 2020 PTB-XL copy only as a waveform fallback. CPSC 2018, CPSC-Extra, Georgia, and the combined Chapman-Shaoxing-Ningbo source are counted as separate named sources. "
        "Cross-source patient de-duplication is unavailable because only PTB-XL exposes a patient identifier.",
        "",
        "## 1. Dataset label systems",
        "",
        markdown_table(dataset_info, [
            ("dataset", "Dataset"),
            ("metadata_files_used", "Metadata file(s) used"),
            ("record_identifier", "ECG/record identifier"),
            ("patient_identifier", "Patient identifier"),
            ("diagnosis_fields", "Diagnosis field(s)"),
            ("coding_system", "Representation"),
            ("multi_diagnosis", "Multiple diagnoses?"),
            ("normal_annotation", "NORMAL handling"),
        ]),
        "",
        "## 2. Label mapping table",
        "",
        markdown_table(mappings, [
            ("Target Label", "Target Label"),
            ("Dataset", "Dataset"),
            ("Original Label/Code", "Original Label/Code"),
            ("Original Description", "Original Description"),
            ("Include?", "Include?"),
            ("Records carrying code_numeric", "Records carrying code"),
            ("Reason", "Reason"),
        ]),
        "",
        "The `Records carrying code` column is not the final target count because records may carry more than one included code.",
        "",
        "## 3. Unique usable ECG records",
        "",
        markdown_table(counts, [
            ("Dataset", "Dataset"),
            ("NORMAL_numeric", "NORMAL"),
            ("AF_numeric", "AF"),
            ("LVH_numeric", "LVH"),
            ("LBBB_numeric", "LBBB"),
            ("ISCHEMIC_STT_numeric", "Ischemic ST-T"),
            ("NONSPECIFIC_STT_numeric", "Non-specific ST-T"),
            ("LONG_QT_numeric", "Long QT"),
            ("Total ECGs_numeric", "Total ECGs"),
        ]),
        "",
        "### File-pair usability audit",
        "",
        markdown_table(audits, [
            ("dataset", "Dataset"),
            ("metadata_or_header_records_numeric", "Annotated/header records"),
            ("usable_records_numeric", "Usable records"),
            ("missing_waveform_or_header_numeric", "Missing pair/diagnosis"),
            ("native_high_resolution_pairs_used_numeric", "Native PTB-XL pairs used"),
            ("challenge_ptbxl_fallback_pairs_used_numeric", "PTB-XL fallback pairs"),
            ("notes", "Notes"),
        ]),
        "",
        "## 4. Unique ECG records and patients",
        "",
        markdown_table(patients, [
            ("Dataset", "Dataset"),
            ("Label", "Label"),
            ("Unique ECG Records_numeric", "Unique ECG Records"),
            ("Unique Patients_numeric", "Unique Patients"),
        ]),
        "",
        "Unique-patient count unavailable for CPSC 2018, CPSC-Extra, Georgia, and Chapman-Shaoxing-Ningbo.",
        "",
        "## 5. All observed target-label combinations",
        "",
        markdown_table(combinations, [("Label Combination", "Label Combination"), ("Records_numeric", "Records")]),
        "",
        "### Abnormal multi-label summary",
        "",
        markdown_table(multilabel, [("Metric", "Metric"), ("Records_numeric", "Records"), ("Percentage", "Percentage of abnormal target-positive ECGs")]),
        "",
        "## 6. Pairwise overlap matrix",
        "",
        "Counts (unique ECG records positive for both labels):",
        "",
        markdown_table(pair_counts, [("Label", "Label"), *[(label, DISPLAY[label]) for label in ABNORMAL_TARGETS]]),
        "",
        "Directional percentages (each cell is the percentage of the row label that also contains the column label):",
        "",
        markdown_table(pair_percents, [("Label", "Label"), *[(label, DISPLAY[label]) for label in ABNORMAL_TARGETS]]),
        "",
        "## 7. NORMAL audit",
        "",
        markdown_table(normal_rows, [
            ("Dataset", "Dataset"),
            ("Raw Explicit Normal Assertions_numeric", "Raw explicit normal assertions"),
            ("Clean NORMAL Count_numeric", "Clean NORMAL count"),
            ("Normal Conflicts_numeric", "Normal conflicts"),
            ("Sinus Rhythm Assertions (not NORMAL)_numeric", "Sinus-rhythm assertions (not NORMAL)"),
            ("Sinus Rhythm With Other Diagnoses_numeric", "Sinus rhythm with other diagnoses"),
            ("Sinus Rhythm Only Proxy (not counted)_numeric", "Sinus rhythm only proxy (not counted)"),
        ]),
        "",
        "Every PTB-XL conflict is preserved in `normal_annotation_conflicts.csv`; the raw `NORM` assertion also remains in the record-level file.",
        "",
        "## 8. Label-source bias",
        "",
        markdown_table(bias, [
            ("Label", "Label"),
            ("Dataset", "Dataset"),
            ("Positive Records_numeric", "Positive Records"),
            ("% of All Positives Display", "% of All Positives"),
            ("Risk Flag", "Risk Flag"),
            ("No. of Source Datasets_numeric", "Source datasets"),
        ]),
        "",
        "A target is flagged when more than 70% of its positives come from one dataset or when only one dataset contributes positives.",
        "",
        "## 9. Trainability assessment",
        "",
        markdown_table(trainability_rows, [
            ("Target", "Target"),
            ("Total Records_numeric", "Total Records"),
            ("Unique Patients", "Unique Patients"),
            ("Multi-label Records_numeric", "Multi-label Records"),
            ("No. of Source Datasets_numeric", "Source Datasets"),
            ("Largest Source Contribution", "Largest Source Contribution"),
            ("Recommendation", "Recommendation"),
            ("Rationale", "Rationale"),
        ]),
        "",
        "## 10. Ischemic versus non-specific ST-T",
        "",
        markdown_table(stt_rows, [
            ("Target", "Target"),
            ("Unique Records_numeric", "Unique Records"),
            ("Unique Patients", "Unique Patients"),
            ("Datasets Supporting", "Datasets Supporting"),
            ("Exact Included Labels", "Exact Included Labels"),
            ("Overlap With Other ST-T Target_numeric", "Overlap with other ST-T target"),
            ("Overlap With LVH_numeric", "Overlap with LVH"),
            ("Overlap With LBBB_numeric", "Overlap with LBBB"),
            ("Overlap With AF_numeric", "Overlap with AF"),
            ("Isolated Cases_numeric", "Isolated Cases"),
            ("Multi-label Cases_numeric", "Multi-label Cases"),
            ("Annotation Consistency", "Annotation Consistency"),
        ]),
        "",
        "Dataset recommendation: keep `NONSPECIFIC_STT` as the more practical cross-source candidate, with caution about heterogeneous PTB-XL form versus diagnostic statements. "
        "Treat `ISCHEMIC_STT` as a PTB-XL-only exploratory target unless a second independently rendered dataset with an explicitly ischemic ST-T annotation is added.",
        "",
        "## Duplicate and de-duplication checks",
        "",
        markdown_table(duplicate_rows, [("scope", "Scope"), ("check", "Check"), ("result_numeric", "Result"), ("interpretation", "Interpretation")]),
        "",
        "## Material limitations",
        "",
        "- CPSC 2018, CPSC-Extra, Georgia, and Chapman-Shaoxing-Ningbo do not expose patient IDs in the local WFDB headers, so patient-level independence and cross-source patient duplication cannot be verified.",
        "- The combined Chapman-Shaoxing-Ningbo distribution does not expose a reliable institution/source flag per record in the inspected headers; it is therefore treated as one dataset source.",
        "- The Chapman codebook does not define every SNOMED code observed in its headers. Undefined near-miss codes such as `55827005` are excluded rather than inferred from waveforms.",
        "- `LBBB` is deliberately broad: PTB-XL complete and incomplete LBBB plus SNOMED generic and complete LBBB are included. Run a sensitivity analysis that excludes incomplete LBBB if a morphology-homogeneous class is required.",
        "- `NONSPECIFIC_STT` includes PTB-XL statements explicitly named non-specific even when they are form statements. The record-level file preserves the source codes so a diagnostic-only sensitivity cohort can be created.",
        "",
        "## Secondary terminology verification",
        "",
        "Local annotations remain the primary evidence for counts. The following terminology pages were used only to resolve exact standardized concept names:",
        "",
        "- SNOMED 428750005: Nonspecific ST-T abnormality — https://www.ncbi.nlm.nih.gov/medgen/744959",
        "- SNOMED 426783006: ECG: sinus rhythm — https://www.ncbi.nlm.nih.gov/medgen/742926",
        "- SNOMED 111975006: Prolonged QT interval — https://www.ncbi.nlm.nih.gov/medgen/57494",
        "",
        "## Generated files",
        "",
        "The output directory contains the record-level CSV, all summary tables as CSV, this Markdown report, and an XLSX workbook containing the principal reader-facing and audit tables.",
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    challenge_train = (
        dataset_root
        / "Classification of 12-lead ECGs: The PhysioNet | Computing in Cardiology Challenge 2020"
        / "training"
    )
    scp = load_scp_statements(dataset_root / "ptb-xl-dataset" / "scp_statements.csv")
    snomed_descriptions, snomed_acronyms = load_snomed_codebook(
        dataset_root / "chapman-dataset" / "ConditionNames_SNOMED-CT.csv"
    )

    records, ptb_audit = ptbxl_records(dataset_root, challenge_train, scp)
    audits = [ptb_audit]
    snomed_configs = [
        DatasetConfig("CPSC 2018", challenge_train / "cpsc_2018"),
        DatasetConfig("CPSC-Extra", challenge_train / "cpsc_2018_extra"),
        DatasetConfig("Georgia", challenge_train / "georgia"),
        DatasetConfig("Chapman-Shaoxing-Ningbo", dataset_root / "chapman-dataset" / "WFDBRecords"),
    ]
    for config in snomed_configs:
        source_records, audit = snomed_records(config, snomed_descriptions, snomed_acronyms)
        records.extend(source_records)
        audits.append(audit)

    dataset_order = ["PTB-XL", "CPSC 2018", "CPSC-Extra", "Georgia", "Chapman-Shaoxing-Ningbo"]
    order_index = {name: index for index, name in enumerate(dataset_order)}
    records.sort(key=lambda row: (order_index[row["dataset"]], row["record_id"]))

    inventory = code_inventory(records, scp, snomed_descriptions, snomed_acronyms)
    mappings = mapping_rows(inventory, scp, snomed_descriptions)
    info = dataset_info_rows(dataset_root, challenge_train)
    counts = dataset_counts(records)
    patients = patient_counts(records)
    combinations, combinations_by_dataset = combination_rows(records)
    multilabel = multilabel_summary(records)
    pair_counts, pair_percents, pair_long = pairwise_outputs(records)
    bias = source_bias(records)
    trainability_rows = trainability(records, bias)
    stt_rows = stt_comparison(records, mappings)
    conflicts = normal_conflict_rows(records)
    duplicate_rows = duplicate_audit(records, audits)
    normal_rows = normal_summary(records)

    record_fields = [
        "dataset",
        "record_id",
        "patient_id",
        "original_diagnosis_labels",
        "original_diagnosis_descriptions",
        *TARGETS,
        "normal_asserted_raw",
        "normal_conflict",
        "normal_conflict_labels",
        "sinus_rhythm_asserted",
        "target_label_count",
        "abnormal_target_count",
        "target_label_combination",
        "abnormal_label_combination",
        "metadata_path",
        "header_path",
        "waveform_path",
        "alternate_header_path",
        "alternate_waveform_path",
    ]
    write_csv(output_dir / "ecg_target_labels_record_level.csv", records, record_fields)
    write_csv(output_dir / "dataset_label_systems.csv", info)
    write_csv(output_dir / "source_code_inventory.csv", inventory)
    write_csv(output_dir / "label_mapping_table.csv", mappings)
    write_csv(output_dir / "dataset_label_counts.csv", counts)
    write_csv(output_dir / "dataset_label_patient_counts.csv", patients)
    write_csv(output_dir / "label_combinations_all.csv", combinations)
    write_csv(output_dir / "label_combinations_by_dataset.csv", combinations_by_dataset)
    write_csv(output_dir / "multilabel_summary.csv", multilabel)
    write_csv(output_dir / "pairwise_overlap_counts.csv", pair_counts)
    write_csv(output_dir / "pairwise_overlap_percentages.csv", pair_percents)
    write_csv(output_dir / "pairwise_overlap_long.csv", pair_long)
    write_csv(output_dir / "label_source_bias.csv", bias)
    write_csv(output_dir / "trainability_assessment.csv", trainability_rows)
    write_csv(output_dir / "stt_candidate_comparison.csv", stt_rows)
    write_csv(output_dir / "normal_annotation_conflicts.csv", conflicts)
    write_csv(output_dir / "normal_handling_summary.csv", normal_rows)
    write_csv(output_dir / "source_usability_audit.csv", audits)
    write_csv(output_dir / "duplicate_audit.csv", duplicate_rows)

    report = report_text(
        info,
        audits,
        mappings,
        counts,
        patients,
        combinations,
        multilabel,
        pair_counts,
        pair_percents,
        bias,
        trainability_rows,
        stt_rows,
        normal_rows,
        duplicate_rows,
        records,
    )
    (output_dir / "ecg_dataset_label_audit_report.md").write_text(report, encoding="utf-8")

    manifest = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "target_columns": TARGETS,
        "normal_policy": "Explicit PTB-XL NORM only; exclude NORM records carrying non-neutral structured findings from clean NORMAL.",
        "waveform_inference_used": False,
        "generated_files": sorted(
            path.name
            for path in output_dir.iterdir()
            if path.is_file() and not path.name.endswith(".inspect.ndjson")
        ),
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"records": len(records), "counts": counts, "audits": audits}, indent=2))


if __name__ == "__main__":
    main()
