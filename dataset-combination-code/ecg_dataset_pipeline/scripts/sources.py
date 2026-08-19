"""
CardioSentry -- per-source readers.

Each reader turns one dataset into a uniform list of `RawRecord`s. Readers
know about FILE LAYOUT only; they never decide what a label means. Label
semantics come from config/label_mapping.yaml via common.LabelResolver.
"""
from __future__ import annotations

import ast
import csv
import glob
import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from common import HeaderInfo, assumption, info, parse_header, warn


@dataclass
class RawRecord:
    source: str
    original_record_id: str
    patient_id: str | None
    raw_labels: list                      # source label tokens
    raw_label_text: str                   # human-readable original diagnosis
    signal_path: str                      # path WITHOUT extension (WFDB stem)
    header_path: str
    data_path: str
    age: str = ""
    sex: str = ""
    extra: dict = field(default_factory=dict)


# ==========================================================================
# PTB-XL
# ==========================================================================

class PTBXLSource:
    key = "ptbxl"
    prefix = "PTBXL"

    def __init__(self, cfg: dict):
        self.root = Path(cfg["root"])
        self.db_csv = self.root / cfg.get("database_csv", "ptbxl_database.csv")
        self.scp_csv = self.root / cfg.get("scp_statements_csv", "scp_statements.csv")
        self.filename_col = cfg.get("filename_column", "filename_hr")

    def available(self) -> bool:
        return self.db_csv.exists()

    def scp_statements(self) -> dict:
        out = {}
        with open(self.scp_csv, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                code = (row.get("") or row.get("Unnamed: 0") or "").strip()
                if not code:
                    code = list(row.values())[0]
                out[code] = row
        return out

    def read(self) -> list:
        recs = []
        with open(self.db_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    codes = ast.literal_eval(row["scp_codes"])
                except (ValueError, SyntaxError):
                    codes = {}
                stem = row[self.filename_col]
                full = str(self.root / stem)
                recs.append(RawRecord(
                    source="PTBXL",
                    original_record_id=str(row["ecg_id"]),
                    patient_id=f"PTBXL_P{int(float(row['patient_id']))}" if row.get("patient_id") else None,
                    raw_labels=list(codes.keys()),
                    raw_label_text=row["scp_codes"],
                    signal_path=full,
                    header_path=full + ".hea",
                    data_path=full + ".dat",
                    age=row.get("age", ""),
                    sex=row.get("sex", ""),
                    extra={
                        "scp_likelihoods": codes,
                        "report": row.get("report", ""),
                        "infarction_stadium1": row.get("infarction_stadium1", ""),
                        "infarction_stadium2": row.get("infarction_stadium2", ""),
                        "validated_by_human": row.get("validated_by_human", ""),
                        "recording_date": row.get("recording_date", ""),
                        "device": row.get("device", ""),
                        "site": row.get("site", ""),
                        "strat_fold": row.get("strat_fold", ""),
                        "heart_axis": row.get("heart_axis", ""),
                        "pacemaker": row.get("pacemaker", ""),
                        "baseline_drift": row.get("baseline_drift", ""),
                        "static_noise": row.get("static_noise", ""),
                        "burst_noise": row.get("burst_noise", ""),
                        "electrodes_problems": row.get("electrodes_problems", ""),
                    },
                ))
        return recs


# ==========================================================================
# Chapman-Shaoxing / Ningbo
# ==========================================================================

class ChapmanSource:
    key = "chapman"
    prefix = "CHAPMAN"

    def __init__(self, cfg: dict):
        self.root = Path(cfg["root"])
        self.records_dir = self.root / cfg.get("records_dir", "WFDBRecords")
        self.cond_csv = self.root / cfg.get("condition_names_csv", "ConditionNames_SNOMED-CT.csv")

    def available(self) -> bool:
        return self.records_dir.is_dir()

    def condition_names(self) -> dict:
        """SNOMED code -> (acronym, full name), from the shipped CSV."""
        out = {}
        if not self.cond_csv.exists():
            return out
        raw = open(self.cond_csv, encoding="utf-8-sig").read()
        for row in csv.DictReader(io.StringIO(raw)):
            code = (row.get("Snomed_CT") or "").strip()
            if code:
                out[code] = ((row.get("Acronym Name") or "").strip(),
                             (row.get("Full Name") or "").strip())
        return out

    def header_paths(self) -> list:
        return sorted(glob.glob(str(self.records_dir / "*" / "*" / "*.hea")))

    def read(self) -> list:
        recs = []
        for hea in self.header_paths():
            stem = hea[:-4]
            hi = parse_header(hea)
            dx_raw = hi.comments.get("Dx", "")
            codes = [c.strip() for c in dx_raw.split(",") if c.strip()]
            rec_id = hi.record_name or Path(stem).name
            recs.append(RawRecord(
                source="CHAPMAN",
                # No patient identifier exists in this distribution.
                original_record_id=rec_id,
                patient_id=None,
                raw_labels=codes,
                raw_label_text=dx_raw,
                signal_path=stem,
                header_path=hea,
                data_path=stem + ".mat",
                age=hi.comments.get("Age", ""),
                sex=hi.comments.get("Sex", ""),
                extra={
                    "Rx": hi.comments.get("Rx", ""),
                    "Hx": hi.comments.get("Hx", ""),
                    "Sx": hi.comments.get("Sx", ""),
                    "_header": hi,
                },
            ))
        return recs


# ==========================================================================
# 2026 ACS / STEMI dataset
# ==========================================================================

class STEMISource:
    key = "stemi"
    prefix = "STEMI"

    def __init__(self, cfg: dict):
        self.root = Path(cfg["root"])
        self.train_csv = self.root / cfg.get("train_csv", "CSV/train.csv")
        tc = cfg.get("test_csv")
        self.test_csv = (self.root / tc) if tc else None
        self.signal_dir = self.root / cfg.get("signal_dir", "ECG_row_data/row_data")
        self.median_dir = self.root / cfg.get("median_dir", "ECG_median_data/med_data")

    def available(self) -> bool:
        return self.train_csv.exists()

    def read(self, mapping_src: dict | None = None) -> list:
        meta_cols = set((mapping_src or {}).get(
            "metadata_columns",
            ["Patient_id", "ecg_row_record", "ecg_med_record", "gender", "age", "Time_Interval"]))
        recs = []
        with open(self.train_csv, newline="", encoding="utf-8-sig") as fh:
            rdr = csv.DictReader(fh)
            label_cols = [c for c in (rdr.fieldnames or []) if c not in meta_cols]
            for row in rdr:
                pos = [c for c in label_cols if str(row.get(c, "")).strip() == "1"]
                dat = str(row.get("ecg_row_record", "")).strip()
                stem = str(self.signal_dir / dat[:-4]) if dat.endswith(".dat") else str(self.signal_dir / dat)
                recs.append(RawRecord(
                    source="STEMI",
                    original_record_id=(dat[:-4] if dat.endswith(".dat") else dat),
                    patient_id=str(row.get("Patient_id", "")).strip() or None,
                    raw_labels=pos,
                    raw_label_text=("+".join(pos) if pos else "(all diagnostic columns 0)"),
                    signal_path=stem,
                    header_path=stem + ".hea",
                    data_path=stem + ".dat",
                    age=row.get("age", ""),
                    sex=row.get("gender", ""),
                    extra={
                        "Time_Interval": row.get("Time_Interval", ""),
                        "ecg_med_record": row.get("ecg_med_record", ""),
                        "split": "train",
                        "all_label_columns": label_cols,
                    },
                ))
        return recs

    def unlabeled_test_count(self) -> int:
        if not self.test_csv or not self.test_csv.exists():
            return 0
        with open(self.test_csv, newline="", encoding="utf-8-sig") as fh:
            return sum(1 for _ in csv.DictReader(fh))


def build_sources(paths: dict) -> dict:
    out = {}
    if paths.get("ptbxl", {}).get("enabled", True):
        out["ptbxl"] = PTBXLSource(paths["ptbxl"])
    if paths.get("chapman", {}).get("enabled", True):
        out["chapman"] = ChapmanSource(paths["chapman"])
    if paths.get("stemi", {}).get("enabled", True):
        out["stemi"] = STEMISource(paths["stemi"])
    return out
