"""
CardioSentry / LUDB staging.

LUDB already ships in the project's target representation -- 12 leads, 500 Hz,
5000 samples, WFDB -- so nothing is transcoded. This script only resolves the
cardiologists' diagnosis lines into CardioSentry labels (via the MAIN
pipeline's own `common.LabelResolver`, against config/label_map_ludb.yaml) and
emits harmonized-schema metadata pointing at the original files.

Writes:
  data/ludb_metadata.csv       harmonized-schema, ready to concatenate
  reports/label_decisions.csv  every record, its diagnoses and the decision
  reports/cohort_summary.txt
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_PIPELINE = Path(os.environ.get(
    "CARDIOSENTRY_PIPELINE",
    ROOT.parents[1] / "dataset-combination-code" / "ecg_dataset_pipeline"))
sys.path.insert(0, str(MAIN_PIPELINE / "scripts"))

import yaml                                                     # noqa: E402
from common import TARGET_CLASSES, LabelResolver, parse_header   # noqa: E402

LUDB_ROOT = Path(os.environ.get(
    "LUDB_ROOT",
    ROOT.parents[1] / "dataset-files"
    / "lobachevsky-university-electrocardiography-database-1.0.1"))
MAPPING = ROOT / "config" / "label_map_ludb.yaml"
OUT_META = ROOT / "data" / "ludb_metadata.csv"
OUT_DEC = ROOT / "reports" / "label_decisions.csv"
OUT_SUM = ROOT / "reports" / "cohort_summary.txt"

SOURCE = "LUDB"
NORMAL_TOKEN = "LUDB_NO_PATHOLOGICAL_FINDING"

# Diagnosis lines that are NOT pathology findings. A record whose diagnosis
# list contains only these is the source asserting "nothing found".
NON_PATHOLOGY = re.compile(
    r"^(Rhythm: Sinus (rhythm|bradycardia|tachycardia|arrhythmia)"
    r"|Electric axis of the heart: (normal|vertical|horizontal))$")

META_FIELDS = [
    "global_record_id", "source_dataset", "original_record_id", "patient_id",
    "patient_id_is_synthetic", "split_group_id", "original_labels",
    "original_labels_readable", "accepted_label_mapping", "cardiosentry_label",
    "STEMI", "AF", "LVH", "NORMAL", "n_positive_labels", "sampling_frequency",
    "duration_seconds", "num_leads", "num_samples", "lead_names", "signal_path",
    "header_path", "data_path", "validation_status", "validation_issues",
    "ambiguity_flag", "ambiguity_note", "age", "sex", "source_extra_json",
]


def diagnoses(header_text: str) -> list:
    lines = [l.lstrip("#").strip() for l in header_text.splitlines()
             if l.startswith("#")]
    try:
        i = lines.index("<diagnoses>:")
    except ValueError:
        return []
    out = []
    for l in lines[i + 1:]:
        l = re.sub(r"\s+", " ", l).strip().rstrip(".").strip()
        if l:
            out.append(l)
    return out


def main() -> int:
    if not MAPPING.exists():
        print(f"missing {MAPPING}", file=sys.stderr)
        return 1
    records_file = LUDB_ROOT / "RECORDS"
    if not records_file.exists():
        print(f"LUDB not found at {LUDB_ROOT}", file=sys.stderr)
        return 1

    mapping = yaml.safe_load(MAPPING.read_text(encoding="utf-8"))
    resolver = LabelResolver(mapping, "ludb")

    ids = [l.strip() for l in records_file.read_text().splitlines() if l.strip()]
    ids.sort(key=lambda s: int(re.sub(r"\D", "", s) or 0))

    meta, decisions = [], []
    stats, unknown_terms = Counter(), Counter()
    n_kept = 0

    for rec in ids:
        stem = LUDB_ROOT / rec if (LUDB_ROOT / (rec + ".hea")).exists() \
            else LUDB_ROOT / "data" / Path(rec).name
        hea = Path(str(stem) + ".hea")
        if not hea.exists():
            stats["missing_header"] += 1
            continue

        text = hea.read_text(encoding="utf-8", errors="replace")
        hi = parse_header(str(hea))
        dx = diagnoses(text)
        raw = list(dx)
        if dx and all(NON_PATHOLOGY.match(d) for d in dx):
            raw.append(NORMAL_TOKEN)

        d = resolver.resolve(raw)
        binary, amb_flag, amb_note = resolver.finalize(d)
        positives = [c for c in TARGET_CLASSES if binary[c]]
        for u in d.unknown:
            unknown_terms[u] += 1

        # RECORDS entries look like "data/1"; the bare number is the
        # record id AND, since LUDB is one record per subject, the subject id.
        rec_num = Path(rec).name
        rec_id = f"{SOURCE}_{int(rec_num):06d}"
        accepted = "; ".join(f"{lab}->{cls}" for cls, labs in sorted(d.positives.items())
                             for lab in labs)
        dec = {
            "ludb_id": rec_num, "global_record_id": rec_id,
            "diagnoses": " | ".join(dx),
            "normal_token_emitted": int(NORMAL_TOKEN in raw),
            "STEMI": binary["STEMI"], "AF": binary["AF"], "LVH": binary["LVH"],
            "NORMAL": binary["NORMAL"],
            "cardiosentry_label": "+".join(positives),
            "accepted_label_mapping": accepted,
            "record_excluders": "|".join(sorted(set(d.record_excluders))),
            "unknown_terms": "|".join(sorted(set(d.unknown))),
            "decision": "",
        }

        if d.record_excluders:
            dec["decision"] = "EXCLUDED_RECORD_EXCLUDER"
            stats["excluded_by_label"] += 1
        elif not positives:
            dec["decision"] = "EXCLUDED_NO_TARGET_LABEL"
            stats["excluded_no_label"] += 1
        else:
            dec["decision"] = "CANDIDATE"
            stats["candidates"] += 1
            n_kept += 1
            for c in positives:
                stats[f"class_{c}"] += 1

            issues = []
            if hi.fs != 500:
                issues.append(f"fs={hi.fs}")
            if hi.n_samples != 5000:
                issues.append(f"n_samples={hi.n_samples}")
            if hi.n_sig != 12:
                issues.append(f"n_leads={hi.n_sig}")

            meta.append({
                "global_record_id": rec_id,
                "source_dataset": SOURCE,
                "original_record_id": rec_num,
                # LUDB is 200 records from 200 distinct subjects (README), so
                # record id and subject id are one and the same.
                "patient_id": f"{SOURCE}_P{rec_num}",
                "patient_id_is_synthetic": 0,
                "split_group_id": f"{SOURCE}::{SOURCE}_P{rec_num}",
                "original_labels": "|".join(dx),
                "original_labels_readable": "; ".join(dx),
                "accepted_label_mapping": accepted,
                "cardiosentry_label": "+".join(positives),
                "STEMI": binary["STEMI"], "AF": binary["AF"],
                "LVH": binary["LVH"], "NORMAL": binary["NORMAL"],
                "n_positive_labels": len(positives),
                "sampling_frequency": float(hi.fs),
                "duration_seconds": round(hi.n_samples / hi.fs, 3),
                "num_leads": hi.n_sig,
                "num_samples": hi.n_samples,
                "lead_names": "|".join(hi.lead_names),
                "signal_path": str(stem),
                "header_path": str(hea),
                "data_path": str(stem) + ".dat",
                "validation_status": "VALID" if not issues else "CHECK",
                "validation_issues": "; ".join(issues),
                "ambiguity_flag": amb_flag,
                "ambiguity_note": amb_note,
                "age": hi.comments.get("<age>", ""),
                "sex": {"M": "Male", "F": "Female"}.get(
                    hi.comments.get("<sex>", "").strip(),
                    hi.comments.get("<sex>", "")),
                "source_extra_json": json.dumps({
                    "label_provenance": "HUMAN_CARDIOLOGIST",
                    "ludb_diagnoses": dx,
                    "normal_token_emitted": NORMAL_TOKEN in raw,
                    "note": "signals referenced in place; LUDB already ships "
                            "12-lead 500 Hz 10 s WFDB, no transcoding applied",
                }, ensure_ascii=False),
            })
        decisions.append(dec)

    OUT_DEC.parent.mkdir(parents=True, exist_ok=True)
    with OUT_DEC.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(decisions[0].keys()))
        w.writeheader()
        w.writerows(decisions)
    OUT_META.parent.mkdir(parents=True, exist_ok=True)
    with OUT_META.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=META_FIELDS)
        w.writeheader()
        w.writerows(meta)

    lines = [
        "LUDB -- staged cohort summary",
        "=" * 60,
        f"records in RECORDS               : {len(ids)}",
        f"  dropped, EXCLUDE_RECORD label  : {stats['excluded_by_label']}",
        f"  dropped, no target label       : {stats['excluded_no_label']}",
        f"  CANDIDATES (>=1 target label)  : {stats['candidates']}",
        "",
        "candidates by class (multi-label)",
    ]
    for c in TARGET_CLASSES:
        lines.append(f"  {c:<7} {stats[f'class_{c}']}")
    if unknown_terms:
        lines += ["", "diagnosis terms NOT declared in label_map_ludb.yaml "
                      "(these block NORMAL and create no label -- declare them):"]
        for k, v in unknown_terms.most_common():
            lines.append(f"  {v:>4}  {k}")
    else:
        lines += ["", "every diagnosis term in the database is declared in "
                      "config/label_map_ludb.yaml."]
    OUT_SUM.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT_META} ({len(meta)} rows)\nwrote {OUT_DEC}\nwrote {OUT_SUM}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
