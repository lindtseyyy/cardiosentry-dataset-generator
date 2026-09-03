"""
CardioSentry / CUHK-STEMI staging -- resolve labels, choose the cohort.

Applies config/label_map_cuhk.yaml to every XML in data/index/xml_index.csv,
using the MAIN pipeline's own `common.LabelResolver` -- not a reimplementation
-- so that a CUHK label means exactly what the same status means for PTB-XL,
Chapman and the 2026 ACS/STEMI source.

Writes:
  data/index/candidates.csv            records with >=1 target label: the set
                                       convert_to_wfdb.py will decode
  reports/label_decisions.csv          every XML, its decision and why
  reports/excluded_records.csv         every dropped XML with the reason
  reports/cohort_summary.txt           counts by class, patient and site
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_PIPELINE = Path(os.environ.get(
    "CARDIOSENTRY_PIPELINE",
    ROOT.parents[1] / "dataset-combination-code" / "ecg_dataset_pipeline"))
sys.path.insert(0, str(MAIN_PIPELINE / "scripts"))

import yaml                                                    # noqa: E402
from common import TARGET_CLASSES, LabelResolver               # noqa: E402

INDEX = ROOT / "data" / "index" / "xml_index.csv"
MAPPING = ROOT / "config" / "label_map_cuhk.yaml"
OUT_CAND = ROOT / "data" / "index" / "candidates.csv"
OUT_DEC = ROOT / "reports" / "label_decisions.csv"
OUT_EXCL = ROOT / "reports" / "excluded_records.csv"
OUT_SUM = ROOT / "reports" / "cohort_summary.txt"

SOURCE = "CUHK"


def patient_group(pid: str) -> str:
    """The identity CUHK's anonymisation actually supports.

    Patient ids are anonymised by keeping the first 4 characters and masking
    the rest ("A016XXXX"). A handful of records escaped the mask and carry the
    full id, so the first 4 characters are the ONLY spelling every cohort
    agrees on -- grouping on anything longer would split one patient across
    folds.

    The mask is lossy in the other direction: 934 of 3,813 masked ids carry
    conflicting sex values, i.e. one 4-character prefix covers several real
    patients. That over-groups, which costs a little data but cannot leak a
    patient across a split. Over-grouping is the safe error here.
    """
    return (pid or "")[:4]


def main() -> int:
    if not INDEX.exists():
        print(f"missing {INDEX} -- run scripts/scan_xml_index.py", file=sys.stderr)
        return 1
    if not MAPPING.exists():
        print(f"missing {MAPPING} -- run scripts/build_label_map.py", file=sys.stderr)
        return 1

    mapping = yaml.safe_load(MAPPING.read_text(encoding="utf-8"))
    resolver = LabelResolver(mapping, "cuhk_stemi")

    rows = [r for r in csv.DictReader(INDEX.open(encoding="utf-8"))
            if not r.get("parse_error")]

    # ---- de-duplicate ------------------------------------------------
    # CUHK ships the same acquisition in several cohort folders when a patient
    # belongs to more than one registry (a STEMI patient who also has heart
    # failure appears under both). The Philips document GUID is an exact
    # per-acquisition identity, so it is the dedup key; the cohorts a record
    # appeared under are preserved on the surviving copy.
    # When the same acquisition survives in several folders, the copy kept is
    # the one from the most SPECIFIC registry, so the `cohort` column stays
    # clinically meaningful. `cohorts_all` keeps every membership either way,
    # and the label never depends on this choice -- it comes from the Philips
    # statements, which are identical in every copy.
    COHORT_PRIORITY = ["STEMI", "NSTEMI", "MYOCARDITIS", "PERICARDITIS",
                       "HF_AS", "HF", "MIXED_REDACTED"]
    prio = {c: i for i, c in enumerate(COHORT_PRIORITY)}

    by_guid: dict = {}
    dup_cohorts: dict = {}
    n_raw = len(rows)
    for r in sorted(rows, key=lambda r: (prio.get(r["cohort"], 99),
                                         r["source_folder"], r["xml_file"])):
        g = r["document_guid"]
        dup_cohorts.setdefault(g, []).append(r["cohort"])
        if g not in by_guid:
            by_guid[g] = r
    rows = list(by_guid.values())
    n_dupes = n_raw - len(rows)
    for r in rows:
        cs = sorted(set(dup_cohorts[r["document_guid"]]))
        r["cohorts_all"] = "|".join(cs)
        r["n_cohorts"] = str(len(cs))

    rows.sort(key=lambda r: (patient_group(r["patient_id"]), r["patient_id"],
                             r["acquisition_date"], r["acquisition_time"],
                             r["xml_file"]))
    print(f"{n_raw} scanned records -> {len(rows)} unique acquisitions "
          f"({n_dupes} duplicate copies across cohort folders removed)")

    decisions, candidates, excluded = [], [], []
    stats = Counter()
    per_class_patients = defaultdict(set)

    for i, r in enumerate(rows, 1):
        codes = [c for c in (r["statement_codes"] or "").split("|") if c and c != "?"]
        d = resolver.resolve(codes)
        binary, amb_flag, amb_note = resolver.finalize(d)
        positives = [c for c in TARGET_CLASSES if binary[c]]
        accepted = "; ".join(
            f"{lab}->{cls}" for cls, labs in sorted(d.positives.items()) for lab in labs)

        dec = {
            "xml_file": r["xml_file"],
            "cohort": r["cohort"],
            "cohorts_all": r["cohorts_all"],
            "site": r["site"],
            "patient_id": r["patient_id"],
            "acquisition_date": r["acquisition_date"],
            "philips_codes": r["statement_codes"],
            "STEMI": binary["STEMI"], "AF": binary["AF"],
            "LVH": binary["LVH"], "NORMAL": binary["NORMAL"],
            "n_positive_labels": len(positives),
            "cardiosentry_label": "+".join(positives),
            "accepted_label_mapping": accepted,
            "record_excluders": "|".join(sorted(set(d.record_excluders))),
            "ambiguity_flag": amb_flag,
            "ambiguity_note": amb_note,
            "decision": "",
        }

        if d.record_excluders:
            dec["decision"] = "EXCLUDED_RECORD_EXCLUDER"
            stats["excluded_by_label"] += 1
            excluded.append({**dec, "exclusion_reason":
                             "carries EXCLUDE_RECORD label(s): "
                             + "|".join(sorted(set(d.record_excluders)))})
        elif not positives:
            dec["decision"] = "EXCLUDED_NO_TARGET_LABEL"
            stats["excluded_no_label"] += 1
            excluded.append({**dec, "exclusion_reason": "NO_TARGET_LABEL"})
        else:
            dec["decision"] = "CANDIDATE"
            stats["candidates"] += 1
            for c in positives:
                stats[f"class_{c}"] += 1
                per_class_patients[c].add(patient_group(r["patient_id"]))
            candidates.append({
                "global_record_id": "",              # assigned below, in order
                "source_dataset": SOURCE,
                "original_record_id": Path(r["xml_file"]).stem,
                "patient_id": f"{SOURCE}_P{patient_group(r['patient_id'])}",
                "patient_id_is_synthetic": 0,
                "split_group_id":
                    f"{SOURCE}::{SOURCE}_P{patient_group(r['patient_id'])}",
                "cuhk_patient_id_raw": r["patient_id"],
                "document_guid": r["document_guid"],
                "cohort": r["cohort"],
                "cohorts_all": r["cohorts_all"],
                "site": r["site"],
                "xml_path": r["xml_path"],
                "original_labels": r["statement_codes"],
                "original_labels_readable": r["statement_text"],
                "accepted_label_mapping": accepted,
                "cardiosentry_label": "+".join(positives),
                "STEMI": binary["STEMI"], "AF": binary["AF"],
                "LVH": binary["LVH"], "NORMAL": binary["NORMAL"],
                "n_positive_labels": len(positives),
                "ambiguity_flag": amb_flag,
                "ambiguity_note": amb_note,
                "age": "" if r["age_default_used"] == "1" else r["age_years"],
                "sex": r["sex"],
                "acquisition_date": r["acquisition_date"],
                "acquisition_time": r["acquisition_time"],
                "philips_qa_action_code": r["qa_action_code"],
                "philips_heartrate": r["heartrate"],
                "native_resolution_uv": r["resolution_uv"],
                "native_duration_ms": r["duration_ms"],
                "native_lead_labels": r["lead_labels"],
            })
        decisions.append(dec)

    # Stable, patient-then-time ordered ids.
    for n, c in enumerate(candidates, 1):
        c["global_record_id"] = f"{SOURCE}_{n:06d}"

    OUT_DEC.parent.mkdir(parents=True, exist_ok=True)
    with OUT_DEC.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(decisions[0].keys()))
        w.writeheader()
        w.writerows(decisions)
    with OUT_EXCL.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(excluded[0].keys()))
        w.writeheader()
        w.writerows(excluded)
    with OUT_CAND.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(candidates[0].keys()))
        w.writeheader()
        w.writerows(candidates)

    # ---- summary -----------------------------------------------------
    excl_reasons = Counter()
    for e in excluded:
        if e["decision"] == "EXCLUDED_RECORD_EXCLUDER":
            for lab in e["record_excluders"].split("|"):
                if lab:
                    excl_reasons[f"EXCLUDE_RECORD:{lab}"] += 1
        else:
            excl_reasons["NO_TARGET_LABEL"] += 1

    combos = Counter(c["cardiosentry_label"] for c in candidates)
    per_cohort = defaultdict(Counter)
    for c in candidates:
        for cls in c["cardiosentry_label"].split("+"):
            per_cohort[c["cohort"]][cls] += 1
        per_cohort[c["cohort"]]["_records"] += 1

    lines = [
        "CUHK ECG collection -- staged cohort summary",
        "=" * 68,
        f"XML files scanned                : {n_raw}",
        f"  duplicate copies removed       : {n_dupes} "
        f"(same acquisition in >1 cohort folder)",
        f"unique acquisitions              : {len(rows)}",
        f"  dropped, EXCLUDE_RECORD label  : {stats['excluded_by_label']}",
        f"  dropped, no target label       : {stats['excluded_no_label']}",
        f"  CANDIDATES (>=1 target label)  : {stats['candidates']}",
        "",
        "candidates by class (multi-label: a record may appear in several)",
    ]
    for c in TARGET_CLASSES:
        lines.append(f"  {c:<7} records={stats[f'class_{c}']:<6} "
                     f"patients={len(per_class_patients[c])}")
    lines += [
        "",
        "MULTI-LABEL BREAKDOWN (exact label combination per record)",
        f"  {'combination':<22}{'records':>9}{'% of candidates':>18}",
    ]
    for k, v in sorted(combos.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k:<22}{v:>9}{100.0 * v / max(len(candidates), 1):>17.1f}%")
    n_multi = sum(v for k, v in combos.items() if "+" in k)
    lines += [
        f"  {'-' * 46}",
        f"  {'single-label':<22}{len(candidates) - n_multi:>9}",
        f"  {'multi-label (>=2)':<22}{n_multi:>9}"
        f"{100.0 * n_multi / max(len(candidates), 1):>17.1f}%",
        "",
        "BY SOURCE COHORT (the folder a record came from -- provenance, NOT a",
        "label; every label above comes from the Philips statements only)",
        f"  {'cohort':<16}{'records':>9}{'STEMI':>8}{'AF':>8}{'LVH':>8}",
    ]
    for coh in sorted(per_cohort, key=lambda c: -per_cohort[c]["_records"]):
        p = per_cohort[coh]
        lines.append(f"  {coh:<16}{p['_records']:>9}{p['STEMI']:>8}"
                     f"{p['AF']:>8}{p['LVH']:>8}")
    lines += [
        "",
        f"distinct patient groups in candidate set : "
        f"{len({c['patient_id'] for c in candidates})}",
        f"distinct patient groups in whole dataset : "
        f"{len({patient_group(r['patient_id']) for r in rows})}",
        "  (a 'patient group' is the 4-character anonymised id prefix -- see",
        "   patient_group() in this script: it over-groups, never under-groups)",
        "",
        "records per patient group in the candidate set (serial ECGs -- these",
        "MUST be grouped by `split_group_id` when splitting):",
    ]
    ppc = Counter(c["patient_id"] for c in candidates)
    vals = sorted(ppc.values())
    if vals:
        lines.append(f"  min={vals[0]} median={vals[len(vals)//2]} "
                     f"max={vals[-1]} mean={sum(vals)/len(vals):.2f}")
    lines += ["", "top exclusion reasons:"]
    for k, v in excl_reasons.most_common(25):
        lines.append(f"  {v:>6}  {k}")

    OUT_SUM.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT_CAND}\nwrote {OUT_DEC}\nwrote {OUT_EXCL}\nwrote {OUT_SUM}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
