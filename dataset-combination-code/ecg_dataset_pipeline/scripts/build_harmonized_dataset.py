#!/usr/bin/env python3
"""
STEPS 5, 7, 8, 9 -- Build the MAXIMUM VALID harmonized candidate pool.

Joins the extraction, signal-validation and duplicate reports into one
auditable metadata table. Nothing is balanced, downsampled or randomly
discarded here: this is the honest ceiling of what the data supports.

Outputs
  data/harmonized/harmonized_ecg_metadata.csv   the dataset table
  reports/harmonization_report.csv              per-record inclusion/exclusion ledger
  reports/final_class_counts.csv                STEP 8 class x source counts
  reports/balancing_recommendation.csv          STEP 9 recommendation (no action taken)
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (TARGET_CLASSES, TARGET_DURATION_S, TARGET_FS,
                    TARGET_N_SAMPLES, banner, harmonized_dir, info, load_mapping,
                    load_paths, reports_dir, warn, write_csv)

HARMONIZED_FIELDS = [
    "global_record_id", "source_dataset", "original_record_id", "patient_id",
    "patient_id_is_synthetic", "split_group_id",
    "original_labels", "original_labels_readable", "accepted_label_mapping",
    "cardiosentry_label", "STEMI", "AF", "LVH", "NORMAL", "n_positive_labels",
    "sampling_frequency", "duration_seconds", "num_leads", "num_samples",
    "lead_names", "signal_path", "header_path", "data_path",
    "validation_status", "validation_issues", "ambiguity_flag", "ambiguity_note",
    "age", "sex", "source_extra_json",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", default=None)
    ap.add_argument("--mapping", default=None)
    ap.add_argument("--copy-signals", action="store_true",
                    help="physically copy each valid signal file into data/harmonized/signals/ "
                         "(~2.5 GB). Off by default: the metadata references the ORIGINAL "
                         "read-only files, which are never modified either way.")
    args = ap.parse_args(argv)

    paths = load_paths(args.paths)
    mapping = load_mapping(args.mapping)
    rdir = reports_dir(paths)
    hdir = harmonized_dir(paths)

    def _read(name, required=True):
        p = rdir / name
        if not p.exists():
            if required:
                warn(f"{p} not found -- run the earlier pipeline steps first")
                sys.exit(1)
            return []
        with open(p, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    cands = _read("extracted_candidates.csv")
    sig = {r["global_record_id"]: r for r in _read("signal_compatibility_report.csv")}
    dups = _read("duplicate_patient_report.csv", required=False)

    # duplicate flags per record
    dup_flags = collections.defaultdict(list)
    for d in dups:
        for gid in d["global_record_ids"].split("|"):
            if gid:
                dup_flags[gid].append(f"{d['finding_type']}({d['severity']})")

    harmonized, ledger = [], []
    counts = collections.Counter()          # (class, source) -> valid
    cand_counts = collections.Counter()     # (class, source) -> candidate
    excl_counts = collections.Counter()     # (class, source) -> excluded
    excl_reasons = collections.Counter()

    for r in cands:
        gid = r["global_record_id"]
        s = sig.get(gid, {})
        vstatus = s.get("validation_status", "NOT_VALIDATED")
        classes = [c for c in TARGET_CLASSES if r.get(c) == "1"]
        for c in classes:
            cand_counts[(c, r["source_dataset"])] += 1

        keep = vstatus.startswith("VALID")
        if not keep:
            for c in classes:
                excl_counts[(c, r["source_dataset"])] += 1
            excl_reasons[vstatus] += 1
            ledger.append({
                "global_record_id": gid, "source_dataset": r["source_dataset"],
                "original_record_id": r["original_record_id"], "patient_id": r["patient_id"],
                "original_labels": r["original_labels"],
                "cardiosentry_label": r["cardiosentry_label"],
                "included": 0, "validation_status": vstatus,
                "reason": s.get("issues", "") or "record failed signal validation",
                "duplicate_flags": "|".join(dup_flags.get(gid, [])),
            })
            continue

        for c in classes:
            counts[(c, r["source_dataset"])] += 1

        harmonized.append({
            "global_record_id": gid,
            "source_dataset": r["source_dataset"],
            "original_record_id": r["original_record_id"],
            "patient_id": r["patient_id"],
            "patient_id_is_synthetic": r["patient_id_is_synthetic"],
            # Records sharing a split_group_id MUST land in the same
            # train/val/test fold. For sources with real patient ids this is
            # the patient; for Chapman it degenerates to the record itself.
            "split_group_id": f"{r['source_dataset']}::{r['patient_id']}",
            "original_labels": r["original_labels"],
            "original_labels_readable": r["original_labels_readable"],
            "accepted_label_mapping": r["accepted_label_mapping"],
            "cardiosentry_label": r["cardiosentry_label"],
            "STEMI": r["STEMI"], "AF": r["AF"], "LVH": r["LVH"], "NORMAL": r["NORMAL"],
            "n_positive_labels": r["n_positive_labels"],
            "sampling_frequency": s.get("sampling_frequency", ""),
            "duration_seconds": s.get("duration_seconds", ""),
            "num_leads": s.get("num_leads", ""),
            "num_samples": s.get("num_samples", ""),
            "lead_names": s.get("lead_names", ""),
            "signal_path": r["signal_path"],
            "header_path": r["header_path"],
            "data_path": r["data_path"],
            "validation_status": vstatus,
            "validation_issues": s.get("issues", ""),
            "ambiguity_flag": r["ambiguity_flag"],
            "ambiguity_note": r["ambiguity_note"],
            "age": r["age"], "sex": r["sex"],
            "source_extra_json": r["source_extra_json"],
        })
        ledger.append({
            "global_record_id": gid, "source_dataset": r["source_dataset"],
            "original_record_id": r["original_record_id"], "patient_id": r["patient_id"],
            "original_labels": r["original_labels"],
            "cardiosentry_label": r["cardiosentry_label"],
            "included": 1, "validation_status": vstatus,
            "reason": "passed label mapping and signal validation",
            "duplicate_flags": "|".join(dup_flags.get(gid, [])),
        })

    out_csv = hdir / "harmonized_ecg_metadata.csv"
    write_csv(out_csv, harmonized, HARMONIZED_FIELDS)
    write_csv(rdir / "harmonization_report.csv", ledger,
              ["global_record_id", "source_dataset", "original_record_id", "patient_id",
               "original_labels", "cardiosentry_label", "included", "validation_status",
               "reason", "duplicate_flags"])

    # ---- optional physical copy ---------------------------------------
    # Renaming a .hea on disk is not enough: a WFDB header names its own record
    # on line 1 and names its signal file on every signal line, so a renamed
    # copy whose contents still say `JS00001.mat` is unreadable by wfdb. The
    # rewrite lives in export_dataset.py, which is the fuller tool for this.
    if args.copy_signals:
        from export_dataset import export_record
        sdir = hdir / "signals"
        sdir.mkdir(parents=True, exist_ok=True)
        info(f"copying {len(harmonized):,} signal pairs into {sdir} (originals untouched) ...")
        n_bad = 0
        for i, h in enumerate(harmonized, 1):
            try:
                export_record(h, sdir)
            except Exception as exc:              # noqa: BLE001 - report, continue
                n_bad += 1
                if n_bad <= 10:
                    warn(f"  {h['global_record_id']}: {type(exc).__name__}: {exc}")
            if i % 2000 == 0:
                info(f"  {i:,}/{len(harmonized):,}")
        if n_bad:
            warn(f"{n_bad} records could not be copied")

    # ---- STEP 8: final class counts -----------------------------------
    src_of = {"STEMI": ["STEMI"], "AF": ["PTBXL", "CHAPMAN"],
              "LVH": ["PTBXL", "CHAPMAN"], "NORMAL": ["PTBXL", "CHAPMAN"]}
    rows8 = []
    for cls in TARGET_CLASSES:
        for s in src_of[cls]:
            cand = cand_counts[(cls, s)]
            val = counts[(cls, s)]
            if cand == 0 and val == 0:
                continue
            rows8.append({
                "class": cls, "source": s, "candidate": cand, "valid": val,
                "excluded": cand - val,
                "pct_retained": f"{100*val/cand:.2f}" if cand else "",
            })
    for cls in TARGET_CLASSES:
        cand = sum(cand_counts[(cls, s)] for s in src_of[cls])
        val = sum(counts[(cls, s)] for s in src_of[cls])
        rows8.append({"class": cls, "source": "TOTAL", "candidate": cand, "valid": val,
                      "excluded": cand - val,
                      "pct_retained": f"{100*val/cand:.2f}" if cand else ""})
    write_csv(rdir / "final_class_counts.csv", rows8,
              ["class", "source", "candidate", "valid", "excluded", "pct_retained"])

    valid_tot = {c: sum(counts[(c, s)] for s in src_of[c]) for c in TARGET_CLASSES}

    # ---- STEP 9: balancing recommendation ------------------------------
    pref = {"STEMI": 1442, "AF": 2800, "LVH": 2700, "NORMAL": 2800}
    stemi_valid = valid_tot["STEMI"]

    # Requirement: STEMI stays the smallest class and nothing falls below it.
    rec = {}
    for c in TARGET_CLASSES:
        avail = valid_tot[c]
        if c == "STEMI":
            rec[c] = avail                     # never discard the scarcest class
        else:
            t = min(pref[c], avail)
            t = max(t, min(stemi_valid, avail))  # never below STEMI (unless impossible)
            rec[c] = t

    rows9 = []
    for c in TARGET_CLASSES:
        avail, tgt = valid_tot[c], rec[c]
        pref_t = pref[c]
        rows9.append({
            "class": c,
            "available_valid_records": avail,
            "current_ratio_to_stemi": f"{avail/stemi_valid:.3f}" if stemi_valid else "",
            "preferred_target_from_brief": pref_t,
            "preferred_target_achievable": "yes" if pref_t <= avail else "NO",
            "recommended_target": tgt,
            "would_be_retained": tgt,
            "would_be_discarded": avail - tgt,
            "pct_retained": f"{100*tgt/avail:.2f}" if avail else "",
            "recommended_ratio_to_stemi": f"{tgt/rec['STEMI']:.3f}" if rec["STEMI"] else "",
            "note": ("scarcest class -- kept in full, defines the floor"
                     if c == "STEMI" else
                     (f"preferred target {pref_t} exceeds the {avail} valid records available; "
                      f"capped at {tgt}" if pref_t > avail else
                      f"preferred target {pref_t} is achievable")),
        })
    rows9.append({
        "class": "TOTAL",
        "available_valid_records": sum(valid_tot.values()),
        "current_ratio_to_stemi": "",
        "preferred_target_from_brief": sum(pref.values()),
        "preferred_target_achievable": "",
        "recommended_target": sum(rec.values()),
        "would_be_retained": sum(rec.values()),
        "would_be_discarded": sum(valid_tot.values()) - sum(rec.values()),
        "pct_retained": f"{100*sum(rec.values())/sum(valid_tot.values()):.2f}",
        "recommended_ratio_to_stemi": "",
        "note": "no balancing has been performed -- this is a recommendation only",
    })
    write_csv(rdir / "balancing_recommendation.csv", rows9, list(rows9[0].keys()))

    # ---- print ---------------------------------------------------------
    print(banner("STEP 7-8 -- MAXIMUM VALID HARMONIZED DATASET"))
    print(f"  harmonized records : {len(harmonized):,}")
    print(f"  written to         : {out_csv}")
    print(f"  signal files       : {'copied into data/harmonized/signals/' if args.copy_signals else 'referenced in place (originals never modified)'}")
    print()
    print(f"  {'Class':7s} {'Source':9s} {'Candidate':>10s} {'Valid':>8s} {'Excluded':>9s} {'%kept':>7s}")
    print(f"  {'-'*7} {'-'*9} {'-'*10} {'-'*8} {'-'*9} {'-'*7}")
    for r in rows8:
        sep = "  " if r["source"] == "TOTAL" else "  "
        print(f"{sep}{r['class']:7s} {r['source']:9s} {r['candidate']:10,d} {r['valid']:8,d} "
              f"{r['excluded']:9,d} {r['pct_retained']:>7s}")

    print("\n  exclusions by validation status:")
    for k, n in excl_reasons.most_common():
        print(f"    {k:28s} {n:6,d}")

    ml = collections.Counter(h["cardiosentry_label"] for h in harmonized)
    multi = {k: v for k, v in ml.items() if "+" in k}
    print(f"\n  multi-label records (STEP 11): {sum(multi.values()):,}")
    for k, v in sorted(multi.items(), key=lambda x: -x[1]):
        print(f"    {k:20s} {v:6,d}")
    if not multi:
        print("    (none)")

    print(banner("STEP 9 -- BALANCING RECOMMENDATION (nothing performed)"))
    print(f"  {'Class':8s} {'Available':>10s} {'Ratio':>7s} {'Preferred':>10s} {'Recommended':>12s} "
          f"{'Discard':>8s} {'%kept':>7s}")
    print(f"  {'-'*8} {'-'*10} {'-'*7} {'-'*10} {'-'*12} {'-'*8} {'-'*7}")
    for r in rows9[:-1]:
        print(f"  {r['class']:8s} {r['available_valid_records']:10,d} "
              f"{r['current_ratio_to_stemi']:>7s} {r['preferred_target_from_brief']:10,d} "
              f"{r['recommended_target']:12,d} {r['would_be_discarded']:8,d} {r['pct_retained']:>7s}")
    t = rows9[-1]
    print(f"  {'TOTAL':8s} {t['available_valid_records']:10,d} {'':>7s} "
          f"{t['preferred_target_from_brief']:10,d} {t['recommended_target']:12,d} "
          f"{t['would_be_discarded']:8,d} {t['pct_retained']:>7s}")
    for r in rows9[:-1]:
        if r["preferred_target_achievable"] == "NO":
            print(f"\n  !! {r['class']}: {r['note']}")
    print(f"\n  Constraint check -- STEMI is the smallest recommended class: "
          f"{'YES' if all(rec['STEMI'] <= rec[c] for c in TARGET_CLASSES) else 'NO'}")

    for f in ("harmonization_report.csv", "final_class_counts.csv", "balancing_recommendation.csv"):
        info(f"wrote {rdir/f}")
    info(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
