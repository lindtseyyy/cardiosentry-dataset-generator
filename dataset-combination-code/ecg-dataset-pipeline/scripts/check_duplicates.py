#!/usr/bin/env python3
"""
STEP 6 -- Duplicate detection and patient-overlap analysis.

Checks:
  * duplicate global / original recording IDs
  * duplicate patient IDs (i.e. repeated ECGs from one patient)
  * duplicate ECG signal FILES (exact, by content hash)
  * near-identical signals (optional, --near-duplicates)
  * cross-source record-id and patient-id collisions

*** Nothing is deleted. *** Repeated ECGs from the same patient are
legitimate clinical data; they are reported so that train/val/test splitting
can be done at the PATIENT level.

Outputs
  reports/duplicate_patient_report.csv
  reports/patient_summary.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import banner, info, load_paths, reports_dir, warn, write_csv

DUP_FIELDS = [
    "finding_type", "scope", "key", "n_records", "source_datasets",
    "global_record_ids", "cardiosentry_labels", "severity", "action", "note",
]


def _hash_file(args):
    gid, path = args
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return gid, h.hexdigest(), ""
    except OSError as exc:
        return gid, "", str(exc)


def _signal_fingerprint(args):
    """Coarse content fingerprint for near-duplicate detection.

    Rounds each lead's mean/std/min/max to 3 decimals. Two recordings sharing
    a fingerprint are near-certainly the same signal (possibly re-scaled or
    re-encoded); they still need eyeballing before anything is removed.
    """
    gid, path = args
    try:
        import numpy as np
        import wfdb
        sig = np.asarray(wfdb.rdrecord(str(path)).p_signal, dtype=np.float64)
        if sig.ndim != 2:
            return gid, "", "not 2-D"
        with np.errstate(invalid="ignore"):
            feats = np.concatenate([
                np.nanmean(sig, axis=0), np.nanstd(sig, axis=0),
                np.nanmin(sig, axis=0), np.nanmax(sig, axis=0),
            ])
        key = ",".join(f"{v:.3f}" for v in np.nan_to_num(feats))
        return gid, hashlib.sha256(key.encode()).hexdigest()[:32], ""
    except Exception as exc:                       # noqa: BLE001
        return gid, "", f"{type(exc).__name__}: {exc}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--near-duplicates", action="store_true",
                    help="also fingerprint every signal to find near-identical "
                         "recordings (slow: reads all samples again)")
    ap.add_argument("--valid-only", action="store_true", default=True,
                    help="restrict to records that passed signal validation (default)")
    ap.add_argument("--all-records", dest="valid_only", action="store_false")
    args = ap.parse_args(argv)

    paths = load_paths(args.paths)
    rdir = reports_dir(paths)

    cand_csv = rdir / "extracted_candidates.csv"
    if not cand_csv.exists():
        warn(f"{cand_csv} not found -- run extract_records.py first")
        return 1
    with open(cand_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    status = {}
    sig_csv = rdir / "signal_compatibility_report.csv"
    if sig_csv.exists():
        with open(sig_csv, newline="", encoding="utf-8") as fh:
            status = {r["global_record_id"]: r["validation_status"] for r in csv.DictReader(fh)}
    elif args.valid_only:
        warn("signal_compatibility_report.csv not found -- analysing ALL candidates")

    if args.valid_only and status:
        rows = [r for r in rows if status.get(r["global_record_id"], "").startswith("VALID")]
    info(f"analysing {len(rows):,} records")

    findings = []

    # ---- 1. duplicate global ids --------------------------------------
    by_gid = collections.defaultdict(list)
    for r in rows:
        by_gid[r["global_record_id"]].append(r)
    for k, v in by_gid.items():
        if len(v) > 1:
            findings.append({
                "finding_type": "DUPLICATE_GLOBAL_RECORD_ID", "scope": "pipeline", "key": k,
                "n_records": len(v), "source_datasets": "|".join(sorted({x["source_dataset"] for x in v})),
                "global_record_ids": "|".join(x["global_record_id"] for x in v),
                "cardiosentry_labels": "|".join(sorted({x["cardiosentry_label"] for x in v})),
                "severity": "ERROR", "action": "MUST FIX -- global ids must be unique",
                "note": "bug in id assignment",
            })

    # ---- 2. duplicate original record ids ------------------------------
    by_orig = collections.defaultdict(list)
    for r in rows:
        by_orig[(r["source_dataset"], r["original_record_id"])].append(r)
    for (src, oid), v in by_orig.items():
        if len(v) > 1:
            findings.append({
                "finding_type": "DUPLICATE_ORIGINAL_RECORD_ID", "scope": src, "key": oid,
                "n_records": len(v), "source_datasets": src,
                "global_record_ids": "|".join(x["global_record_id"] for x in v),
                "cardiosentry_labels": "|".join(sorted({x["cardiosentry_label"] for x in v})),
                "severity": "WARNING", "action": "REVIEW -- same source record appears twice",
                "note": "the source metadata lists this record id more than once",
            })

    # ---- 3. cross-source original-id collisions ------------------------
    by_oid_only = collections.defaultdict(set)
    for r in rows:
        by_oid_only[r["original_record_id"]].add(r["source_dataset"])
    for oid, srcs in by_oid_only.items():
        if len(srcs) > 1:
            v = [r for r in rows if r["original_record_id"] == oid]
            findings.append({
                "finding_type": "CROSS_SOURCE_RECORD_ID_COLLISION", "scope": "cross-dataset",
                "key": oid, "n_records": len(v), "source_datasets": "|".join(sorted(srcs)),
                "global_record_ids": "|".join(x["global_record_id"] for x in v),
                "cardiosentry_labels": "|".join(sorted({x["cardiosentry_label"] for x in v})),
                "severity": "INFO",
                "action": "NO ACTION -- the global_record_id prefix already disambiguates",
                "note": "different datasets happen to reuse the same local record number; "
                        "these are NOT the same recording",
            })

    # ---- 4. repeated ECGs per patient ----------------------------------
    by_pid = collections.defaultdict(list)
    for r in rows:
        by_pid[r["patient_id"]].append(r)
    real_multi = 0
    for pid, v in by_pid.items():
        if len(v) > 1:
            synth = any(x["patient_id_is_synthetic"] == "1" for x in v)
            real_multi += 0 if synth else 1
            findings.append({
                "finding_type": "REPEATED_ECG_SAME_PATIENT", "scope": v[0]["source_dataset"],
                "key": pid, "n_records": len(v),
                "source_datasets": "|".join(sorted({x["source_dataset"] for x in v})),
                "global_record_ids": "|".join(x["global_record_id"] for x in v),
                "cardiosentry_labels": "|".join(sorted({x["cardiosentry_label"] for x in v})),
                "severity": "INFO",
                "action": "KEEP ALL -- but these records MUST stay in the same split",
                "note": "legitimate repeated recordings from one patient; never split across "
                        "train/val/test or the model leaks patient identity",
            })

    # ---- 5. cross-source patient id collisions -------------------------
    pid_src = collections.defaultdict(set)
    for r in rows:
        if r["patient_id_is_synthetic"] != "1":
            pid_src[r["patient_id"]].add(r["source_dataset"])
    for pid, srcs in pid_src.items():
        if len(srcs) > 1:
            findings.append({
                "finding_type": "CROSS_SOURCE_PATIENT_ID_COLLISION", "scope": "cross-dataset",
                "key": pid, "n_records": 0, "source_datasets": "|".join(sorted(srcs)),
                "global_record_ids": "", "cardiosentry_labels": "", "severity": "WARNING",
                "action": "REVIEW", "note": "same patient id string in two datasets",
            })

    # ---- 6. exact duplicate signal files -------------------------------
    info("hashing signal files ...")
    tasks = [(r["global_record_id"], r["data_path"]) for r in rows if os.path.exists(r["data_path"])]
    hashes = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for gid, h, err in ex.map(_hash_file, tasks, chunksize=64):
            if h:
                hashes[gid] = h
    gid_row = {r["global_record_id"]: r for r in rows}
    by_hash = collections.defaultdict(list)
    for gid, h in hashes.items():
        by_hash[h].append(gid)
    n_exact = 0
    for h, gids in by_hash.items():
        if len(gids) > 1:
            n_exact += 1
            v = [gid_row[g] for g in gids]
            srcs = sorted({x["source_dataset"] for x in v})
            findings.append({
                "finding_type": "IDENTICAL_SIGNAL_FILE", "scope": "cross-dataset" if len(srcs) > 1 else srcs[0],
                "key": h[:16], "n_records": len(gids), "source_datasets": "|".join(srcs),
                "global_record_ids": "|".join(sorted(gids)),
                "cardiosentry_labels": "|".join(sorted({x["cardiosentry_label"] for x in v})),
                "severity": "ERROR" if len(srcs) > 1 else "WARNING",
                "action": "REVIEW BEFORE SPLITTING -- byte-identical signal files",
                "note": "same recording stored twice" + (" IN DIFFERENT DATASETS" if len(srcs) > 1 else ""),
            })

    # ---- 7. near-identical signals (optional) --------------------------
    n_near = 0
    if args.near_duplicates:
        info("fingerprinting signals for near-duplicate detection (slow) ...")
        tasks = [(r["global_record_id"], r["signal_path"]) for r in rows]
        fps = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (gid, fp, err) in enumerate(ex.map(_signal_fingerprint, tasks, chunksize=32), 1):
                if fp:
                    fps.setdefault(fp, []).append(gid)
                if i % 5000 == 0:
                    info(f"  {i:,}/{len(tasks):,}")
        for fp, gids in fps.items():
            if len(gids) > 1:
                same_hash = len({hashes.get(g) for g in gids}) == 1 and all(g in hashes for g in gids)
                if same_hash:
                    continue                       # already reported as exact duplicate
                n_near += 1
                v = [gid_row[g] for g in gids]
                srcs = sorted({x["source_dataset"] for x in v})
                findings.append({
                    "finding_type": "NEAR_IDENTICAL_SIGNAL", "scope": "cross-dataset" if len(srcs) > 1 else srcs[0],
                    "key": fp[:16], "n_records": len(gids), "source_datasets": "|".join(srcs),
                    "global_record_ids": "|".join(sorted(gids)),
                    "cardiosentry_labels": "|".join(sorted({x["cardiosentry_label"] for x in v})),
                    "severity": "WARNING", "action": "REVIEW -- inspect before keeping both",
                    "note": "identical per-lead mean/std/min/max to 3 dp",
                })

    write_csv(rdir / "duplicate_patient_report.csv", findings, DUP_FIELDS)

    # ---- patient summary ----------------------------------------------
    summary = []
    for src in sorted({r["source_dataset"] for r in rows}):
        sub = [r for r in rows if r["source_dataset"] == src]
        synth = all(r["patient_id_is_synthetic"] == "1" for r in sub)
        pids = collections.Counter(r["patient_id"] for r in sub)
        multi = {p: n for p, n in pids.items() if n > 1}
        summary.append({
            "source_dataset": src,
            "n_records": len(sub),
            "n_unique_patients": len(pids),
            "patient_ids_are_real": "no (synthetic, one per record)" if synth else "yes",
            "n_patients_with_multiple_ecgs": len(multi),
            "max_ecgs_per_patient": max(pids.values()) if pids else 0,
            "mean_ecgs_per_patient": round(len(sub) / len(pids), 4) if pids else 0,
            "patient_level_splitting_possible": "NO -- no patient identifier in source" if synth else "YES",
        })
    tot_rows = len(rows)
    tot_pid = len({r["patient_id"] for r in rows})
    summary.append({
        "source_dataset": "ALL", "n_records": tot_rows, "n_unique_patients": tot_pid,
        "patient_ids_are_real": "mixed",
        "n_patients_with_multiple_ecgs": sum(s["n_patients_with_multiple_ecgs"] for s in summary),
        "max_ecgs_per_patient": max((s["max_ecgs_per_patient"] for s in summary), default=0),
        "mean_ecgs_per_patient": round(tot_rows / tot_pid, 4) if tot_pid else 0,
        "patient_level_splitting_possible": "PARTIAL -- see per-source rows",
    })
    write_csv(rdir / "patient_summary.csv", summary, list(summary[0].keys()))

    # ---- print ---------------------------------------------------------
    print(banner("STEP 6 -- DUPLICATES AND PATIENT OVERLAP"))
    print(f"  records analysed                 : {tot_rows:,}")
    print(f"  unique patient identifiers       : {tot_pid:,}")
    print(f"  ECG recordings                   : {tot_rows:,}")
    print(f"  mean recordings per patient      : {tot_rows/tot_pid:.3f}" if tot_pid else "")
    print()
    for s in summary:
        if s["source_dataset"] == "ALL":
            continue
        print(f"  {s['source_dataset']:10s} {s['n_records']:7,d} ECGs / "
              f"{s['n_unique_patients']:7,d} patients  "
              f"(patients with >1 ECG: {s['n_patients_with_multiple_ecgs']:,}, "
              f"max {s['max_ecgs_per_patient']})")
        print(f"  {'':10s} patient-level splitting: {s['patient_level_splitting_possible']}")
    print()
    ft = collections.Counter(f["finding_type"] for f in findings)
    print("  findings:")
    for k, n in ft.most_common():
        print(f"    {k:36s} {n:6,d}")
    if not ft:
        print("    (none)")
    print(f"\n  exact duplicate signal-file groups : {n_exact:,}")
    if args.near_duplicates:
        print(f"  near-identical signal groups       : {n_near:,}")
    else:
        print("  near-identical signals             : NOT CHECKED (pass --near-duplicates)")

    print("\n  LIMITATION (stated explicitly, as required):")
    print("    Patient identifiers are NOT comparable across the three datasets. They come")
    print("    from different institutions and use unrelated id schemes, and none of them")
    print("    ships a linkage key. Cross-dataset patient overlap can therefore be neither")
    print("    confirmed nor excluded. Any residual overlap is assumed negligible because")
    print("    the cohorts are geographically and temporally distinct -- this is an")
    print("    ASSUMPTION, not a verified fact, and should be stated as such in the thesis.")
    synth_srcs = [s["source_dataset"] for s in summary if s["patient_ids_are_real"].startswith("no")]
    if synth_srcs:
        print(f"    {', '.join(synth_srcs)} ships no patient id at all, so repeated ECGs from one")
        print("    patient WITHIN that source cannot be detected or kept together in a split.")
    info(f"wrote {rdir/'duplicate_patient_report.csv'}")
    info(f"wrote {rdir/'patient_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
