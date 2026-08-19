#!/usr/bin/env python3
"""
STEP 10 -- Create the final BALANCED dataset (separate, optional, reproducible).

This script is deliberately NOT part of run_pipeline.py's default flow. Run it
only after you have inspected the harmonized counts and are happy with them.

Guarantees
  * The original datasets are never touched.
  * data/harmonized/harmonized_ecg_metadata.csv is read-only input.
  * Selection is seeded (default 42) -- rerunning reproduces the same cohort.
  * Whole PATIENT GROUPS are selected together, so no patient can be split
    across the balanced cohort boundary.
  * Multi-label records (e.g. AF+LVH) are counted toward every class they
    carry, so the per-class totals are honest.

Configure targets on the command line or by editing TARGETS below.

Outputs
  data/balanced/balanced_ecg_metadata.csv
  data/balanced/signals/                (with --copy-signals)
  reports/balancing_applied_report.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import TARGET_CLASSES, banner, info, load_paths, reports_dir, warn, write_csv

# --------------------------------------------------------------------------
# CONFIGURATION -- edit here, or override with --target CLASS=N
# --------------------------------------------------------------------------
RANDOM_SEED = 42

TARGETS = {
    "STEMI": 1442,
    "AF": 2800,
    "LVH": 2700,
    "NORMAL": 2800,
}
# Use 0 or None for "take everything available in this class".
# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", default=None)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED,
                    help="random seed (default %(default)s)")
    ap.add_argument("--target", action="append", default=[], metavar="CLASS=N",
                    help="override a class target, e.g. --target NORMAL=2544 "
                         "(repeatable)")
    ap.add_argument("--use-recommended", action="store_true",
                    help="take targets from reports/balancing_recommendation.csv "
                         "instead of the TARGETS dict")
    ap.add_argument("--copy-signals", action="store_true",
                    help="copy the selected .hea/.dat/.mat pairs into data/balanced/signals/")
    ap.add_argument("--allow-multilabel-overflow", action="store_true", default=True,
                    help="permit a class to exceed its target when the excess comes "
                         "purely from multi-label records selected for another class "
                         "(default: on)")
    args = ap.parse_args(argv)

    paths = load_paths(args.paths)
    rdir = reports_dir(paths)
    bdir = Path(paths["output"]["balanced_dir"])
    bdir.mkdir(parents=True, exist_ok=True)

    hpath = Path(paths["output"]["harmonized_dir"]) / "harmonized_ecg_metadata.csv"
    if not hpath.exists():
        warn(f"{hpath} not found -- run build_harmonized_dataset.py first")
        return 1
    with open(hpath, newline="", encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))
        fieldnames = list(records[0].keys()) if records else []
    info(f"read {len(records):,} harmonized records from {hpath} (read-only)")

    # ---- resolve targets ----------------------------------------------
    targets = dict(TARGETS)
    if args.use_recommended:
        rp = rdir / "balancing_recommendation.csv"
        if not rp.exists():
            warn(f"{rp} not found")
            return 1
        with open(rp, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["class"] in TARGET_CLASSES:
                    targets[r["class"]] = int(r["recommended_target"])
        info("targets taken from balancing_recommendation.csv")
    for spec in args.target:
        if "=" not in spec:
            warn(f"ignoring malformed --target {spec!r} (expected CLASS=N)")
            continue
        k, v = spec.split("=", 1)
        k = k.strip().upper()
        if k not in TARGET_CLASSES:
            warn(f"ignoring unknown class {k!r}")
            continue
        targets[k] = int(v)

    available = {c: sum(1 for r in records if r.get(c) == "1") for c in TARGET_CLASSES}
    for c in TARGET_CLASSES:
        if not targets.get(c):
            targets[c] = available[c]
        if targets[c] > available[c]:
            warn(f"{c}: target {targets[c]:,} exceeds {available[c]:,} available "
                 f"-- capping at {available[c]:,}")
            targets[c] = available[c]

    # ---- group by patient ---------------------------------------------
    groups = collections.defaultdict(list)
    for r in records:
        groups[r["split_group_id"]].append(r)

    def group_classes(g):
        s = set()
        for r in g:
            s.update(c for c in TARGET_CLASSES if r.get(c) == "1")
        return s

    rng = random.Random(args.seed)
    selected_groups, selected = set(), []
    got = collections.Counter()

    # Fill the scarcest class first: its records are the hardest to replace,
    # and multi-label spill-over then counts toward the commoner classes.
    order = sorted(TARGET_CLASSES, key=lambda c: available[c])
    info(f"filling classes in scarcity order: {' -> '.join(order)}")

    for cls in order:
        pool = [gid for gid, g in groups.items()
                if gid not in selected_groups and cls in group_classes(g)]
        rng.shuffle(pool)
        # prefer single-label groups so multi-label records stay available
        pool.sort(key=lambda gid: len(group_classes(groups[gid])))
        for gid in pool:
            if got[cls] >= targets[cls]:
                break
            g = groups[gid]
            add = collections.Counter()
            for r in g:
                for c in TARGET_CLASSES:
                    if r.get(c) == "1":
                        add[c] += 1
            if not args.allow_multilabel_overflow:
                if any(got[c] + add[c] > targets[c] for c in add):
                    continue
            selected_groups.add(gid)
            selected.extend(g)
            got.update(add)

    selected.sort(key=lambda r: r["global_record_id"])

    # ---- write ---------------------------------------------------------
    out_csv = bdir / "balanced_ecg_metadata.csv"
    write_csv(out_csv, selected, fieldnames)

    rep = []
    for c in TARGET_CLASSES:
        rep.append({
            "class": c,
            "available_in_harmonized": available[c],
            "target": targets[c],
            "selected": got[c],
            "discarded": available[c] - got[c],
            "pct_retained": f"{100*got[c]/available[c]:.2f}" if available[c] else "",
            "ratio_to_stemi": f"{got[c]/got['STEMI']:.3f}" if got["STEMI"] else "",
            "target_met": "yes" if got[c] >= targets[c] else "NO",
            "note": "exceeds target via multi-label spill-over" if got[c] > targets[c] else "",
        })
    rep.append({
        "class": "TOTAL_RECORDS", "available_in_harmonized": len(records),
        "target": sum(targets.values()), "selected": len(selected),
        "discarded": len(records) - len(selected),
        "pct_retained": f"{100*len(selected)/len(records):.2f}" if records else "",
        "ratio_to_stemi": "", "target_met": "",
        "note": f"seed={args.seed}; patient groups kept intact; "
                f"{len(selected_groups):,} groups selected",
    })
    write_csv(rdir / "balancing_applied_report.csv", rep, list(rep[0].keys()))

    # ---- optional physical copy ---------------------------------------
    # Renaming a .hea on disk is not enough: a WFDB header names its own record
    # on line 1 and names its signal file on every signal line, so a renamed
    # copy whose contents still say `JS00001.mat` is unreadable by wfdb. The
    # rewrite lives in export_dataset.py, which is the fuller tool for this.
    if args.copy_signals:
        from export_dataset import export_record
        sdir = bdir / "signals"
        sdir.mkdir(parents=True, exist_ok=True)
        info(f"copying {len(selected):,} signal pairs into {sdir} ...")
        n, n_bad = 0, 0
        for i, r in enumerate(selected, 1):
            try:
                export_record(r, sdir)
                n += 2
            except Exception as exc:              # noqa: BLE001 - report, continue
                n_bad += 1
                if n_bad <= 10:
                    warn(f"  {r['global_record_id']}: {type(exc).__name__}: {exc}")
            if i % 1000 == 0:
                info(f"  {i:,}/{len(selected):,}")
        info(f"copied {n:,} files (originals untouched)"
             + (f"; {n_bad} failed" if n_bad else ""))

    # ---- print ---------------------------------------------------------
    print(banner("STEP 10 -- BALANCED DATASET CREATED"))
    print(f"  random seed        : {args.seed}")
    print(f"  source (read-only) : {hpath}")
    print(f"  output             : {out_csv}")
    print(f"  records selected   : {len(selected):,} of {len(records):,}")
    print(f"  patient groups     : {len(selected_groups):,}")
    print(f"  signal files       : {'copied into data/balanced/signals/' if args.copy_signals else 'referenced in place (pass --copy-signals to materialise)'}")
    print()
    print(f"  {'Class':8s} {'Available':>10s} {'Target':>8s} {'Selected':>9s} {'Discarded':>10s} "
          f"{'%kept':>7s} {'ratio':>7s}")
    print(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*9} {'-'*10} {'-'*7} {'-'*7}")
    for r in rep[:-1]:
        print(f"  {r['class']:8s} {r['available_in_harmonized']:10,d} {r['target']:8,d} "
              f"{r['selected']:9,d} {r['discarded']:10,d} {r['pct_retained']:>7s} "
              f"{r['ratio_to_stemi']:>7s}")
    print(f"\n  STEMI remains the smallest class: "
          f"{'YES' if all(got['STEMI'] <= got[c] for c in TARGET_CLASSES) else 'NO'}")
    by_src = collections.Counter(r["source_dataset"] for r in selected)
    print(f"  source composition: " + ", ".join(f"{k}={v:,}" for k, v in by_src.most_common()))
    info(f"wrote {rdir/'balancing_applied_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
