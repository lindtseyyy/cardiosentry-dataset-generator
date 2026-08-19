#!/usr/bin/env python3
"""
STEP 14 -- Assign train/val/test ONCE and freeze it into the release metadata.

The split is a property of the DATASET, not of any downstream build. Once
written it must never be recomputed, or a model evaluated today is not
comparable with one evaluated tomorrow. This script therefore:

  * groups on `split_group_id` -- the patient wherever a patient id exists, so
    every ECG from one patient lands in one fold and cannot leak;
  * honours PTB-XL's own stratified folds (9 -> val, 10 -> test), which are
    already patient-consistent upstream;
  * assigns the remainder 80/10/10 over whole patient groups, stratified by
    CardioSentry class;
  * writes a `split` column into the harmonized AND balanced CSVs of a release,
    keeping the two consistent by construction (balanced is a subset, so it
    inherits the harmonized assignment record for record);
  * REFUSES to overwrite an existing split column unless --reassign is given.

Downstream stages read the frozen column; they do not re-derive it.

    python scripts/freeze_splits.py --release v2
    python scripts/freeze_splits.py --release v2 --verify      # check only
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import TARGET_CLASSES, banner, info, load_paths, reports_dir, warn, write_csv

SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}
SEED = 20260818
USE_PTBXL_FOLDS = True
PTBXL_VAL_FOLDS = {9}
PTBXL_TEST_FOLDS = {10}


def _ptbxl_fold(row: dict) -> int | None:
    if row.get("source_dataset", "").upper() != "PTBXL":
        return None
    try:
        fold = json.loads(row.get("source_extra_json") or "{}").get("strat_fold")
        return int(float(fold)) if fold not in (None, "") else None
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _class_of(row: dict) -> str:
    pos = [c for c in TARGET_CLASSES if row.get(c) == "1"]
    return "+".join(pos) if pos else "NONE"


def assign(rows: list[dict], rng: np.random.Generator) -> dict:
    """Return {split_group_id: split}. Whole groups only, stratified by class."""
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[r["split_group_id"]].append(r)

    # A group is stratified by the class it mostly carries. Splitting a group
    # finer would break the no-leak guarantee, which matters more than perfect
    # class proportions.
    strat: dict[str, list[str]] = collections.defaultdict(list)
    forced: dict[str, str] = {}
    for key, members in groups.items():
        cls = collections.Counter(_class_of(m) for m in members).most_common(1)[0][0]
        strat[cls].append(key)
        if USE_PTBXL_FOLDS:
            folds = {f for f in (_ptbxl_fold(m) for m in members) if f}
            if folds & PTBXL_TEST_FOLDS:
                forced[key] = "test"
            elif folds & PTBXL_VAL_FOLDS:
                forced[key] = "val"

    out: dict[str, str] = {}
    for cls, keys in strat.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        free = [k for k in keys if k not in forced]
        # PTB-XL's forced folds already consume part of the val/test budget for
        # this class; only top up the remainder from the free groups.
        have_test = sum(1 for k in keys if forced.get(k) == "test")
        have_val = sum(1 for k in keys if forced.get(k) == "val")
        need_test = max(0, round(len(keys) * SPLITS["test"]) - have_test)
        need_val = max(0, round(len(keys) * SPLITS["val"]) - have_val)
        for i, k in enumerate(free):
            if i < need_test:
                out[k] = "test"
            elif i < need_test + need_val:
                out[k] = "val"
            else:
                out[k] = "train"
        out.update({k: v for k, v in forced.items() if k in keys})
    return out


def _read(p: Path) -> tuple[list[dict], list[str]]:
    with open(p, newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        return list(rdr), list(rdr.fieldnames or [])


def _report(rows: list[dict], name: str) -> list[str]:
    L = [f"  {name}: {len(rows):,} records"]
    sc = collections.Counter(r["split"] for r in rows)
    tot = len(rows) or 1
    L.append(f"    {'split':6s} {'records':>9s} {'%':>7s}   " +
             "  ".join(f"{c:>7s}" for c in TARGET_CLASSES))
    for s in ("train", "val", "test"):
        per = [sum(1 for r in rows if r["split"] == s and r.get(c) == "1")
               for c in TARGET_CLASSES]
        L.append(f"    {s:6s} {sc[s]:9,d} {100*sc[s]/tot:6.2f}%   " +
                 "  ".join(f"{v:7,d}" for v in per))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", default=None)
    ap.add_argument("--release", default="v2", help="release under data/releases/ (default v2)")
    ap.add_argument("--reassign", action="store_true",
                    help="overwrite an existing frozen split (breaks comparability)")
    ap.add_argument("--verify", action="store_true", help="check only, write nothing")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    paths = load_paths(args.paths)
    root = Path(paths["output"]["harmonized_dir"]).parent / "releases" / args.release
    hpath = root / "harmonized" / "harmonized_ecg_metadata.csv"
    bpath = root / "balanced" / "balanced_ecg_metadata.csv"
    for p in (hpath, bpath):
        if not p.exists():
            warn(f"{p} not found")
            return 1

    H, hfields = _read(hpath)
    B, bfields = _read(bpath)

    frozen = "split" in hfields and all(r.get("split") for r in H)
    if frozen and not args.reassign:
        info("split column already frozen -- reusing it (pass --reassign to overwrite)")
        mapping = {r["split_group_id"]: r["split"] for r in H}
    elif args.verify:
        warn("no frozen split present; nothing to verify")
        return 1
    else:
        if frozen:
            warn("OVERWRITING an existing frozen split: results from the old split "
                 "are no longer comparable with results from the new one.")
        mapping = assign(H, np.random.default_rng(args.seed))
        info(f"assigned 80/10/10 over {len(mapping):,} patient groups (seed {args.seed})")

    for r in H:
        r["split"] = mapping[r["split_group_id"]]
    missing = [r for r in B if r["split_group_id"] not in mapping]
    for r in B:
        r["split"] = mapping.get(r["split_group_id"], "")

    # ---------------- integrity checks --------------------------------
    print(banner("STEP 14 -- FROZEN TRAIN/VAL/TEST SPLIT"))
    ok = True

    seen: dict[str, str] = {}
    leaks = []
    for r in H + B:
        prev = seen.setdefault(r["split_group_id"], r["split"])
        if prev != r["split"]:
            leaks.append((r["split_group_id"], prev, r["split"]))
    print(f"  patient-group leakage (harmonized+balanced): "
          f"{'FAIL - ' + str(len(leaks)) if leaks else 'NONE'}")
    ok &= not leaks

    pid_split: dict[str, set] = collections.defaultdict(set)
    for r in H:
        if r.get("patient_id_is_synthetic") != "1":
            pid_split[r["patient_id"]].add(r["split"])
    bad = {p: s for p, s in pid_split.items() if len(s) > 1}
    print(f"  real patient ids spanning >1 split          : "
          f"{'FAIL - ' + str(len(bad)) if bad else 'NONE'}  "
          f"({len(pid_split):,} real patients checked)")
    ok &= not bad

    print(f"  balanced records missing an assignment      : "
          f"{'FAIL - ' + str(len(missing)) if missing else 'NONE'}")
    ok &= not missing

    multi = sum(1 for r in H if r["patient_id_is_synthetic"] != "1")
    print(f"  grouping key                                : split_group_id "
          f"(= SOURCE::patient_id)")
    print(f"    records grouped by a REAL patient id      : {multi:,}")
    syn = collections.Counter(r["source_dataset"] for r in H
                              if r.get("patient_id_is_synthetic") == "1")
    for s, n in syn.items():
        print(f"    records with a SYNTHETIC per-record group : {n:,}  ({s})")

    print()
    for line in _report(H, "harmonized"):
        print(line)
    print()
    for line in _report(B, "balanced"):
        print(line)

    if syn:
        print()
        print("  LIMITATION (must be reported):")
        for s, n in syn.items():
            print(f"    {s} ships no patient identifier. Its {n:,} records are grouped")
            print(f"    one-per-record -- the safest available grouping, but if the same")
            print(f"    patient was recorded twice in {s}, those two ECGs CAN land in")
            print(f"    different folds. This residual leakage cannot be detected or")
            print(f"    prevented from the data as distributed.")

    if args.verify:
        print("\n  verify-only: nothing written")
        return 0 if ok else 1
    if not ok:
        warn("integrity checks failed -- refusing to write")
        return 1

    for path, rows, fields in ((hpath, H, hfields), (bpath, B, bfields)):
        f = fields + ["split"] if "split" not in fields else fields
        write_csv(path, rows, f)
        info(f"froze split column into {path}")

    rdir = reports_dir(paths)
    srows = []
    for name, rows in (("harmonized", H), ("balanced", B)):
        for s in ("train", "val", "test"):
            sub = [r for r in rows if r["split"] == s]
            srows.append({
                "cohort": name, "split": s, "records": len(sub),
                "pct": f"{100*len(sub)/max(len(rows),1):.2f}",
                "patient_groups": len({r["split_group_id"] for r in sub}),
                **{c: sum(1 for r in sub if r.get(c) == "1") for c in TARGET_CLASSES},
            })
    write_csv(rdir / "split_assignment_report.csv", srows, list(srows[0].keys()))
    info(f"wrote {rdir/'split_assignment_report.csv'}")
    print("\n  The split is now FROZEN. Downstream stages read this column; they")
    print("  must not recompute it. Re-running this script is a no-op.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
