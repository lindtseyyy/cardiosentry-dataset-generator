#!/usr/bin/env python3
"""
STEP 12 -- Run the whole CardioSentry data-preparation pipeline.

    python scripts/run_pipeline.py

Runs, in order:
    1. inspect_datasets.py          -- what is actually in the files
    2. extract_records.py           -- apply config/label_mapping.yaml
    3. validate_signals.py          -- 12 leads / 500 Hz / 10 s / 5000 samples
    4. check_duplicates.py          -- duplicates + patient-level grouping
    5. build_harmonized_dataset.py  -- maximum valid harmonized pool

create_balanced_dataset.py is NOT run: balancing is a deliberate, separate
decision you take after inspecting these results.
"""
from __future__ import annotations

import argparse
import collections
import csv
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import TARGET_CLASSES, banner, info, load_paths, reports_dir, warn

HERE = Path(__file__).resolve().parent

STEPS = [
    ("inspect_datasets.py", [], "STEP 1  inspect datasets"),
    ("extract_records.py", [], "STEP 2-3  label mapping + extraction"),
    ("validate_signals.py", [], "STEP 4  signal compatibility"),
    ("check_duplicates.py", [], "STEP 6  duplicates + patient overlap"),
    ("build_harmonized_dataset.py", [], "STEP 7-9  harmonize + counts + balancing advice"),
]


def _rows(path):
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def final_summary(paths) -> str:
    rdir = reports_dir(paths)
    hdir = Path(paths["output"]["harmonized_dir"])

    counts = _rows(rdir / "final_class_counts.csv")
    cand = {r["class"]: int(r["candidate"]) for r in counts if r["source"] == "TOTAL"}
    valid = {r["class"]: int(r["valid"]) for r in counts if r["source"] == "TOTAL"}

    sig = _rows(rdir / "signal_compatibility_report.csv")
    ok = [r for r in sig if r["validation_status"].startswith("VALID")]

    def _all(key):
        return bool(ok) and all(str(r[key]) == "1" for r in ok)

    psum = _rows(rdir / "patient_summary.csv")
    per_src = [p for p in psum if p["source_dataset"] != "ALL"]
    yes = [p["source_dataset"] for p in per_src
           if p["patient_level_splitting_possible"].startswith("YES")]
    no = [p["source_dataset"] for p in per_src
          if p["patient_level_splitting_possible"].startswith("NO")]

    rec = {r["class"]: r for r in _rows(rdir / "balancing_recommendation.csv")}

    L = []
    L.append("=== CARDIOSENTRY DATASET SUMMARY ===")
    L.append("")
    for c in TARGET_CLASSES:
        L.append(f"{c}:")
        L.append(f"  Candidate: {cand.get(c, 0)}")
        L.append(f"  Valid: {valid.get(c, 0)}")
        L.append("")
    L.append("Signal compatibility:")
    L.append(f"  12-lead: {'YES' if _all('leads_ok') else 'NO'}")
    L.append(f"  500 Hz: {'YES' if _all('fs_ok') else 'NO'}")
    L.append(f"  10 seconds: {'YES' if _all('duration_ok') else 'NO'}")
    L.append(f"  5000 samples: {'YES' if _all('n_samples_ok') else 'NO'}")
    L.append(f"  lead order identical: {'YES' if _all('lead_order_ok') else 'NO'}")
    L.append("")
    L.append("Patient-level splitting possible:")
    if yes and no:
        L.append(f"  PARTIAL - YES for {', '.join(yes)}; NO for {', '.join(no)}")
        L.append(f"           ({', '.join(no)} ships no patient identifier; each of its")
        L.append("            records is treated as its own split group.)")
    elif yes:
        L.append("  YES")
    else:
        L.append("  NO")
    L.append("")
    L.append("Recommended final targets:")
    for c in TARGET_CLASSES:
        r = rec.get(c)
        if r:
            L.append(f"  {c}: {r['recommended_target']}")
    L.append("")
    L.append("No balancing has been performed yet.")
    L.append("")
    L.append(f"Harmonized dataset: {hdir/'harmonized_ecg_metadata.csv'}")
    L.append(f"Reports:            {rdir}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", default=None)
    ap.add_argument("--mapping", default=None)
    ap.add_argument("--skip", action="append", default=[],
                    help="skip a step by script name, e.g. --skip validate_signals.py")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--near-duplicates", action="store_true",
                    help="also run near-identical signal detection (slower)")
    ap.add_argument("--copy-signals", action="store_true",
                    help="materialise signal copies under data/harmonized/signals/")
    ap.add_argument("--quiet", action="store_true", help="only print the final summary")
    args = ap.parse_args(argv)

    paths = load_paths(args.paths)
    t_all = time.time()

    for script, extra, title in STEPS:
        if script in args.skip:
            info(f"skipping {script}")
            continue
        cmd = [sys.executable, str(HERE / script), *extra]
        if args.paths:
            cmd += ["--paths", args.paths]
        if args.mapping and script in ("inspect_datasets.py", "extract_records.py",
                                       "build_harmonized_dataset.py"):
            cmd += ["--mapping", args.mapping]
        if args.workers and script in ("validate_signals.py", "check_duplicates.py"):
            cmd += ["--workers", str(args.workers)]
        if args.near_duplicates and script == "check_duplicates.py":
            cmd += ["--near-duplicates"]
        if args.copy_signals and script == "build_harmonized_dataset.py":
            cmd += ["--copy-signals"]

        print(banner(title, "#"))
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=args.quiet, text=True)
        if r.returncode != 0:
            warn(f"{script} failed with exit code {r.returncode}")
            if args.quiet and r.stdout:
                print(r.stdout[-4000:])
            if args.quiet and r.stderr:
                print(r.stderr[-4000:])
            return r.returncode
        info(f"{script} finished in {time.time()-t0:.1f}s")

    summary = final_summary(paths)
    print("\n" + summary + "\n")
    out = reports_dir(paths) / "cardiosentry_summary.txt"
    out.write_text(summary + "\n", encoding="utf-8")
    info(f"wrote {out}")
    info(f"pipeline finished in {time.time()-t_all:.1f}s")
    print("\nNext step: review reports/, then optionally run")
    print("  python scripts/create_balanced_dataset.py --use-recommended --copy-signals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
