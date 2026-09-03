#!/usr/bin/env python3
"""
Inventory the CUHK corpus: how many ECGs carry a usable label, from how many
patients, and how many of those can be decoded without an XLI decoder.

    python inventory.py                       # full corpus, ~4 min on 4 cores
    python inventory.py --limit 500           # quick look
    python inventory.py --csv out.csv         # per-record rows for further work

The patient column is the one that matters and the one that is easy to miss.
The corpus holds 16,755 ECGs from about 1,600 subjects, recorded 2005-2019 -
it is a longitudinal registry, not 16,755 independent cases. Counting ECGs
alone overstates what is here by roughly tenfold, and any split built on it has
to group by patient or it leaks a subject across the boundary.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import philips_xml as P          # noqa: E402

DEFAULT_ROOT = (Path(__file__).resolve().parents[2]
                / "dataset-files" / "cuhk_dataset")
SUBDIRS = ["STEMI XML AB", "STEMI XML CE", "STEMI XML GZ"]


def probe(path: str) -> dict | None:
    """One record, reduced to the flags the inventory reports on."""
    try:
        r = P.read(path, with_beats=True)
    except Exception:                            # noqa: BLE001
        return None
    c = r.codes
    beats = r.median_beats
    return {
        "path": path,
        "batch": Path(path).parent.name,
        "patient": r.patient,
        "date": r.date,
        "severity": r.severity,
        "acute": int(bool(c & P.ACUTE_INFARCT)),
        "ste": int(bool(c & P.ST_ELEVATION)),
        "mimic": int(bool(c & P.BENIGN_STE_MIMIC)),
        "af": int(bool(c & P.AF)),
        "aflut": int(bool(c & P.ATRIAL_FLUTTER)),
        "lvh": int(bool(c & P.LVH)),
        # "Decodable today" means the representative beat is readable without
        # solving XLI. It does NOT mean renderable - see FINDINGS.md.
        "beats": int(beats is not None),
        "beat_samples": int(beats.shape[1]) if beats is not None else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--csv", default=None, help="write per-record rows here")
    ap.add_argument("-j", "--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    root = Path(args.root)
    files: list[str] = []
    for d in SUBDIRS:
        files += sorted(glob.glob(str(root / d / "*.xml")))
    if not files:
        sys.exit(f"no XML under {root}")
    if args.limit:
        files = files[:args.limit]

    print(f"scanning {len(files)} files on {args.workers} workers ...", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(probe, files, chunksize=64)):
            if r:
                rows.append(r)
            if (i + 1) % 2500 == 0:
                print(f"  {i+1}/{len(files)}", flush=True)
    print(f"readable: {len(rows)}/{len(files)}\n")

    def report(name, pred):
        sub = [r for r in rows if pred(r)]
        dec = [r for r in sub if r["beats"]]
        print(f"{name:<42}{len(sub):>7}{len(set(r['patient'] for r in sub)):>8}"
              f"{len(dec):>10}{len(set(r['patient'] for r in dec)):>8}")

    print(f"{'':<42}{'ECGs':>7}{'pts':>8}{'beats-ok':>10}{'pts':>8}")
    print("-" * 75)
    report("acute infarct asserted", lambda r: r["acute"])
    report("ST elevation / injury (not acute)", lambda r: r["ste"] and not r["acute"])
    report("EITHER acute or ST elevation", lambda r: r["acute"] or r["ste"])
    report("benign ST-elevation mimic", lambda r: r["mimic"])
    report("atrial fibrillation", lambda r: r["af"])
    report("atrial flutter (NOT AF)", lambda r: r["aflut"])
    report("LVH", lambda r: r["lvh"])
    report("severity NORMAL", lambda r: r["severity"] == "NO")
    report("severity DEFECTIVE (drop these)", lambda r: r["severity"] == "DE")
    print("-" * 75)
    report("TOTAL", lambda r: True)

    pos = [r for r in rows if r["acute"] or r["ste"]]
    per = collections.Counter(r["patient"] for r in pos)
    print(f"\nECGs per patient among the {len(pos)} ST-elevation/acute records:")
    print("   ", dict(sorted(collections.Counter(per.values()).items())[:10]))
    print(f"    -> one ECG per patient would yield {len(per)} independent records")

    lens = collections.Counter(r["beat_samples"] for r in rows if r["beats"])
    print(f"\nrepresentative-beat length (samples @500Hz): {dict(lens.most_common(5))}")
    print("    600 samples = 1.2 s. The image pipeline needs 5000 (10 s).")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
