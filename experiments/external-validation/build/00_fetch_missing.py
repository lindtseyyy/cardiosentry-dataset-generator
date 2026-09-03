#!/usr/bin/env python3
"""
Fetch the pieces of the source data this build needs and the local copies lack.

Two things, both small and both idempotent:

  1. `data/ptbxl_selected.csv` from dgedon/nstemi_prediction - the 275 hand-
     inspected PTB-XL ids (200 normal / 75 mi) that define the proxy cohort.

  2. The individual PTB-XL records500 files that the local dataset copy is
     missing. The local tree holds ~19,490 of PTB-XL's 21,799 records, and 35
     of the gaps happen to fall inside the curated selection - 14 of them on
     the MI arm. Losing 14 of 59 proxy positives to an incomplete download
     would be a silly reason to weaken the only cohort large enough to carry a
     confidence interval, so they are fetched one file at a time rather than
     re-downloading the 3 GB archive.

Downloads only. Nothing here is uploaded anywhere, and existing files are never
overwritten.

    python 00_fetch_missing.py            # fetch what is missing
    python 00_fetch_missing.py --check    # report, download nothing
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE.parent / "data"
PTBXL = REPO / "dataset-files" / "ptb-xl-dataset"

SELECTION_URL = ("https://raw.githubusercontent.com/dgedon/nstemi_prediction/"
                 "main/data/ptbxl_selected.csv")
PHYSIONET = "https://physionet.org/files/ptb-xl/1.0.3/"

UA = {"User-Agent": "cardiosentry-external-validation/1.0"}


def fetch(url: str, dest: Path, timeout: int = 60) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f"    FAILED {url}\n           {exc}")
        return False
    # Write via a temp file so an interrupted download cannot leave a truncated
    # .dat behind that later looks present and reads as a corrupt signal.
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(body)
    tmp.rename(dest)
    return True


def ensure_selection(check: bool) -> Path:
    dest = DATA / "ptbxl_selected.csv"
    if dest.exists():
        print(f"[1/2] selection CSV present: {dest}")
        return dest
    if check:
        print(f"[1/2] selection CSV MISSING: {dest}")
        return dest
    print(f"[1/2] fetching {SELECTION_URL}")
    if not fetch(SELECTION_URL, dest):
        sys.exit("could not fetch the curated selection; download it by hand from\n"
                 f"  {SELECTION_URL}\nand save it to {dest}")
    print(f"      -> {dest}")
    return dest


def ensure_records(selection: Path, check: bool) -> None:
    if not selection.exists():
        print("[2/2] skipped - no selection CSV yet")
        return
    with open(selection, newline="", encoding="utf-8") as fh:
        stems = [r["filename_hr"] for r in csv.DictReader(fh)]

    missing: list[tuple[str, Path]] = []
    for stem in stems:
        for ext in (".hea", ".dat"):
            rel = stem + ext
            local = PTBXL / rel
            if not local.exists():
                missing.append((rel, local))

    if not missing:
        print(f"[2/2] all {len(stems)} selected records present locally")
        return

    recs = sorted({re.sub(r"\.(hea|dat)$", "", rel) for rel, _ in missing})
    print(f"[2/2] {len(missing)} file(s) missing across {len(recs)} record(s)")
    if check:
        for rel in recs[:10]:
            print(f"        {rel}")
        if len(recs) > 10:
            print(f"        ... and {len(recs) - 10} more")
        return

    ok = 0
    for i, (rel, local) in enumerate(missing, start=1):
        print(f"    [{i}/{len(missing)}] {rel}")
        if fetch(PHYSIONET + rel, local):
            ok += 1
    print(f"      fetched {ok}/{len(missing)}")
    if ok < len(missing):
        print("      the remainder will simply be dropped by stage 1; the build "
              "still works, with a smaller proxy cohort.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only")
    args = ap.parse_args()
    sel = ensure_selection(args.check)
    ensure_records(sel, args.check)
    print("\nnext: python 01_build_metadata.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
