#!/usr/bin/env python3
"""
Tar the finished corpus for upload to Drive, and record what is in it.

    python 04_package.py

Produces, beside the corpus:

    external_testset.tar          images + index.csv
    external_testset.tar.sha256   checksum
    external_testset_PROVENANCE.json

The provenance file is the point of this script, not the tar. The corpus is
about to leave this machine and be scored somewhere else, and the numbers it
produces will be quoted in a writeup. Six months from now the only way to know
what "8 gold STEMI positives" referred to is a record written at build time -
so the cohort counts, the label rules in force, the exclusions and the render
settings are all captured here and travel inside the tar.

Uncompressed on purpose: the payload is JPEG, so gzip buys a percent or two and
costs minutes at both ends.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
LABELS = ["STEMI", "AF", "LVH", "NORMAL"]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", default=str(DATA / "external_testset"))
    ap.add_argument("-o", "--out", default=str(DATA / "external_testset.tar"))
    args = ap.parse_args()

    testset, tar_path = Path(args.testset), Path(args.out)
    index = testset / "index.csv"
    if not index.exists():
        sys.exit(f"missing {index} - run 03_merge.py")

    with open(index, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    def n_pos(label, pred=lambda r: True):
        return sum(1 for r in rows if r[label] == "1" and pred(r))

    cohorts = {}
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        cohorts[src] = {"sheets": len(sub),
                        **{l: sum(1 for r in sub if r[l] == "1") for l in LABELS}}

    excluded = []
    exc_path = DATA / "excluded_records.csv"
    if exc_path.exists():
        with open(exc_path, newline="", encoding="utf-8") as fh:
            excluded = list(csv.DictReader(fh))

    prov = {
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "purpose": (
            "Break the source/label confound on the STEMI head. In the training "
            "corpus every STEMI positive came from the Chongqing ACS set and no "
            "other source contributed one, so 'detects ST elevation' and 'detects "
            "the Chongqing scanner' were indistinguishable. This corpus supplies "
            "STEMI positives from sources that never contributed a STEMI label."),
        "sheets": len(rows),
        "cohorts": cohorts,
        "totals": {l: n_pos(l) for l in LABELS},
        "hard_negatives": sum(1 for r in rows if r["n_positive_labels"] == "0"),
        "gold_stemi": n_pos("STEMI", lambda r: r["source"] == "LUDB"),
        "proxy_stemi": n_pos("STEMI", lambda r: r["source"] == "PTBXL_MI"),
        "patients_seen_in_training": sum(1 for r in rows
                                         if str(r["seen_in_training"]) == "1"),
        "label_rules": {
            "STEMI__LUDB": "header report contains an asserted 'STEMI: <wall>'; "
                           "'supp.NSTEMI' is explicitly NOT matched",
            "STEMI__PTBXL_MI": "PROXY: Gedon et al. manual 'mi' label AND PTB-XL "
                               "infarction_stadium1 == 'Stadium I' (acute). Not an "
                               "asserted ST-elevation label - do not merge with the "
                               "LUDB figure",
            "AF": "atrial fibrillation only; atrial flutter excluded, per "
                  "label_mapping.yaml:AFLT",
            "LVH": "left ventricular hypertrophy only; left atrial hypertrophy, "
                   "left ventricular overload and RVH all excluded",
            "NORMAL__LUDB": "assertion by absence - a full cardiologist report "
                            "listing sinus rhythm, an axis and no findings. Weaker "
                            "than PTB-XL's explicit NORM statement",
            "NORMAL__PTBXL": "explicit SCP 'NORM' statement",
        },
        "exclusions": {
            "rule": "PTB-XL records whose PATIENT appears in the balanced training "
                    "corpus are dropped; the corpus split on patient, so record-level "
                    "exclusion would be insufficient",
            "count": len(excluded),
            "by_reason": dict(Counter(e["reason"] for e in excluded)),
        },
        "render": {
            "pipeline": "dataset-combination-code/ecg_dataset_pipeline/image_pipeline, "
                        "stages 1-4, unmodified",
            "note": "the SAME renderer, DPI, layout, grid palettes and augmentation "
                    "as the training corpus. A separate implementation would make a "
                    "score drop unattributable",
            "variants_per_record": 1,
            "sheet_numbering": "shuffled before staging, so the ID printed on the "
                               "sheet does not encode source or label",
        },
        "known_limits": [
            "8 gold STEMI positives - enough to detect a collapse, not enough to "
            "certify a head works. Quote intervals, never point estimates",
            "the PTB-XL arm is a proxy label AND a training source; its value is "
            "one-directional (PTB-XL never contributed a STEMI label, so firing "
            "there cannot be explained by paper)",
            "both sides are synthetic renders, so this measures generalisation "
            "across ECG databases, not across real printouts or phone photographs",
            "LVH prevalence is ~54% in LUDB against a much lower training rate; "
            "AUPRC/F1/precision are not comparable across cohorts, AUROC is",
        ],
    }

    (testset / "PROVENANCE.json").write_text(json.dumps(prov, indent=2))

    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        tar.add(testset, arcname=testset.name)

    digest = sha256(tar_path)
    tar_path.with_suffix(".tar.sha256").write_text(f"{digest}  {tar_path.name}\n")

    mb = tar_path.stat().st_size / 1e6
    print(f"{tar_path}  ({mb:.1f} MB)")
    print(f"sha256 {digest}")
    print()
    print(f"  sheets           {prov['sheets']}")
    print(f"  gold STEMI       {prov['gold_stemi']} (LUDB, asserted)")
    print(f"  proxy STEMI      {prov['proxy_stemi']} (PTB-XL acute MI)")
    print(f"  hard negatives   {prov['hard_negatives']}")
    print(f"  seen in training {prov['patients_seen_in_training']}")
    print()
    print("upload to Drive, e.g.")
    print("  ~/MyDrive/cardiosentry_dataset/external_testset.tar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
