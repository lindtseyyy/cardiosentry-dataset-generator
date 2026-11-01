#!/usr/bin/env python3
"""Export test ECGs a clean12 run classified exactly right, per condition, re-rendered.

Reads ``<run>/<architecture>/test_predictions.csv`` (``record_id`` is the HEEDB
record path, e.g. ``I0006/WFDB/2012/MUSE_..._30000.hea``) and, for each of the
12 clean12 conditions, keeps the single-condition test ECGs whose thresholded
prediction matches the ground truth on all 12 labels: the model flagged that
condition and nothing else. STEMI never occurs alone in clean12 (STE is
propagated from every STEMI), so its folder holds exact STEMI+STE records.

Per condition it picks ``--per-label`` records (distinct patients first,
deterministic for ``--seed``), downloads only those records' ``.hea`` + signal
file from the HEEDB (BDSP) S3 access point (key ``ECG/<record_id>``), and
renders them with the clean12 dataset's own worker and SHA-256-pinned
ecg-image-kit bundle (``export_cardiologist_samples.render``), i.e. the same
1024x768 3x4 + Lead II sheets the model was trained and tested on.

Output layout::

    <out>/README.txt
    <out>/index.csv                       one row per ECG: labels, all 12 probabilities
    <out>/AF_Atrial_fibrillation/01_<site>_<file_id>_clean.png
    <out>/STEMI_STEMI/01_<site>_<file_id>_clean.png   (STEMI+STE records)
    <out>/_work/                          WFDB records + kit (a rerun skips the download)

Usage::

    AWS_PROFILE=lindtsey-cardiosentry python export_correct_predictions.py \\
        --run ../../prototype-results/new_runs/convnext_tiny_v1_rfpn_clean12_colab_a100_seed42
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath

import export_cardiologist_samples as base

LABELS = ["NORM", "SB", "STACH", "LBBB", "LQT", "AF", "AFL", "3AVB", "2AVB", "1AVB", "STEMI", "STE"]
# Label set each condition folder is drawn from; STEMI always carries STE in clean12.
TARGET_SETS = {code: frozenset({code}) for code in LABELS}
TARGET_SETS["STEMI"] = frozenset({"STEMI", "STE"})


def load_predictions(run: Path, architecture: str, policy: str) -> list[dict]:
    """Test rows with true/predicted label sets and per-label probabilities."""
    patients = {}
    with (run / "record_split.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            patients[row["record_path"]] = row["patient_key"]
    rows = []
    with (run / architecture / "test_predictions.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "record_path": row["record_id"], "site": row["source"],
                "file_id": PurePosixPath(row["record_id"]).stem,
                "patient_key": patients[row["record_id"]],
                "true": frozenset(c for c in LABELS if row[f"true_{c}"] == "1"),
                "pred": frozenset(c for c in LABELS if row[f"pred_{policy}_{c}"] == "1"),
                "prob": {c: float(row[f"prob_{c}"]) for c in LABELS},
            })
    return rows


def select(rows: list[dict], per_label: int, seed: int) -> dict[str, list[dict]]:
    """Exact-match records per condition, distinct patients first."""
    picked = {}
    for code, target in TARGET_SETS.items():
        members = sorted((r for r in rows if r["true"] == target and r["pred"] == target),
                         key=lambda r: hashlib.sha256(f"{seed}:{r['record_path']}".encode()).hexdigest())
        seen, first, repeat = set(), [], []
        for row in members:
            (repeat if row["patient_key"] in seen else first).append(row)
            seen.add(row["patient_key"])
        picked[code] = (first + repeat)[:per_label]
        print(f"  {code:<6} {len(members):>5} exact-match records -> {len(picked[code])} picked")
        if len(picked[code]) < per_label:
            print(f"WARNING: only {len(picked[code])} correct {code} records available")
    return picked


README = """Correctly classified clean12 test ECGs, one folder per condition
==================================================================

Run: {run}
Model: {architecture}; threshold policy: {policy} (thresholds below).
Test set: clean12_patient_split_v1, {n_test:,} ECGs.

Each folder holds single-condition test ECGs whose thresholded prediction
matched the ground truth on all 12 labels (the condition flagged, nothing
else). STEMI never occurs without STE in clean12, so STEMI_STEMI holds
STEMI+STE ECGs predicted as exactly STEMI+STE. Picks are a seeded draw
(seed {seed}), distinct patients first, not the most confident cases.

Images are re-rendered from the HEEDB WFDB records with the clean12 dataset
renderer (ecg-image-kit bundle ecg_image_v1_code.tar.gz): 1024x768, 3x4 leads
+ Lead II rhythm strip, 25 mm/s, 10 mm/mV, the sheet the model was scored on.

Folder: available exact-match records -> exported
{counts}

Thresholds
{thresholds}

index.csv: folder, image, record path, patient key, true and predicted labels,
and the model probability for every label.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="run group folder (holds record_split.csv)")
    parser.add_argument("--architecture", default="convnext_tiny_rfpn")
    parser.add_argument("--policy", choices=["validation_tuned", "0_5"], default="validation_tuned",
                        help="which thresholded predictions must match the ground truth")
    parser.add_argument("--out", type=Path, help="default: <run>/correct_single_label_samples")
    parser.add_argument("--per-label", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmented-variants", type=int, default=0,
                        help="also write N paper/photo augmented copies (the model saw clean only)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--heedb-access-point", default=base.HEEDB_ACCESS_POINT)
    parser.add_argument("--notebook", type=Path,
                        default=base.RENDER_CODE_DIR / "Clean12_ECG_Signal_Images_EC2.ipynb")
    parser.add_argument("--kit-bundle", type=Path, default=base.RENDER_CODE_DIR / "ecg_image_v1_code.tar.gz")
    parser.add_argument("--augment-code", type=Path, default=base.AUGMENT_CODE_DIR)
    args = parser.parse_args()
    if args.per_label < 1 or args.workers < 1 or args.augmented_variants < 0:
        parser.error("--per-label and --workers must be >= 1, --augmented-variants >= 0")

    run = args.run.expanduser().resolve()
    out = (args.out or run / "correct_single_label_samples").expanduser().resolve()
    work = out / "_work"
    signals_dir = work / "signals"

    rows = load_predictions(run, args.architecture, args.policy)
    print(f"{len(rows):,} test predictions; exact matches under {args.policy} thresholds:")
    picked = select(rows, args.per_label, args.seed)

    jobs = []
    for code, members in picked.items():
        name = base.LABEL_NAMES[code].replace("/", "-").replace(" ", "_")
        folder = f"{code}_{name}"
        for number, row in enumerate(members, 1):
            jobs.append({"rel": row["record_path"], "folder": folder, "dir": str(out / folder),
                         "stem": f"{number:02d}_{row['site']}_{row['file_id']}", "row": row})
    base.download_records([job["row"] for job in jobs], signals_dir, args.heedb_access_point)

    for code in LABELS:
        for stale in out.glob(f"{code}_*"):
            shutil.rmtree(stale, ignore_errors=True)
    results = base.render(jobs, signals_dir, work, args.notebook.resolve(), args.kit_bundle.resolve(),
                          args.augment_code.resolve(), args.workers, args.seed, args.augmented_variants)
    shutil.rmtree(work / "kit_scratch", ignore_errors=True)

    index_rows, failures = [], []
    for job in jobs:
        result, row = results[job["rel"]], job["row"]
        if "error" in result:
            failures.append(f"{job['rel']}: {result['error']}")
            continue
        index_rows.append({
            "folder": job["folder"], "clean_image": f"{job['folder']}/{result['clean']}",
            "augmented_images": ";".join(f"{job['folder']}/{a['image']}" for a in result["augmented"]),
            "true_labels": "+".join(c for c in LABELS if c in row["true"]),
            "pred_labels": "+".join(c for c in LABELS if c in row["pred"]),
            "site": row["site"], "file_id": row["file_id"], "patient_key": row["patient_key"],
            "record_path": row["record_path"],
            **{f"prob_{c}": f"{row['prob'][c]:.4f}" for c in LABELS},
        })
    if failures:
        raise SystemExit(f"{len(failures)} records failed to render:\n  " + "\n  ".join(failures))
    with (out / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    if args.policy == "validation_tuned":
        thresholds = json.loads((run / args.architecture / "validation_tuned_thresholds.json").read_text())
    else:
        thresholds = {c: 0.5 for c in LABELS}
    available = {code: sum(r["true"] == t and r["pred"] == t for r in rows) for code, t in TARGET_SETS.items()}
    (out / "README.txt").write_text(README.format(
        run=run.name, architecture=args.architecture, policy=args.policy, n_test=len(rows),
        seed=args.seed,
        counts="\n".join(f"  {code:<6} {available[code]:>5} -> {len(picked[code])}" for code in LABELS),
        thresholds="\n".join(f"  {c:<6} {thresholds[c]:.3f}" for c in LABELS)))
    print(f"Done: {len(index_rows)} ECGs in {len({r['folder'] for r in index_rows})} condition folders under {out}")


if __name__ == "__main__":
    main()
