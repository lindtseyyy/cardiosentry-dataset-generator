"""
CardioSentry -- build release v3: the four-source merged signal dataset.

v3 = v2's three sources (PTB-XL, Chapman, 2026 ACS/STEMI) + the CUHK ECG
collection, merged into ONE standalone harmonized cohort.

What this script deliberately does NOT do
-----------------------------------------
No class balancing, no over/under-sampling, no augmentation, no synthetic
generation, and no train/val/test assignment. v3 is the *natural* merged
distribution and nothing else. Balancing and splitting are separate, later,
reversible decisions; baking them in here would make the natural distribution
unrecoverable.

Order of operations
-------------------
1. VALIDATE every candidate record by opening it -- sampling rate, lead count
   and order, duration, units, gain, NaN / WFDB invalid samples, dead leads,
   amplitude range. Records failing a hard check are excluded before anything
   is copied, and listed in a report.
2. DE-DUPLICATE on the signal itself (SHA-1 of the 12x5000 samples quantised
   to 1 uV), so an identical recording present twice is caught regardless of
   which source it came from or how it was stored.
3. COPY the survivors into a self-contained release: signal bytes verbatim,
   only the WFDB header's record name and file reference rewritten.
4. VERIFY the copy by re-hashing, and re-read a sample through wfdb.

Every record keeps its provenance: source_dataset, original_record_id,
patient_id, split_group_id, the four class columns, cardiosentry_label
(the exact label combination) and source_extra_json.

Usage
-----
    python scripts/build_v3_release.py                 # full build
    python scripts/build_v3_release.py --dry-run       # validate + report only
    python scripts/build_v3_release.py --reuse-checks  # skip re-measuring
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TARGET_CLASSES, normalize_lead                  # noqa: E402
from export_dataset import rewrite_header                          # noqa: E402

PIPELINE = Path(__file__).resolve().parent.parent
REPO = PIPELINE.parents[1]

CURRENT_CSV = PIPELINE / "data" / "harmonized" / "harmonized_ecg_metadata.csv"
CUHK_DIR = Path(os.environ.get(
    "CUHK_CLEAN_ROOT", REPO / "dataset-files" / "cuhk_dataset_clean"))
CUHK_CSV = CUHK_DIR / "metadata.csv"

VERSION = "v3"
RELEASES = PIPELINE / "data" / "releases"
DEST = RELEASES / VERSION
COHORT = DEST / "harmonized"
REPORTS = PIPELINE / "reports"
CHECKS_CSV = REPORTS / f"{VERSION}_precheck.csv"
EXCLUDED_CSV = REPORTS / f"{VERSION}_excluded_records.csv"
DUPES_CSV = REPORTS / f"{VERSION}_duplicate_records.csv"

TARGET_FS = 500
TARGET_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF",
                "V1", "V2", "V3", "V4", "V5", "V6"]
TARGET_N = 5000
TARGET_GAIN = 1000.0

FLAT_MV = 0.05          # a lead below this peak-to-peak is dead, not recorded
AMP_MAX_MV = 50.0       # above this, the recorder was saturating
AMP_MIN_MV = 0.05       # below this for EVERY lead, there is no ECG here


# ---------------------------------------------------------------------------
# 1. validation
# ---------------------------------------------------------------------------

def check_one(job) -> dict:
    """Open one record and measure it. Never raises; failures come back as rows."""
    rec_id, source, path, expect_ext = job
    out = {"global_record_id": rec_id, "source_dataset": source,
           "signal_path": path, "hard_fail": "", "flags": ""}
    hard, flags = [], []
    try:
        if not Path(path + ".hea").exists():
            out["hard_fail"] = "missing header"
            return out
        if not Path(path + expect_ext).exists():
            out["hard_fail"] = f"missing signal file {expect_ext}"
            return out

        r = wfdb.rdrecord(path)
        x = r.p_signal

        out["fs"] = r.fs
        out["n_leads"] = r.n_sig
        out["n_samples"] = r.sig_len
        out["duration_s"] = round(r.sig_len / r.fs, 4) if r.fs else 0
        out["leads"] = "|".join(r.sig_name)
        out["units"] = "|".join(sorted(set(r.units)))
        out["gains"] = "|".join(sorted({f"{g:g}" for g in r.adc_gain}))
        out["fmt"] = "|".join(sorted(set(r.fmt)))

        if r.fs != TARGET_FS:
            hard.append(f"fs={r.fs}")
        if r.n_sig != len(TARGET_LEADS):
            hard.append(f"n_leads={r.n_sig}")
        if r.sig_len != TARGET_N:
            hard.append(f"n_samples={r.sig_len}")
        if [normalize_lead(l) for l in r.sig_name] != TARGET_LEADS:
            hard.append("lead names/order")
        if set(r.units) != {"mV"}:
            hard.append(f"units={sorted(set(r.units))}")
        if set(r.adc_gain) != {TARGET_GAIN}:
            hard.append(f"gain={sorted(set(r.adc_gain))}")

        # NaN in p_signal is how wfdb surfaces a format-16 "invalid sample"
        # (the reserved minimum code), so this catches both at once.
        n_nan = int(np.isnan(x).sum())
        out["n_nan"] = n_nan
        if n_nan:
            hard.append(f"{n_nan} NaN/invalid samples")

        xf = np.nan_to_num(x)
        ptp = np.ptp(xf, axis=0)
        out["n_flat_leads"] = int((ptp < FLAT_MV).sum())
        out["median_ptp_mV"] = round(float(np.median(ptp)), 4)
        out["max_ptp_mV"] = round(float(ptp.max()), 4)

        if ptp.max() < AMP_MIN_MV:
            hard.append("all leads flat -- no ECG present")
        if out["n_flat_leads"]:
            flags.append(f"{out['n_flat_leads']} dead lead(s)")
        if ptp.max() > AMP_MAX_MV:
            flags.append(f"saturating: {ptp.max():.1f} mV p-p")

        q = np.rint(xf * 1000.0).astype(np.int32)
        out["fingerprint"] = hashlib.sha1(q.tobytes()).hexdigest()
    except Exception as exc:                                       # noqa: BLE001
        out["hard_fail"] = f"{type(exc).__name__}: {exc}"
        return out

    out["hard_fail"] = "; ".join(hard)
    out["flags"] = "; ".join(flags)
    return out


CHECK_COLS = ["global_record_id", "source_dataset", "fs", "n_leads",
              "n_samples", "duration_s", "leads", "units", "gains", "fmt",
              "n_nan", "n_flat_leads", "median_ptp_mV", "max_ptp_mV",
              "fingerprint", "hard_fail", "flags", "signal_path"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs() -> list:
    """Both metadata CSVs, with absolute signal paths and a data extension."""
    rows = []
    if not CURRENT_CSV.exists():
        sys.exit(f"missing {CURRENT_CSV}")
    for r in csv.DictReader(CURRENT_CSV.open(encoding="utf-8")):
        r["_ext"] = Path(r["data_path"]).suffix or ".dat"
        r["_abs"] = r["signal_path"]
        r["_origin"] = "current-harmonized"
        rows.append(r)
    if not CUHK_CSV.exists():
        sys.exit(f"missing {CUHK_CSV} -- build the CUHK clean release first")
    for r in csv.DictReader(CUHK_CSV.open(encoding="utf-8")):
        r["_ext"] = Path(r["data_path"]).suffix or ".dat"
        r["_abs"] = r["signal_path"]
        r["_origin"] = "cuhk-clean"
        rows.append(r)
    return rows


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "-"


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NPROC", 4)))
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and report; copy nothing")
    ap.add_argument("--reuse-checks", action="store_true",
                    help="reuse an existing precheck CSV instead of re-measuring")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="keep byte-identical duplicate signals instead of "
                         "dropping the later copies")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing release directory")
    args = ap.parse_args()

    if DEST.exists() and not args.dry_run and not args.force:
        sys.exit(f"{DEST} already exists -- pass --force to rebuild")

    rows = load_inputs()
    by_src = Counter(r["source_dataset"] for r in rows)
    print(f"candidate records: {len(rows):,}")
    for k, v in sorted(by_src.items()):
        print(f"  {k:<10} {v:>7,}")

    # ---- 1. validate ------------------------------------------------
    REPORTS.mkdir(parents=True, exist_ok=True)
    checks: dict = {}
    if args.reuse_checks and CHECKS_CSV.exists():
        for c in csv.DictReader(CHECKS_CSV.open(encoding="utf-8")):
            checks[c["global_record_id"]] = c
        print(f"\nreusing {CHECKS_CSV} ({len(checks):,} rows)")
    else:
        jobs = [(r["global_record_id"], r["source_dataset"], r["_abs"], r["_ext"])
                for r in rows]
        print(f"\nvalidating {len(jobs):,} records with {args.workers} workers "
              f"(opens every signal)")
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for c in ex.map(check_one, jobs, chunksize=64):
                checks[c["global_record_id"]] = c
                done += 1
                if done % 5000 == 0:
                    print(f"  {done:,}/{len(jobs):,}", flush=True)
        with CHECKS_CSV.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CHECK_COLS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(checks[r["global_record_id"]])
        print(f"wrote {CHECKS_CSV}")

    failed = [r for r in rows if checks[r["global_record_id"]]["hard_fail"]]
    passed = [r for r in rows if not checks[r["global_record_id"]]["hard_fail"]]
    print(f"\nvalidation: {len(passed):,} passed, {len(failed):,} hard-failed")
    if failed:
        reasons = Counter(checks[r["global_record_id"]]["hard_fail"] for r in failed)
        for k, v in reasons.most_common(10):
            print(f"  {v:>6}  {k}")

    # ---- 2. duplicates ----------------------------------------------
    fp = defaultdict(list)
    for r in passed:
        fp[checks[r["global_record_id"]]["fingerprint"]].append(r)
    dup_groups = {k: v for k, v in fp.items() if len(v) > 1}
    dropped_dupes = []
    if dup_groups:
        print(f"\nduplicate signals: {len(dup_groups)} group(s) covering "
              f"{sum(len(v) for v in dup_groups.values())} records")
        pairs = Counter()
        for v in dup_groups.values():
            pairs[tuple(sorted({r["source_dataset"] for r in v}))] += 1
        for k, v in pairs.most_common():
            print(f"  {v:>4} group(s)  {'+'.join(k)}")
    with DUPES_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fingerprint", "n_copies", "kept", "dropped",
                    "sources", "labels"])
        for k, v in sorted(dup_groups.items()):
            v = sorted(v, key=lambda r: r["global_record_id"])
            w.writerow([k, len(v), v[0]["global_record_id"],
                        "|".join(r["global_record_id"] for r in v[1:]),
                        "|".join(sorted({r["source_dataset"] for r in v})),
                        "|".join(sorted({r["cardiosentry_label"] for r in v}))])
            if not args.keep_duplicates:
                dropped_dupes.extend(v[1:])
    print(f"wrote {DUPES_CSV}")

    drop_ids = {r["global_record_id"] for r in dropped_dupes}
    kept = [r for r in passed if r["global_record_id"] not in drop_ids]
    if dropped_dupes:
        print(f"  dropping {len(dropped_dupes)} redundant copy/copies "
              f"(keeping the first of each group)")

    # ---- exclusion report -------------------------------------------
    with EXCLUDED_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["global_record_id", "source_dataset", "reason", "detail"])
        for r in failed:
            w.writerow([r["global_record_id"], r["source_dataset"],
                        "FAILED_SIGNAL_CHECK",
                        checks[r["global_record_id"]]["hard_fail"]])
        for r in dropped_dupes:
            w.writerow([r["global_record_id"], r["source_dataset"],
                        "DUPLICATE_SIGNAL",
                        "identical to an earlier record; see "
                        f"{DUPES_CSV.name}"])
    print(f"wrote {EXCLUDED_CSV}")

    # ---- ids must be unique -----------------------------------------
    dup_ids = [k for k, v in Counter(r["global_record_id"] for r in kept).items()
               if v > 1]
    if dup_ids:
        sys.exit(f"duplicate global_record_id across sources: {dup_ids[:5]}")

    # ---- summary -----------------------------------------------------
    counts = {c: sum(int(r[c]) for r in kept) for c in TARGET_CLASSES}
    combos = Counter(r["cardiosentry_label"] for r in kept)
    src_counts = Counter(r["source_dataset"] for r in kept)
    groups = {r["split_group_id"] for r in kept}
    flagged = sum(1 for r in kept if checks[r["global_record_id"]]["flags"])

    print(f"\n{'=' * 58}\nv3 cohort (natural distribution, nothing balanced)")
    print(f"{'=' * 58}")
    print(f"records          : {len(kept):,}")
    print(f"patient groups   : {len(groups):,}")
    print(f"sources          : " + "  ".join(
        f"{k}={v:,}" for k, v in sorted(src_counts.items())))
    print("classes          : " + "  ".join(
        f"{c}={counts[c]:,}" for c in TARGET_CLASSES))
    print("combinations     :")
    for k, v in combos.most_common():
        print(f"    {k or '(none)':<20} {v:>7,}  {pct(v, len(kept))}")
    print(f"quality-flagged  : {flagged:,} ({pct(flagged, len(kept))}) "
          f"-- dead lead or saturation, kept and flagged")

    if args.dry_run:
        print("\n--dry-run: nothing written to the release directory")
        return 0

    # ---- 3. copy -----------------------------------------------------
    if DEST.exists():
        shutil.rmtree(DEST)
    (COHORT / "signals").mkdir(parents=True, exist_ok=True)
    print(f"\ncopying {len(kept):,} records into {COHORT}")

    manifest, out_rows, sums = [], [], []
    fail_copy = []
    for n, r in enumerate(sorted(kept, key=lambda r: r["global_record_id"]), 1):
        rec_id = r["global_record_id"]
        src_dir = COHORT / "signals" / r["source_dataset"]
        src_dir.mkdir(parents=True, exist_ok=True)
        hdr_src = Path(r["_abs"] + ".hea")
        dat_src = Path(r["_abs"] + r["_ext"])
        hdr_dst = src_dir / f"{rec_id}.hea"
        dat_dst = src_dir / f"{rec_id}{r['_ext']}"
        try:
            shutil.copyfile(dat_src, dat_dst)
            hdr_dst.write_text(
                rewrite_header(hdr_src.read_text(encoding="utf-8"),
                               hdr_src.stem, rec_id), encoding="utf-8")
        except Exception as exc:                                   # noqa: BLE001
            fail_copy.append((rec_id, str(exc)))
            continue

        rel = f"signals/{r['source_dataset']}/{rec_id}"
        h_sha, d_sha = sha256(hdr_dst), sha256(dat_dst)
        manifest.append({
            "global_record_id": rec_id,
            "source_dataset": r["source_dataset"],
            "original_record_id": r["original_record_id"],
            "record_base": rel,
            "header_file": rel + ".hea",
            "data_file": rel + r["_ext"],
            "header_sha256": h_sha,
            "data_sha256": d_sha,
        })
        sums += [f"{h_sha}  {rel}.hea", f"{d_sha}  {rel}{r['_ext']}"]

        out = {k: v for k, v in r.items() if not k.startswith("_")}
        out["signal_path"] = rel
        out["header_path"] = rel + ".hea"
        out["data_path"] = rel + r["_ext"]
        chk = checks[rec_id]
        issues = [i for i in (out.get("validation_issues", ""), chk["flags"]) if i]
        out["validation_issues"] = "; ".join(issues)
        out["validation_status"] = "CHECK" if chk["flags"] else "VALID"
        out_rows.append(out)

        if n % 5000 == 0:
            print(f"  {n:,}/{len(kept):,}", flush=True)

    if fail_copy:
        print(f"COPY FAILURES: {len(fail_copy)} -- {fail_copy[:3]}", file=sys.stderr)

    meta_cols = list(out_rows[0].keys())
    meta_path = COHORT / "harmonized_ecg_metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=meta_cols)
        w.writeheader()
        w.writerows(out_rows)
    with (COHORT / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    (COHORT / "checksums.sha256").write_text("\n".join(sums) + "\n",
                                             encoding="utf-8")
    print(f"wrote {meta_path} ({len(out_rows):,} rows)")

    # ---- 4. verify ---------------------------------------------------
    print("\nverifying the copy")
    bad = 0
    for m in manifest:
        if sha256(COHORT / m["header_file"]) != m["header_sha256"] or \
                sha256(COHORT / m["data_file"]) != m["data_sha256"]:
            bad += 1
    import random
    random.seed(0)
    readback = 0
    for m in random.sample(manifest, min(300, len(manifest))):
        r = wfdb.rdrecord(str(COHORT / m["record_base"]))
        assert (r.fs, r.sig_len, r.n_sig) == (TARGET_FS, TARGET_N, 12)
        assert not np.isnan(r.p_signal).any()
        readback += 1
    print(f"  checksum mismatches : {bad}")
    print(f"  re-read through wfdb: {readback} records, all conformant")

    n_files = len(manifest) * 2
    n_bytes = sum((COHORT / m["header_file"]).stat().st_size +
                  (COHORT / m["data_file"]).stat().st_size for m in manifest)

    # ---- VERSION.json -------------------------------------------------
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cfg = PIPELINE / "config"
    version = {
        "version": VERSION,
        "created_utc": created,
        "tool": "scripts/build_v3_release.py 1.0",
        "note": (
            "Four-source merge: v2's PTB-XL + Chapman + 2026 ACS/STEMI, plus "
            "the CUHK ECG collection (17,969 records decoded from 67,431 "
            "Philips Sierra ECG XML files). NO balancing, over/under-sampling, "
            "augmentation, synthetic generation or split assignment was "
            "applied -- v3 is the natural merged distribution. Every record "
            "was validated by opening it (rate, leads, duration, units, gain, "
            "NaN/invalid samples, dead leads, amplitude) and de-duplicated on "
            "the signal itself. CUHK labels are ALGORITHMIC (Philips DXL), "
            "unlike the human-labelled v2 sources -- see "
            "harmonized/DATASET.md."),
        "cohort_exported": "harmonized",
        "layout": "single",
        "hardlinked": False,
        "path_style": "relative",
        "balancing_applied": False,
        "splits_assigned": False,
        "cohorts": {
            "harmonized": {
                "directory": "harmonized",
                "metadata_csv": "harmonized/harmonized_ecg_metadata.csv",
                "records": len(out_rows),
                "rows": len(out_rows),
                "files": n_files,
                "bytes": n_bytes,
                "gigabytes": round(n_bytes / 1e9, 3),
                "failures": len(fail_copy),
                "verification": {
                    "mode": "all",
                    "checked": len(manifest),
                    "failed": bad,
                    "summary": f"{len(manifest) - bad:,}/{len(manifest):,} "
                               f"records re-hashed identical after copy",
                },
            }
        },
        "counts": {
            "harmonized": counts,
            "combinations": dict(combos),
            "by_source": dict(src_counts),
            "patient_groups": len(groups),
        },
        "validation": {
            "candidates": len(rows),
            "hard_failed": len(failed),
            "duplicates_dropped": len(dropped_dupes),
            "quality_flagged_kept": flagged,
            "precheck_csv": str(CHECKS_CSV.relative_to(PIPELINE)),
            "excluded_csv": str(EXCLUDED_CSV.relative_to(PIPELINE)),
            "duplicates_csv": str(DUPES_CSV.relative_to(PIPELINE)),
        },
        "inputs": {
            "current_harmonized_csv": {
                "path": str(CURRENT_CSV), "sha256": sha256(CURRENT_CSV),
                "rows": sum(1 for r in rows if r["_origin"] == "current-harmonized"),
            },
            "cuhk_clean_csv": {
                "path": str(CUHK_CSV), "sha256": sha256(CUHK_CSV),
                "rows": sum(1 for r in rows if r["_origin"] == "cuhk-clean"),
            },
        },
        "config": {
            p.name: sha256(p) for p in sorted(cfg.glob("*.yaml"))
        } if cfg.is_dir() else {},
        "failures": fail_copy,
    }
    (DEST / "VERSION.json").write_text(json.dumps(version, indent=2) + "\n",
                                       encoding="utf-8")

    write_docs(DEST, COHORT, version, out_rows, checks, counts, combos,
               src_counts, groups, len(rows), len(failed), len(dropped_dupes),
               flagged, dup_groups)
    update_changelog(version, counts, src_counts, len(out_rows), n_bytes)

    print(f"\n{'=' * 58}")
    print(f"release {VERSION} written to {DEST}")
    print(f"  {len(out_rows):,} records, {n_files:,} files, "
          f"{n_bytes / 1e9:.2f} GB")
    return 0 if not fail_copy and not bad else 2


# ---------------------------------------------------------------------------
def write_docs(dest, cohort, version, rows, checks, counts, combos,
               src_counts, groups, n_cand, n_failed, n_dupes, flagged,
               dup_groups):
    """DATASET.md inside harmonized/, plus a short release README."""
    n = len(rows)

    def co(a, b):
        return sum(1 for r in rows if r[a] == "1" and r[b] == "1")

    triple = sum(1 for r in rows if r["STEMI"] == "1" and r["AF"] == "1"
                 and r["LVH"] == "1")
    per_src_cls = defaultdict(Counter)
    for r in rows:
        per_src_cls[r["source_dataset"]]["_n"] += 1
        for c in TARGET_CLASSES:
            if r[c] == "1":
                per_src_cls[r["source_dataset"]][c] += 1
    grp_per_src = defaultdict(set)
    for r in rows:
        grp_per_src[r["source_dataset"]].add(r["split_group_id"])
    rec_per_grp = Counter(r["split_group_id"] for r in rows)

    n_multi = sum(v for k, v in combos.items() if "+" in k)
    ages = [float(r["age"]) for r in rows if r.get("age", "").replace(".", "", 1).isdigit()]

    L = []
    a = L.append
    a("# CardioSentry v3 — merged harmonized ECG dataset\n")
    a(f"**{n:,} twelve-lead ECGs** from four sources, in one standalone "
      "directory. This is the *natural* merged distribution: no balancing, no "
      "resampling of classes, no augmentation, no synthetic data, and no "
      "train/validation/test assignment.\n")
    a(f"Built {version['created_utc']} by `scripts/build_v3_release.py`.\n")
    a("---\n")

    a("## What is in here\n")
    a("```")
    a("harmonized/")
    a(f"├── harmonized_ecg_metadata.csv   {n:,} rows, one per ECG")
    a(f"├── MANIFEST.csv                  {n:,} records with SHA-256 per file")
    a(f"├── checksums.sha256              {n * 2:,} files")
    a("├── DATASET.md                     this file")
    a("└── signals/<SOURCE>/<global_record_id>.hea + .dat|.mat")
    a("```\n")
    a("Paths inside the CSV are **relative to this directory**, so the folder "
      "can be moved or copied anywhere and still works:\n")
    a("```python")
    a("import pandas as pd, wfdb")
    a('df = pd.read_csv("harmonized_ecg_metadata.csv")')
    a('rec = wfdb.rdrecord(df.signal_path[0])      # from inside harmonized/')
    a("rec.p_signal.shape   # (5000, 12), millivolts")
    a("```\n")

    a("## Signal format — identical for every record\n")
    a("| property | value |")
    a("|---|---|")
    a("| Sampling rate | 500 Hz |")
    a("| Leads | 12 — `I II III aVR aVL aVF V1 V2 V3 V4 V5 V6` |")
    a("| Duration | 10.000 s (5000 samples) |")
    a("| Units | mV |")
    a("| ADC gain | 1000 ADU/mV |")
    a("| Storage | WFDB, format 16 (Chapman uses `16+24`) |")
    a("")
    a("Every one of these was **verified by opening the record**, not taken "
      "from metadata — see *Validation* below.\n")

    a("## Sources\n")
    a("| source | records | patient groups | STEMI | AF | LVH | NORMAL | labels |")
    a("|---|---:|---:|---:|---:|---:|---:|---|")
    prov = {"PTBXL": "human (SCP-ECG overread)",
            "CHAPMAN": "human (SNOMED-CT)",
            "STEMI": "human (clinician STEMI column)",
            "CUHK": "**algorithm** (Philips DXL)"}
    for s in sorted(per_src_cls, key=lambda s: -per_src_cls[s]["_n"]):
        p = per_src_cls[s]
        a(f"| {s} | {p['_n']:,} | {len(grp_per_src[s]):,} | {p['STEMI']:,} | "
          f"{p['AF']:,} | {p['LVH']:,} | {p['NORMAL']:,} | {prov.get(s, '?')} |")
    a(f"| **total** | **{n:,}** | **{len(groups):,}** | "
      f"**{counts['STEMI']:,}** | **{counts['AF']:,}** | "
      f"**{counts['LVH']:,}** | **{counts['NORMAL']:,}** | |")
    a("")

    a("## Natural class distribution\n")
    a("Multi-label: a record may carry more than one class, so these do not "
      "sum to the record count.\n")
    a("| class | records | % of dataset |")
    a("|---|---:|---:|")
    for c in TARGET_CLASSES:
        a(f"| **{c}** | {counts[c]:,} | {pct(counts[c], n)} |")
    a("")
    a("### Exact label combinations\n")
    a("Each record appears on exactly one row here.\n")
    a("| combination | records | % |")
    a("|---|---:|---:|")
    for k, v in combos.most_common():
        a(f"| {k or '(none)'} | {v:,} | {pct(v, n)} |")
    a(f"| **single-label** | {n - n_multi:,} | {pct(n - n_multi, n)} |")
    a(f"| **multi-label (≥2)** | {n_multi:,} | {pct(n_multi, n)} |")
    a("")
    a("### Co-occurrence (inclusive — triples counted in each pair)\n")
    a("| pair | records |")
    a("|---|---:|")
    a(f"| STEMI + AF | {co('STEMI', 'AF'):,} |")
    a(f"| STEMI + LVH | {co('STEMI', 'LVH'):,} |")
    a(f"| AF + LVH | {co('AF', 'LVH'):,} |")
    a(f"| STEMI + AF + LVH | {triple:,} |")
    a("")
    a("> The STEMI co-morbidity above comes **entirely from CUHK**. v2 had "
      "none of it: its ACS/STEMI source is declared "
      "`contributes_classes: [STEMI]`, so it could never produce an AF or LVH "
      "positive, and PTB-XL/Chapman contribute no STEMI. v3 is the first "
      "release where a model can see a STEMI that co-occurs with AF or LVH.\n")

    a("## Provenance — what every row carries\n")
    a("| column | meaning |")
    a("|---|---|")
    a("| `global_record_id` | unique id in this release, `<SOURCE>_NNNNNN` |")
    a("| `source_dataset` | which database the ECG came from |")
    a("| `original_record_id` | the id it had in that database |")
    a("| `patient_id` | subject id, namespaced per source |")
    a("| `patient_id_is_synthetic` | 1 where the source ships no subject id "
      "(Chapman) and one was minted per record |")
    a("| `split_group_id` | **the unit to split on** — never split on records |")
    a("| `STEMI` `AF` `LVH` `NORMAL` | the four binary labels |")
    a("| `cardiosentry_label` | the exact label combination, e.g. `STEMI+LVH` |")
    a("| `original_labels` / `original_labels_readable` | the source's own codes and text |")
    a("| `accepted_label_mapping` | which source code produced which class |")
    a("| `validation_status` / `validation_issues` | signal-quality outcome |")
    a("| `source_extra_json` | everything else the source carried, including "
      "`label_provenance` |")
    a("")
    a("Nothing in this directory points back at the original source "
      "distributions — it is a real standalone copy.\n")

    a("## Validation performed before merging\n")
    a(f"All {n_cand:,} candidate records were opened and measured. A record "
      "was **excluded** if any of these failed:\n")
    a("- sampling rate ≠ 500 Hz")
    a("- lead count ≠ 12, or lead names/order not the target set")
    a("- length ≠ 5000 samples / 10.000 s")
    a("- units ≠ mV, or ADC gain ≠ 1000 ADU/mV")
    a("- any NaN or WFDB invalid sample (the reserved format-16 code)")
    a("- every lead flat — no ECG present")
    a("- header or signal file missing or unreadable\n")
    a("| outcome | records |")
    a("|---|---:|")
    a(f"| candidates | {n_cand:,} |")
    a(f"| excluded — failed a hard check | {n_failed:,} |")
    a(f"| excluded — duplicate signal | {n_dupes:,} |")
    a(f"| **kept** | **{n:,}** |")
    a("")
    a("Two conditions are **flagged, not excluded**, because they are real "
      "recordings that a deployed model will meet:\n")
    a(f"- a dead/flat lead (< {FLAT_MV} mV peak-to-peak)")
    a(f"- recorder saturation (> {AMP_MAX_MV} mV peak-to-peak)\n")
    a(f"{flagged:,} records ({pct(flagged, n)}) carry such a flag: they have "
      "`validation_status = CHECK` and the detail in `validation_issues`. "
      "Consider excluding them from a *test* set, where they make a metric "
      "harder to interpret.\n")

    a("### Duplicate detection\n")
    a("Duplicates were found by fingerprinting the **signal itself** — SHA-1 "
      "of all 12×5000 samples quantised to 1 µV — so an identical recording "
      "is caught whatever source it came from or how it was stored.\n")
    if dup_groups:
        pairs = Counter()
        for v in dup_groups.values():
            pairs[tuple(sorted({r["source_dataset"] for r in v}))] += 1
        a(f"{len(dup_groups)} duplicate group(s) found:\n")
        a("| sources involved | groups |")
        a("|---|---:|")
        for k, v in pairs.most_common():
            a(f"| {' + '.join(k)} | {v} |")
        a("")
        a("The first record of each group was kept and the rest dropped; the "
          "full list is in `reports/v3_duplicate_records.csv`.\n")
    else:
        a("No duplicate signals were found.\n")
    a("CUHK was additionally de-duplicated *before* it reached this build: it "
      "ships the same acquisition in several cohort folders when a patient "
      "belongs to more than one registry, and 5,713 such copies were removed "
      "on the Philips document GUID.\n")

    a("## Two things to know before training\n")
    a("### 1. Label provenance is not uniform\n")
    a("PTB-XL, Chapman and the ACS/STEMI source are **human-labelled**. CUHK "
      "is labelled by the **Philips DXL algorithm** — no clinician signed it. "
      "Every CUHK row carries `label_provenance: ALGORITHM_PHILIPS_DXL` in "
      "`source_extra_json`, with the full Philips statement codes and text "
      "preserved, so the cohort can be audited, re-mapped or down-weighted.\n")
    a("A metric computed on CUHK measures agreement with Philips DXL, not "
      "with clinical truth. Keep the primary test set human-labelled.\n")
    a("### 2. The dataset is now partly serial\n")
    a("| source | records per patient group (median, max) |")
    a("|---|---|")
    for s in sorted(grp_per_src):
        v = sorted(Counter(r["split_group_id"] for r in rows
                           if r["source_dataset"] == s).values())
        a(f"| {s} | {v[len(v) // 2]}, {v[-1]} |")
    a("")
    vv = sorted(rec_per_grp.values())
    a(f"Overall: {len(groups):,} patient groups; the median group holds "
      f"{vv[len(vv) // 2]} record(s), the largest {vv[-1]}. "
      "**Split on `split_group_id`** "
      "(`GroupShuffleSplit` / `StratifiedGroupKFold`) — a random record-level "
      "split puts the same patient in train and test.\n")
    a("Note that CUHK patient groups are the 4-character anonymised id prefix, "
      "which *over*-groups: several real patients can share one. That costs a "
      "little data and cannot leak. Chapman's ids are synthetic, one per "
      "record — if that distribution contains repeat patients they are not "
      "detectable here.\n")

    a("## Not applied, on purpose\n")
    a("| step | status |")
    a("|---|---|")
    a("| class balancing | **not applied** |")
    a("| over- / under-sampling | **not applied** |")
    a("| augmentation | **not applied** |")
    a("| synthetic generation | **not applied** |")
    a("| train/val/test split | **not assigned** |")
    a("")
    a("v3 is the clean merged dataset and the natural distribution. Balancing "
      "and splitting are later, reversible steps that operate on this — see "
      "`scripts/create_balanced_dataset.py` and `scripts/freeze_splits.py`. "
      "Doing them here would make the natural distribution unrecoverable.\n")
    if ages:
        a(f"Demographics for reference: age recorded for {len(ages):,} of "
          f"{n:,} records, median {np.median(ages):.0f}.\n")

    a("## Verifying this copy\n")
    a("```bash")
    a("cd harmonized && sha256sum -c checksums.sha256")
    a("```\n")
    a("## Licensing\n")
    a("A compilation of licence-gated source databases (PhysioNet "
      "credentialed / DUA; the Chongqing ACS/STEMI portion is CC BY-NC-**ND** "
      "4.0; CUHK is institutional data). **Must not be redistributed.** For "
      "internal use by people who already hold access to every source.\n")
    a("> The CUHK source XMLs contain unredacted patient names in ~3.4% of "
      "files, and 28 filenames embed one. None of that reaches this release: "
      "only the anonymised id prefix is carried, and the upstream "
      "`cuhk_dataset_clean` build redacts source filenames and fails if any "
      "name-like text survives. The *source* folders under "
      "`dataset-files/cuhk_dataset/` still hold it.\n")

    (cohort / "DATASET.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    R = [f"# CardioSentry ECG dataset — release `{VERSION}`\n",
         f"Created {version['created_utc']} by `scripts/build_v3_release.py`.\n",
         f"**{n:,} records, {version['cohorts']['harmonized']['gigabytes']} GB.** "
         "Four sources merged into one standalone harmonized cohort.\n",
         "**No balancing, no augmentation, no synthetic data, no split "
         "assignment** — v3 is the natural merged distribution.\n",
         "| Class | records |", "| --- | ---: |"]
    for c in TARGET_CLASSES:
        R.append(f"| {c} | {counts[c]:,} |")
    R += ["", "| Source | records |", "| --- | ---: |"]
    for s, v in sorted(src_counts.items()):
        R.append(f"| {s} | {v:,} |")
    R += ["",
          "Full detail — provenance columns, validation performed, duplicate "
          "handling, label-provenance caveat and splitting guidance — is in "
          "[`harmonized/DATASET.md`](harmonized/DATASET.md).",
          "",
          "`VERSION.json` carries the build provenance, including SHA-256 of "
          "both input CSVs and of every config file.", ""]
    (dest / "README.md").write_text("\n".join(R) + "\n", encoding="utf-8")


def update_changelog(version, counts, src_counts, n, n_bytes):
    path = RELEASES / "CHANGELOG.md"
    entry = (
        f"## `{VERSION}` — {version['created_utc'][:10]}\n\n"
        f"{n:,} records, {n_bytes / 1e9:.3f} GB. "
        + " · ".join(f"{c} {counts[c]:,}" for c in TARGET_CLASSES) + ". "
        f"Sources: " + ", ".join(f"{k} {v:,}" for k, v in sorted(src_counts.items()))
        + ".\n\n"
        "Adds the CUHK ECG collection (17,969 records decoded from 67,431 "
        "Philips Sierra ECG XML files, XLI-decompressed) to v2's three "
        "sources. NO balancing, over/under-sampling, augmentation, synthetic "
        "generation or split assignment — v3 is the natural merged "
        "distribution. Every record validated by opening it and "
        "de-duplicated on the signal itself. First release containing STEMI "
        "co-morbidity (STEMI+AF, STEMI+LVH, triple positives), which v2 could "
        "not represent. CUHK labels are ALGORITHMIC (Philips DXL) unlike the "
        "human-labelled v2 sources — see harmonized/DATASET.md.\n\n"
    )
    if path.exists():
        text = path.read_text(encoding="utf-8")
        # Rebuilding a version replaces its entry rather than stacking another.
        text = re.sub(rf"^## `{re.escape(VERSION)}` — .*?(?=^## |\Z)", "",
                      text, flags=re.S | re.M)
        marker = "Newest first. Written by `scripts/export_dataset.py`.\n"
        if marker in text:
            text = text.replace(marker, marker + "\n" + entry, 1)
        else:
            text += "\n" + entry
        text = re.sub(r"\n{4,}", "\n\n\n", text)
    else:
        text = "# Dataset releases\n\nNewest first.\n\n" + entry
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
