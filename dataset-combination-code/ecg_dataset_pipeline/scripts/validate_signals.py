#!/usr/bin/env python3
"""
STEP 4 -- Verify signal compatibility before anything is combined.

For every extracted candidate this checks, from the real files:
  * 12 leads present
  * lead ordering (case-normalised: PTB-XL writes AVR, others write aVR)
  * sampling frequency
  * recording duration / number of samples
  * signal amplitude units + ADC gain (orientation/scale convention)
  * missing leads, NaN/Inf values, all-zero or flat-line leads
  * unreadable / truncated / corrupted recordings
  * unusual signal lengths

*** This script NEVER resamples, pads, crops or otherwise modifies a signal. ***
If a source does not match the 12-lead / 500 Hz / 10 s / 5000-sample target,
the incompatibility is reported together with the preprocessing that would be
required to fix it. Deciding whether to apply that preprocessing is yours.

Outputs
  reports/signal_compatibility_report.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import (NORM_TARGET_LEADS, TARGET_DURATION_S, TARGET_FS,
                    TARGET_LEADS, TARGET_N_SAMPLES, banner, info, load_paths,
                    normalize_lead, parse_header, reports_dir, warn, write_csv)

FIELDS = [
    "global_record_id", "source_dataset", "original_record_id", "cardiosentry_label",
    "header_exists", "data_exists", "readable",
    "num_leads", "leads_ok", "lead_names", "lead_order_ok", "missing_leads", "extra_leads",
    "sampling_frequency", "fs_ok", "num_samples", "n_samples_ok",
    "duration_seconds", "duration_ok",
    "adc_gain", "adc_units", "units_ok",
    "n_nan", "n_inf", "n_flat_leads", "flat_lead_names", "all_zero",
    "signal_min", "signal_max", "signal_absmax_mv",
    "validation_status", "issues", "required_preprocessing",
]


# A recording with this many dead (constant) leads is treated as corrupt
# rather than merely noisy: at 6, half the 12-lead montage carries no
# information. Override with --max-flat-leads (use 12 to disable).
DEFAULT_MAX_FLAT_LEADS = 6


def _check_one(row, max_flat_leads=DEFAULT_MAX_FLAT_LEADS):
    """Validate one record. Runs in a worker process."""
    out = {k: "" for k in FIELDS}
    out.update({
        "global_record_id": row["global_record_id"],
        "source_dataset": row["source_dataset"],
        "original_record_id": row["original_record_id"],
        "cardiosentry_label": row["cardiosentry_label"],
    })
    issues, prep = [], []

    hea, dat = row["header_path"], row["data_path"]
    out["header_exists"] = int(os.path.exists(hea))
    out["data_exists"] = int(os.path.exists(dat))

    if not out["header_exists"] or not out["data_exists"]:
        miss = []
        if not out["header_exists"]:
            miss.append("header (.hea)")
        if not out["data_exists"]:
            miss.append(f"signal ({Path(dat).suffix})")
        out["readable"] = 0
        out["validation_status"] = "INVALID_MISSING_FILE"
        out["issues"] = "missing " + " and ".join(miss)
        out["required_preprocessing"] = "none possible -- re-download the source distribution"
        return out

    hi = parse_header(hea)
    if hi.parse_error:
        issues.append(f"malformed header: {hi.parse_error}")

    out["num_leads"] = hi.n_sig if hi.n_sig is not None else ""
    out["sampling_frequency"] = hi.fs if hi.fs is not None else ""
    out["num_samples"] = hi.n_samples if hi.n_samples is not None else ""
    out["duration_seconds"] = round(hi.duration_s, 4) if hi.duration_s else ""
    out["lead_names"] = "|".join(hi.lead_names)
    out["adc_gain"] = "|".join(str(g) for g in sorted(set(hi.adc_gain))) if hi.adc_gain else ""
    out["adc_units"] = "|".join(sorted(set(u for u in hi.adc_units if u))) or ""

    # ---- lead checks -------------------------------------------------
    norm = [normalize_lead(l) for l in hi.lead_names]
    out["leads_ok"] = int(hi.n_sig == 12)
    if hi.n_sig != 12:
        issues.append(f"expected 12 leads, header declares {hi.n_sig}")
        prep.append("cannot be used as a 12-lead record")
    missing = [l for l in NORM_TARGET_LEADS if l not in norm]
    extra = [l for l in norm if l not in NORM_TARGET_LEADS]
    out["missing_leads"] = "|".join(missing)
    out["extra_leads"] = "|".join(extra)
    out["lead_order_ok"] = int(norm == NORM_TARGET_LEADS)
    if missing:
        issues.append("missing leads: " + ",".join(missing))
    if extra:
        issues.append("unexpected leads: " + ",".join(extra))
    if not missing and not extra and norm != NORM_TARGET_LEADS:
        issues.append("lead ORDER differs from target")
        prep.append(f"reorder leads to {','.join(TARGET_LEADS)}")

    # ---- rate / length checks ----------------------------------------
    out["fs_ok"] = int(hi.fs == TARGET_FS)
    if hi.fs != TARGET_FS:
        issues.append(f"sampling frequency {hi.fs} != {TARGET_FS} Hz")
        prep.append(f"resample {hi.fs} Hz -> {TARGET_FS} Hz (NOT performed)")
    out["n_samples_ok"] = int(hi.n_samples == TARGET_N_SAMPLES)
    out["duration_ok"] = int(hi.duration_s is not None and
                             abs(hi.duration_s - TARGET_DURATION_S) < 1e-6)
    if hi.n_samples != TARGET_N_SAMPLES:
        issues.append(f"{hi.n_samples} samples != {TARGET_N_SAMPLES}")
        if hi.duration_s and hi.duration_s > TARGET_DURATION_S:
            prep.append(f"crop {hi.duration_s}s -> {TARGET_DURATION_S}s (NOT performed)")
        elif hi.duration_s:
            prep.append(f"pad/reject {hi.duration_s}s < {TARGET_DURATION_S}s (NOT performed)")

    out["units_ok"] = int(all(u.lower() == "mv" for u in hi.adc_units if u))
    if not out["units_ok"]:
        issues.append("amplitude units are not mV: " + out["adc_units"])
        prep.append("convert amplitude units to mV")

    # ---- read the actual samples -------------------------------------
    try:
        import wfdb
        rec = wfdb.rdrecord(str(row["signal_path"]))
        sig = np.asarray(rec.p_signal, dtype=np.float64)
        out["readable"] = 1
    except Exception as exc:                      # noqa: BLE001 - report, never crash
        out["readable"] = 0
        out["validation_status"] = "INVALID_UNREADABLE"
        out["issues"] = "; ".join(issues + [f"read failed: {type(exc).__name__}: {exc}"])
        out["required_preprocessing"] = "; ".join(prep) or "none possible -- file is corrupt"
        return out

    if sig.ndim != 2:
        out["validation_status"] = "INVALID_CORRUPT"
        out["issues"] = "; ".join(issues + [f"signal array has {sig.ndim} dimensions"])
        return out

    n_samp, n_lead = sig.shape
    if n_lead != (hi.n_sig or n_lead):
        issues.append(f"header declares {hi.n_sig} leads, data holds {n_lead}")
    if n_samp != (hi.n_samples or n_samp):
        issues.append(f"header declares {hi.n_samples} samples, data holds {n_samp}")

    n_nan = int(np.isnan(sig).sum())
    n_inf = int(np.isinf(sig).sum())
    out["n_nan"], out["n_inf"] = n_nan, n_inf
    if n_nan:
        issues.append(f"{n_nan} NaN samples")
    if n_inf:
        issues.append(f"{n_inf} Inf samples")

    finite = sig[np.isfinite(sig)]
    if finite.size:
        out["signal_min"] = round(float(finite.min()), 5)
        out["signal_max"] = round(float(finite.max()), 5)
        out["signal_absmax_mv"] = round(float(np.abs(finite).max()), 5)

    with np.errstate(invalid="ignore"):
        stds = np.nanstd(sig, axis=0)
    flat = [i for i, s in enumerate(stds) if not np.isfinite(s) or s == 0.0]
    out["n_flat_leads"] = len(flat)
    out["flat_lead_names"] = "|".join(
        hi.lead_names[i] if i < len(hi.lead_names) else str(i) for i in flat)
    if flat:
        issues.append(f"{len(flat)} flat/constant lead(s): {out['flat_lead_names']}")
    out["all_zero"] = int(finite.size > 0 and not np.any(finite != 0))
    if out["all_zero"]:
        issues.append("entire recording is zero")

    # ---- verdict ------------------------------------------------------
    fatal = (not out["leads_ok"] or missing or n_nan or n_inf or out["all_zero"]
             or len(flat) >= max_flat_leads or hi.parse_error)
    if flat and len(flat) >= max_flat_leads:
        issues.append(f"{len(flat)} of 12 leads carry no signal -- treated as corrupt "
                      f"(threshold --max-flat-leads={max_flat_leads})")
    if fatal:
        out["validation_status"] = "INVALID_SIGNAL"
    elif issues:
        out["validation_status"] = ("VALID_NEEDS_PREPROCESSING" if prep else "VALID_WITH_WARNINGS")
    else:
        out["validation_status"] = "VALID"
    out["issues"] = "; ".join(issues)
    out["required_preprocessing"] = "; ".join(prep) or ("none -- already 12-lead / 500 Hz / 10 s / 5000 samples"
                                                        if not issues else "")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="validate only the first N (debugging)")
    ap.add_argument("--max-flat-leads", type=int, default=DEFAULT_MAX_FLAT_LEADS,
                    help="a record with this many constant/dead leads is marked "
                         "INVALID_SIGNAL (default %(default)s; pass 13 to disable)")
    args = ap.parse_args(argv)

    paths = load_paths(args.paths)
    rdir = reports_dir(paths)
    cand_csv = rdir / "extracted_candidates.csv"
    if not cand_csv.exists():
        warn(f"{cand_csv} not found -- run extract_records.py first")
        return 1

    with open(cand_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]

    info(f"validating {len(rows):,} candidate signals with {args.workers} workers "
         f"(reads every sample -- takes a few minutes) ...")

    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_check_one, r, args.max_flat_leads): r["global_record_id"]
                    for r in rows}
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if i % 2000 == 0:
                    info(f"  {i:,}/{len(rows):,}")
    else:
        for i, r in enumerate(rows, 1):
            results.append(_check_one(r, args.max_flat_leads))
            if i % 2000 == 0:
                info(f"  {i:,}/{len(rows):,}")

    results.sort(key=lambda d: d["global_record_id"])
    write_csv(rdir / "signal_compatibility_report.csv", results, FIELDS)

    by_status = collections.Counter(r["validation_status"] for r in results)
    by_src = collections.defaultdict(collections.Counter)
    for r in results:
        by_src[r["source_dataset"]][r["validation_status"]] += 1

    print(banner("STEP 4 -- SIGNAL COMPATIBILITY"))
    print(f"  target representation: 12 leads / {TARGET_FS} Hz / {TARGET_DURATION_S:.0f} s / "
          f"{TARGET_N_SAMPLES} samples per lead")
    print(f"  lead order           : {', '.join(TARGET_LEADS)}\n")
    print("  overall:")
    for st, n in by_status.most_common():
        print(f"    {st:28s} {n:7,d}")
    print("\n  per source:")
    for s in sorted(by_src):
        parts = ", ".join(f"{k}={v:,}" for k, v in by_src[s].most_common())
        print(f"    {s:10s} {parts}")

    nflat = sum(1 for r in results if str(r["n_flat_leads"]) not in ("", "0"))
    if nflat:
        print(f"\n  {nflat:,} records carry at least one flat/dead lead "
              f"(>= {args.max_flat_leads} dead leads => INVALID_SIGNAL).")

    ok = [r for r in results if r["validation_status"].startswith("VALID")]
    print("\n  conformance among readable records:")
    for lbl, key in (("12 leads", "leads_ok"), ("lead order matches", "lead_order_ok"),
                     ("500 Hz", "fs_ok"), ("5000 samples", "n_samples_ok"),
                     ("10 seconds", "duration_ok"), ("mV units", "units_ok")):
        n = sum(1 for r in ok if str(r[key]) == "1")
        print(f"    {lbl:20s} {n:7,d} / {len(ok):,}  "
              f"{'YES (all)' if n == len(ok) else 'NO -- see report'}")

    needs = [r for r in results if r["required_preprocessing"] and
             not r["required_preprocessing"].startswith("none")]
    if needs:
        print(f"\n  !! {len(needs):,} records would need preprocessing. NONE was applied.")
        for p, n in collections.Counter(r["required_preprocessing"] for r in needs).most_common(5):
            print(f"       {n:6,d}  {p}")
    else:
        print("\n  No resampling, cropping or padding is required for any valid record.")
    info(f"wrote {rdir/'signal_compatibility_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
