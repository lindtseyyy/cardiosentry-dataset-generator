"""
CardioSentry / CUHK-STEMI staging -- PASS 2: XML -> WFDB.

Decodes the Philips XLI-compressed waveform in each candidate XML and writes
a WFDB record in EXACTLY the representation the main pipeline targets:

    12 leads, order I II III aVR aVL aVF V1..V6
    500 Hz, 5000 samples (10.000 s)
    format 16 (int16 LE, multiplexed), gain 1000 ADU/mV, ADC zero 0

Two representation changes are made, both declared in the header and in the
metadata, neither of them lossy:

  * DURATION.  The source records are 11.000 s (5500 samples). The first
    10.000 s are kept and the trailing 1 s discarded. The tail is where the
    Philips recorder stops maintaining the derived-lead relationships
    (III = II - I etc.), so it is the correct end to drop.
  * UNITS.  The source stores 5 uV per LSB (gain 200 ADU/mV). Samples are
    multiplied by 5 so the record carries the project-wide gain of 1000
    ADU/mV. This is an exact integer rescale, not a resample: the true
    quantisation step remains 5 uV and is recorded as
    `native_resolution_uv` in the metadata.
  * LIMB LEADS.  III, aVR, aVL and aVF are recomputed from I and II rather
    than taken from the file. See the long comment in `convert_one` -- these
    are 10-wire acquisitions, so those four leads are arithmetic by
    construction, and the stored residual is unreliable in 26% of files.

Writes:
  data/signals/<global_record_id>.hea + .dat
  data/cuhk_metadata.csv           harmonized-schema metadata, one row per
                                   converted record, ready to concatenate
  data/signals/checksums.sha256
  reports/conversion_report.csv    per record: status, checks, any issue
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import sierraecg

ROOT = Path(__file__).resolve().parents[1]
CAND = ROOT / "data" / "index" / "candidates.csv"
SIGDIR = ROOT / "data" / "signals"
OUT_META = ROOT / "data" / "cuhk_metadata.csv"
OUT_REPORT = ROOT / "reports" / "conversion_report.csv"

TARGET_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                "V1", "V2", "V3", "V4", "V5", "V6"]
TARGET_FS = 500
TARGET_N = 5000
TARGET_GAIN = 1000.0          # ADU/mV
NATIVE_UV_PER_LSB = 5         # Philips <resolution>
SCALE = int(TARGET_GAIN * NATIVE_UV_PER_LSB / 1000)   # = 5

# European vs American precordial naming; both occur in this dataset.
LEAD_ALIASES = {f"C{i}": f"V{i}" for i in range(1, 7)}


def wfdb_checksum(x: np.ndarray) -> int:
    s = int(np.sum(x.astype(np.int64))) % 65536
    return s - 65536 if s >= 32768 else s


def write_header(path: Path, rec_id: str, x: np.ndarray) -> None:
    """x: (12, TARGET_N) int16, already in ADU at TARGET_GAIN."""
    lines = [f"{rec_id} {len(TARGET_LEADS)} {TARGET_FS} {TARGET_N}"]
    for i, lead in enumerate(TARGET_LEADS):
        lines.append(
            f"{rec_id}.dat 16 {TARGET_GAIN:.1f}(0)/mV 16 0 "
            f"{int(x[i, 0])} {wfdb_checksum(x[i])} 0 {lead}")
    lines.append(f"# source: CUHK STEMI registry, Philips Sierra ECG XML "
                 f"(XLI-compressed), decoded with sierraecg")
    lines.append(f"# native: 500 Hz, 5500 samples (11 s), {NATIVE_UV_PER_LSB} uV/LSB; "
                 f"first {TARGET_N} samples kept, samples x{SCALE} to reach "
                 f"{TARGET_GAIN:.0f} ADU/mV")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_one(row: dict) -> dict:
    rec_id = row["global_record_id"]
    out = {"global_record_id": rec_id, "original_record_id": row["original_record_id"],
           "xml_path": row["xml_path"], "status": "OK", "issues": ""}
    issues = []
    try:
        ecg = sierraecg.read_file(row["xml_path"])
        labels = [LEAD_ALIASES.get(l.label.upper(), l.label) for l in ecg.leads]
        labels = [{"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}.get(l.upper(), l)
                  for l in labels]
        if labels != TARGET_LEADS:
            out["status"] = "FAILED"
            out["issues"] = f"lead order {labels}"
            return out

        raw = np.asarray([l.samples for l in ecg.leads], dtype=np.int64)
        out["native_n_samples"] = int(raw.shape[1])
        if raw.shape[1] < TARGET_N:
            out["status"] = "FAILED"
            out["issues"] = f"only {raw.shape[1]} samples, need {TARGET_N}"
            return out

        sig = raw[:, :TARGET_N].astype(np.float64)

        # ---- the four limb leads are RECOMPUTED, not trusted ---------
        # Every record in this dataset is `acquisitiontype = 10-WIRE`, i.e.
        # only 8 leads (I, II, V1-V6) are physically acquired; III, aVR, aVL
        # and aVF are arithmetic functions of I and II and cannot carry
        # independent information.
        #
        # XLI stores those four as a residual against that derivation. In 74%
        # of these files the residual is zero to within 2 LSB (10 uV); in the
        # other 26% the decoded residual is large (hundreds of uV) with no
        # intermediate cases -- a bimodal split that is a decoder artefact,
        # not physiology. Independent checks (decoded heart rate vs the
        # Philips-reported rate, per-lead amplitudes vs the Philips lead
        # measurements) confirm I, II and V1-V6 decode correctly in BOTH
        # groups, so only the residual channels are affected.
        #
        # The four leads are therefore always recomputed from I and II. For
        # the 74% this changes nothing; for the 26% it replaces a corrupt
        # residual with the only values physically consistent with the
        # acquisition. The discrepancy against the file's own reconstruction
        # is recorded per record as `max_lead_relation_err_uv`.
        I, II = sig[0], sig[1]
        derived = {2: II - I,               # III
                   3: -(I + II) / 2,        # aVR
                   4: I - II / 2,           # aVL
                   5: II - I / 2}           # aVF
        errs = [np.abs(sig[i] - d).max() for i, d in derived.items()]
        out["max_lead_relation_err_uv"] = round(float(max(errs)) * NATIVE_UV_PER_LSB, 1)
        for i, d in derived.items():
            sig[i] = d
        if max(errs) * NATIVE_UV_PER_LSB > 100:
            issues.append(
                f"limb leads recomputed: file's own reconstruction differed by "
                f"{max(errs) * NATIVE_UV_PER_LSB:.0f} uV")

        x = np.rint(sig * SCALE)
        if np.abs(x).max() > 32767:
            issues.append(f"clipped: |max|={int(np.abs(x).max())} ADU")
            # Clip to -32767, NOT -32768: WFDB reserves the minimum value of a
            # format-16 signal as the "invalid sample" sentinel, which reads
            # back as NaN. Saturated records are already flagged; they should
            # not also acquire holes.
            x = np.clip(x, -32767, 32767)
        x = x.astype(np.int16)

        # ---- signal sanity checks (recorded, not enforced) -----------
        ptp_uv = np.ptp(sig, axis=1) * NATIVE_UV_PER_LSB
        flat = int((ptp_uv < 50).sum())
        if flat:
            issues.append(f"{flat} lead(s) near-flat (<0.05 mV peak-to-peak)")
        if ptp_uv.max() > 20000:
            issues.append(f"implausible amplitude {ptp_uv.max()/1000:.1f} mV")
        out["max_ptp_mV"] = round(float(ptp_uv.max()) / 1000, 3)
        out["n_flat_leads"] = flat

        SIGDIR.mkdir(parents=True, exist_ok=True)
        # WFDB format 16 is sample-multiplexed: all 12 leads of sample 0, then
        # all 12 of sample 1, ...  -> transpose before flattening.
        (SIGDIR / f"{rec_id}.dat").write_bytes(
            np.ascontiguousarray(x.T).astype("<i2").tobytes())
        write_header(SIGDIR / f"{rec_id}.hea", rec_id, x)

    except Exception as exc:                                   # noqa: BLE001
        out["status"] = "FAILED"
        out["issues"] = f"{type(exc).__name__}: {exc}"
        return out

    out["issues"] = "; ".join(issues)
    if issues:
        out["status"] = "OK_WITH_WARNINGS"
    return out


META_FIELDS = [
    "global_record_id", "source_dataset", "original_record_id", "patient_id",
    "patient_id_is_synthetic", "split_group_id", "original_labels",
    "original_labels_readable", "accepted_label_mapping", "cardiosentry_label",
    "STEMI", "AF", "LVH", "NORMAL", "n_positive_labels", "sampling_frequency",
    "duration_seconds", "num_leads", "num_samples", "lead_names", "signal_path",
    "header_path", "data_path", "validation_status", "validation_issues",
    "ambiguity_flag", "ambiguity_note", "age", "sex", "source_extra_json",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="convert only the first N candidates (smoke test)")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("NPROC", 4)))
    ap.add_argument("--only", default="",
                    help="comma-separated global_record_ids to re-convert; "
                         "their rows in the conversion report are updated in "
                         "place and everything else is left alone")
    ap.add_argument("--metadata-only", action="store_true",
                    help="rebuild the metadata CSV from an existing "
                         "reports/conversion_report.csv, without re-decoding "
                         "(the XLI decode is ~0.8 s/file)")
    args = ap.parse_args()

    if not CAND.exists():
        print(f"missing {CAND} -- run scripts/build_manifest.py", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(CAND.open(encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]
    print(f"converting {len(rows)} records with {args.workers} workers")

    if args.only:
        if not OUT_REPORT.exists():
            print(f"--only needs an existing {OUT_REPORT}", file=sys.stderr)
            return 1
        results = {r["global_record_id"]: r
                   for r in csv.DictReader(OUT_REPORT.open(encoding="utf-8"))}
        wanted = {i.strip() for i in args.only.split(",") if i.strip()}
        todo = [r for r in rows if r["global_record_id"] in wanted]
        missing = wanted - {r["global_record_id"] for r in todo}
        if missing:
            print(f"not in candidates.csv: {sorted(missing)}", file=sys.stderr)
            return 1
        print(f"re-converting {len(todo)} record(s), reusing the rest")
        for r in todo:
            res = convert_one(r)
            results[res["global_record_id"]] = res
            print(f"  {res['global_record_id']}: {res['status']} {res['issues']}")
    elif args.metadata_only:
        if not OUT_REPORT.exists():
            print(f"--metadata-only needs {OUT_REPORT}", file=sys.stderr)
            return 1
        results = {r["global_record_id"]: r
                   for r in csv.DictReader(OUT_REPORT.open(encoding="utf-8"))}
        print(f"reusing {OUT_REPORT} ({len(results)} rows), not re-decoding")
    else:
        results = {}
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(convert_one, rows, chunksize=8):
                results[res["global_record_id"]] = res
                done += 1
                if done % 250 == 0:
                    print(f"  {done}/{len(rows)}", flush=True)

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    rep_fields = ["global_record_id", "original_record_id", "status", "issues",
                  "native_n_samples", "max_ptp_mV", "n_flat_leads",
                  "max_lead_relation_err_uv", "xml_path"]
    if not args.metadata_only:
        with OUT_REPORT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=rep_fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(results[r["global_record_id"]])

    meta = []
    for r in rows:
        res = results[r["global_record_id"]]
        if res["status"] == "FAILED":
            continue
        rec_id = r["global_record_id"]
        stem = str(SIGDIR / rec_id)
        qa = r["philips_qa_action_code"]
        issues = [res["issues"]] if res["issues"] else []
        if qa and qa != "ECG OK":
            issues.append(f"philips_qa={qa}")
        # "limb leads recomputed" is an expected, fully handled note (see
        # convert_one); saturation, clipping and dead leads are not, so those
        # records are marked CHECK for the user to filter or inspect.
        blob = " ".join(issues)
        needs_check = any(k in blob for k in
                          ("implausible amplitude", "clipped", "near-flat"))
        meta.append({
            "global_record_id": rec_id,
            "source_dataset": r["source_dataset"],
            "original_record_id": r["original_record_id"],
            "patient_id": r["patient_id"],
            "patient_id_is_synthetic": 0,
            "split_group_id": r["split_group_id"],
            "original_labels": r["original_labels"],
            "original_labels_readable": r["original_labels_readable"],
            "accepted_label_mapping": r["accepted_label_mapping"],
            "cardiosentry_label": r["cardiosentry_label"],
            "STEMI": r["STEMI"], "AF": r["AF"], "LVH": r["LVH"],
            "NORMAL": r["NORMAL"],
            "n_positive_labels": r["n_positive_labels"],
            "sampling_frequency": float(TARGET_FS),
            "duration_seconds": TARGET_N / TARGET_FS,
            "num_leads": len(TARGET_LEADS),
            "num_samples": TARGET_N,
            "lead_names": "|".join(TARGET_LEADS),
            "signal_path": stem,
            "header_path": stem + ".hea",
            "data_path": stem + ".dat",
            "validation_status": "CHECK" if needs_check else "VALID",
            "validation_issues": "; ".join(i for i in issues if i),
            "ambiguity_flag": r["ambiguity_flag"],
            "ambiguity_note": r["ambiguity_note"],
            "age": r["age"],
            "sex": r["sex"],
            "source_extra_json": json.dumps({
                "label_provenance": "ALGORITHM_PHILIPS_DXL",
                "philips_statement_codes": r["original_labels"],
                "philips_statement_text": r["original_labels_readable"],
                "philips_qa_action_code": qa,
                "philips_heartrate": r["philips_heartrate"],
                "cohort": r["cohort"],
                "cohorts_all": r["cohorts_all"],
                "site": r["site"],
                "acquisition_date": r["acquisition_date"],
                "acquisition_time": r["acquisition_time"],
                "document_guid": r["document_guid"],
                "cuhk_patient_group": r["patient_id"].split("_P", 1)[-1],
                "cuhk_patient_id_raw": r["cuhk_patient_id_raw"],
                "native_sampling_rate_hz": 500,
                "native_duration_ms": r["native_duration_ms"],
                "native_resolution_uv": int(r["native_resolution_uv"]),
                "native_lead_labels": r["native_lead_labels"],
                "trimmed_to_first_n_samples": TARGET_N,
                "amplitude_rescale_factor": SCALE,
                "limb_leads_recomputed_from_I_II": True,
                "file_limb_lead_discrepancy_uv": res.get(
                    "max_lead_relation_err_uv"),
                "max_ptp_mV": res.get("max_ptp_mV"),
                "age_is_missing": r["age"] == "",
                "source_xml": Path(r["xml_path"]).name,
            }, ensure_ascii=False),
        })

    with OUT_META.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=META_FIELDS)
        w.writeheader()
        w.writerows(meta)

    # checksums over what was actually written
    sums = []
    for m in sorted(meta, key=lambda r: r["global_record_id"]):
        for p in (Path(m["header_path"]), Path(m["data_path"])):
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            sums.append(f"{h}  {p.name}")
    (SIGDIR / "checksums.sha256").write_text("\n".join(sums) + "\n", encoding="utf-8")

    n_ok = sum(1 for v in results.values() if v["status"] == "OK")
    n_warn = sum(1 for v in results.values() if v["status"] == "OK_WITH_WARNINGS")
    n_fail = sum(1 for v in results.values() if v["status"] == "FAILED")
    print(f"\nconverted OK        : {n_ok}")
    print(f"converted, warnings : {n_warn}")
    print(f"FAILED              : {n_fail}")
    if n_fail:
        for v in results.values():
            if v["status"] == "FAILED":
                print(f"  {v['global_record_id']}: {v['issues']}")
    print(f"\nwrote {OUT_META} ({len(meta)} rows)")
    print(f"wrote {OUT_REPORT}")
    print(f"wrote {SIGDIR}/*.hea + *.dat  and checksums.sha256")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
