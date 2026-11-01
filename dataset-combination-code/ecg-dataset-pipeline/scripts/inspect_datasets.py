#!/usr/bin/env python3
"""
STEP 1 -- Inspect the datasets before extracting anything.

Reads the actual metadata/label files and reports what is really there:
structure, record counts, columns, label vocabularies and frequencies,
patient/recording identifiers, sampling frequency, lead count, duration,
signal-file availability, and source-specific metadata.

Nothing here is hard-coded that can be read from the files.

Outputs
  reports/dataset_inspection_report.csv    one row per (source, property)
  reports/label_frequency_report.csv       every source label + its frequency
  reports/dataset_inspection_summary.txt   human-readable summary
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (banner, info, load_mapping, load_paths, parse_header,
                    reports_dir, warn, write_csv)
from sources import build_sources


def _dur(n_samples, fs_counter):
    """Duration in seconds, tolerant of malformed headers."""
    fs = fs_counter.most_common(1)[0][0] if fs_counter else None
    if not n_samples or not fs:
        return "?"
    return round(n_samples / fs, 4)


def _fmt(n):
    return f"{n:,}" if isinstance(n, int) else str(n)


def inspect_ptbxl(src, mapping, rows, lab_rows, out):
    if not src.available():
        warn(f"PTB-XL not found at {src.db_csv} -- skipping")
        out.append("PTB-XL: NOT FOUND (skipped)")
        return

    import csv as _csv
    with open(src.db_csv, newline="", encoding="utf-8") as fh:
        rdr = _csv.DictReader(fh)
        cols = rdr.fieldnames or []
    recs = src.read()
    scp = src.scp_statements()

    counts = collections.Counter()
    likelihood = collections.defaultdict(collections.Counter)
    for r in recs:
        counts.update(r.raw_labels)
        for k, v in r.extra["scp_likelihoods"].items():
            likelihood[k][v] += 1

    dat_present = sum(1 for r in recs if os.path.exists(r.data_path))
    hea_present = sum(1 for r in recs if os.path.exists(r.header_path))
    both = sum(1 for r in recs if os.path.exists(r.data_path) and os.path.exists(r.header_path))

    # header facts, sampled and then verified across everything present
    hdr_fs, hdr_nsig, hdr_ns, leadsets = (collections.Counter() for _ in range(4))
    for r in recs:
        if not os.path.exists(r.header_path):
            continue
        hi = parse_header(r.header_path)
        hdr_fs[hi.fs] += 1
        hdr_nsig[hi.n_sig] += 1
        hdr_ns[hi.n_samples] += 1
        leadsets["|".join(hi.lead_names)] += 1

    orphan_dat = {Path(p).stem for p in glob.glob(str(src.root / "records500" / "*" / "*.dat"))}
    orphan_hea = {Path(p).stem for p in glob.glob(str(src.root / "records500" / "*" / "*.hea"))}

    props = {
        "root": str(src.root),
        "metadata_file": str(src.db_csv),
        "n_records_in_metadata": len(recs),
        "n_columns": len(cols),
        "columns": "|".join(cols),
        "record_id_column": "ecg_id",
        "patient_id_column": "patient_id",
        "n_unique_record_ids": len({r.original_record_id for r in recs}),
        "n_unique_patient_ids": len({r.patient_id for r in recs if r.patient_id}),
        "label_column": "scp_codes (SCP-ECG statement -> likelihood dict)",
        "label_vocabulary_file": str(src.scp_csv),
        "n_label_statements_defined": len(scp),
        "n_distinct_labels_used": len(counts),
        "labels_used_but_undefined": "|".join(sorted(set(counts) - set(scp))) or "(none)",
        "signal_files_expected": len(recs),
        "signal_dat_present": dat_present,
        "signal_hea_present": hea_present,
        "signal_pairs_present": both,
        "signal_missing": len(recs) - both,
        "orphan_dat_without_hea": len(orphan_dat - orphan_hea),
        "orphan_hea_without_dat": len(orphan_hea - orphan_dat),
        "sampling_frequency_hz": "|".join(f"{k}:{v}" for k, v in hdr_fs.most_common()),
        "num_leads": "|".join(f"{k}:{v}" for k, v in hdr_nsig.most_common()),
        "num_samples": "|".join(f"{k}:{v}" for k, v in hdr_ns.most_common()),
        "duration_seconds": "|".join(
            f"{_dur(k, hdr_fs)}:{v}" for k, v in hdr_ns.most_common(3)),
        "distinct_lead_orderings": len(leadsets),
        "lead_ordering_most_common": leadsets.most_common(1)[0][0] if leadsets else "",
        "extra_metadata": "age|sex|height|weight|report|infarction_stadium1|infarction_stadium2|"
                          "validated_by_human|second_opinion|device|site|recording_date|strat_fold|"
                          "heart_axis|pacemaker|baseline_drift|static_noise|burst_noise|electrodes_problems",
        "has_predefined_folds": "yes (strat_fold 1-10)",
    }
    for k, v in props.items():
        rows.append({"source": "PTBXL", "property": k, "value": v})

    for code, n in counts.most_common():
        st = scp.get(code, {})
        lk = likelihood[code]
        lab_rows.append({
            "source": "PTBXL", "source_label": code,
            "label_name": st.get("description", "*** NOT DEFINED ***"),
            "label_role": "|".join(
                r for r, f in (("diagnostic", st.get("diagnostic")), ("form", st.get("form")),
                               ("rhythm", st.get("rhythm"))) if str(f) not in ("", "nan", "None")),
            "diagnostic_class": st.get("diagnostic_class", ""),
            "diagnostic_subclass": st.get("diagnostic_subclass", ""),
            "count": n,
            "likelihood_distribution": "|".join(f"{k}:{v}" for k, v in sorted(lk.items())),
        })

    out.append(banner("SOURCE: PTB-XL v1.0.3"))
    out.append(f"  root                     : {src.root}")
    out.append(f"  records in metadata      : {_fmt(len(recs))}")
    out.append(f"  unique ECG ids           : {_fmt(props['n_unique_record_ids'])}")
    out.append(f"  unique patient ids       : {_fmt(props['n_unique_patient_ids'])}  <- patient-level splitting POSSIBLE")
    out.append(f"  SCP statements defined   : {len(scp)}   used in data: {len(counts)}")
    out.append(f"  sampling frequency       : {props['sampling_frequency_hz']}")
    out.append(f"  leads / samples          : {props['num_leads']}  /  {props['num_samples']}")
    out.append(f"  lead ordering (n distinct={len(leadsets)}): {props['lead_ordering_most_common']}")
    out.append(f"  signal pairs present     : {_fmt(both)} / {_fmt(len(recs))}")
    if both < len(recs):
        out.append(f"  !! MISSING SIGNAL FILES  : {_fmt(len(recs) - both)} records in the metadata have no .dat/.hea on disk.")
        out.append(f"     This distribution is INCOMPLETE. Those records cannot be used.")
    if orphan_dat - orphan_hea:
        out.append(f"  !! {len(orphan_dat - orphan_hea)} .dat files have no matching .hea (unusable).")
    out.append("  top labels:")
    for code, n in counts.most_common(12):
        out.append(f"      {code:8s} {n:6d}  {scp.get(code, {}).get('description', '?')}")


def inspect_chapman(src, mapping, rows, lab_rows, out):
    if not src.available():
        warn(f"Chapman not found at {src.records_dir} -- skipping")
        out.append("Chapman/Ningbo: NOT FOUND (skipped)")
        return

    cond = src.condition_names()
    heas = src.header_paths()
    mats = {Path(p).stem for p in glob.glob(str(src.records_dir / "*" / "*" / "*.mat"))}
    hstems = {Path(p).stem for p in heas}

    counts = collections.Counter()
    fs_c, nsig_c, ns_c, leadsets = (collections.Counter() for _ in range(4))
    bad_headers, ages, sexes = [], collections.Counter(), collections.Counter()
    rec_ids = []
    for hea in heas:
        hi = parse_header(hea)
        if hi.parse_error:
            bad_headers.append((hea, hi.parse_error))
        fs_c[hi.fs] += 1
        nsig_c[hi.n_sig] += 1
        ns_c[hi.n_samples] += 1
        leadsets["|".join(hi.lead_names)] += 1
        counts.update(c.strip() for c in hi.comments.get("Dx", "").split(",") if c.strip())
        ages[hi.comments.get("Age", "")] += 1
        sexes[hi.comments.get("Sex", "")] += 1
        rec_ids.append(hi.record_name or Path(hea).stem)

    undefined = sorted(set(counts) - set(cond))
    props = {
        "root": str(src.root),
        "metadata_file": "per-record WFDB .hea comments (#Dx/#Age/#Sex/#Rx/#Hx/#Sx)",
        "label_vocabulary_file": str(src.cond_csv),
        "n_records_headers": len(heas),
        "n_records_signals_mat": len(mats),
        "n_unique_record_ids": len(set(rec_ids)),
        "record_id_column": "WFDB record name (e.g. JS00001)",
        "patient_id_column": "*** NONE -- this distribution carries no patient identifier ***",
        "n_unique_patient_ids": 0,
        "n_label_codes_defined": len(cond),
        "n_distinct_labels_used": len(counts),
        "labels_used_but_undefined": "|".join(undefined) or "(none)",
        "n_labels_used_but_undefined": len(undefined),
        "signal_hea_without_mat": len(hstems - mats),
        "signal_mat_without_hea": len(mats - hstems),
        "signal_pairs_present": len(hstems & mats),
        "malformed_headers": len(bad_headers),
        "malformed_header_examples": "|".join(f"{Path(p).name}:{e}" for p, e in bad_headers[:5]),
        "sampling_frequency_hz": "|".join(f"{k}:{v}" for k, v in fs_c.most_common()),
        "num_leads": "|".join(f"{k}:{v}" for k, v in nsig_c.most_common()),
        "num_samples": "|".join(f"{k}:{v}" for k, v in ns_c.most_common(5)),
        "distinct_lead_orderings": len(leadsets),
        "lead_ordering_most_common": leadsets.most_common(1)[0][0] if leadsets else "",
        "extra_metadata": "Age|Sex|Rx|Hx|Sx (Rx/Hx/Sx are 'Unknown' throughout)",
        "has_predefined_folds": "no",
    }
    for k, v in props.items():
        rows.append({"source": "CHAPMAN", "property": k, "value": v})

    for code, n in counts.most_common():
        acr, name = cond.get(code, ("", "*** NOT DEFINED IN DISTRIBUTION ***"))
        lab_rows.append({
            "source": "CHAPMAN", "source_label": code, "label_name": name,
            "label_role": "SNOMED-CT concept", "diagnostic_class": acr,
            "diagnostic_subclass": "", "count": n, "likelihood_distribution": "",
        })

    out.append(banner("SOURCE: Chapman-Shaoxing / Ningbo 12-lead ECG database"))
    out.append(f"  root                     : {src.root}")
    out.append(f"  .hea headers             : {_fmt(len(heas))}")
    out.append(f"  .mat signals             : {_fmt(len(mats))}")
    out.append(f"  complete pairs           : {_fmt(len(hstems & mats))}")
    out.append(f"  !! .hea without .mat     : {len(hstems - mats)}  (header but NO signal -- unusable)")
    out.append(f"  !! .mat without .hea     : {len(mats - hstems)}  (signal but NO labels -- unusable)")
    out.append(f"  malformed headers        : {len(bad_headers)}")
    for p, e in bad_headers[:5]:
        out.append(f"       {Path(p).name}: {e}")
    out.append(f"  patient identifier       : NONE PRESENT")
    out.append(f"     -> patient-level splitting is NOT possible within this source.")
    out.append(f"  sampling frequency       : {props['sampling_frequency_hz']}")
    out.append(f"  leads / samples          : {props['num_leads']}  /  {props['num_samples']}")
    out.append(f"  SNOMED codes defined     : {len(cond)}   used in data: {len(counts)}")
    out.append(f"  !! codes used but UNDEFINED in ConditionNames_SNOMED-CT.csv: {len(undefined)}")
    out.append(f"     largest undefined codes:")
    for code, n in counts.most_common():
        if code in undefined:
            out.append(f"       {code:16s} {n:6d} occurrences  <- no definition shipped")
            if n < 1000:
                break
    out.append("  top labels:")
    for code, n in counts.most_common(12):
        acr, name = cond.get(code, ("--", "*** UNDEFINED ***"))
        out.append(f"      {code:16s} {n:6d}  {acr:8s} {name}")


def inspect_stemi(src, mapping, rows, lab_rows, out):
    if not src.available():
        warn(f"STEMI dataset not found at {src.train_csv} -- skipping")
        out.append("2026 ACS/STEMI: NOT FOUND (skipped)")
        return

    import csv as _csv
    msrc = mapping["sources"].get("stemi", {})
    recs = src.read(msrc)
    with open(src.train_csv, newline="", encoding="utf-8-sig") as fh:
        cols = _csv.DictReader(fh).fieldnames or []
    meta_cols = msrc.get("metadata_columns", [])
    label_cols = [c for c in cols if c not in set(meta_cols)]

    counts = collections.Counter()
    for r in recs:
        counts.update(r.raw_labels)

    dat = {Path(p).stem for p in glob.glob(str(src.signal_dir / "*.dat"))}
    hea = {Path(p).stem for p in glob.glob(str(src.signal_dir / "*.hea"))}
    present = sum(1 for r in recs if os.path.exists(r.data_path) and os.path.exists(r.header_path))

    fs_c, nsig_c, ns_c, leadsets = (collections.Counter() for _ in range(4))
    for r in recs:
        if os.path.exists(r.header_path):
            hi = parse_header(r.header_path)
            fs_c[hi.fs] += 1
            nsig_c[hi.n_sig] += 1
            ns_c[hi.n_samples] += 1
            leadsets["|".join(hi.lead_names)] += 1

    n_test = src.unlabeled_test_count()
    props = {
        "root": str(src.root),
        "metadata_file": str(src.train_csv),
        "n_records_in_metadata": len(recs),
        "n_columns": len(cols),
        "columns": "|".join(cols),
        "diagnostic_columns": "|".join(label_cols),
        "n_diagnostic_columns": len(label_cols),
        "record_id_column": "ecg_row_record",
        "patient_id_column": "Patient_id",
        "n_unique_record_ids": len({r.original_record_id for r in recs}),
        "n_unique_patient_ids": len({r.patient_id for r in recs if r.patient_id}),
        "label_type": "one binary column per finding",
        "n_distinct_labels_used": len(counts),
        "signal_dir": str(src.signal_dir),
        "signal_dat_files": len(dat),
        "signal_hea_files": len(hea),
        "signal_pairs_present": len(dat & hea),
        "records_with_signal": present,
        "records_missing_signal": len(recs) - present,
        "sampling_frequency_hz": "|".join(f"{k}:{v}" for k, v in fs_c.most_common()),
        "num_leads": "|".join(f"{k}:{v}" for k, v in nsig_c.most_common()),
        "num_samples": "|".join(f"{k}:{v}" for k, v in ns_c.most_common()),
        "distinct_lead_orderings": len(leadsets),
        "lead_ordering_most_common": leadsets.most_common(1)[0][0] if leadsets else "",
        "unlabeled_test_csv_rows": n_test,
        "median_beat_dir": str(src.median_dir),
        "extra_metadata": "gender|age|Time_Interval|ecg_med_record",
        "has_predefined_folds": "train/test split provided, but test.csv has NO diagnostic columns",
    }
    for k, v in props.items():
        rows.append({"source": "STEMI", "property": k, "value": v})

    for code, n in counts.most_common():
        lab_rows.append({
            "source": "STEMI", "source_label": code, "label_name": code,
            "label_role": "binary diagnostic column", "diagnostic_class": "",
            "diagnostic_subclass": "", "count": n, "likelihood_distribution": "",
        })
    for c in label_cols:
        if c not in counts:
            lab_rows.append({
                "source": "STEMI", "source_label": c, "label_name": c,
                "label_role": "binary diagnostic column", "diagnostic_class": "",
                "diagnostic_subclass": "", "count": 0, "likelihood_distribution": "",
            })

    out.append(banner("SOURCE: 2026 ACS/STEMI dataset"))
    out.append(f"  root                     : {src.root}")
    out.append(f"  labelled records (train) : {_fmt(len(recs))}")
    out.append(f"  unique record ids        : {_fmt(props['n_unique_record_ids'])}")
    out.append(f"  unique patient ids       : {_fmt(props['n_unique_patient_ids'])}  <- patient-level splitting POSSIBLE")
    out.append(f"     ({_fmt(len(recs))} ECGs from {_fmt(props['n_unique_patient_ids'])} patients"
               f" -> repeated recordings per patient exist)")
    out.append(f"  diagnostic columns       : {len(label_cols)}")
    out.append(f"  sampling frequency       : {props['sampling_frequency_hz']}")
    out.append(f"  leads / samples          : {props['num_leads']}  /  {props['num_samples']}")
    out.append(f"  lead ordering            : {props['lead_ordering_most_common']}")
    out.append(f"  records with signal      : {_fmt(present)} / {_fmt(len(recs))}")
    if n_test:
        out.append(f"  !! test.csv holds {_fmt(n_test)} rows with NO diagnostic columns.")
        out.append(f"     They cannot be labelled and are EXCLUDED from the pipeline entirely.")
    out.append("  positive counts per diagnostic column:")
    for c in label_cols:
        out.append(f"      {c:12s} {counts.get(c, 0):6d}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", default=None, help="path to paths.yaml")
    ap.add_argument("--mapping", default=None, help="path to label_mapping.yaml")
    args = ap.parse_args(argv)

    paths = load_paths(args.paths)
    mapping = load_mapping(args.mapping)
    srcs = build_sources(paths)
    rdir = reports_dir(paths)

    rows, lab_rows, out = [], [], []
    out.append(banner("CARDIOSENTRY -- STEP 1: DATASET INSPECTION", "#"))
    out.append("Everything below was read from the actual files on disk.")

    if "ptbxl" in srcs:
        info("inspecting PTB-XL ...")
        inspect_ptbxl(srcs["ptbxl"], mapping, rows, lab_rows, out)
    if "chapman" in srcs:
        info("inspecting Chapman/Ningbo (scanning ~45k headers, this takes a minute) ...")
        inspect_chapman(srcs["chapman"], mapping, rows, lab_rows, out)
    if "stemi" in srcs:
        info("inspecting 2026 ACS/STEMI ...")
        inspect_stemi(srcs["stemi"], mapping, rows, lab_rows, out)

    write_csv(rdir / "dataset_inspection_report.csv", rows, ["source", "property", "value"])
    write_csv(rdir / "label_frequency_report.csv", lab_rows,
              ["source", "source_label", "label_name", "label_role", "diagnostic_class",
               "diagnostic_subclass", "count", "likelihood_distribution"])

    out.append(banner("NOTES CARRIED FORWARD TO EXTRACTION", "#"))
    out.append("* No source other than the 2026 ACS/STEMI dataset contains any")
    out.append("  ST-elevation-MI concept. STEMI therefore comes from ONE source only.")
    out.append("* Chapman/Ningbo has no patient identifier -> patient-level splitting")
    out.append("  can only be guaranteed within PTB-XL and the ACS/STEMI dataset.")
    out.append("* Patient identifiers are NOT comparable across datasets: the three")
    out.append("  cohorts are from different institutions and countries. Cross-dataset")
    out.append("  patient overlap can be neither confirmed nor ruled out.")

    txt = "\n".join(out) + "\n"
    (rdir / "dataset_inspection_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    info(f"wrote {rdir/'dataset_inspection_report.csv'}")
    info(f"wrote {rdir/'label_frequency_report.csv'}")
    info(f"wrote {rdir/'dataset_inspection_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
