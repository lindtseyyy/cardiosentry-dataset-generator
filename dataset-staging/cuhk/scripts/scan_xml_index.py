"""
CardioSentry / CUHK-STEMI staging -- PASS 1: inventory scan.

Reads every Philips Sierra ECG XML in the CUHK dataset WITHOUT decoding the
XLI-compressed waveform (fast: regex over the raw UTF-16 bytes) and writes:

  data/index/xml_index.csv        one row per XML: ids, acquisition params,
                                  Philips statement codes, global measurements
  data/index/code_inventory.csv   statement-code frequency table, for building
                                  the explicit label mapping

Nothing here decides what a label MEANS -- that is config/label_map_cuhk.yaml.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUHK = Path(os.environ.get(
    "CUHK_ROOT",
    ROOT.parents[1] / "dataset-files" / "cuhk_dataset"))

# The STEMI cohort: three sites, one folder each.
STEMI_SUBDIRS = ["STEMI XML CE", "STEMI XML AB", "STEMI XML GZ"]

# Everything else CUHK ships under `others/`. Same Philips Sierra ECG XML,
# same acquisition -- different referral cohorts. The folder name is the ONLY
# statement of what cohort a record came from; it is recorded per record as
# `cohort` and is NOT used as a diagnosis (see README).
OTHERS_DIR = "others"

OUT_INDEX = ROOT / "data" / "index" / "xml_index.csv"
OUT_CODES = ROOT / "data" / "index" / "code_inventory.csv"


def discover(root: Path, subdirs: list | None, include_others: bool) -> list:
    """-> [(relative subdir, filename)] for every XML to scan."""
    jobs = []
    for sub in (subdirs or []):
        d = root / sub
        if not d.is_dir():
            print(f"missing: {d}", file=sys.stderr)
            continue
        jobs += [(sub, f) for f in sorted(os.listdir(d)) if f.lower().endswith(".xml")]
    if include_others:
        od = root / OTHERS_DIR
        if od.is_dir():
            for sub in sorted(os.listdir(od)):
                d = od / sub
                if not d.is_dir():
                    continue
                names = [f for f in sorted(os.listdir(d)) if f.lower().endswith(".xml")]
                if names:
                    jobs += [(f"{OTHERS_DIR}/{sub}", f) for f in names]
                else:
                    print(f"  (skipped {OTHERS_DIR}/{sub}: no XML files)")
    return jobs


def u16(s: str) -> bytes:
    return s.encode("utf-16-le")


def _rx(pattern: str, flags=re.S) -> re.Pattern:
    """Compile a regex whose literal parts are UTF-16LE encoded."""
    return re.compile(pattern.encode("utf-16-le"), flags)


# Element text: <tag>value</tag>  (UTF-16LE aware, built per tag)
_TEXT_CACHE: dict[str, re.Pattern] = {}


def text_of(buf: bytes, tag: str) -> str:
    rx = _TEXT_CACHE.get(tag)
    if rx is None:
        rx = re.compile(u16("<" + tag + ">") + b"(.{0,400}?)" + u16("</" + tag + ">"), re.S)
        _TEXT_CACHE[tag] = rx
    m = rx.search(buf)
    return m.group(1).decode("utf-16-le", "replace").strip() if m else ""


ATTR_RX = re.compile(r'([\w:]+)="([^"]*)"')
WAVE_RX = re.compile(u16("<parsedwaveforms") + b"(.{0,2000}?)" + u16(">"), re.S)
STMT_RX = re.compile(u16("<unparsedstatement") + b"(.{0,6000}?)" + u16("</unparsedstatement>"), re.S)
ACQ_RX = re.compile(u16("<dataacquisition") + b"(.{0,400}?)" + u16(">"), re.S)
SEVERITY_RX = re.compile(u16("<severity>") + b"(.{0,200}?)" + u16("</severity>"), re.S)


def scan_one(args) -> dict:
    subdir, fname = args
    path = CUHK / subdir / fname
    row = {"cohort": cohort_of(subdir), "site": site_of(subdir),
           "source_folder": subdir, "xml_file": fname,
           "xml_path": str(path), "parse_error": ""}
    try:
        buf = path.read_bytes()
    except OSError as exc:
        row["parse_error"] = f"read: {exc}"
        return row

    row["file_bytes"] = len(buf)

    m = WAVE_RX.search(buf)
    if m:
        a = dict(ATTR_RX.findall(m.group(1).decode("utf-16-le", "replace")))
        for k_out, k_in in [("sampling_frequency", "samplespersecond"),
                            ("duration_ms", "durationperchannel"),
                            ("num_leads", "numberofleads"),
                            ("lead_labels", "leadlabels"),
                            ("resolution_uv", "resolution"),
                            ("bits_per_sample", "bitspersample"),
                            ("compression", "compression"),
                            ("data_encoding", "dataencoding"),
                            ("signal_signed", "signalsigned"),
                            ("signal_offset", "signaloffset")]:
            row[k_out] = a.get(k_in, "")
    else:
        row["parse_error"] = "no <parsedwaveforms>"

    m = ACQ_RX.search(buf)
    if m:
        a = dict(ATTR_RX.findall(m.group(1).decode("utf-16-le", "replace")))
        row["acquisition_date"] = a.get("date", "")
        row["acquisition_time"] = a.get("time", "")

    row["patient_id"] = text_of(buf, "patientid")
    # The Philips document GUID. Unlike (patient, date, time) this is an exact
    # identity for one acquisition, so it is what de-duplication keys on: CUHK
    # ships the same ECG in several cohort folders when a patient belongs to
    # more than one registry.
    row["document_guid"] = text_of(buf, "documentname")
    row["doc_version"] = text_of(buf, "documentversion")
    row["machine_desc"] = text_of(buf, "machine") or ""
    row["acquisition_type"] = text_of(buf, "acquisitiontype")
    row["electrode_placement"] = text_of(buf, "electrodeplacement")
    row["sex"] = text_of(buf, "sex")
    row["pace_status"] = text_of(buf, "pacestatus")
    row["qa_action_code"] = text_of(buf, "qaactioncode")
    row["heartrate"] = text_of(buf, "heartrate")
    row["print"] = text_of(buf, "print")
    row["qrsdur"] = text_of(buf, "qrsdur")
    row["qtint"] = text_of(buf, "qtint")
    row["qtcb"] = text_of(buf, "qtcb")
    row["severity"] = SEVERITY_RX.search(buf).group(1).decode("utf-16-le", "replace").strip() \
        if SEVERITY_RX.search(buf) else ""

    # <age defaultage="50"> means the true age was NOT entered.
    m = re.search(u16("<age") + b"(.{0,120}?)" + u16(">"), buf, re.S)
    age_attr = dict(ATTR_RX.findall(m.group(1).decode("utf-16-le", "replace"))) if m else {}
    row["age_default_used"] = "1" if "defaultage" in age_attr else "0"
    row["age_years"] = text_of(buf, "years") or age_attr.get("defaultage", "")

    # Statements are kept CODE-ALIGNED: one entry per <unparsedstatement>,
    # in document order, so `statement_codes[i]` always describes
    # `statement_text[i]`. A statement with no code is recorded as "?".
    codes, texts = [], []
    for sm in STMT_RX.finditer(buf):
        t = sm.group(1).decode("utf-16-le", "replace")
        a = dict(ATTR_RX.findall(t))
        codes.append(a.get("code", "").strip() or "?")
        lhs = re.search(r"<lhsstatement>(.*?)</lhsstatement>", t, re.S)
        rhs = re.search(r"<rhsstatement>(.*?)</rhsstatement>", t, re.S)
        parts = [re.sub(r"\s+", " ", g.group(1)).strip()
                 for g in (lhs, rhs) if g and g.group(1).strip()]
        texts.append(" ".join(parts).replace("|", "/").replace(";", ","))
    row["statement_codes"] = "|".join(codes)
    row["statement_text"] = "|".join(texts)[:2000]
    row["n_statements"] = len(codes)
    return row


# folder name -> (cohort, site). The cohort is provenance, not a label.
COHORTS = [
    ("STEMI XML ",                       "STEMI"),
    ("others/NSTEMI XML",                "NSTEMI"),
    ("others/HF_Additional_",            "HF"),
    ("others/HF AS XML",                 "HF_AS"),
    ("others/Redacted Version 2",        "MIXED_REDACTED"),
    ("others/Peri-inflammation XML ",    "PERICARDITIS"),
    ("others/Myoinflammation XML",       "MYOCARDITIS"),
]


def cohort_of(subdir: str) -> str:
    for prefix, name in COHORTS:
        if subdir.startswith(prefix):
            return name
    return subdir.split("/")[-1]


def site_of(subdir: str) -> str:
    """The letter suffix CUHK uses to shard a cohort across folders."""
    leaf = subdir.split("/")[-1]
    for prefix, _ in COHORTS:
        p = prefix.split("/")[-1]
        if leaf.startswith(p) and len(leaf) > len(p):
            return leaf[len(p):].strip()
    return ""


FIELDS = ["cohort", "site", "source_folder", "xml_file", "document_guid",
          "patient_id", "acquisition_date", "acquisition_time",
          "sex", "age_years", "age_default_used", "pace_status",
          "sampling_frequency", "duration_ms", "num_leads", "lead_labels",
          "resolution_uv", "bits_per_sample", "compression", "data_encoding",
          "signal_signed", "signal_offset", "acquisition_type",
          "electrode_placement", "machine_desc", "doc_version",
          "qa_action_code", "severity", "heartrate", "print", "qrsdur",
          "qtint", "qtcb", "n_statements", "statement_codes", "statement_text",
          "file_bytes", "xml_path", "parse_error"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stemi-only", action="store_true",
                    help="scan only the three STEMI XML folders")
    ap.add_argument("--others-only", action="store_true",
                    help="scan only cuhk_dataset/others/*")
    args = ap.parse_args()

    jobs = discover(CUHK,
                    None if args.others_only else STEMI_SUBDIRS,
                    not args.stemi_only)
    print(f"scanning {len(jobs)} XML files from {CUHK}")
    from collections import Counter as _C
    for k, v in sorted(_C(cohort_of(s) for s, _ in jobs).items()):
        print(f"  {v:>6}  {k}")

    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    code_counts, code_text = Counter(), {}
    n = 0
    with open(OUT_INDEX, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        with ProcessPoolExecutor(max_workers=int(os.environ.get("NPROC", 4))) as ex:
            for row in ex.map(scan_one, jobs, chunksize=32):
                w.writerow(row)
                n += 1
                for c in (row.get("statement_codes") or "").split("|"):
                    if c:
                        code_counts[c] += 1
                if n % 2000 == 0:
                    print(f"  {n}/{len(jobs)}", flush=True)

    # second pass: attach example statement texts per code (codes and texts
    # are positionally aligned, so this pairing is exact).
    from collections import defaultdict
    code_examples = defaultdict(Counter)
    with open(OUT_INDEX, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cs = (row["statement_codes"] or "").split("|")
            ts = (row["statement_text"] or "").split("|")
            if len(cs) != len(ts):
                continue          # text was truncated at 2000 chars
            for c, t in zip(cs, ts):
                if c and t:
                    code_examples[c][t[:140]] += 1
    for c, ctr in code_examples.items():
        code_text[c] = " || ".join(t for t, _ in ctr.most_common(2))

    with open(OUT_CODES, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["philips_code", "n_records", "pct_of_records", "example_statement"])
        for c, k in code_counts.most_common():
            w.writerow([c, k, f"{100.0 * k / max(n, 1):.3f}", code_text.get(c, "")])

    print(f"wrote {OUT_INDEX} ({n} rows)")
    print(f"wrote {OUT_CODES} ({len(code_counts)} distinct codes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
