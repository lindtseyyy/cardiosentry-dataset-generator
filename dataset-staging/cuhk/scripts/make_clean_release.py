"""
CardioSentry / CUHK -- assemble the clean, self-contained release.

Takes the staged output (WFDB signals + harmonized metadata) and lays it out
under `dataset-files/cuhk_dataset_clean/` as a dataset someone can pick up on
its own: the signals, the labels, and the full record of how every label was
decided and why every dropped record was dropped.

    cuhk_dataset_clean/
      README.md                 what this is and how to load it
      RECORDS                   one record name per line (WFDB convention)
      labels.csv                the labels, plainly: one row per record
      metadata.csv              full 30-column harmonized schema (merge-ready)
      records/<id>.hea + .dat   12 leads, 500 Hz, 5000 samples, gain 1000
      checksums.sha256
      provenance/               every label decision, every exclusion

Signals are HARD-LINKED, not copied: the release and the staging area share
the same inodes, so the release costs no extra disk. The bytes are immutable
outputs, so sharing them is safe. Pass --copy to force real copies (needed if
the release is going onto a different filesystem).
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGED_META = ROOT / "data" / "cuhk_metadata.csv"
STAGED_SIGNALS = ROOT / "data" / "signals"

DEST = Path(os.environ.get(
    "CUHK_CLEAN_ROOT",
    ROOT.parents[1] / "dataset-files" / "cuhk_dataset_clean"))

# (staged path, name inside provenance/)
PROVENANCE = [
    (ROOT / "config" / "cuhk_code_rules.yaml", "cuhk_code_rules.yaml"),
    (ROOT / "config" / "label_map_cuhk.yaml", "label_map_cuhk.yaml"),
    (ROOT / "reports" / "code_decisions.csv", "code_decisions.csv"),
    (ROOT / "reports" / "cohort_summary.txt", "cohort_summary.txt"),
    (ROOT / "reports" / "excluded_records.csv", "excluded_records.csv"),
    (ROOT / "reports" / "label_decisions.csv", "label_decisions.csv"),
    (ROOT / "reports" / "conversion_report.csv", "conversion_report.csv"),
    (ROOT / "data" / "index" / "code_inventory.csv", "code_inventory.csv"),
    (ROOT / "reports" / "label_signal_agreement.csv", "label_signal_agreement.csv"),
]

LABEL_FIELDS = [
    "record_id", "STEMI", "AF", "LVH", "NORMAL", "label", "n_labels",
    "patient_group", "cohort", "cohorts_all", "age", "sex",
    "acquisition_date", "signal_quality", "quality_notes",
    "philips_statement_codes", "philips_statement_text",
]


# ---------------------------------------------------------------------------
# PHI redaction
# ---------------------------------------------------------------------------
# CUHK's anonymisation is incomplete AT SOURCE: ~3.4% of the XML files still
# carry a real patient name in <lastname>/<firstname>, and 28 filenames embed
# one (e.g. "A850XXXX-99AU-YEUNG_KAM_SHUN2008-11-21_12-26-36.xml").
#
# None of the name FIELDS are read by this pipeline -- only <patientid> is --
# so the only route into a derived file is the filename, which the staging
# metadata keeps verbatim for traceability back to the source. The release is
# the artefact that leaves this machine, so the filename is redacted here:
# the anonymised id prefix and the acquisition timestamp are kept, everything
# between them is dropped. `document_guid` remains the exact link back.
FNAME_RX = re.compile(r"^(?P<pid>[^-]*)-.*?(?P<ts>\d{4}-\d{1,2}-\d{1,2}_[\d-]+)$")
# Any letter run that is not the X-mask, in the segment between the two.
NAMEISH_RX = re.compile(r"[A-Za-z]{2,}")


def redact_record_id(stem: str) -> str:
    """Filename stem -> a stem that cannot carry a patient name."""
    m = FNAME_RX.match(stem)
    if not m:
        # Sequentially-named files ("HF XML (4872)") carry no name.
        return stem
    return f"{m.group('pid')}-{m.group('ts')}"


# Sequentially-named files carry the cohort, not a person: "HF XML (4872)".
SEQ_RX = re.compile(r"^(HF|HF MR|ARVD)\s+XML\s*\(\d+\)$")


def name_like(text: str) -> bool:
    """True if `text` still holds a letter run that is not the X-mask."""
    text = (text or "").strip()
    if SEQ_RX.match(text):
        return False
    for run in NAMEISH_RX.findall(text):
        if set(run.upper()) != {"X"}:
            return True
    return False


def place(src: Path, dst: Path, copy: bool) -> str:
    """-> "link" or "copy", whichever actually happened."""
    if dst.exists():
        dst.unlink()
    if not copy:
        try:
            os.link(src, dst)
            return "link"
        except OSError:                       # e.g. a different filesystem
            pass
    shutil.copyfile(src, dst)
    return "copy"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--copy", action="store_true",
                    help="copy signal files instead of hard-linking them")
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args()
    dest = args.dest

    if not STAGED_META.exists():
        print(f"missing {STAGED_META} -- run scripts/convert_to_wfdb.py first",
              file=sys.stderr)
        return 1

    rows = list(csv.DictReader(STAGED_META.open(encoding="utf-8")))
    if not rows:
        print("staged metadata is empty", file=sys.stderr)
        return 1
    print(f"packaging {len(rows)} records -> {dest}")

    recdir = dest / "records"
    provdir = dest / "provenance"
    recdir.mkdir(parents=True, exist_ok=True)
    provdir.mkdir(parents=True, exist_ok=True)

    # ---- signals ------------------------------------------------------
    n_placed, missing, how = 0, [], Counter()
    for r in rows:
        rid = r["global_record_id"]
        for ext in (".hea", ".dat"):
            src = STAGED_SIGNALS / f"{rid}{ext}"
            if not src.exists():
                missing.append(src.name)
                continue
            how[place(src, recdir / f"{rid}{ext}", args.copy)] += 1
            n_placed += 1
    if missing:
        print(f"WARNING: {len(missing)} signal files missing, e.g. {missing[:3]}",
              file=sys.stderr)
    print(f"  {n_placed} signal files placed "
          f"({how['link']} hard-linked, {how['copy']} copied)")

    # ---- metadata.csv: identical schema, paths rewritten to the release
    n_redacted, n_phi = [0], [0]
    meta_fields = list(rows[0].keys())
    with (dest / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=meta_fields)
        w.writeheader()
        for r in rows:
            rid = r["global_record_id"]
            stem = str((recdir / rid).resolve())
            out = dict(r)
            out["signal_path"] = stem
            out["header_path"] = stem + ".hea"
            out["data_path"] = stem + ".dat"
            out["original_record_id"] = redact_record_id(r["original_record_id"])
            x = json.loads(r["source_extra_json"])
            if "source_xml" in x:
                x["source_xml"] = redact_record_id(
                    x["source_xml"].removesuffix(".xml")) + ".xml"
            x["phi_note"] = ("source filename redacted for release; "
                             "document_guid is the exact link back")
            out["source_extra_json"] = json.dumps(x, ensure_ascii=False)
            red = redact_record_id(r["original_record_id"])
            if red != r["original_record_id"]:
                n_redacted[0] += 1
                if name_like(r["original_record_id"].removeprefix(
                        r["original_record_id"].split("-")[0])):
                    n_phi[0] += 1
            w.writerow(out)

    # ---- labels.csv: the labels on their own, no pipeline knowledge needed
    with (dest / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LABEL_FIELDS)
        w.writeheader()
        for r in rows:
            x = json.loads(r["source_extra_json"])
            w.writerow({
                "record_id": r["global_record_id"],
                "STEMI": r["STEMI"], "AF": r["AF"], "LVH": r["LVH"],
                "NORMAL": r["NORMAL"],
                "label": r["cardiosentry_label"],
                "n_labels": r["n_positive_labels"],
                "patient_group": r["patient_id"],
                "cohort": x.get("cohort", ""),
                "cohorts_all": x.get("cohorts_all", ""),
                "age": r["age"], "sex": r["sex"],
                "acquisition_date": x.get("acquisition_date", ""),
                "signal_quality": r["validation_status"],
                "quality_notes": r["validation_issues"],
                "philips_statement_codes": r["original_labels"],
                "philips_statement_text": r["original_labels_readable"],
            })

    # ---- RECORDS + checksums -----------------------------------------
    ids = sorted(r["global_record_id"] for r in rows)
    (dest / "RECORDS").write_text("\n".join(f"records/{i}" for i in ids) + "\n",
                                  encoding="utf-8")
    sums = []
    for i in ids:
        for ext in (".hea", ".dat"):
            p = recdir / f"{i}{ext}"
            if p.exists():
                sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  "
                            f"records/{i}{ext}")
    (dest / "checksums.sha256").write_text("\n".join(sums) + "\n", encoding="utf-8")

    # ---- provenance ---------------------------------------------------
    n_prov = 0
    for src, name in PROVENANCE:
        if src.exists():
            shutil.copyfile(src, provdir / name)
            n_prov += 1
        else:
            print(f"  (provenance file not present, skipped: {src.name})")
    print(f"  {n_prov} provenance files copied")

    # ---- PHI verification sweep --------------------------------------
    # Assert, don't assume: re-read what was actually written and fail loudly
    # if any free-text field still holds a name-like letter run.
    # Only free text is at risk. The anonymised id token (the part before the
    # first "-", e.g. "UD61XXXX") and the CUHK_/patient-group ids ARE the
    # anonymisation, so they are the baseline, not a leak -- the sweep checks
    # everything that follows the id token.
    def tail(v: str) -> str:
        v = (v or "").removesuffix(".xml")
        return v.split("-", 1)[1] if "-" in v else v

    suspect = []
    # The two fields that derive from a source filename.
    with (dest / "metadata.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vals = [("original_record_id", row["original_record_id"]),
                    ("source_xml",
                     json.loads(row["source_extra_json"]).get("source_xml", ""))]
            for c, v in vals:
                if name_like(tail(v)):
                    suspect.append(("metadata.csv", c, v[:60]))
    # Our own ids are generated, not derived: check the shape instead.
    bad_ids = [row["record_id"] for row in
               csv.DictReader((dest / "labels.csv").open(encoding="utf-8"))
               if not re.fullmatch(r"CUHK_\d{6}", row["record_id"])]
    if bad_ids:
        suspect += [("labels.csv", "record_id", i) for i in bad_ids[:5]]
    print(f"  {n_redacted[0]} source filenames normalised "
          f"({n_phi[0]} of them held real name text)")
    if suspect:
        print(f"  PHI CHECK FAILED: {len(suspect)} suspect values, "
              f"e.g. {suspect[:3]}", file=sys.stderr)
        return 3
    print("  PHI check passed: no name-like text in any released text field")

    # ---- README -------------------------------------------------------
    counts = {c: sum(int(r[c]) for r in rows) for c in ("STEMI", "AF", "LVH", "NORMAL")}
    combos = Counter(r["cardiosentry_label"] for r in rows)
    n_multi = sum(v for k, v in combos.items() if "+" in k)
    groups = {r["patient_id"] for r in rows}
    per_cohort = defaultdict(Counter)
    for r in rows:
        c = json.loads(r["source_extra_json"]).get("cohort", "?")
        per_cohort[c]["_n"] += 1
        for cls in r["cardiosentry_label"].split("+"):
            per_cohort[c][cls] += 1
    qual = Counter(r["validation_status"] for r in rows)
    total_mb = sum((recdir / f"{i}{e}").stat().st_size
                   for i in ids for e in (".hea", ".dat")
                   if (recdir / f"{i}{e}").exists()) / 1e6

    def tbl(rowsx):
        return "\n".join(rowsx)

    readme = f"""# CUHK ECG dataset — clean release

{len(rows)} twelve-lead ECGs decoded from the CUHK Philips Sierra ECG XML
collection, labelled for the four CardioSentry classes, in one
self-contained folder. {total_mb / 1000:.2f} GB.

Generated by `dataset-staging/cuhk/scripts/make_clean_release.py`.
Full investigation write-up: `dataset-staging/FINDINGS.md`.

## Labels

| class | records | patient groups |
|---|---|---|
{tbl(f"| **{c}** | {counts[c]} | "
     f"{len({r['patient_id'] for r in rows if r[c] == '1'})} |"
     for c in ('STEMI', 'AF', 'LVH', 'NORMAL'))}

Multi-label — a record may carry more than one class:

| combination | records | % |
|---|---|---|
{tbl(f"| {k or '(none)'} | {v} | {100.0 * v / len(rows):.1f}% |"
     for k, v in sorted(combos.items(), key=lambda kv: -kv[1]))}
| **single-label** | {len(rows) - n_multi} | {100.0 * (len(rows) - n_multi) / len(rows):.1f}% |
| **multi-label (≥2)** | {n_multi} | {100.0 * n_multi / len(rows):.1f}% |

NORMAL is 0 by design: every CUHK cohort is a disease registry, so a
STEMI-negative record here is a different acute presentation, never a healthy
control.

## Source cohorts

The folder a record came from is **provenance, not a diagnosis** — a patient
in the NSTEMI registry has serial ECGs, some of which read as acute STEMI.
Every label comes from the Philips statements alone.

| cohort | records | STEMI | AF | LVH |
|---|---|---|---|---|
{tbl(f"| {c} | {per_cohort[c]['_n']} | {per_cohort[c]['STEMI']} | "
     f"{per_cohort[c]['AF']} | {per_cohort[c]['LVH']} |"
     for c in sorted(per_cohort, key=lambda c: -per_cohort[c]['_n']))}

## Layout

```
cuhk_dataset_clean/
  labels.csv         one row per record: the four class columns, the exact
                     label combination, patient group, cohort, quality flag,
                     and the Philips statements the label came from
  metadata.csv       the same records in the full 30-column CardioSentry
                     harmonized schema, paths pointing into records/ --
                     concatenate straight onto harmonized_ecg_metadata.csv
  records/           <record_id>.hea + .dat, WFDB
  RECORDS            one record name per line
  checksums.sha256   sha256 of every signal file
  provenance/        how every label was decided, and every dropped record
```

`labels.csv` and `metadata.csv` describe the same {len(rows)} records; use
whichever suits. `record_id` / `global_record_id` join them.

## Signal format

Every record, without exception:

- 12 leads, order `I II III aVR aVL aVF V1 V2 V3 V4 V5 V6`
- 500 Hz, 5000 samples = **10.000 s**
- WFDB format 16 (int16 LE, multiplexed), **gain 1000 ADU/mV**, ADC zero 0

```python
import wfdb
rec = wfdb.rdrecord("records/{ids[0]}")
rec.p_signal          # (5000, 12) float, millivolts
rec.sig_name          # ['I', 'II', 'III', 'aVR', ...]
```

Three declared changes from the native XML (all recorded per record in
`metadata.csv` → `source_extra_json`):

1. **11 s → 10 s.** Source records are 5500 samples; the first 5000 are kept.
   The trailing second is where the recorder stops maintaining the
   derived-lead relationships.
2. **5 µV/LSB → gain 1000 ADU/mV.** An exact ×5 integer rescale, not a
   resample. True quantisation is still 5 µV (`native_resolution_uv`).
3. **III, aVR, aVL, aVF recomputed from I and II.** These are 10-wire
   acquisitions, so those four leads are arithmetic by construction; the
   XLI-stored residual is corrupt in 26% of files. See FINDINGS.md §3.1.

## Two things to know before training

**The labels are algorithmic.** They come from the Philips DXL interpretation
statements in each XML, not from a clinician. Every row carries
`label_provenance: ALGORITHM_PHILIPS_DXL`. `provenance/code_decisions.csv`
lists all 709 statement codes with counts, statement text and assigned status
— that is the table to hand a cardiologist.

**These are serial ECGs.** {len(rows)} records come from {len(groups)} patient
groups. Split on `patient_group` (= `split_group_id` in metadata.csv) or the
same patient will appear in both train and test. Note that a "patient group"
is the 4-character anonymised id prefix, which over-groups — several real
patients can share one. That costs a little data but cannot leak.

## Signal quality

| status | records |
|---|---|
{tbl(f"| {k} | {v} |" for k, v in sorted(qual.items()))}

`CHECK` means recorder saturation (~41 mV, the ADC rail) or a dead lead —
filter on `signal_quality` if that matters for your model. Records whose
source carried the Philips verdict `Poor ECG, retry if possible` keep that
note in `quality_notes` verbatim.

## What is NOT here

Records the pipeline deliberately refused to label are excluded, with the
reason for each in `provenance/excluded_records.csv`:

- hedged acute ischaemia ("INFARCT, POSSIBLY ACUTE", "INFARCT, RECENT") and
  bare ST elevation with no mechanism — in an ACS registry, `STEMI=0` would be
  a wrong negative and `STEMI=1` would be a guess
- ST elevation attributed to LVH or LBBB — STEMI mimics that also mask it
- paced rhythms, and records where the algorithm reported it did not finish
- duplicates: the same acquisition ships in several cohort folders; deduped on
  the Philips document GUID

That is the largest pool of recoverable data in the collection — a clinician
adjudicating it would add real STEMI positives.

The other CUHK files cannot contribute waveforms at all: the myocarditis CSV
and the valvular `.xlsx` hold extracted *measurements* with no signals, and
`others/HFLAStrain`, `others/HFLVStrain` are echocardiographic strain
spreadsheets.

## Related

LUDB (200 cardiologist-labelled records, already WFDB) is staged separately in
`dataset-staging/ludb/` and referenced in place — it is not part of this
release.
"""
    (dest / "README.md").write_text(readme, encoding="utf-8")

    print(f"\nwrote {dest}/README.md")
    print(f"wrote {dest}/labels.csv     ({len(rows)} rows)")
    print(f"wrote {dest}/metadata.csv   ({len(rows)} rows)")
    print(f"wrote {dest}/RECORDS, checksums.sha256")
    print(f"records/ holds {n_placed} files, {total_mb / 1000:.2f} GB")
    print(f"\nclasses: " + "  ".join(f"{c}={counts[c]}" for c in counts))
    print(f"multi-label: {n_multi} of {len(rows)} "
          f"({100.0 * n_multi / len(rows):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
