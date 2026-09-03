#!/usr/bin/env python3
"""
Stage 0 of the external validation build.

Turns two ECG catalogues into ONE CSV in the exact schema the existing image
pipeline consumes (`balanced_ecg_metadata.csv`), with `split` pinned to "test"
on every row.

WHY THIS EXISTS
---------------
The trained corpus carries a source confound the evaluation report could not
resolve: every one of the 1,442 STEMI positives came from the Chongqing ACS
set, and no other source contributed a single STEMI. Within that corpus the
STEMI head has no negatives from Chongqing and no positives from anywhere else,
so "recognises ST elevation" and "recognises the Chongqing scanner" predict the
identical confusion matrix. The question is not answerable with that data.

It becomes answerable the moment a STEMI positive arrives from a DIFFERENT
source. That is the entire purpose of this file.

THE TWO COHORTS, AND WHY THEY ARE NOT INTERCHANGEABLE
-----------------------------------------------------
  LUDB      Lobachevsky University Database, Nizhny Novgorod. 200 records,
            Schiller Cardiovit AT-101, 12-lead, 500 Hz, 10 s. Its cardiologist
            reports use the literal word "STEMI" with a wall, so the label is
            ASSERTED, not inferred. A source the models have never seen, in any
            class. This is the gold cohort. It is also small: 8 STEMI.

  PTBXL_MI  The 275 records hand-inspected by Gedon et al. and published in
            dgedon/nstemi_prediction as `data/ptbxl_selected.csv`
            (200 "normal" / 75 "mi").

            READ THIS BEFORE QUOTING ANY NUMBER FROM THIS COHORT. It is NOT a
            new database - it is PTB-XL, which is already a training source.
            Two consequences, in opposite directions:

              * IN ITS FAVOUR: PTB-XL contributed zero STEMI to training.
                label_mapping.yaml refuses PTB-XL's MI codes for STEMI because
                they encode infarct LOCATION, not ST elevation. So the models
                have seen this scanner extensively and have never once been
                told a PTB-XL sheet is STEMI. If a model fires here it cannot
                be firing on the paper.

              * AGAINST IT: the label is "mi", not "STEMI". All 75 are
                infarction_stadium1 == "Stadium I" (acute), which is the stage
                where ST elevation appears - but "acute MI" still admits
                NSTEMI. This is a PROXY. It is carried as STEMI=1 so the head
                can be scored, and every row is tagged so the report can keep
                it apart from LUDB. Do not merge the two into one sensitivity
                figure.

LEAKAGE
-------
58 of the 267 selected PTB-XL patients are in the balanced corpus the models
trained on. The corpus split on patient, so exclusion here is on PATIENT, not
on record id. Excluded rows are written out with a reason rather than dropped
silently, so the count is auditable.

TWO OUTPUT CSVs, AND WHY
------------------------
`stage1_manifest_from_csv.py` drops any record with no positive label
("skip_no_label"). For building a TRAINING corpus that is right - a record
belonging to no class has nothing to teach. For an EVALUATION corpus it is
actively harmful, because it deletes exactly the negatives worth having.

In LUDB it deletes 59 records, and they are the hardest STEMI negatives in the
entire external set:

    31   non-specific repolarization abnormalities  (ST-T change, not STEMI)
     8   early repolarization syndrome              (BENIGN ST ELEVATION - the
                                                     textbook STEMI mimic)
     8   undefined ischemia / scar / supp.NSTEMI
     7   scar formation
     5   III degree AV block, 13 IRBBB, 6 paced, ...

A model that has learnt "Chongqing paper" scores these correctly by accident. A
model that has learnt "abnormal-looking ECG" fails the repolarization and early-
repolarization records specifically. Losing them costs the most discriminating
comparison available.

So the build runs twice. The second pass writes these records to a separate CSV
with a SENTINEL NORMAL=1 purely so stage 1 will carry them, and `03_merge.py`
restores the true all-zero labels afterwards. This is safe to do because the
label columns do not reach the renderer: stage 1 uses them only for `cls` (a
directory and reporting name) and for `n_renders`, and config.RENDERS_PER_RECORD
is 1 for every class, so both passes render one sheet per record either way. The
pixels do not depend on the sentinel. `03_merge.py` re-derives every label from
this file rather than trusting either index.csv, so the sentinel cannot survive
into the evaluation set.

Nothing clinical is decided anywhere else in this folder. If you disagree with
a labelling call, it is in this file and only this file.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "dataset-files"
LUDB_DIR = DATASETS / "lobachevsky-university-electrocardiography-database-1.0.1" / "data"
PTBXL_DIR = DATASETS / "ptb-xl-dataset"
BALANCED = (REPO / "dataset-combination-code" / "ecg_dataset_pipeline"
            / "data" / "balanced" / "balanced_ecg_metadata.csv")
SELECTED = Path(__file__).resolve().parents[1] / "data" / "ptbxl_selected.csv"

LABELS = ["STEMI", "AF", "LVH", "NORMAL"]

# The schema `stage1_manifest_from_csv.py` reads, plus `split`. Column names
# come from image_pipeline/config.py:CSV_COLUMNS - do not rename them here.
FIELDS = [
    "global_record_id", "source_dataset", "original_record_id", "patient_id",
    "patient_id_is_synthetic", "split_group_id", "original_labels",
    "original_labels_readable", "accepted_label_mapping", "cardiosentry_label",
    *LABELS, "n_positive_labels", "sampling_frequency", "duration_seconds",
    "num_leads", "num_samples", "lead_names", "signal_path", "header_path",
    "data_path", "validation_status", "validation_issues", "ambiguity_flag",
    "ambiguity_note", "age", "sex", "source_extra_json", "split",
]

LEAD_NAMES = "I|II|III|aVR|aVL|aVF|V1|V2|V3|V4|V5|V6"


# ======================================================================
# LUDB
# ======================================================================
#
# Every rule below is a direct transcription of label_mapping.yaml's semantics
# onto LUDB's report vocabulary. Where LUDB says something the mapping has an
# explicit ruling on, that ruling wins.

# ASSERTED ST-elevation MI. The negative lookbehind is load-bearing: LUDB also
# writes "Undefined ischemia/scar/supp.NSTEMI: <wall>", and a naive "STEMI"
# substring match would sweep those in as positives. `\b` does not fire between
# the N and the S of NSTEMI, so \bSTEMI: matches only the real thing.
RE_STEMI = re.compile(r"\bSTEMI:")

# AF is atrial fibrillation ONLY. label_mapping.yaml:AFLT excludes atrial
# flutter from AF explicitly ("a distinct macro-reentrant atrial arrhythmia"),
# so LUDB's "Atrial flutter, typical" is a NORMAL disqualifier and nothing else.
RE_AF = re.compile(r"Atrial fibrillation", re.I)
RE_AFLT = re.compile(r"Atrial flutter", re.I)

# LVH is the LEFT ventricle, and hypertrophy rather than overload:
#   * "Right ventricular hypertrophy"  -> RVH, EXCLUDE_NOT_TARGET (wrong chamber)
#   * "Left ventricular overload"      -> overload, not a hypertrophy diagnosis;
#                                         treated like LAO/LAE, which the mapping
#                                         refuses because it is not ventricular
#                                         hypertrophy. Disqualifies NORMAL only.
#   * "Left atrial hypertrophy"        -> atrial, not ventricular. 102/200 carry
#                                         it; mapping it would inflate LVH by 5x.
RE_LVH = re.compile(r"Left ventricular hypertrophy", re.I)

# Anything at all in the report beyond rhythm and electrical axis is a finding,
# and a finding disqualifies NORMAL. Handled structurally rather than by
# enumerating disqualifiers, so a diagnosis this file has never seen fails
# CLOSED (record is not normal) instead of open.
NORMAL_RHYTHMS = re.compile(r"Rhythm:\s*Sinus (rhythm|bradycardia|arrhythmia)", re.I)


def ludb_diagnoses(hea_text: str) -> list[str]:
    """The report lines after the #<diagnoses>: marker, cleaned of WFDB syntax."""
    if "#<diagnoses>:" not in hea_text:
        return []
    block = hea_text.split("#<diagnoses>:", 1)[1]
    out = []
    for line in block.splitlines():
        line = line.strip().lstrip("#").strip().rstrip(".").strip()
        if line:
            out.append(line)
    return out


def ludb_age_sex(hea_text: str) -> tuple[str, str]:
    age = re.search(r"#<age>:\s*(\S+)", hea_text)
    sex = re.search(r"#<sex>:\s*(\S+)", hea_text)
    return (age.group(1).strip() if age else "",
            sex.group(1).strip().upper()[:1] if sex else "")


def build_ludb() -> tuple[list[dict], list[dict]]:
    rows, skipped = [], []
    heas = sorted(LUDB_DIR.glob("*.hea"), key=lambda p: int(p.stem))
    if not heas:
        sys.exit(f"no LUDB headers under {LUDB_DIR}")

    for hea in heas:
        text = hea.read_text(encoding="utf-8", errors="replace")
        rid = hea.stem
        dx = ludb_diagnoses(text)
        joined = " | ".join(dx)
        age, sex = ludb_age_sex(text)

        # The kit's --print_header does a bare dict lookup for Age and Sex and
        # raises KeyError when either is missing, so a record without both is
        # unrenderable. LUDB has them on all 200; this is a backstop.
        if not age or sex not in {"M", "F"}:
            skipped.append({"cohort": "LUDB", "record": rid,
                            "reason": "missing_age_or_sex", "detail": f"{age!r}/{sex!r}"})
            continue

        stemi = 1 if RE_STEMI.search(joined) else 0
        af = 1 if RE_AF.search(joined) else 0
        lvh = 1 if RE_LVH.search(joined) else 0

        findings = [d for d in dx
                    if not d.startswith("Rhythm:")
                    and not d.startswith("Electric axis of the heart:")]

        # NORMAL, and the one place this file departs from a literal reading of
        # label_mapping.yaml. That rule demands a POSITIVE normal assertion
        # (PTB-XL's NORM statement, Chapman's "sinus rhythm" SNOMED code), and
        # LUDB publishes no such statement. What it publishes instead is a
        # COMPLETE cardiologist report per record - so a report listing sinus
        # rhythm, an axis, and nothing else is that cardiologist stating they
        # found no abnormality. LUDB's README confirms the cohort includes
        # healthy volunteers.
        #
        # That is an assertion by absence, which is weaker than PTB-XL's NORM,
        # and it is tagged as such in `accepted_label_mapping` so the report can
        # discount it. It yields 25 records. It never overrides a finding:
        # `findings` must be empty, so this fails closed.
        normal = 1 if (not findings and NORMAL_RHYTHMS.search(joined)) else 0

        # rules.ambiguity.on_conflict = drop_normal. Cannot trigger given the
        # empty-findings test above, but asserted rather than assumed.
        if normal and (stemi or af or lvh):
            normal = 0

        rows.append({
            "global_record_id": f"LUDB_{int(rid):06d}",
            "source_dataset": "LUDB",
            "original_record_id": rid,
            # LUDB is one record per subject, so record id IS the subject.
            "patient_id": f"LUDB_P{rid}",
            "patient_id_is_synthetic": "0",
            "split_group_id": f"LUDB::LUDB_P{rid}",
            "original_labels": joined,
            "original_labels_readable": joined,
            "accepted_label_mapping": _ludb_mapping_note(stemi, af, lvh, normal),
            "cardiosentry_label": "+".join(l for l, v in
                                           zip(LABELS, (stemi, af, lvh, normal)) if v),
            "STEMI": stemi, "AF": af, "LVH": lvh, "NORMAL": normal,
            "n_positive_labels": stemi + af + lvh + normal,
            "sampling_frequency": "500.0", "duration_seconds": "10.0",
            "num_leads": "12", "num_samples": "5000", "lead_names": LEAD_NAMES,
            "signal_path": str(LUDB_DIR / rid),
            "header_path": str(hea),
            "data_path": str(LUDB_DIR / f"{rid}.dat"),
            "validation_status": "VALID", "validation_issues": "",
            "ambiguity_flag": "", "ambiguity_note": "",
            "age": age,
            # Written as M/F rather than a source-specific code on purpose:
            # config.SEX_DECODERS has no LUDB entry, and stage 1 falls back to a
            # first-letter test that only works on words. Adding a decoder would
            # mean editing the shared pipeline config, which this build does not
            # touch.
            "sex": sex,
            "source_extra_json": json.dumps({
                "cohort": "LUDB",
                # Per record, not per cohort: only the 8 records that actually
                # carry an asserted "STEMI: <wall>" earn the gold-STEMI tag. The
                # other 133 labelled LUDB records are ordinary AF/LVH/NORMAL
                # rows from an unseen source, and calling them "gold_asserted_
                # stemi" would make the column useless for exactly the filtering
                # it exists to support.
                "label_confidence": ("gold_asserted_stemi" if stemi
                                     else "gold_ludb_nonstemi"),
                "seen_in_training": 0,
                "diagnoses": dx,
                "atrial_flutter": bool(RE_AFLT.search(joined)),
            }),
            "split": "test",
        })
    return rows, skipped


def _ludb_mapping_note(stemi, af, lvh, normal) -> str:
    bits = []
    if stemi:
        bits.append("LUDB 'STEMI: <wall>'->STEMI")
    if af:
        bits.append("LUDB 'Atrial fibrillation'->AF")
    if lvh:
        bits.append("LUDB 'Left ventricular hypertrophy'->LVH")
    if normal:
        bits.append("LUDB report with rhythm+axis only->NORMAL (assertion by absence)")
    return "; ".join(bits) or "(no target class)"


# ======================================================================
# PTB-XL, curated selection
# ======================================================================

def _ecg_id(filename_hr: str) -> str:
    """'records500/19000/19269_hr' -> '19269'."""
    m = re.search(r"/(\d+)_hr", filename_hr)
    if not m:
        raise ValueError(f"unparseable filename_hr: {filename_hr!r}")
    return m.group(1).lstrip("0")


def trained_ptbxl_patients() -> set[str]:
    """Numeric PTB-XL patient ids present in the corpus the models trained on.

    The corpus split on patient, so a record is contaminated when its PATIENT
    appears - not merely when its own ecg_id does. Checking record ids alone
    would let a second ECG from a training patient through.
    """
    if not BALANCED.exists():
        sys.exit(f"balanced corpus CSV not found: {BALANCED}\n"
                 "  It is needed to exclude records the models trained on.")
    pats = set()
    with open(BALANCED, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("source_dataset") == "PTBXL":
                pats.add(str(row.get("patient_id", "")).replace("PTBXL_P", ""))
    return pats


def build_ptbxl(keep_seen: bool) -> tuple[list[dict], list[dict]]:
    if not SELECTED.exists():
        sys.exit(f"{SELECTED} not found - run 00_fetch_selection.sh first")

    db = {}
    with open(PTBXL_DIR / "ptbxl_database.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            db[row["ecg_id"]] = row

    seen_patients = trained_ptbxl_patients()
    rows, skipped = [], []

    with open(SELECTED, newline="", encoding="utf-8") as fh:
        selection = list(csv.DictReader(fh))

    for sel in selection:
        eid = _ecg_id(sel["filename_hr"])
        pid = str(sel["patient_id"]).strip()
        rec = db.get(eid)
        if rec is None:
            skipped.append({"cohort": "PTBXL", "record": eid,
                            "reason": "not_in_ptbxl_database", "detail": ""})
            continue

        contaminated = pid in seen_patients
        if contaminated and not keep_seen:
            skipped.append({"cohort": "PTBXL", "record": eid,
                            "reason": "patient_in_training_corpus",
                            "detail": f"patient {pid}"})
            continue

        codes = ast.literal_eval(rec["scp_codes"]) if rec.get("scp_codes") else {}
        is_mi = sel["label"].strip().lower() == "mi"

        # STEMI: proxy only, and only on the curated MI arm. See module docstring.
        stemi = 1 if is_mi else 0

        # The other three heads come from PTB-XL's own SCP codes under the
        # project's existing rulings, NOT from Gedon's binary label. Leaving
        # them at 0 would score a genuine AFIB record as an AF false positive.
        af = 1 if "AFIB" in codes else 0
        lvh = 1 if "LVH" in codes else 0
        normal = 1 if (not is_mi and "NORM" in codes) else 0
        if normal and (stemi or af or lvh):
            normal = 0                      # rules.ambiguity: drop_normal

        stadium = (rec.get("infarction_stadium1") or "").strip()
        rows.append({
            "global_record_id": f"PTBXLX_{int(eid):06d}",
            # Cohort rides in the source column because that is the one field
            # stage 1 carries all the way through to index.csv, which is what
            # the evaluator groups on. Keeping the arms separate here is what
            # lets the report avoid quoting one sensitivity across both.
            "source_dataset": "PTBXL_MI" if is_mi else "PTBXL_NORM",
            "original_record_id": eid,
            "patient_id": f"PTBXL_P{pid}",
            "patient_id_is_synthetic": "0",
            "split_group_id": f"PTBXL::PTBXL_P{pid}",
            "original_labels": "|".join(codes.keys()),
            "original_labels_readable": rec.get("scp_codes", ""),
            "accepted_label_mapping": _ptbxl_mapping_note(is_mi, af, lvh, normal, stadium),
            "cardiosentry_label": "+".join(l for l, v in
                                           zip(LABELS, (stemi, af, lvh, normal)) if v),
            "STEMI": stemi, "AF": af, "LVH": lvh, "NORMAL": normal,
            "n_positive_labels": stemi + af + lvh + normal,
            "sampling_frequency": "500.0", "duration_seconds": "10.0",
            "num_leads": "12", "num_samples": "5000", "lead_names": LEAD_NAMES,
            "signal_path": str(PTBXL_DIR / sel["filename_hr"]),
            "header_path": str(PTBXL_DIR / (sel["filename_hr"] + ".hea")),
            "data_path": str(PTBXL_DIR / (sel["filename_hr"] + ".dat")),
            "validation_status": "VALID", "validation_issues": "",
            "ambiguity_flag": "", "ambiguity_note": "",
            "age": rec.get("age", ""),
            # PTB-XL encodes 0=male, 1=female. Decoded here for the same reason
            # LUDB's is: the cohort source names are not in config.SEX_DECODERS.
            "sex": {"0": "M", "1": "F"}.get(str(sel.get("sex", "")).strip(), ""),
            "source_extra_json": json.dumps({
                "cohort": "PTBXL_MI" if is_mi else "PTBXL_NORM",
                "label_confidence": "proxy_acute_mi" if is_mi else "gold_norm",
                "seen_in_training": int(contaminated),
                "gedon_label": sel["label"],
                "infarction_stadium1": stadium,
                "scp_codes": list(codes.keys()),
            }),
            "split": "test",
        })
    return rows, skipped


def _ptbxl_mapping_note(is_mi, af, lvh, normal, stadium) -> str:
    bits = []
    if is_mi:
        bits.append(f"Gedon manual 'mi' + PTB-XL {stadium or 'stadium?'}"
                    f"->STEMI (PROXY, not an asserted ST-elevation label)")
    if af:
        bits.append("PTB-XL AFIB->AF")
    if lvh:
        bits.append("PTB-XL LVH->LVH")
    if normal:
        bits.append("PTB-XL NORM->NORMAL")
    return "; ".join(bits) or "(no target class)"


# ======================================================================

def summarise(rows: list[dict]) -> str:
    by_source = Counter(r["source_dataset"] for r in rows)
    lines = [f"{'cohort':<12} {'n':>5}  " + "  ".join(f"{l:>6}" for l in LABELS)]
    lines.append("-" * len(lines[0]))
    for src in sorted(by_source):
        sub = [r for r in rows if r["source_dataset"] == src]
        counts = "  ".join(f"{sum(r[l] for r in sub):>6}" for l in LABELS)
        lines.append(f"{src:<12} {len(sub):>5}  {counts}")
    total = "  ".join(f"{sum(r[l] for r in rows):>6}" for l in LABELS)
    lines.append("-" * len(lines[0]))
    lines.append(f"{'TOTAL':<12} {len(rows):>5}  {total}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=str(Path(__file__).resolve().parents[1]
                                               / "data" / "external_test_metadata.csv"))
    ap.add_argument("--keep-seen", action="store_true",
                    help="keep PTB-XL records whose PATIENT is in the training "
                         "corpus (they are excluded by default)")
    ap.add_argument("--ludb-only", action="store_true",
                    help="build the gold cohort alone")
    args = ap.parse_args()

    rows, skipped = build_ludb()
    if not args.ludb_only:
        p_rows, p_skip = build_ptbxl(args.keep_seen)
        rows += p_rows
        skipped += p_skip

    # Shuffled before writing, and this is not cosmetic.
    #
    # Stage 2 renames every record to ECG%06d in the order it reads them, and
    # the kit PRINTS that name on the sheet as "ID: ECG000123". Written in
    # source order, the printed number would run LUDB first, then PTBXL_MI,
    # then PTBXL_NORM - making the ID a near-perfect linear proxy for both
    # source and label, rendered directly into the pixels a model reads.
    #
    # This experiment exists to ask whether a model keys on source rather than
    # on the trace. Handing it a fresh source cue on the evaluation sheets
    # would make the answer uninterpretable. Fixed seed, so the build stays
    # reproducible.
    import random
    random.Random(20260820).shuffle(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Split on "has a positive label", the exact test stage 1 applies. See the
    # module docstring for why the second group is rescued rather than dropped.
    labelled = [r for r in rows if r["n_positive_labels"] > 0]
    unlabelled = [r for r in rows if r["n_positive_labels"] == 0]

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(labelled)

    unlab_path = out.with_name(out.stem + "_unlabelled.csv")
    with open(unlab_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in unlabelled:
            r = dict(r)
            # SENTINEL. Stripped again by 03_merge.py, which re-derives labels
            # from THIS file's true values and never reads them back from an
            # index.csv. Recorded in the row so it is visible if anyone opens
            # the intermediate CSV and wonders why a hard negative says NORMAL.
            r["NORMAL"] = 1
            r["n_positive_labels"] = 1
            r["cardiosentry_label"] = "NORMAL"
            r["ambiguity_flag"] = "SENTINEL_LABEL"
            r["ambiguity_note"] = ("NORMAL=1 is a render-pass sentinel, not a "
                                   "clinical label; true labels are all-zero")
            w.writerow(r)

    skip_path = out.with_name("excluded_records.csv")
    with open(skip_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["cohort", "record", "reason", "detail"])
        w.writeheader()
        w.writerows(skipped)

    print(summarise(rows))
    print()
    if skipped:
        print("excluded:")
        for reason, n in Counter(s["reason"] for s in skipped).most_common():
            print(f"  {n:>4}  {reason}")
        print(f"  -> {skip_path}")
    print(f"\nwrote {len(labelled):>4} labelled rows   -> {out}")
    print(f"wrote {len(unlabelled):>4} unlabelled rows -> {unlab_path}")
    print("      (unlabelled = true negatives on all four heads; they carry a "
          "sentinel\n       NORMAL=1 so stage 1 will render them, stripped "
          "again by 03_merge.py)")

    # A build with no STEMI positive from LUDB has lost the only thing it is
    # for, and would still produce a plausible-looking corpus. Fail loudly.
    gold = sum(r["STEMI"] for r in rows if r["source_dataset"] == "LUDB")
    if gold == 0:
        print("\nFATAL: zero LUDB STEMI positives - the external test cannot "
              "answer the question it exists for.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
