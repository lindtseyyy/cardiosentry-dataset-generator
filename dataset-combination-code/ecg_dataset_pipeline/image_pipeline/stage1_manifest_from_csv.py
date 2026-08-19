"""Stage 1 - build the render manifest from an already-generated dataset CSV.

This replaces stages 1-2 of the original prototype (which resolved labels from
the three raw catalogues and then subsampled them). The dataset pipeline has
already done that work, so this stage only *adapts* its CSV:

  * decodes age and sex into the one form the kit can print,
  * resolves each record to a readable WFDB base path,
  * decides how many render variants each record contributes,
  * assigns a patient-grouped train/val/test split, unless the CSV brings one.

Nothing clinical is decided here. The four binary label columns are copied
through verbatim and stay the source of truth all the way to index.csv; the
`cls` column is only a directory and reporting name.

    python stage1_manifest_from_csv.py
    python stage1_manifest_from_csv.py --input-csv ../data/harmonized/harmonized_ecg_metadata.csv
    python stage1_manifest_from_csv.py --limit 40          # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import config as C

COL = C.CSV_COLUMNS

FIELDS = ([
    "record_id", "source", "original_id", "signal_path", "cls",
] + C.LABEL_COLUMNS + [
    "patient_id", "age", "sex", "split", "n_renders", "note",
])


# ------------------------------------------------------------------ metadata
def clean_age(value) -> str:
    """Return the printable age, or "" if it cannot be used.

    PTB-XL masks any age over 89 as 300; the patient is genuinely over 89, so
    the sentinel is rendered as 90 rather than dropping the record.
    """
    try:
        age = float(value)
    except (TypeError, ValueError):
        return ""
    if age != age:                      # NaN; Chapman writes NaN for unknown age
        return ""
    if age >= C.PTBXL_AGE_SENTINEL:
        return str(C.AGE_MASKED_AS)
    if age <= 0:
        return ""
    return str(int(round(age)))


def clean_sex(value, source: str) -> str:
    """Decode a source's sex encoding to "M"/"F", or "" if unknown.

    The three catalogues disagree and two of them use bare integers with
    *opposite* conventions, so the mapping has to be per source - see
    config.SEX_DECODERS.
    """
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "unknown", "none"}:
        return ""

    table = C.SEX_DECODERS.get(source.upper(), {})
    for key in (raw, raw.lower()):
        if key in table:
            return table[key]
    # Numeric codes are ambiguous without a declared source; words are not.
    low = raw.lower()
    if low.startswith("m"):
        return "M"
    if low.startswith("f"):
        return "F"
    return ""


# ------------------------------------------------------------------ paths
def resolve_signal(row: dict, csv_dir: Path) -> tuple[str, str]:
    """Return (wfdb base path, reason-if-unusable).

    wfdb.rdrecord() wants the path without an extension and reads .dat and
    .mat alike through the header, so the base path is all stage 2 needs.

    A relative path is resolved against the CSV's own directory, which is what
    makes a versioned release from `scripts/export_dataset.py` work as input:
    its CSVs carry portable paths like `signals/CHAPMAN/CHAPMAN_000001`.
    """
    base = (row.get(COL["signal_path"]) or "").strip()
    if not base:
        header = (row.get(COL["header_path"]) or "").strip()
        if not header:
            return "", "no_path"
        base = header[:-4] if header.endswith(".hea") else header
    path = Path(base).expanduser()
    if not path.is_absolute():
        path = csv_dir / path
    base = str(path)
    if base.endswith((".hea", ".dat", ".mat")):
        base = base.rsplit(".", 1)[0]

    if not Path(base + ".hea").exists():
        return base, "missing_header"
    if not (Path(base + ".dat").exists() or Path(base + ".mat").exists()):
        return base, "missing_signal"
    return base, ""


# ------------------------------------------------------------------ labels
def positive_classes(row: dict) -> list[str]:
    out = []
    for cls in C.LABEL_COLUMNS:
        try:
            if int(float(row.get(cls, 0) or 0)) == 1:
                out.append(cls)
        except (TypeError, ValueError):
            continue
    return out


def class_name(row: dict, positives: list[str]) -> str:
    """Directory / reporting name. Prefer the CSV's own, fall back to joining."""
    given = (row.get(COL["label"]) or "").strip()
    if given:
        return given
    return "+".join(positives)


def render_count(positives: list[str], override: int | None) -> int:
    if override:
        return override
    if not positives:
        return 1
    return max(C.RENDERS_PER_RECORD.get(cls, 1) for cls in positives)


# ------------------------------------------------------------------ splits
def _ptbxl_fold(row: dict) -> int | None:
    """PTB-XL ships its own stratified folds; honour 9 and 10 as val/test.

    The dataset pipeline keeps them inside source_extra_json rather than as a
    column, so this digs them out and shrugs if the shape ever changes.
    """
    blob = row.get("source_extra_json")
    if not blob:
        return None
    try:
        fold = json.loads(blob).get("strat_fold")
        return int(float(fold)) if fold not in (None, "") else None
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def assign_splits(rows: list[dict], rng: np.random.Generator) -> None:
    """80/10/10 over whole patient groups, stratified by class.

    Grouped, because a patient on both sides of the evaluation boundary
    invalidates it. Stratified per class, because assigning globally lets
    PTB-XL's forced folds eat the test budget and starves the classes PTB-XL
    does not contribute to.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["patient_id"]].append(r)

    # A patient group is stratified by the class it mostly carries; splitting
    # it further would break the grouping guarantee, which matters more.
    strata: dict[str, list[str]] = defaultdict(list)
    for key, members in groups.items():
        cls = Counter(m["cls"] for m in members).most_common(1)[0][0]
        strata[cls].append(key)

    for keys in strata.values():
        keys = sorted(keys)
        rng.shuffle(keys)

        forced: dict[str, str] = {}
        if C.USE_PTBXL_FOLDS:
            for key in keys:
                folds = {m["_fold"] for m in groups[key]
                         if m["source"].upper() == "PTBXL" and m["_fold"]}
                if folds & C.PTBXL_TEST_FOLDS:
                    forced[key] = "test"
                elif folds & C.PTBXL_VAL_FOLDS:
                    forced[key] = "val"

        free = [k for k in keys if k not in forced]
        need_test = max(0, round(len(keys) * C.SPLITS["test"])
                        - sum(v == "test" for v in forced.values()))
        need_val = max(0, round(len(keys) * C.SPLITS["val"])
                       - sum(v == "val" for v in forced.values()))

        plan = {k: "test" for k in free[:need_test]}
        plan.update({k: "val" for k in free[need_test:need_test + need_val]})
        for key in keys:
            split = forced.get(key) or plan.get(key, "train")
            for member in groups[key]:
                member["split"] = split


# ------------------------------------------------------------------ main
def build(rows_in: list[dict], renders_override: int | None,
          check_files: bool, csv_dir: Path) -> tuple[list[dict], Counter]:
    stats = Counter()
    out: list[dict] = []

    for row in rows_in:
        record_id = (row.get(COL["record_id"]) or "").strip()
        source = (row.get(COL["source"]) or "").strip()

        status = (row.get(COL["validation"]) or "").strip()
        if C.ACCEPT_VALIDATION is not None and status and status not in C.ACCEPT_VALIDATION:
            stats[f"skip_validation_{status}"] += 1
            continue

        positives = positive_classes(row)
        if not positives:
            stats["skip_no_label"] += 1
            continue

        age = clean_age(row.get(COL["age"]))
        sex = clean_sex(row.get(COL["sex"]), source)
        if not age or not sex:
            # Not cosmetic: the kit's --print_header does a bare dict lookup
            # for 'Age' and 'Sex' and raises KeyError if either is absent.
            stats["skip_missing_metadata"] += 1
            continue

        base, reason = ("", "")
        if check_files:
            base, reason = resolve_signal(row, csv_dir)
            if reason:
                stats[f"skip_{reason}"] += 1
                continue
        else:
            base = (row.get(COL["signal_path"]) or "").strip()

        patient = (row.get(COL["patient"]) or "").strip() or record_id
        cls = class_name(row, positives)

        entry = {
            "record_id": record_id,
            "source": source,
            "original_id": (row.get(COL["original_id"]) or "").strip(),
            "signal_path": base,
            "cls": cls,
            "patient_id": patient,
            "age": age,
            "sex": sex,
            "split": (row.get(COL["split"]) or "").strip(),
            "n_renders": render_count(positives, renders_override),
            "note": status,
            "_fold": _ptbxl_fold(row),
        }
        for label in C.LABEL_COLUMNS:
            entry[label] = 1 if label in positives else 0

        out.append(entry)
        stats[f"ok_{cls}"] += 1

    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-csv", type=Path, default=None,
                    help=f"default: {C.INPUT_CSV}. Relative signal paths are "
                         f"resolved against this file's directory, so a release "
                         f"from scripts/export_dataset.py works directly.")
    ap.add_argument("--renders", type=int, default=0,
                    help="force this many render variants for every record")
    ap.add_argument("--limit", type=int, default=0,
                    help="keep only the first N records (smoke test)")
    ap.add_argument("--no-file-check", action="store_true",
                    help="trust the CSV's paths instead of stat-ing them")
    ap.add_argument("--reassign-splits", action="store_true",
                    help="ignore any split column in the CSV and assign fresh")
    args = ap.parse_args()

    src = args.input_csv or C.INPUT_CSV
    if not src.exists() and args.input_csv is None and C.FALLBACK_INPUT_CSV.exists():
        print(f"{src} not found, falling back to {C.FALLBACK_INPUT_CSV.name}")
        src = C.FALLBACK_INPUT_CSV
    if not src.exists():
        print(f"input CSV not found: {src}", file=sys.stderr)
        return 1

    rows_in = list(csv.DictReader(src.open()))
    if not rows_in:
        print(f"{src} is empty", file=sys.stderr)
        return 1
    missing = [c for c in (COL["record_id"], COL["source"], COL["signal_path"])
               if c not in rows_in[0]]
    if missing:
        print(f"{src} is missing required columns: {missing}", file=sys.stderr)
        print("Adjust CSV_COLUMNS in config.py if your CSV names them differently.",
              file=sys.stderr)
        return 1

    print(f"reading {len(rows_in)} rows from {src}")
    rows, stats = build(rows_in, args.renders or None, not args.no_file_check,
                        src.resolve().parent)
    if not rows:
        print("no usable records", file=sys.stderr)
        return 1
    if args.limit:
        rows = rows[: args.limit]

    have_split = all(r["split"] for r in rows)
    if not have_split and not args.reassign_splits:
        print(
            "\nREFUSING TO INVENT A SPLIT.\n"
            f"  {args.input_csv or C.INPUT_CSV} has no populated `split` column.\n"
            "  The train/val/test split is frozen into the dataset release so that\n"
            "  results stay comparable across rebuilds. Freeze it first:\n\n"
            "      python ../scripts/freeze_splits.py --release v2\n\n"
            "  ...or pass --reassign-splits to assign a throwaway one here (this\n"
            "  will NOT match the frozen split and must not be used for reported\n"
            "  results).",
            file=sys.stderr)
        return 1
    if have_split and not args.reassign_splits:
        print("using the split column already FROZEN in the CSV "
              "(scripts/freeze_splits.py) - not recomputing it")
    else:
        assign_splits(rows, np.random.default_rng(C.SEED))
        print(f"assigned {'/'.join(f'{int(v*100)}' for v in C.SPLITS.values())} "
              f"splits over patient groups (seed {C.SEED})")

    C.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with C.MANIFEST.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)

    # ---- report -------------------------------------------------------
    print(f"\nmanifest: {len(rows)} records -> {C.MANIFEST}")
    skipped = {k: v for k, v in stats.items() if k.startswith("skip_")}
    if skipped:
        print("skipped:")
        for k in sorted(skipped):
            print(f"    {k:28s} {skipped[k]:6d}")

    # Evaluation splits get exactly one render, and stage 4 pins them to
    # exactly one augmented variant. Enforced in both places on purpose: the
    # guarantee must not depend on a single file being edited correctly.
    pinned = 0
    for r in rows:
        if r["split"] in C.EVAL_SPLITS and int(r["n_renders"]) != 1:
            r["n_renders"] = 1
            pinned += 1
    if pinned:
        print(f"pinned {pinned} val/test records back to n_renders=1")

    cls_split = Counter((r["cls"], r["split"]) for r in rows)
    images = Counter()
    for r in rows:
        # One render becomes VARIANTS_PER_SPLIT[split] images in stage 4.
        per = 1 if r["split"] in C.EVAL_SPLITS else max(
            1, int(C.VARIANTS_PER_SPLIT.get(r["split"], 1)))
        images[r["cls"]] += int(r["n_renders"]) * per
    print(f"\n    {'class':10s}{'train':>8s}{'val':>7s}{'test':>7s}"
          f"{'recs':>8s}{'images':>9s}")
    for cls in sorted({r["cls"] for r in rows}):
        tr, va, te = (cls_split[(cls, s)] for s in ("train", "val", "test"))
        print(f"    {cls:10s}{tr:8d}{va:7d}{te:7d}{tr+va+te:8d}{images[cls]:9d}")
    print(f"    {'TOTAL':10s}{'':22s}{len(rows):8d}{sum(images.values()):9d}")

    per_split = Counter(r["split"] for r in rows)
    print(f"\n    images per split (variants: "
          f"{', '.join(f'{k}={v}' for k, v in C.VARIANTS_PER_SPLIT.items())}):")
    for sp in ("train", "val", "test"):
        per = 1 if sp in C.EVAL_SPLITS else max(
            1, int(C.VARIANTS_PER_SPLIT.get(sp, 1)))
        print(f"      {sp:6s} {per_split[sp]:7d} records x {per} "
              f"= {per_split[sp]*per:8d} images"
              + ("   <- exactly one per ECG" if sp in C.EVAL_SPLITS else ""))

    by_source = Counter(r["source"] for r in rows)
    print("\n    by source: " + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))
    print("    Reminder: STEMI is single-source. Whatever the leakage probe in")
    print("    stage 5 says, report per-source metrics as well.")

    # Leakage assertion: no patient group may appear in two splits.
    seen: dict[str, str] = {}
    for r in rows:
        prev = seen.setdefault(r["patient_id"], r["split"])
        if prev != r["split"]:
            print(f"    FAIL: patient {r['patient_id']} in {prev} and {r['split']}")
            return 1
    print("    patient-split integrity: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
