"""Stage 5 - check the corpus before anything is trained on it.

Four checks, cheapest first. The last one is the important one: if a small model
can recover which hospital produced an image, then the STEMI class is still
partly encoded by acquisition fingerprint rather than by ST elevation, and any
headline accuracy is not trustworthy.

Adapted from the prototype's stage6 for the multi-label CSV: `cls` here can be
a combination like "AF+LVH", so balance is reported both per label column and
per combination.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import config as C


def load_index() -> list[dict]:
    if not C.INDEX_CSV.exists():
        sys.exit(f"{C.INDEX_CSV.name} missing - run stage4 first")
    return list(csv.DictReader(C.INDEX_CSV.open()))


def _positive(row: dict, label: str) -> bool:
    try:
        return int(float(row.get(label) or 0)) == 1
    except ValueError:
        return False


# ------------------------------------------------------------------ 1. integrity
def check_integrity(rows: list[dict]) -> bool:
    ok = True
    print("\n[1] integrity")

    patient_splits = defaultdict(set)
    record_classes = defaultdict(set)
    for r in rows:
        patient_splits[r["patient_id"]].add(r["split"])
        record_classes[r["record"]].add(r["cls"])

    bad_patients = {p: s for p, s in patient_splits.items() if len(s) > 1}
    bad_records = {t: c for t, c in record_classes.items() if len(c) > 1}
    missing = [r["image_path"] for r in rows if not Path(r["image_path"]).exists()]
    unlabelled = [r["record"] for r in rows
                  if not any(_positive(r, lab) for lab in C.LABEL_COLUMNS)]

    # NORMAL is mutually exclusive with the other three by construction
    # upstream; if that ever stops being true, it should be loud.
    both = [r["record"] for r in rows if _positive(r, "NORMAL")
            and any(_positive(r, lab) for lab in C.LABEL_COLUMNS if lab != "NORMAL")]

    for label, bad in (("patients spanning splits", bad_patients),
                       ("records in two classes", bad_records),
                       ("missing image files", missing),
                       ("images with no label", unlabelled),
                       ("NORMAL + pathology", both)):
        status = "OK" if not bad else f"FAIL ({len(bad)})"
        print(f"    {label:28s} {status}")
        ok &= not bad
    return ok


# ------------------------------------------------------------------ 2. balance
def check_balance(rows: list[dict]) -> bool:
    print("\n[2] class / split / source balance")

    print(f"    {'label':10s}{'train':>8s}{'val':>7s}{'test':>7s}{'total':>8s}")
    for label in C.LABEL_COLUMNS:
        counts = Counter(r["split"] for r in rows if _positive(r, label))
        tr, va, te = (counts[s] for s in ("train", "val", "test"))
        print(f"    {label:10s}{tr:8d}{va:7d}{te:7d}{tr+va+te:8d}")

    combos = Counter(r["cls"] for r in rows)
    multi = {k: v for k, v in combos.items() if "+" in k}
    if multi:
        print("\n    multi-label combinations: "
              + ", ".join(f"{k}={v}" for k, v in sorted(multi.items())))

    sources = sorted({r["source"] for r in rows})
    print(f"\n    {'label':10s}" + "".join(f"{s:>10s}" for s in sources))
    grid = Counter((label, r["source"]) for r in rows for label in C.LABEL_COLUMNS
                   if _positive(r, label))
    for label in C.LABEL_COLUMNS:
        print(f"    {label:10s}" + "".join(f"{grid[(label, s)]:10d}" for s in sources))
    print("\n    Reminder: STEMI is single-source by construction. The NORMAL row")
    print("    is what keeps source from being a clean proxy for the label.")
    return True


# ------------------------------------------------------------------ 3. geometry
def check_images(rows: list[dict], n: int = 200) -> bool:
    """Sanity-check pixel statistics; catches gain and rendering blow-ups."""
    print(f"\n[3] image statistics (sample of {n})")
    from PIL import Image

    rng = np.random.default_rng(C.SEED)
    sample = [rows[i] for i in rng.choice(len(rows), min(n, len(rows)), replace=False)]

    stats = defaultdict(list)
    sizes = set()
    for r in sample:
        im = Image.open(r["image_path"])
        sizes.add(im.size)
        arr = np.asarray(im.convert("L"), dtype=np.float32)
        stats[r["source"]].append((arr.mean(), (arr < 100).mean()))

    print(f"    distinct image sizes: {sorted(sizes)}")
    print(f"    {'source':10s}{'mean lum':>10s}{'ink frac':>10s}")
    for src in sorted(stats):
        vals = np.array(stats[src])
        print(f"    {src:10s}{vals[:,0].mean():10.1f}{vals[:,1].mean():10.4f}")
    print("    Ink fraction should be similar across sources; a large gap means")
    print("    one source is rendering at the wrong gain.")
    return len(sizes) == 1


# ------------------------------------------------------------------ 4. leakage
def check_leakage(rows: list[dict], per_source: int = 400, size: int = 64) -> bool:
    """Can a weak model tell the source from the pixels? It should barely manage.

    Deliberately a small model on tiny thumbnails: the question is whether
    source identity is trivially recoverable, not whether it is recoverable at
    all with unlimited capacity.
    """
    print(f"\n[4] source-leakage probe ({per_source}/source, {size}x{size} grey)")
    try:
        from PIL import Image
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError:
        print("    skipped: scikit-learn not installed")
        return True

    rng = np.random.default_rng(C.SEED)
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    if len(by_source) < 2:
        print("    skipped: fewer than two sources")
        return True

    X, y, groups = [], [], []
    for src, items in by_source.items():
        idx = rng.choice(len(items), min(per_source, len(items)), replace=False)
        for i in idx:
            im = Image.open(items[i]["image_path"]).convert("L").resize((size, size))
            X.append(np.asarray(im, dtype=np.float32).ravel() / 255.0)
            y.append(src)
            groups.append(items[i]["patient_id"])
    X, y, groups = np.array(X), np.array(y), np.array(groups)

    n_groups = len(set(groups))
    if n_groups < 40:
        print(f"    skipped: only {n_groups} patients sampled - probe is underpowered")
        return True

    # Group by patient, or the probe just memorises records: a record may be
    # rendered more than once, so an image-level split puts near-duplicates on
    # both sides and reports leakage that is really duplication.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=C.SEED)
    tr, te = next(splitter.split(X, y, groups))
    clf = LogisticRegression(max_iter=400)
    clf.fit(X[tr], y[tr])
    acc = clf.score(X[te], y[te])
    chance = max(Counter(y[te]).values()) / len(y[te])

    print(f"    source accuracy {acc:.3f}   (chance {chance:.3f})")
    if acc > chance + 0.25:
        print("    FAIL: source is trivially readable from the pixels. Harmonisation")
        print("    did not hold - do not trust STEMI numbers from this corpus.")
        return False
    if acc > chance + 0.10:
        print("    WARN: some source signal remains; report per-source metrics.")
    else:
        print("    OK: source is not trivially recoverable.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()

    rows = load_index()
    print(f"index: {len(rows)} images")

    ok = check_integrity(rows)
    check_balance(rows)
    ok &= check_images(rows)
    if not args.skip_probe:
        ok &= check_leakage(rows)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
