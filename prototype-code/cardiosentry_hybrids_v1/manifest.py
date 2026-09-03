"""
Dataset manifest loading and integrity enforcement.

Pure pandas - no torch - so it can be imported and checked anywhere.

The splits in index.csv were FROZEN once by scripts/freeze_splits.py and are
patient-grouped. Nothing here ever recomputes or reassigns them; it reads the
`split` column and verifies the properties that make it trustworthy. If an
assertion in check_integrity() fires, the experiment is invalid - do not
"fix" it by resampling.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

import config as C

REQUIRED_COLUMNS = {"rel_path", "record", "split", "source", "patient_id", *C.LABELS}


def load(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    idx = root / "index.csv"
    if not idx.exists():
        # Distinguish the three ways this goes wrong - they need different fixes.
        if not root.exists():
            parent = root.parent
            siblings = ([p.name for p in parent.iterdir() if p.is_dir()][:12]
                        if parent.exists() else [])
            raise FileNotFoundError(
                f"the folder {root} DOES NOT EXIST.\n"
                f"  On Colab, /content is wiped when the runtime restarts, while\n"
                f"  /content/drive survives. Re-run the cell that extracts the tar.\n"
                f"  directories in {parent}: {siblings or '(none)'}"
            )
        contents = [p.name for p in root.iterdir()][:12]
        raise FileNotFoundError(
            f"{root} exists but contains no index.csv.\n"
            f"  found instead: {contents or '(empty directory)'}\n"
            f"  If this is a resized copy, the resize did not finish - rerun\n"
            f"  resize_dataset.py. Otherwise re-extract the tar."
        )
    df = pd.read_csv(idx)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"index.csv is missing columns: {sorted(missing)}")

    # image_path is an absolute path from the build machine and will not resolve
    # anywhere else. rel_path is the portable one. Drop the trap outright.
    df = df.drop(columns=[c for c in ("image_path",) if c in df.columns])

    df["abs_path"] = df["rel_path"].map(lambda p: str(root / p))
    for lab in C.LABELS:
        df[lab] = df[lab].astype("float32")
    return df


def check_integrity(df: pd.DataFrame, verify_files: bool = False) -> dict:
    """Verify the properties a fair, leak-free comparison depends on.

    Raises on anything that would invalidate results. Returns a report dict.
    """
    problems: list[str] = []

    # 1. no patient may appear in more than one split
    spans = df.groupby("patient_id")["split"].nunique()
    leaked = spans[spans > 1]
    if len(leaked):
        problems.append(
            f"{len(leaked)} patient(s) appear in more than one split - "
            f"results would be inflated by leakage. e.g. {list(leaked.index[:3])}"
        )

    # 2. no record may appear in more than one split
    rspans = df.groupby("record")["split"].nunique()
    if (rspans > 1).any():
        problems.append(f"{int((rspans > 1).sum())} record(s) appear in more than one split")

    # 3. val/test must hold exactly one image per record, or evaluation silently
    #    weights some ECGs more heavily than others
    for split in ("val", "test"):
        sub = df[df.split == split]
        if len(sub):
            per = sub.groupby("record").size()
            if (per != 1).any():
                problems.append(
                    f"{split}: {int((per != 1).sum())} record(s) have more than one image; "
                    f"evaluation must be exactly 1 canonical image per ECG"
                )

    # 4. NORMAL is mutually exclusive with the pathologies by construction
    path_cols = [l for l in C.LABELS if l != "NORMAL"]
    both = df[(df["NORMAL"] == 1) & (df[path_cols].sum(axis=1) > 0)]
    if len(both):
        problems.append(f"{len(both)} row(s) are NORMAL and also carry a pathology label")

    # 5. every row must carry at least one positive label
    unlabelled = df[df[C.LABELS].sum(axis=1) == 0]
    if len(unlabelled):
        problems.append(f"{len(unlabelled)} row(s) have no positive label")

    # 6. all three splits present
    for split in ("train", "val", "test"):
        if not (df.split == split).any():
            problems.append(f"split '{split}' is empty")

    if verify_files:
        miss = [p for p in df["abs_path"] if not Path(p).exists()]
        if miss:
            problems.append(f"{len(miss)} image file(s) missing on disk, e.g. {miss[:3]}")

    if problems:
        raise RuntimeError(
            "DATASET INTEGRITY CHECK FAILED - results would not be valid:\n  - "
            + "\n  - ".join(problems)
        )

    return {
        "rows": int(len(df)),
        "records": int(df["record"].nunique()),
        "patients": int(df["patient_id"].nunique()),
        "images_per_split": df.split.value_counts().to_dict(),
        "records_per_split": df.groupby("split")["record"].nunique().to_dict(),
        "sources": df.source.value_counts().to_dict(),
        "positives": {l: int(df[l].sum()) for l in C.LABELS},
    }


def pos_weight(df: pd.DataFrame) -> list[float]:
    """neg/pos per head, computed on TRAIN only.

    Passed to BCEWithLogitsLoss so the minority heads (STEMI, NORMAL) are not
    drowned out. Derived at runtime so it stays correct across prototype_500 /
    _800 / _1152, which have different class priors.
    """
    tr = df[df.split == "train"]
    out = []
    for lab in C.LABELS:
        pos = float(tr[lab].sum())
        neg = float(len(tr) - pos)
        out.append(neg / pos if pos > 0 else 1.0)
    return out


def describe(df: pd.DataFrame) -> str:
    lines = []
    tr = df[df.split == "train"]
    lines.append(f"  images  train={len(tr):,}  val={int((df.split=='val').sum()):,}  "
                 f"test={int((df.split=='test').sum()):,}")
    lines.append(f"  records {df['record'].nunique():,}   patients {df['patient_id'].nunique():,}")
    lines.append("  per-head positive rate on TRAIN:")
    for lab, pw in zip(C.LABELS, pos_weight(df)):
        pos = int(tr[lab].sum())
        lines.append(f"      {lab:7} {pos:6,} / {len(tr):,} = {pos/len(tr):6.1%}   pos_weight={pw:5.2f}")
    lines.append(f"  sources: {df.source.value_counts().to_dict()}")
    return "\n".join(lines)
