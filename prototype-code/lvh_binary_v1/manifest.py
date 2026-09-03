"""
Dataset manifest loading and integrity enforcement - LVH binary task.

Pure pandas - no torch - so it can be imported and checked anywhere.

The splits in index.csv were FROZEN once and are patient-grouped; they are the
SAME splits the four-head reference run used. Nothing here ever recomputes or
reassigns them; it reads the `split` column and verifies the properties that
make it trustworthy. If an assertion in check_integrity() fires, the experiment
is invalid - do not "fix" it by resampling.

DIFFERENCES FROM THE MULTI-LABEL MANIFEST, ALL FORCED BY THE TASK

  * Only the LVH column is loaded as the label; STEMI/AF/NORMAL stay in the
    index but are never read by training.
  * The round-2 check "every row must carry at least one positive label" is a
    multi-label property and does not exist here. Its binary replacement: every
    row must carry a valid 0/1 LVH label (no NaNs), and no row may be NORMAL
    and LVH-positive at once - the same exclusivity the label audit guarantees.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

import config as C

REQUIRED_COLUMNS = {"rel_path", "record", "split", "source", "patient_id", *C.LABELS}

# Columns the index carries but the binary task does not use. Kept in the
# manifest so check_integrity can still police the NORMAL/LVH exclusivity.
_INDEX_PATHOLOGY_COLUMNS = ("STEMI", "AF", "NORMAL")


def load(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    idx = root / "index.csv"
    if not idx.exists():
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
            f"  This is the 768px corpus built by the hybrids round - re-extract\n"
            f"  prototype_1152_768.tar, or rebuild it with the hybrids round's\n"
            f"  0_Prepare notebook."
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

    # 4. binary label validity: every row carries a 0/1 LVH label (this is the
    #    binary replacement for round 2's "at least one positive label" rule,
    #    which is a multi-label property that does not exist here)
    lvh = df["LVH"]
    if lvh.isna().any():
        problems.append(f"{int(lvh.isna().sum())} row(s) have a missing LVH label")
    bad = ~lvh.isin([0.0, 1.0])
    if bad.any():
        problems.append(f"{int(bad.sum())} row(s) have an LVH label that is not 0/1")

    # 5. NORMAL and LVH are mutually exclusive by the label audit; a row that
    #    is both is a contradiction worth stopping on even in the binary task
    if "NORMAL" in df.columns:
        both = df[(df["NORMAL"] == 1) & (df["LVH"] == 1)]
        if len(both):
            problems.append(f"{len(both)} row(s) are NORMAL and also carry an LVH label")

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

    For the single LVH head this is the SAME number the four-head run used
    (2.5524): identical formula, identical train split, identical label.
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
    lines.append("  LVH binary label on TRAIN:")
    for lab, pw in zip(C.LABELS, pos_weight(df)):
        pos = int(tr[lab].sum())
        lines.append(f"      {lab:7} {pos:6,} / {len(tr):,} = {pos/len(tr):6.1%}   pos_weight={pw:5.2f}")
    lines.append(f"  sources: {df.source.value_counts().to_dict()}")
    return "\n".join(lines)
