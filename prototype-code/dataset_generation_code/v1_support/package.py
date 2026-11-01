"""Deterministic TAR packaging + SHA-256 sidecars (PLAN 16)."""
from __future__ import annotations

import csv
import hashlib
import tarfile
from pathlib import Path

from .config import SUBSETS, TAR_MTIME
from .util import V1Error, log, sha256_file


def _reset(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    ti.mtime = TAR_MTIME
    ti.uid = 0
    ti.gid = 0
    ti.uname = ""
    ti.gname = ""
    ti.pax_headers = {}
    return ti


def build_tar(dataset_dir: str | Path, tar_path: str | Path,
              top_entries: list[str]) -> dict:
    """Create a deterministic uncompressed TAR of dataset/<entries>.

    top_entries: e.g. ["train","val_clean","val_stress","test_clean",
    "test_stress","splits","full_metadata.csv",...].
    """
    dataset_dir, tar_path = Path(dataset_dir), Path(tar_path)
    for e in top_entries:
        if not (dataset_dir / e).exists():
            raise V1Error(f"TAR entry missing: {e}")
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tar_path.with_suffix(".tar.tmp")
    if tmp.exists():
        tmp.unlink()
    members: list[str] = []
    with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as tf:
        for e in sorted(top_entries):
            src = dataset_dir / e
            if src.is_dir():
                # Sort by arcname string (LC_ALL=C byte order per PLAN §16),
                # NOT by Path: Path-tuple ordering differs from byte order
                # ('a/b' vs 'a-b') and would silently break the contract.
                files = sorted(
                    (f"{e}/{f.relative_to(src)}", f)
                    for f in src.rglob("*") if f.is_file())
                for arc, f in files:
                    ti = tf.gettarinfo(str(f), arcname=arc)
                    ti = _reset(ti)
                    with open(f, "rb") as fh:
                        tf.addfile(ti, fh)
                    members.append(arc)
            else:
                ti = tf.gettarinfo(str(src), arcname=e)
                ti = _reset(ti)
                with open(src, "rb") as fh:
                    tf.addfile(ti, fh)
                members.append(e)
    tmp.replace(tar_path)
    digest = sha256_file(tar_path)
    (tar_path.parent / (tar_path.name + ".sha256")).write_text(
        f"{digest}  {tar_path.name}\n")
    log(f"TAR: {tar_path.name} ({len(members)} members, "
        f"{tar_path.stat().st_size} bytes, sha256={digest[:16]}…)")
    return {"path": str(tar_path), "sha256": digest,
            "bytes": tar_path.stat().st_size, "members": members}


def verify_tar_listing(tar_path: str | Path, full_metadata_csv: str | Path,
                       extra_members: set[str]) -> None:
    """Every metadata image_path is in the TAR; no unexpected members."""
    with open(full_metadata_csv, newline="") as fh:
        want = {r["image_path"] for r in csv.DictReader(fh)}
    with tarfile.open(tar_path, "r") as tf:
        have = set(tf.getnames())
    imgs = {m for m in have if m.endswith(".png")}
    if imgs != want:
        raise V1Error(f"TAR/metadata image mismatch: "
                      f"missing={sorted(want - imgs)[:3]} "
                      f"extra={sorted(imgs - want)[:3]}")
    rest = have - want - extra_members
    # splits/* + top-level sidecars + the five per-subset metadata.csv
    # files are allowed; anything else (kit droppings, strays) fails.
    subset_meta = {f"{s}/metadata.csv" for s in SUBSETS}
    allowed = {m for m in rest if m.startswith("splits/") or "/" not in m
               or m in subset_meta}
    if rest - allowed:
        raise V1Error(f"unexpected TAR members: {sorted(rest - allowed)[:5]}")
    log(f"TAR listing verified: {len(imgs)} images + {len(have - imgs)} sidecars")
