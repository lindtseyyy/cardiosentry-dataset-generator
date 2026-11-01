"""Scratch WFDB staging (PLAN 0.5 format path, no filter).

Reads each accepted record (mat or dat) to mV via wfdbio, writes canonical
scratch .dat (fmt 16, gain 1000, canonical lead order) for the kit, and
verifies the int16 round-trip. Source files are only read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .util import V1Error, ensure_dir, log
from . import wfdbio


def stage_record(record: dict, out_dir: str | Path) -> dict:
    """Stage one record; return staging info. Verifies round-trip exactly."""
    out_dir = ensure_dir(out_dir)
    sig, meta = wfdbio.read_record_mv(record["subset_output_path"])
    base = out_dir / record["record_id"]
    hea, dat = wfdbio.write_dat(
        base, sig, meta["fs"],
        comments={"Age": meta["age"], "Sex": meta["sex"]})
    # round-trip: staged int16 must equal source int16 exactly
    back, _ = wfdbio.read_record_mv(hea)
    a = np.round(sig * 1000).astype(np.int64)
    b = np.round(back * 1000).astype(np.int64)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise V1Error(f"{record['record_id']}: staging round-trip mismatch")
    return {"record_id": record["record_id"], "staged_hea": str(hea),
            "staged_dat": str(dat), "fs": meta["fs"], "nsamp": meta["nsamp"]}


def stage_chunk(records: list[dict], out_dir: str | Path) -> list[dict]:
    out = []
    for r in records:
        try:
            out.append(stage_record(r, out_dir))
        except V1Error as exc:
            raise V1Error(f"staging failed for {r['record_id']}: {exc}") from exc
    log(f"staged {len(out)} records -> {out_dir}")
    return out
