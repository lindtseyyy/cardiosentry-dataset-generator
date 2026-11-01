"""Minimal WFDB I/O for the V1 subset (PLAN 0.5, stage format path).

Deliberately dependency-free (numpy + scipy only): the support package must
stay importable before the render venv exists, and explicit gain handling is
exactly what the kit's .mat path gets wrong. Only what the subset contains
is supported — anything else fails loudly:

  read:  .mat ('val' int16, header fmt '16x1+24') and .dat (fmt 16, int16 LE)
  write: .dat fmt 16 + matching .hea (scratch staging for the kit)

Physical units: millivolts, physical = raw_int16 / ADC_GAIN with ADC_GAIN =
1000 and adc_zero = 0 (asserted on every file; the subset was verified to
use exactly this convention).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .config import ADC_GAIN, CANONICAL_LEADS
from .util import V1Error


_CANON_UPPER = {lead.upper(): lead for lead in CANONICAL_LEADS}


def canonical_lead(name: str) -> str:
    try:
        return _CANON_UPPER[name.strip().upper()]
    except KeyError:
        raise V1Error(f"non-canonical lead name {name!r}") from None


def parse_header(hea_path: str | Path) -> dict:
    """Parse a WFDB header into first-line fields, per-lead rows, comments."""
    lines = Path(hea_path).read_text().splitlines()
    if not lines:
        raise V1Error(f"empty header {hea_path}")
    first = lines[0].split()
    if len(first) < 4:
        raise V1Error(f"malformed header first line in {hea_path}: {lines[0]!r}")
    rec, nleads, fs, nsamp = first[0], int(first[1]), float(first[2]), int(first[3])
    leads = []
    comments = {}
    for line in lines[1:]:
        if line.startswith("#"):
            body = line[1:].strip()
            if ":" in body:
                k, v = body.split(":", 1)
                comments[k.strip()] = v.strip()
            continue
        if not line.strip():
            continue
        f = line.split()
        if len(f) < 9:
            raise V1Error(f"malformed lead line in {hea_path}: {line!r}")
        adc_field = f[2]
        gain = float(adc_field.split("(")[0])
        leads.append({"file": f[0], "fmt": f[1], "gain": gain,
                      "adc_field": adc_field, "adc_zero": f[4],
                      "lead": f[-1]})
    return {"record": rec, "nleads": nleads, "fs": fs, "nsamp": nsamp,
            "leads": leads, "comments": comments, "path": str(hea_path)}


def _assert_convention(meta: dict) -> None:
    if meta["nleads"] != 12:
        raise V1Error(f"{meta['path']}: nleads={meta['nleads']}, need 12")
    for row in meta["leads"]:
        if abs(row["gain"] - ADC_GAIN) > 1e-9 or row["adc_zero"] != "0":
            raise V1Error(f"{meta['path']}: lead {row['lead']}: gain={row['gain']}"
                          f" adc_zero={row['adc_zero']}, need 1000/0")


def lead_order_index(meta: dict) -> list[int]:
    """Column indices reordering file order -> canonical 12-lead order."""
    names = [canonical_lead(r["lead"]) for r in meta["leads"]]
    if sorted(names) != sorted(CANONICAL_LEADS):
        raise V1Error(f"{meta['path']}: lead set {names} is not canonical 12")
    pos = {name: i for i, name in enumerate(names)}
    return [pos[lead] for lead in CANONICAL_LEADS]


def read_record_mv(hea_path: str | Path) -> tuple[np.ndarray, dict]:
    """Read one subset record. Returns (sig_mV float64 (12, N), meta).

    meta gains: fs, nsamp, duration_s, lead_names (canonical), fmt, age, sex.
    No filtering, no resampling, no amplitude change — int16/1000 only.
    """
    hea_path = Path(hea_path)
    meta = parse_header(hea_path)
    _assert_convention(meta)
    order = lead_order_index(meta)
    fmts = {r["fmt"] for r in meta["leads"]}
    if fmts == {"16x1+24"}:
        mat_path = hea_path.with_suffix(".mat")
        if not mat_path.exists():
            # file field may carry the real name
            mat_path = hea_path.parent / meta["leads"][0]["file"]
        arr = np.asarray(loadmat(str(mat_path))["val"])
        if arr.shape == (meta["nsamp"], 12):
            arr = arr.T
        if arr.shape != (12, meta["nsamp"]):
            raise V1Error(f"{hea_path}: .mat shape {arr.shape}, "
                          f"need (12, {meta['nsamp']})")
        raw = arr.astype(np.int64)
        fmt = "mat"
    elif fmts == {"16"}:
        dat_path = hea_path.with_suffix(".dat")
        if not dat_path.exists():
            dat_path = hea_path.parent / meta["leads"][0]["file"]
        n = 12 * meta["nsamp"]
        raw = np.fromfile(str(dat_path), dtype="<i2", count=n).astype(np.int64)
        if raw.size != n:
            raise V1Error(f"{hea_path}: .dat holds {raw.size} samples, need {n}")
        raw = raw.reshape(meta["nsamp"], 12).T
        fmt = "dat"
    else:
        raise V1Error(f"{hea_path}: mixed/unsupported fmts {sorted(fmts)}")
    sig = raw[order, :] / ADC_GAIN
    meta_out = {"fs": meta["fs"], "nsamp": meta["nsamp"],
                "duration_s": meta["nsamp"] / meta["fs"],
                "lead_names": list(CANONICAL_LEADS), "fmt": fmt,
                "age": meta["comments"].get("Age", ""),
                "sex": meta["comments"].get("Sex", "")}
    return sig.astype(np.float64), meta_out


def signal_int16_hash_mv(sig_mv: np.ndarray) -> str:
    """Content hash of the int16 ADC representation (patient join key).

    Inverse of the read path: round(mV * 1000) -> int16 -> lead-major bytes.
    Used to verify precomputed content joins without the upstream copy.
    """
    import hashlib
    raw = np.ascontiguousarray(np.round(sig_mv * ADC_GAIN).astype("<i2"))
    return hashlib.sha256(raw.tobytes()).hexdigest()


def write_dat(base_path: str | Path, sig_mv: np.ndarray, fs: float,
              comments: dict | None = None) -> tuple[Path, Path]:
    """Write scratch staging .hea + .dat (fmt 16, gain 1000, adc_zero 0).

    sig_mv: (12, N) float64 in canonical lead order. Values are rounded to
    int16 counts (exact for subset signals, which came from int16/1000).
    """
    base = Path(base_path)
    if sig_mv.shape[0] != 12:
        raise V1Error(f"{base}: shape {sig_mv.shape}, need (12, N)")
    n = sig_mv.shape[1]
    raw = np.round(sig_mv * ADC_GAIN)
    # True int16 range: -32768 (INT16_MIN, ADC negative rail) is legal —
    # abs() would wrongly reject it (abs(-32768) = 32768). v1.13.
    if raw.min() < -32768 or raw.max() > 32767:
        raise V1Error(f"{base}: signal exceeds int16 at gain 1000")
    interleaved = np.ascontiguousarray(raw.astype("<i2").T)
    hea, dat = base.with_suffix(".hea"), base.with_suffix(".dat")
    name = base.name
    lines = [f"{name} 12 {int(fs) if float(fs).is_integer() else fs} {n}"]
    for lead in CANONICAL_LEADS:
        lines.append(f"{name}.dat 16 1000.0(0)/mV 16 0 0 0 0 {lead}")
    for key in ("Age", "Sex"):
        if comments and comments.get(key):
            lines.append(f"# {key}: {comments[key]}")
    hea.write_text("\n".join(lines) + "\n")
    interleaved.tofile(str(dat))
    return hea, dat
