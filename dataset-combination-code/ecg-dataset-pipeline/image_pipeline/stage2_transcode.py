"""Stage 2 - harmonise every source into one WFDB dialect.

Three jobs, all of them load-bearing:

1. Format.   The kit loads .mat via loadmat()['val'], which returns raw ADC
   counts, while .dat goes through wfdb.rdrecord().p_signal and returns
   millivolts. The gain is read into a variable in extract_leads.py and then
   never applied, so Chapman renders 1000x over-scale and crashes on write.
   Reading everything through wfdb and writing .dat sidesteps it entirely.

2. Signal.   One identical zero-phase band-pass across all three sources. This
   is the main lever against the source/class confound, so it must never be
   applied per-source. Amplitude is deliberately left alone: normalising it
   would erase the voltage criteria that define LVH.

3. Identity. Records are renamed to an anonymous scheme and given #Age/#Sex
   comments. The kit prints "ID: <record name>" on the sheet, and the native
   names announce the source; PTB-XL and the STEMI dataset also carry no header
   comments at all, so metadata printing yields nothing until they are injected.

The dataset pipeline verified that all three sources are natively 12-lead,
500 Hz, 10 s, mV - so this stage resamples nothing, and refuses anything that
does not match rather than silently reshaping it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
from collections import Counter

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt

import config as C


def _bandpass_sos(fs: int):
    low, high = C.BANDPASS
    nyq = fs / 2.0
    return butter(2, [low / nyq, high / nyq], btype="band", output="sos")


def _canonical_leads(sig_name: list[str]) -> list[int]:
    """Return column indices that put the signal into standard 12-lead order."""
    # Case-folded because PTB-XL writes AVR/AVL/AVF where the other two
    # sources write aVR/aVL/aVF.
    lookup = {name.strip().upper(): i for i, name in enumerate(sig_name)}
    return [lookup[lead.upper()] for lead in C.LEAD_ORDER]


def transcode(row: dict, out_name: str, out_dir, sos) -> str | None:
    """Return None on success, or a short reason string on failure.

    Never raises: a single unreadable record must not abandon a build that is
    already thousands of records in. Stage 1 screens for file existence, but a
    truncated or corrupt signal file can still surface here.
    """
    try:
        return _transcode(row, out_name, out_dir, sos)
    except Exception as exc:                     # noqa: BLE001 - report and continue
        return f"error_{type(exc).__name__}"


def _transcode(row: dict, out_name: str, out_dir, sos) -> str | None:
    src = row["signal_path"]
    if src.endswith((".hea", ".dat", ".mat")):
        src = src.rsplit(".", 1)[0]
    rec = wfdb.rdrecord(src)                     # mV regardless of .dat/.mat

    sig = np.asarray(rec.p_signal, dtype=np.float64)
    if sig.shape[0] != C.N_SAMPLES or sig.shape[1] != len(C.LEAD_ORDER):
        return "bad_shape"
    if not np.isfinite(sig).all():
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        sig = sig[:, _canonical_leads(rec.sig_name)]
    except KeyError:
        return "bad_leads"

    # Zero-phase so the ST segment is not shifted in time - the whole STEMI
    # class depends on ST morphology surviving this step intact.
    sig = sosfiltfilt(sos, sig, axis=0)
    sig = sig - np.median(sig, axis=0, keepdims=True)

    # Keep well inside int16 at 1000 ADU/mV; real ECGs never reach +-32 mV.
    sig = np.clip(sig, -32.0, 32.0)

    # Age/Sex are mandatory ONLY when the header is printed: the kit's
    # --print_header does a bare dict lookup for both and raises KeyError if
    # either is absent. With PRINT_HEADER off nothing reads them.
    #
    # This guard has to agree with the one in stage 1. It did not: stage 1 was
    # relaxed for the header-off build and this backstop was left requiring
    # both, so every age-less record survived stage 1 and was then silently
    # dropped here - which would have thrown away 98.5% of CUHK, the only
    # source of STEMI co-morbidity, while the log showed nothing but an
    # aggregate `missing_metadata` count.
    have_meta = bool(row.get("age")) and row.get("sex") in C.SEX_WORD
    if C.PRINT_HEADER and not have_meta:
        return "missing_metadata"
    comments = ([f"Age: {row['age']}", f"Sex: {C.SEX_WORD[row['sex']]}"]
                if have_meta else [])

    out_dir.mkdir(parents=True, exist_ok=True)
    wfdb.wrsamp(
        record_name=out_name,
        fs=C.FS,
        units=["mV"] * len(C.LEAD_ORDER),
        sig_name=list(C.LEAD_ORDER),
        p_signal=sig,
        fmt=["16"] * len(C.LEAD_ORDER),
        adc_gain=[C.ADC_GAIN] * len(C.LEAD_ORDER),
        baseline=[0] * len(C.LEAD_ORDER),
        comments=comments,
        write_dir=str(out_dir),
    )
    return None


STAGED_FIELDS = ([
    "record", "record_id", "original_id", "cls", "chunk", "source",
] + C.LABEL_COLUMNS + [
    "patient_id", "split", "n_renders", "age", "sex", "note",
    # The band this record was filtered with, carried per row. Redundant across
    # a build by construction, and that is the point: the filter is the one
    # setting that changes the signal invisibly - two corpora built at 40 Hz and
    # 150 Hz render to sheets that look identical, and the only other way to
    # tell them apart is to re-filter the source and diff. Recording it here
    # makes a mis-set ECG_LOWPASS_HZ auditable from the artefact.
    "bandpass_hz",
])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stage only N records")
    ap.add_argument("--force-restage", action="store_true",
                    help="restage even if staged/ already matches the manifest")
    ap.add_argument("--keep-staged", action="store_true",
                    help="do not clear staged/ first (resume a partial staging; "
                         "only safe if the manifest has not changed)")
    ap.add_argument("--fail-above", type=float, default=2.0,
                    help="abort if more than this %% of records fail to stage")
    args = ap.parse_args()

    if not C.MANIFEST.exists():
        print(f"{C.MANIFEST.name} missing - run stage1 first", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(C.MANIFEST.open()))
    if args.limit:
        rows = rows[: args.limit]

    # staged/ is a pure function of the manifest, so it is fingerprinted rather
    # than rebuilt blindly. Same manifest and a complete tree -> skip entirely,
    # which keeps a resumed build cheap. Anything else -> clear and restage,
    # because leftovers from a DIFFERENT manifest are what made stage 3 render
    # thousands of records it was never asked for.
    # The stamp must cover the BAND as well as the manifest. staged/ is a pure
    # function of (manifest, band); fingerprinting the manifest alone meant that
    # re-running with a different ECG_LOWPASS_HZ found a matching stamp, skipped
    # stage 2 entirely, and rendered the previous band's signals under the new
    # build's name - the exact silent failure the printout above guards against.
    _key = C.MANIFEST.read_bytes() + f"|band={C.BANDPASS}".encode()
    fingerprint = hashlib.sha256(_key).hexdigest()
    stamp = C.STAGED / ".manifest.sha256"
    if not args.limit and not args.force_restage and stamp.exists():
        try:
            prev, prev_n = stamp.read_text().split()
            staged_n = sum(1 for _ in C.STAGED.rglob("*.dat"))
            if prev == fingerprint and staged_n == int(prev_n):
                print(f"staged/ already matches this manifest "
                      f"({staged_n} records) - skipping stage 2")
                return 0
        except (ValueError, OSError):
            pass
    if C.STAGED.is_dir() and not args.keep_staged:
        stale = [p for p in C.STAGED.rglob("*") if p.is_file()]
        if stale:
            print(f"clearing {len(stale)} file(s) from a previous staging run "
                  f"({C.STAGED})")
            shutil.rmtree(C.STAGED, ignore_errors=True)

    lo, hi = C.BANDPASS
    src = "ECG_LOWPASS_HZ" if "ECG_LOWPASS_HZ" in os.environ else "config default"
    print(f"band-pass: {lo:g}-{hi:g} Hz, zero-phase 2nd-order Butterworth "
          f"({hi:g} Hz from {src})")
    # Warn on a band the SHEET cannot carry, not on the default. At C.DPI and
    # 25 mm/s the paper holds DPI/25.4*25 px/s, so image Nyquist is half that -
    # 73.8 Hz at 150 DPI. Anything above it cannot be represented and only
    # aliases against the 1 mm grid.
    nyq_img = (C.DPI / 25.4 * 25) / 2.0
    if hi > nyq_img:
        print(f"  WARNING: {hi:g} Hz exceeds the image Nyquist of {nyq_img:.1f} Hz "
              f"at {C.DPI} DPI / 25 mm/s.\n"
              f"  Content above {nyq_img:.1f} Hz cannot appear on the sheet and will "
              f"alias against the 1 mm grid.\n"
              f"  On this corpus a wide band also renders mains hum that is "
              f"concentrated in CUHK (26% of its\n"
              f"  records carry >5% power at 45-65 Hz, vs 0.4% for PTB-XL), and CUHK "
              f"is the only source of\n"
              f"  STEMI co-morbidity - so the artefact correlates with the label. "
              f"Deliberate variant builds only.")
    sos = _bandpass_sos(C.FS)
    stats = Counter()
    mapping = []

    per_class = Counter()
    for i, row in enumerate(rows, start=1):
        cls = row["cls"]
        # Shard by class, then into fixed-size chunks, so stage 3 has many
        # independent batches to spread across cores.
        chunk = per_class[cls] // C.CHUNK_SIZE
        per_class[cls] += 1
        out_dir = C.STAGED / C.slug(cls) / f"c{chunk:03d}"
        out_name = C.RECORD_NAME_FMT.format(i)

        err = transcode(row, out_name, out_dir, sos)
        if err:
            stats[err] += 1
            continue

        stats[f"ok_{cls}"] += 1
        entry = {"record": out_name, "chunk": f"c{chunk:03d}",
                 "bandpass_hz": f"{C.BANDPASS[0]:g}-{C.BANDPASS[1]:g}"}
        entry.update({k: row.get(k, "") for k in STAGED_FIELDS if k in row})
        mapping.append({k: entry.get(k, "") for k in STAGED_FIELDS})

        if i % 1000 == 0:
            print(f"  {i}/{len(rows)}")

    if not mapping:
        print("nothing staged", file=sys.stderr)
        return 1

    C.STAGED_MAP.parent.mkdir(parents=True, exist_ok=True)
    with C.STAGED_MAP.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=STAGED_FIELDS)
        w.writeheader()
        w.writerows(mapping)

    print(f"\nstaged {len(mapping)} of {len(rows)} records -> {C.STAGED}")
    for k in sorted(stats):
        print(f"    {k:22s} {stats[k]:6d}")
    if not args.limit:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(f"{fingerprint} {len(mapping)}")
    print(f"wrote {C.STAGED_MAP}")

    failed = len(rows) - len(mapping)
    if failed:
        pct = 100 * failed / len(rows)
        print(f"\n{failed} records ({pct:.1f}%) could not be staged.")
        if pct > args.fail_above:
            print("That is high enough to skew the class balance - investigate")
            print("before rendering rather than after.")
            return 1
        print("Small enough to ignore; class balance is essentially unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
