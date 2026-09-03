"""
CardioSentry / CUHK-STEMI staging -- does the label agree with the waveform?

The CUHK labels come from an algorithm reading the same waveform, so they are
not independent evidence. This script asks the one question that can still be
answered from the data alone: do the records labelled STEMI actually carry ST
elevation, measured straight off the converted WFDB signals by code that never
sees a Philips statement?

Method (deliberately simple and inspectable):
  * R peaks from a squared-derivative detector on lead II
  * per lead, a median beat over those peaks
  * PR baseline  = mean over R-80ms .. R-40ms
  * ST amplitude = mean over R+60ms .. R+80ms, minus that baseline
  * per record, the maximum ST amplitude over the 8 independent leads
    (I, II, V1-V6 -- the derived limb leads add no information)

Prints the distribution for STEMI=1 vs the labelled-but-STEMI-negative records
(AF/LVH positives), and the fraction of each meeting the conventional
>= 0.1 mV threshold in at least two leads.

Writes reports/label_signal_agreement.csv (one row per record).
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "cuhk_metadata.csv"
OUT = ROOT / "reports" / "label_signal_agreement.csv"

FS = 500
INDEP = [0, 1, 6, 7, 8, 9, 10, 11]        # I, II, V1..V6


def r_peaks(sig: np.ndarray) -> np.ndarray:
    d = np.diff(sig)
    e = d * d
    w = int(0.10 * FS)
    ma = np.convolve(e, np.ones(w) / w, mode="same")
    thr = 0.35 * np.percentile(ma, 99.5)
    if not np.isfinite(thr) or thr <= 0:
        return np.array([], dtype=int)
    above = ma > thr
    starts = np.nonzero(np.diff(above.astype(np.int8)) == 1)[0]
    med = np.median(sig)
    peaks = []
    for s in starts:
        seg = sig[s: s + int(0.16 * FS)]
        if len(seg):
            peaks.append(s + int(np.argmax(np.abs(seg - med))))
    return np.array(sorted(set(peaks)), dtype=int)


def st_profile(p_signal: np.ndarray) -> np.ndarray | None:
    """p_signal: (n_samples, 12) in mV. Returns per-lead ST amplitude in mV."""
    pk = r_peaks(p_signal[:, 1])
    pre, post = int(0.08 * FS), int(0.10 * FS)
    pk = pk[(pk > pre) & (pk < len(p_signal) - post)]
    if len(pk) < 3:
        return None
    b0, b1 = int(0.08 * FS), int(0.04 * FS)          # baseline window, pre-R
    s0, s1 = int(0.06 * FS), int(0.08 * FS)          # ST window, post-R
    out = np.empty(p_signal.shape[1])
    for i in range(p_signal.shape[1]):
        beats = np.array([p_signal[q - pre: q + post, i] for q in pk])
        m = np.median(beats, axis=0)
        base = m[pre - b0: pre - b1].mean()
        out[i] = m[pre + s0: pre + s1].mean() - base
    return out


def main() -> int:
    if not META.exists():
        print(f"missing {META} -- run scripts/convert_to_wfdb.py", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(META.open(encoding="utf-8")))
    print(f"measuring ST on {len(rows)} converted records")

    results = []
    for n, r in enumerate(rows, 1):
        try:
            rec = wfdb.rdrecord(r["signal_path"])
            st = st_profile(rec.p_signal)
        except Exception:                                      # noqa: BLE001
            st = None
        if st is None:
            results.append({**{k: r[k] for k in ("global_record_id", "STEMI", "AF", "LVH")},
                            "max_st_mV": "", "n_leads_ge_0.1mV": "", "note": "no beats detected"})
        else:
            ind = st[INDEP]
            results.append({
                "global_record_id": r["global_record_id"],
                "STEMI": r["STEMI"], "AF": r["AF"], "LVH": r["LVH"],
                "max_st_mV": round(float(ind.max()), 4),
                "n_leads_ge_0.1mV": int((ind >= 0.1).sum()),
                "note": "",
            })
        if n % 500 == 0:
            print(f"  {n}/{len(rows)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    ok = [x for x in results if x["max_st_mV"] != ""]
    pos = np.array([x["max_st_mV"] for x in ok if x["STEMI"] == "1"])
    neg = np.array([x["max_st_mV"] for x in ok if x["STEMI"] == "0"])
    npos = np.array([x["n_leads_ge_0.1mV"] for x in ok if x["STEMI"] == "1"])
    nneg = np.array([x["n_leads_ge_0.1mV"] for x in ok if x["STEMI"] == "0"])

    print(f"\nmeasurable records: {len(ok)}/{len(results)}")
    print(f"{'group':<22}{'n':>6}{'median max ST':>15}{'p25':>9}{'p75':>9}"
          f"{'>=0.1mV in >=2 leads':>24}")
    for name, a, b in [("STEMI = 1", pos, npos),
                       ("STEMI = 0 (AF/LVH)", neg, nneg)]:
        if len(a) == 0:
            continue
        print(f"{name:<22}{len(a):>6}{np.median(a):>15.3f}"
              f"{np.percentile(a, 25):>9.3f}{np.percentile(a, 75):>9.3f}"
              f"{100 * (b >= 2).mean():>23.0f}%")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
