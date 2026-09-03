#!/usr/bin/env python3
"""
Pick the winning input geometry from a finished screen, on VALIDATION only.

    python pick_dim.py --runs runs/dims            # prints the winner's name
    python pick_dim.py --runs runs/dims --report   # ...and the table it came from

Exists so the whole study can run unattended: run_study.sh calls this between
experiment A and experiment B instead of stopping for a human to read a table
and edit a variable.

--------------------------------------------------------------------------------
WHY VALIDATION AND NOT TEST
--------------------------------------------------------------------------------
The input geometry is a HYPERPARAMETER. Choosing it by test macro AUPRC and then
reporting test macro AUPRC for the nine architectures at that geometry is
selection on the test set: the reported number would be the maximum of six draws,
not an estimate of held-out performance, and the bias is invisible in the number
itself.

So the selection statistic is `best_val_monitor` - the validation macro AUPRC of
the best epoch, which train.py already records for early stopping. Test is never
read here. compare.py still prints the test table for every geometry, and you
should still look at it - but the machine picks on val.

--------------------------------------------------------------------------------
TIE-BREAKING
--------------------------------------------------------------------------------
Val AUPRC differences under ~0.005 on a 922-image val split are noise, not
signal. When the top candidates are within TIE_BAND of each other, the cheaper
one wins - "cheaper" being pixel count, because the geometry chosen here gets
paid for nine more times in experiment B. A geometry that wins by 0.002 AUPRC and
costs 33% more compute is not the winner.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

TIE_BAND = 0.005


def load(root: Path) -> list[dict]:
    out = []
    for f in sorted(root.glob("*/result.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        if r.get("smoke") or "geometry" not in r:
            continue
        out.append(r)
    return out


def pick(runs: list[dict]) -> tuple[str, list[dict]]:
    rows = [{
        "dim": r["dim_name"],
        "model": r["model"],
        "val": float(r.get("best_val_monitor", float("nan"))),
        "px": r["geometry"]["pixels"],
        "px_rel": r["geometry"]["px_rel_sq768"],
        "crop": r["geometry"]["crop"],
        "min": r["train_seconds"] / 60,
        # printed for the human, never used to choose
        "test": r["test"]["macro_auprc"],
        "test_stemi": r["test"]["per_label"]["STEMI"]["auprc"],
    } for r in runs]
    rows.sort(key=lambda d: -d["val"])
    best = rows[0]["val"]
    contenders = [d for d in rows if best - d["val"] <= TIE_BAND]
    winner = min(contenders, key=lambda d: d["px"])
    return winner["dim"], rows


def report(rows: list[dict], winner: str) -> str:
    L = ["geometry screen - SELECTION IS ON VALIDATION ONLY", ""]
    L.append(f"  {'dim':10}{'crop':>6}{'val mAUPRC':>12}{'Mpx':>8}{'x sq768':>9}"
             f"{'min':>7}   {'test mAUPRC':>12}{'test STEMI':>12}")
    L.append("  " + "-" * 78)
    for d in rows:
        mark = "  <- winner" if d["dim"] == winner else ""
        L.append(f"  {d['dim']:10}{'yes' if d['crop'] else 'no':>6}{d['val']:12.4f}"
                 f"{d['px']/1e6:8.3f}{d['px_rel']:9.2f}{d['min']:7.1f}   "
                 f"{d['test']:12.4f}{d['test_stemi']:12.4f}{mark}")
    L.append("")
    L.append(f"  test columns are shown for reading, NOT for choosing - the "
             f"geometry is a")
    L.append(f"  hyperparameter, and picking it on test would make experiment B's "
             f"test numbers")
    L.append(f"  the maximum of {len(rows)} draws rather than a held-out estimate.")
    top = max(d["val"] for d in rows)
    tied = [d["dim"] for d in rows if top - d["val"] <= TIE_BAND]
    if len(tied) > 1:
        L.append("")
        L.append(f"  within {TIE_BAND} val AUPRC of the top: {', '.join(tied)}")
        L.append(f"  -> broke the tie on cost, since experiment B pays for this "
                 f"geometry nine more times.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--fallback", default="crop608",
                    help="printed when the screen produced no usable runs")
    a = ap.parse_args()

    runs = load(a.runs)
    if not runs:
        # stdout stays machine-readable: the driver reads exactly one token.
        import sys
        print(f"no usable result.json under {a.runs} - falling back to "
              f"{a.fallback}", file=sys.stderr)
        print(a.fallback)
        return 0

    winner, rows = pick(runs)
    if a.report:
        import sys
        print(report(rows, winner), file=sys.stderr)
    print(winner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
