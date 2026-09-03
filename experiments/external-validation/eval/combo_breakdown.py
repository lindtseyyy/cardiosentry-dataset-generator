#!/usr/bin/env python3
"""
Per-LABEL-COMBINATION breakdown of a run's predictions.

    python eval/combo_breakdown.py                          # default run
    python eval/combo_breakdown.py --run resnet50_cbam@nat768
    python eval/combo_breakdown.py --splits test            # internal test only

WHAT IT ANSWERS
---------------
The four heads are INDEPENDENT binary classifiers, but the sheets are not: AF
and LVH co-occur, and a macro metric averages the co-occurring records into the
same number as the clean ones. So "LVH AUROC 0.966" can hide a head that is
excellent on isolated LVH and close to a coin flip when the rhythm is also
abnormal - which is a different defect, with a different fix, from a head that
is uniformly mediocre.

This splits every metric by the EXACT set of true labels a sheet carries, and
then runs the interference test that motivates the whole file:

    for each label pair (A, B) that co-occurs, is A's recall on {A alone}
    different from A's recall on {A and B}?

Recall is the right statistic for that question because it is computed on
positives only, so the comparison is not contaminated by how many negatives
each group happens to sit beside. The accompanying AUROC scores each positive
group against the SAME negative pool (every sheet lacking the label), which is
what makes the two rows comparable to each other rather than to themselves.

WHAT IT CANNOT ANSWER
---------------------
Whether the interference is the LABEL or the PHYSIOLOGY. An AF+LVH sheet has an
irregular baseline and no P waves, and any of that may be what degrades the LVH
voltage read. This file measures that the degradation exists and how large it
is; separating cause needs the waveform, not the prediction CSV.

Co-occurrence groups are also small - the internal test set has 28 AF+LVH
records - so every interference figure carries a bootstrap interval and none of
them should be quoted without it.

THRESHOLDS come from the run's result.json, frozen on its original validation
split, exactly as everywhere else in this folder. Nothing here re-tunes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

LABELS = ["STEMI", "AF", "LVH", "NORMAL"]
DEFAULT_RUN = "efficientnetv2_s_cbam@nat768"
BOOTSTRAP = 4000


def _repo_root(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / "prototype-code").is_dir() and (p / "prototype-results").is_dir():
            return p
    raise SystemExit(f"no repo root above {start}")


REPO = _repo_root(HERE)


# ------------------------------------------------------------------ loading
def read_predictions(path: Path) -> pd.DataFrame:
    """Normalise the two prediction-CSV conventions onto y_/p_ columns.

    train.py writes `true_X` / `prob_X`; evaluate_run.py writes `y_X` / `p_X`.
    Normalising here is what lets the internal and external sheets be compared
    in one table instead of two scripts.
    """
    d = pd.read_csv(path)
    ren = {}
    for lab in LABELS:
        if f"true_{lab}" in d.columns:
            ren[f"true_{lab}"], ren[f"prob_{lab}"] = f"y_{lab}", f"p_{lab}"
    d = d.rename(columns=ren)
    missing = [c for lab in LABELS for c in (f"y_{lab}", f"p_{lab}")
               if c not in d.columns]
    if missing:
        raise SystemExit(f"{path} is missing {missing}")
    if "source" not in d.columns:
        d["source"] = "?"
    d["combo"] = d.apply(
        lambda r: "+".join(l for l in LABELS if r[f"y_{l}"] == 1) or "(none)", axis=1)
    return d


def collect(run_dir: Path, splits: list[str], external: Path | None) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for sp in splits:
        f = run_dir / "predictions" / f"{sp}_predictions.csv"
        if f.exists():
            out.append((f"internal {sp}", read_predictions(f)))
        else:
            print(f"  (no {f.name} in {run_dir/'predictions'})")
    if external and external.exists():
        out.append(("external", read_predictions(external)))
    return out


# ------------------------------------------------------------------ metrics
def boot_auroc(y, p, seed: int = 42, n: int = BOOTSTRAP) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        v.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v \
        else (float("nan"), float("nan"))


def census(d: pd.DataFrame) -> None:
    print("  combinations present:")
    ct = pd.crosstab(d.combo, d.source)
    for line in ct.to_string().splitlines():
        print("    " + line)


def per_combo(d: pd.DataFrame, thr: dict) -> None:
    """Recall where the combo carries the label, false-positive rate where it does not."""
    print(f"\n  {'combo':12}{'n':>5}  " + "".join(f"{l:>11}" for l in LABELS))
    for cb in sorted(d.combo.unique(), key=lambda c: -len(d[d.combo == c])):
        s = d[d.combo == cb]
        row = f"  {cb:12}{len(s):5d}  "
        for l in LABELS:
            r = float((s[f"p_{l}"] >= thr[l]).mean())
            row += f"{r:11.3f}" if s[f"y_{l}"].sum() > 0 else f"{'(' + format(r, '.3f') + ')':>11}"
        print(row)
    print("  recall where the combo carries the label; (parenthesised) = false-positive rate")

    print(f"\n  mean predicted probability")
    print(f"  {'combo':12}{'n':>5}  " + "".join(f"{'p(' + l + ')':>11}" for l in LABELS))
    for cb in sorted(d.combo.unique(), key=lambda c: -len(d[d.combo == c])):
        s = d[d.combo == cb]
        print(f"  {cb:12}{len(s):5d}  " + "".join(f"{s[f'p_{l}'].mean():11.3f}" for l in LABELS))


def interference(d: pd.DataFrame, thr: dict) -> None:
    """For each co-occurring pair, is the head worse when the partner is present?"""
    pairs = []
    for cb in d.combo.unique():
        parts = cb.split("+")
        if len(parts) > 1:
            for a in parts:
                pairs.append((a, "+".join(x for x in parts if x != a), cb))
    if not pairs:
        print("\n  no co-occurring labels in this split - nothing to test")
        return

    print(f"\n  {'head':7}{'group':16}{'n':>5}{'recall':>9}{'mean p':>9}"
          f"{'AUROC vs all negatives':>26}")

    def row(a, nm, g, neg):
        y = np.r_[np.ones(len(g)), np.zeros(len(neg))]
        p = np.r_[g[f"p_{a}"].values, neg[f"p_{a}"].values]
        lo, hi = boot_auroc(y, p)
        print(f"  {a:7}{nm:16}{len(g):5d}{(g[f'p_{a}'] >= thr[a]).mean():9.3f}"
              f"{g[f'p_{a}'].mean():9.3f}"
              f"{roc_auc_score(y, p):15.4f} [{lo:.3f}, {hi:.3f}]")

    # One block per head: the isolated group once, then each partner under it,
    # so the eye compares down a column instead of across repeated baselines.
    for a in LABELS:
        partners = sorted({(pt, cb) for h, pt, cb in pairs if h == a})
        alone = d[d.combo == a]
        neg = d[d[f"y_{a}"] == 0]
        if not partners or not len(alone):
            continue
        row(a, f"{a} alone", alone, neg)
        for pt, cb in partners:
            both = d[d.combo == cb]
            if len(both):
                row(a, f"{a} + {pt}", both, neg)
    print("  both rows are scored against the SAME negative pool, so they compare directly")


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--runs-root", default=str(REPO / "prototype-results"))
    ap.add_argument("--splits", nargs="*", default=["test", "val"])
    ap.add_argument("--external", default=None,
                    help="external predictions CSV; default: the one "
                         "evaluate_run.py wrote for this run, if present")
    args = ap.parse_args()

    run_dir = Path(args.runs_root) / args.run
    if not run_dir.is_dir():
        run_dir = Path(args.run)
    if not run_dir.is_dir():
        raise SystemExit(f"no run directory {args.run}")

    rj = run_dir / "result.json"
    if not rj.exists():
        raise SystemExit(f"{rj} not found - thresholds must come from the run")
    thr = dict(zip(LABELS, json.loads(rj.read_text())["thresholds"]))

    ext = Path(args.external) if args.external else None
    if ext is None:
        for cand in sorted((EXP / "results").glob(f"*/predictions/{run_dir.name}.csv")):
            ext = cand
            break

    print(f"run         {run_dir}")
    print(f"thresholds  {{" + ", ".join(f"{k} {v:.2f}" for k, v in thr.items()) + "}  (frozen on original val)")

    frames = collect(run_dir, args.splits, ext)
    if not frames:
        raise SystemExit("no prediction CSVs found")

    seen: set[str] = set()
    for name, d in frames:
        print("\n" + "=" * 78)
        print(f"{name}   {len(d)} sheets")
        print("=" * 78)
        census(d)
        per_combo(d, thr)
        interference(d, thr)
        if name.startswith("internal test"):
            seen = set(d.combo.unique())
        elif seen:
            novel = [c for c in d.combo.unique() if c not in seen]
            if novel:
                print(f"\n  COMBINATIONS THE INTERNAL TEST SET NEVER CONTAINED: "
                      f"{', '.join(novel)}")
                print("  the model has no measured behaviour on these at all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
