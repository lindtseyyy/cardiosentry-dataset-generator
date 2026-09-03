"""
AF x LVH subgroup scoring, for every finished hybrid run.

WHY THIS MODULE EXISTS

The AF x LVH subgroup analyses (af_lvh_subgroup_v1, af_lvh_joint_v1) scored the
FROZEN reference model and returned a SUPPORTED verdict: at 768, resnet50_hires
detects AF correctly but disproportionately misses LVH when AF coexists.

    LVH recall on AF+LVH   0.6429  (18/28)      AF recall on AF+LVH   0.9286  (26/28)
    LVH recall on LVH-only 0.8722  (198/227)    AF recall on AF-only  0.9416  (242/257)
    delta -0.2294, Fisher one-sided p = 0.0038  delta -0.0131, Fisher p = 0.5142
    LVH AUPRC on AF+LVH (reference pool) 0.4535

Those analyses were post-hoc - they did not retrain anything, and neither does
this module. Every NEW run in this round is scored the same way, automatically,
from the per-image test predictions train.py already writes, so the question
"does a better feature representation preserve LVH information when AF is
present?" is answered inside the run's own report instead of a side analysis.

METHODOLOGY, PORTED UNCHANGED FROM af_lvh_joint_v1

  * subgroups are defined by GROUND TRUTH on the frozen 938-image test split;
  * decisions are the run's OWN frozen thresholds (the pred_ columns - never
    re-thresholded here);
  * recall is computed INSIDE each subgroup (positives only exist inside it);
  * AUPRC uses the reference-pool design: the subgroup is the positive class
    and the negatives are every test ECG with true_<head> = 0. It is a
    small-sample read (28 positives vs a 653/683-case pool) and is reported
    as asked, not as a verdict read;
  * the one-sided Fisher test compares recall on AF+LVH against the
    AF-only / LVH-only comparand, exactly as in the joint analysis.

The frozen reference numbers live in baseline.AF_LVH_FROZEN, transcribed once
from prototype-results/af_lvh_joint_v1/report.txt, so no later run can move the
bar it is scored against.

    python af_lvh.py --runs <run_dir> [--csv <run_dir>/af_lvh.csv]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import baseline
import config as C

# Subgroup definitions by ground truth. Frozen test-split geometry: sizes are
# asserted at load time, because a run whose test split disagrees is not scored
# on the same subgroup at all.
SUBGROUPS = {
    "af_lvh":   {"AF": 1, "LVH": 1, "display": "AF + LVH"},
    "af_only":  {"AF": 1, "LVH": 0, "display": "AF-only"},
    "lvh_only": {"AF": 0, "LVH": 1, "display": "LVH-only"},
}

# Which subgroup each head is read inside, and its comparand. The AF+LVH
# subgroup is the round's subject; AF-only and LVH-only are what its recall is
# compared against - the same design as af_lvh_joint_v1's R2/R3.
HEAD_ROWS = {
    "AF":  ("af_lvh", "af_only"),
    "LVH": ("af_lvh", "lvh_only"),
}

N_RESAMPLES = 1200      # project convention
SEED = C.SEED

REQUIRED_COLS = {"source", "record", "rel_path", "split"} | {
    f"{p}_{l}" for l in ("AF", "LVH") for p in ("true", "prob", "pred")}


# --------------------------------------------------------------------- loading
def load_predictions(run_dir: Path):
    """Per-image test predictions, or None (smoke runs write none).

    Returns None rather than raising so a half-finished batch still tabulates;
    the caller reports the missing run.
    """
    p = Path(run_dir) / "predictions" / "test_predictions.csv"
    if not p.exists():
        return None
    import pandas as pd
    df = pd.read_csv(p)
    if not REQUIRED_COLS <= set(df.columns):
        return None
    if len(df) != baseline.TEST_N or (df["split"] != "test").any():
        return None
    return df


def subgroup_mask(df, key: str):
    spec = SUBGROUPS[key]
    m = np.ones(len(df), dtype=bool)
    for lab, val in spec.items():
        if lab == "display":
            continue
        m &= (df[f"true_{lab}"].astype(int).values == val)
    return m


# --------------------------------------------------------------------- stats
def fisher_less_p(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p, alternative='less' (subgroup worse than its
    comparand), on [[a, b], [c, d]] - ported from af_lvh_joint_v1/stats.py."""
    n1, n2 = a + b, c + d
    col1 = a + c
    n = n1 + n2
    num = 0
    for x in range(0, a + 1):
        if col1 >= x and n - col1 >= n1 - x:
            num += math.comb(col1, x) * math.comb(n - col1, n1 - x)
    return num / math.comb(n, n1)


def _recall_ci(dec_a: np.ndarray, dec_b: np.ndarray):
    """Delta (a - b) of two groups' recalls with a percentile 95% CI.

    Independent resampling per group, one generator stream so re-runs
    reproduce the exact intervals - same convention as af_lvh_joint_v1.
    """
    a = np.asarray(dec_a).astype(int)
    b = np.asarray(dec_b).astype(int)
    rng = np.random.default_rng(SEED)
    ia = rng.integers(0, len(a), size=(N_RESAMPLES, len(a)))
    ib = rng.integers(0, len(b), size=(N_RESAMPLES, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo = float(np.percentile(d, 2.5))
    hi = float(np.percentile(d, 97.5))
    return (float(a.mean() - b.mean()), lo, hi,
            float(a.mean()), float(b.mean()), int(a.sum()), int(b.sum()))


def _ref_pool_auprc(df, head: str, sub_key: str) -> float:
    """AUPRC with the subgroup as positive class and every true_<head> == 0
    ECG as the negative pool (the af_lvh_joint_v1 reference-pool design)."""
    from sklearn.metrics import average_precision_score
    mask = subgroup_mask(df, sub_key)
    y = mask.astype(int)
    keep = mask | (df[f"true_{head}"].astype(int).values == 0)
    p = df[f"prob_{head}"].values[keep]
    y = y[keep]
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 \
        else float("nan")


# --------------------------------------------------------------------- scoring
def score_run(run_dir: Path, model: str | None = None) -> dict | None:
    """One row per finished run: both heads, inside AF+LVH and its comparand.

    Returns None when the run has no usable per-image predictions.
    """
    df = load_predictions(run_dir)
    if df is None:
        return None

    sizes = {k: int(subgroup_mask(df, k).sum()) for k in SUBGROUPS}
    size_note = [f"{k}={sizes[k]}" for k in SUBGROUPS
                 if sizes[k] != baseline.AF_LVH_FROZEN["n"][k]]

    out = {"model": model or Path(run_dir).name,
           "af_lvh_n": sizes["af_lvh"],
           "subgroup_sizes_match": not size_note,
           "size_note": ", ".join(size_note)}
    for head, (sub_key, comp_key) in HEAD_ROWS.items():
        sub = subgroup_mask(df, sub_key)
        comp = subgroup_mask(df, comp_key)
        ok_sub = df[f"pred_{head}"].astype(int).values[sub]
        ok_comp = df[f"pred_{head}"].astype(int).values[comp]
        a, b = int(ok_sub.sum()), int(len(ok_sub) - ok_sub.sum())
        c, d = int(ok_comp.sum()), int(len(ok_comp) - ok_comp.sum())
        delta, lo, hi, ra, rb, _, _ = _recall_ci(ok_sub, ok_comp)
        key = head.lower()
        out.update({
            f"{key}_recall_af_lvh": float(ok_sub.mean()),
            f"{key}_recall_comparand": float(ok_comp.mean()),
            f"{key}_delta": float(delta),
            f"{key}_delta_ci_low": lo,
            f"{key}_delta_ci_high": hi,
            f"{key}_fisher_p": fisher_less_p(a, b, c, d),
            f"{key}_tp_af_lvh": a,
            f"{key}_fn_af_lvh": b,
            f"{key}_auprc_af_lvh": _ref_pool_auprc(df, head, sub_key),
        })
    return out


# ----------------------------------------------------------------- formatting
def format_section(rows: list[dict], missing: list[str]) -> list[str]:
    F = baseline.AF_LVH_FROZEN
    L = ["", "=" * 108,
         "AF x LVH SUBGROUP - does LVH survive when AF is present?",
         "=" * 108,
         "  subgroups by GROUND TRUTH on the frozen test split; decisions are",
         "  each run's OWN frozen thresholds (pred_ columns, never re-tuned).",
         "  AUPRC is the reference-pool design (subgroup positives vs every",
         "  true_<head> == 0 ECG); recall is computed inside the subgroup.",
         "",
         "  THE FINDING A NEW MODEL HAS TO BEAT (frozen, af_lvh_joint_v1 on",
         f"  {baseline.AF_LVH_REFERENCE}):",
         f"    LVH recall on AF+LVH   {F['lvh']['recall_af_lvh']:.4f}  vs "
         f"{F['lvh']['recall_lvh_only']:.4f} on LVH-only  "
         f"(delta {F['lvh']['delta']:+.4f}, Fisher one-sided p = {F['lvh']['fisher_p']:.4f})",
         f"    AF  recall on AF+LVH   {F['af']['recall_af_lvh']:.4f}  vs "
         f"{F['af']['recall_af_only']:.4f} on AF-only   "
         f"(delta {F['af']['delta']:+.4f}, Fisher p = {F['af']['fisher_p']:.4f})",
         f"    LVH AUPRC on AF+LVH    {F['lvh']['auprc_af_lvh']:.4f}   "
         f"AF AUPRC on AF+LVH {F['af']['auprc_af_lvh']:.4f}   (reference pool)",
         f"    -> the 768 reference detects AF but missed 10 of the 28 AF+LVH",
         f"       cases' LVH. THE QUESTION: does resnet50_vit at 768 preserve",
         f"       that LVH information?",
         ""]
    if not rows:
        L += ["  NO PER-IMAGE PREDICTIONS FOUND - re-run without --smoke, or",
              "  train something first.", ""]
        return L

    hdr = (f"{'model':<22}{'views':>9}{'nA+L':>6}"
           f"{'LVHrec@A+L':>11}{'LVHrec@L':>10}{'d':>8}{'CI95':>17}{'p':>8}"
           f"{'LVH AUPRC@A+L':>15}{'AFrec@A+L':>11}{'AFrec@A':>9}{'d':>8}"
           f"{'p':>8}{'AF AUPRC@A+L':>15}")
    L += [hdr, "-" * len(hdr)]
    for r in rows:
        L.append(
            f"{r['model']:<22}{r['views']:>9}{r['af_lvh_n']:>6}"
            f"{r['lvh_recall_af_lvh']:11.4f}{r['lvh_recall_comparand']:10.4f}"
            f"{r['lvh_delta']:+8.4f} "
            f"[{r['lvh_delta_ci_low']:+.3f},{r['lvh_delta_ci_high']:+.3f}]"
            f"{r['lvh_fisher_p']:8.4f}{r['lvh_auprc_af_lvh']:15.4f}"
            f"{r['af_recall_af_lvh']:11.4f}{r['af_recall_comparand']:9.4f}"
            f"{r['af_delta']:+8.4f}{r['af_fisher_p']:8.4f}"
            f"{r['af_auprc_af_lvh']:15.4f}")
    L += ["", f"  {'model':<22}{'':>9}{'':>6}{'TP':>11}{'':>10}{'':>8}"
              f"{'':>17}{'':>8}{'':>15}{'TP':>11}{'':>9}{'':>8}{'':>8}"]
    for r in rows:
        L.append(
            f"  {r['model']:<22}{'':>9}{'':>6}{r['lvh_tp_af_lvh']:>11}"
            f"{'':>10}{'':>8}{'':>17}{'':>8}{'':>15}{r['af_tp_af_lvh']:>11}")
    L += ["",
          "  LVHrec@A+L / AFrec@A+L are recall INSIDE the 28 AF+LVH cases;",
          "  LVHrec@L / AFrec@A are the LVH-only / AF-only comparands. p is the",
          "  one-sided Fisher test that the AF+LVH subgroup is WORSE (the read",
          "  that matters: a low p for LVH = AF still breaks the LVH head).",
          "  CI95 is the bootstrap CI of the recall delta. AUPRC columns are",
          "  small-sample reads of the reference-pool design, not verdict reads.",
          "",
          "  Reading the verdict: the model preserves LVH under AF if LVHrec@A+L",
          "  rises materially from 0.6429 AND AFrec@A+L does not fall from 0.9286.",
          "  The 28-case subgroup is small - the Fisher p and CI say how much of",
          "  any delta is real. No causal claim is made.",
          ""]
    if missing:
        L += [f"  MISSING per-image predictions (excluded above): "
              f"{', '.join(missing)}", ""]
    if any(r.get("size_note") for r in rows):
        for r in rows:
            if r.get("size_note"):
                L += [f"  WARNING {r['model']}: subgroup sizes differ from the "
                      f"frozen test split ({r['size_note']}) - scored on a "
                      f"different subgroup, not comparable.", ""]
    return L


def format_run_block(row: dict, views: str = "?") -> list[str]:
    """Compact per-model rendering for train.py's report.txt."""
    F = baseline.AF_LVH_FROZEN
    L = [
        f"  {'LVH head':<12} AF+LVH:  TP {row['lvh_tp_af_lvh']:>2} / FN "
        f"{row['lvh_fn_af_lvh']:>2}   recall {row['lvh_recall_af_lvh']:.4f}"
        f"  (n={row['af_lvh_n']})",
        f"  {'':<12} LVH-only: recall {row['lvh_recall_comparand']:.4f}"
        f"   delta {row['lvh_delta']:+.4f}  "
        f"95% CI [{row['lvh_delta_ci_low']:+.3f}, {row['lvh_delta_ci_high']:+.3f}]"
        f"   Fisher one-sided p = {row['lvh_fisher_p']:.4f}",
        f"  {'':<12} LVH AUPRC on AF+LVH (reference pool): "
        f"{row['lvh_auprc_af_lvh']:.4f}",
        f"  {'AF head':<12} AF+LVH:  TP {row['af_tp_af_lvh']:>2} / FN "
        f"{row['af_fn_af_lvh']:>2}   recall {row['af_recall_af_lvh']:.4f}",
        f"  {'':<12} AF-only:  recall {row['af_recall_comparand']:.4f}"
        f"   delta {row['af_delta']:+.4f}   Fisher p = {row['af_fisher_p']:.4f}",
        f"  {'':<12} AF AUPRC on AF+LVH (reference pool): "
        f"{row['af_auprc_af_lvh']:.4f}",
        "",
        f"  reference ({baseline.AF_LVH_REFERENCE}, frozen): LVH recall on AF+LVH "
        f"{F['lvh']['recall_af_lvh']:.4f} vs {F['lvh']['recall_lvh_only']:.4f} "
        f"LVH-only (p={F['lvh']['fisher_p']:.4f});",
        f"  AF recall {F['af']['recall_af_lvh']:.4f} vs {F['af']['recall_af_only']:.4f} "
        f"AF-only (p={F['af']['fisher_p']:.4f}). The 28-case subgroup is small:",
        f"  the Fisher p and CI say how much of any delta is real.",
    ]
    if not row.get("subgroup_sizes_match", True):
        L.insert(0, f"  WARNING: subgroup sizes differ from the frozen test "
                    f"split ({row['size_note']}) - not comparable to the above.")
    return L


def load_rows(runs_dir: Path) -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    for f in sorted(Path(runs_dir).glob("*/predictions/test_predictions.csv")):
        run_dir = f.parent.parent
        row = score_run(run_dir)
        if row is None:
            missing.append(run_dir.name)
            continue
        # views column comes from result.json where available
        rj = run_dir / "result.json"
        if rj.exists():
            import json
            try:
                views = json.loads(rj.read_text()).get("model_info", {}).get("views", [])
                row["views"] = "+".join(map(str, views)) or "?"
            except Exception:
                row["views"] = "?"
        else:
            row["views"] = "?"
        rows.append(row)
    return rows, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", type=Path)
    ap.add_argument("--csv", type=Path, default=None)
    a = ap.parse_args()

    rows, missing = load_rows(a.runs)
    lines = format_section(rows, missing)
    print("\n".join(lines))

    if a.csv and rows:
        import csv as _csv
        fields: list[str] = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with a.csv.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"csv -> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
