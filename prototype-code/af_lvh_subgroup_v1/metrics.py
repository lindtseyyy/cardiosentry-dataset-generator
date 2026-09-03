"""
LVH-head metrics inside one stratum.

WHAT IS MEASURED, AND INSIDE WHICH SET

For one stratum (e.g. every test ECG with true_AF=1) the LVH head is scored
against true_LVH INSIDE that stratum - LVH positives vs LVH negatives of the
stratum. That is the only design under which all six requested metrics
(sensitivity, specificity, FPR, precision, F1, AUPRC) are defined per group.

WHAT IS FROZEN

The threshold is the reference run's val-tuned 0.47, used exactly as reported
and never re-tuned inside a stratum. Re-tuning per stratum would turn a
detector difference into an operating-point difference and make the
comparison meaningless. The threshold-free columns (AUROC, AUPRC, and
sensitivity at FPR <= 0.08) exist so the detector read does not depend on the
operating point at all.

WHY AUPRC ALONE CANNOT BE THE VERDICT METRIC

AUPRC's floor is the prevalence of the positive class. The strata have very
different LVH prevalence by construction (in test: 44.8% in the clean
stratum, 9.8% in the AF stratum), so part of any AUPRC gap is mechanical, not
detector quality. The prevalence-independent reads - AUROC and sensitivity at
matched FPR - are the ones the verdict is built on; AUPRC, precision and F1
are reported beside their strata's prevalence so the mechanical share stays
visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

# The columns every metric table carries. Threshold metrics are computed at
# the frozen 0.47; ranking metrics are threshold-free.
METRIC_COLUMNS = ["n", "n_pos", "prevalence", "auprc", "auroc",
                  "sensitivity", "specificity", "fpr", "precision", "f1",
                  "sens_at_fpr", "tp", "fp", "fn", "tn"]


# ---------------------------------------------------------------- point metrics
def _counts(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def _div(a, b):
    return a / b if b else float("nan")


def _auroc(y, p):
    """Rank-based AUROC, vector-friendly: same value as sklearn."""
    order = np.argsort(p, kind="mergesort")
    ys = y[order]
    npos = int(ys.sum())
    if npos == 0 or npos == len(ys):
        return float("nan")
    ranks = np.where(ys == 1)[0] + 1
    return float((ranks.sum() - npos * (npos + 1) / 2) / (npos * (len(ys) - npos)))


def _auprc(y, p):
    """Average precision by the sorted-scores definition, vector-friendly."""
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    npos = int(ys.sum())
    if npos == 0 or npos == len(ys):
        return float("nan")
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / npos)


def _sens_at_fpr(y, p, target: float = C.MATCHED_FPR):
    """Sensitivity at the highest achievable FPR that does not exceed target.

    Same definition as lvh_binary_v1/subgroup.py.sens_at_fpr - threshold-free,
    so it compares detectors rather than operating points.
    """
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    npos = int(ys.sum())
    nneg = len(ys) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    tp = np.cumsum(ys)
    fp = np.arange(1, len(ys) + 1) - tp
    ok = np.where(fp / nneg <= target)[0]
    if len(ok) == 0:
        return 0.0
    return float(tp[ok[-1]] / npos)


def _auroc_sklearn(y, p):
    """sklearn AUROC - the exact definition the reference harness reported."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def _auprc_sklearn(y, p):
    """sklearn average precision - the exact definition the reference
    harness reported (interpolated, and what baseline.py froze)."""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p))


def _threshold_metrics_only(y: np.ndarray, prob: np.ndarray,
                            thr: float) -> dict:
    """Sensitivity/specificity/FPR/precision/F1 from confusion counts only.

    No sklearn - the bootstrap calls this thousands of times and the ranking
    metrics are computed by the vectorised numpy functions instead."""
    pred = (prob >= thr).astype(int)
    tp, fp, fn, tn = _counts(y, pred)
    sens = _div(tp, tp + fn)
    spec = _div(tn, tn + fp)
    prec = _div(tp, tp + fp)
    f1 = _div(2 * prec * sens, prec + sens) if (prec + sens) > 0 else 0.0
    return {"sensitivity": sens, "specificity": spec,
            "fpr": _div(fp, fp + tn), "precision": prec, "f1": f1}


def stratum_metrics(y: np.ndarray, prob: np.ndarray, thr: float = C.LVH_THRESHOLD,
                    labels: np.ndarray | None = None) -> dict:
    """Every requested metric for the LVH head inside ONE stratum.

    Degenerate strata are legal inputs: a stratum with no positives returns
    NaN for everything except n, false positives and FPR; an empty stratum
    returns n=0 with NaNs. The caller decides which of those get printed.

    AUROC/AUPRC use sklearn, the exact definitions the frozen reference
    numbers were computed with (the cross-check against baseline.py requires
    this - the vectorised numpy versions in this module differ at the 6th
    decimal and are used ONLY inside the bootstrap, where a per-draw
    1e-5 difference is irrelevant).
    """
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob).astype(float)
    n = int(len(y))
    n_pos = int(y.sum())
    if n == 0:
        return {k: float("nan") for k in METRIC_COLUMNS} | {
            "n": 0, "n_pos": 0, "prevalence": float("nan"),
            "tp": 0, "fp": 0, "fn": 0, "tn": 0, "threshold": float(thr)}
    pred = (prob >= thr).astype(int)
    tp, fp, fn, tn = _counts(y, pred)
    sens = _div(tp, tp + fn)
    spec = _div(tn, tn + fp)
    prec = _div(tp, tp + fp)
    f1 = _div(2 * prec * sens, prec + sens) if (prec + sens) > 0 else 0.0
    return {
        "n": n,
        "n_pos": n_pos,
        "prevalence": n_pos / n,
        "auprc": _auprc_sklearn(y, prob) if n_pos not in (0, n) else float("nan"),
        "auroc": _auroc_sklearn(y, prob) if n_pos not in (0, n) else float("nan"),
        "sensitivity": sens,
        "specificity": spec,
        "fpr": _div(fp, fp + tn),
        "precision": prec,
        "f1": f1,
        "sens_at_fpr": _sens_at_fpr(y, prob) if n_pos not in (0, n) else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": float(thr),
    }


def bootstrap_metric_values(metric: str, YY: np.ndarray, PP: np.ndarray,
                            thr: float) -> np.ndarray:
    """Vectorised metric over rows of resampled (Y, P) matrices.

    YY/PP: (n_resamples, n) arrays produced by bootstrap.py. One value per row.
    """
    n_resamples = len(YY)
    if metric == "auprc":
        return np.array([_auprc(YY[k], PP[k]) for k in range(n_resamples)])
    if metric == "auroc":
        return np.array([_auroc(YY[k], PP[k]) for k in range(n_resamples)])
    if metric == "sens_at_fpr":
        return np.array([_sens_at_fpr(YY[k], PP[k]) for k in range(n_resamples)])
    out = np.empty(n_resamples)
    for k in range(n_resamples):
        out[k] = _threshold_metrics_only(YY[k], PP[k], thr)[metric]
    return out


# ---------------------------------------------------------------- formatting
def _f(v, w=9, p=4):
    return f"{'':>{w}}" if v != v else f"{v:{w}.{p}f}"


def format_metrics_table(rows: list[dict]) -> list[str]:
    """rows: [{name, display, metrics}], in display order."""
    L = []
    hdr = (f"{'stratum':<24}{'n':>6}{'LVH+':>7}{'prev':>7}{'AUPRC':>9}"
           f"{'AUROC':>9}{'Sens':>9}{'Spec':>9}{'FPR':>9}{'Prec':>9}"
           f"{'F1':>9}{'s@.08':>9}")
    L += [hdr, "-" * len(hdr)]
    for r in rows:
        m = r["metrics"]
        L.append(f"{r['name']:<24}{m['n']:>6}{m['n_pos']:>7}"
                 f"{_f(m['prevalence'], 7)}"
                 f"{_f(m['auprc'])}{_f(m['auroc'])}{_f(m['sensitivity'])}"
                 f"{_f(m['specificity'])}{_f(m['fpr'])}{_f(m['precision'])}"
                 f"{_f(m['f1'])}{_f(m['sens_at_fpr'])}")
    return L


def format_counts_row(r: dict) -> str:
    m = r["metrics"]
    if m["n"] == 0:
        return f"{r['name']:<24}{'-- no ECGs in this stratum --':>52}"
    return (f"{r['name']:<24}{'TP':>16}{m['tp']:>5}{'FP':>6}{m['fp']:>5}"
            f"{'FN':>6}{m['fn']:>5}{'TN':>6}{m['tn']:>5}")


def metrics_to_df(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        out.append({"stratum": r["name"], **{k: r["metrics"][k]
                                             for k in METRIC_COLUMNS}})
    return pd.DataFrame(out)
