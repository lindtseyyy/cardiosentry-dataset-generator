"""
Per-head evaluation rows and the joint breakdown, from the frozen predictions.

THE REFERENCE-POOL DESIGN (see config.py for the full rationale)

For head H and subgroup S, the evaluation row is:

  positives = the subgroup's ECGs        (all are true-positive for H)
  negatives = every test ECG with true_H = 0   (the frozen pool, identical
              for every subgroup row of that head)
  decisions  = prob_H >= frozen threshold

Every requested metric follows from the resulting confusion counts. Two
facts are structural and declared in the tables:

  * sensitivity/recall is the only metric that changes numerator-wise per
    subgroup; FP/TN (hence specificity) are identical across the rows of one
    head because the negative pool and its decisions do not change;
  * precision, AUPRC and AUROC with the subgroup as positive class are
    small-sample reads: the 28-case positive class meets a 653/683-case
    negative pool. They are reported as asked; the verdict reads are recall,
    the joint breakdown and the exact tests.

THE JOINT BREAKDOWN

For the 28 AF+LVH cases only, each case is classified by both heads at their
frozen thresholds:

  0  AF correct + LVH correct
  1  AF correct + LVH missed
  2  AF missed  + LVH correct
  3  AF missed  + LVH missed
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def _div(a, b):
    return a / b if b else float("nan")


def _auprc(y, p):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 \
        else float("nan")


def _auroc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def subgroup_mask(df: pd.DataFrame, key: str) -> pd.Series:
    """Ground-truth mask for a subgroup of config.SUBGROUPS."""
    spec = C.SUBGROUPS[key]
    m = pd.Series(True, index=df.index)
    for lab, val in spec.items():
        if lab == "display":
            continue
        m &= (df[f"true_{lab}"].astype(int) == val)
    return m


def head_row(df: pd.DataFrame, head: str, subgroup_key: str) -> dict:
    """One evaluation row: subgroup positives vs the head's frozen negative
    pool, at the head's frozen threshold."""
    thr = C.THRESHOLDS[head]
    mask = subgroup_mask(df, subgroup_key)
    pos = df[mask]
    neg = df[df[f"true_{head}"].astype(int) == 0]

    tp = int((pos[f"prob_{head}"] >= thr).sum())
    fn = int((pos[f"prob_{head}"] < thr).sum())
    fp = int((neg[f"prob_{head}"] >= thr).sum())
    tn = int((neg[f"prob_{head}"] < thr).sum())

    sens = _div(tp, tp + fn)
    spec = _div(tn, tn + fp)
    prec = _div(tp, tp + fp)
    f1 = _div(2 * prec * sens, prec + sens) if (prec + sens) > 0 else 0.0

    # ranking metrics: subgroup as the positive class vs the negative pool
    y = np.zeros(len(df))
    y[mask.values] = 1
    keep = (df[f"true_{head}"].astype(int) == 0).values | mask.values
    return {
        "head": head,
        "subgroup": subgroup_key,
        "n_pos": int(len(pos)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "fpr": _div(fp, fp + tn),
        "auprc": _auprc(y[keep], df.loc[keep, f"prob_{head}"].values),
        "auroc": _auroc(y[keep], df.loc[keep, f"prob_{head}"].values),
        "threshold": thr,
        "n_neg_pool": int(len(neg)),
    }


def joint_breakdown(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Category code per AF+LVH case + the 2x2 table.

    Returns (codes, table) where codes are 0..3 in the order
    [af_ok_lvh_ok, af_ok_lvh_miss, af_miss_lvh_ok, af_miss_lvh_miss] and
    table is a dict of the four counts.
    """
    sub = df[subgroup_mask(df, "af_lvh")]
    af_ok = (sub["prob_AF"] >= C.THRESHOLDS["AF"]).values.astype(int)
    lvh_ok = (sub["prob_LVH"] >= C.THRESHOLDS["LVH"]).values.astype(int)
    codes = np.where(af_ok == 1,
                     np.where(lvh_ok == 1, 0, 1),
                     np.where(lvh_ok == 1, 2, 3))
    table = {
        "af_ok_lvh_ok": int((codes == 0).sum()),
        "af_ok_lvh_miss": int((codes == 1).sum()),
        "af_miss_lvh_ok": int((codes == 2).sum()),
        "af_miss_lvh_miss": int((codes == 3).sum()),
    }
    return codes, table


# ---------------------------------------------------------------- formatting
_ROW_COLS = [("n+", "n_pos"), ("TP", "tp"), ("FP", "fp"), ("FN", "fn"),
             ("TN", "tn"), ("Sens", "sensitivity"), ("Spec", "specificity"),
             ("Prec", "precision"), ("F1", "f1"), ("AUPRC", "auprc"),
             ("AUROC", "auroc")]


def format_head_table(rows: list[dict], head: str) -> list[str]:
    L = [f"  {head} head - positives = the subgroup, negatives = the frozen "
         f"{rows[0]['n_neg_pool']}-ECG pool, threshold {C.THRESHOLDS[head]:.2f}",
         "    " + "".join(f"{h:>9}" for h, _ in _ROW_COLS)]
    L.append("    " + "-" * (9 * len(_ROW_COLS)))
    for r in rows:
        L.append(f"    {C.SUBGROUPS[r['subgroup']]['display']:<28}" +
                 "".join(f"{r[k]:>9.4f}" if isinstance(r[k], float) else
                         f"{r[k]:>9d}" for _, k in _ROW_COLS))
    L.append("    (Spec/FP/TN are identical across rows by design: same")
    L.append("     negative pool, same frozen decisions. Prec/AUPRC/AUROC here")
    L.append("     have the 28-case subgroup as their positive class - they")
    L.append("     are reported as asked but are NOT verdict reads.)")
    return L


JOINT_NAMES = {
    "af_ok_lvh_ok": "AF correct + LVH correct",
    "af_ok_lvh_miss": "AF correct + LVH missed",
    "af_miss_lvh_ok": "AF missed  + LVH correct",
    "af_miss_lvh_miss": "AF missed  + LVH missed",
}


def format_joint(bd: dict) -> list[str]:
    n = sum(r["count"] for r in bd.values())
    L = [f"  joint breakdown of the {n} AF+LVH cases (frozen thresholds, "
         f"AF {C.THRESHOLDS['AF']:.2f} / LVH {C.THRESHOLDS['LVH']:.2f})",
         f"    {'category':<28}{'count':>7}{'share':>9}{'95% CI':>19}"]
    L.append("    " + "-" * 64)
    for key, name in JOINT_NAMES.items():
        r = bd[key]
        ci = f"[{r['ci_low']*100:4.1f}%,{r['ci_high']*100:5.1f}%]" \
            if r["ci_low"] == r["ci_low"] else "--"
        L.append(f"    {name:<28}{r['count']:>7}{r['pct']*100:>8.1f}%{ci:>19}")
    L.append("    " + "-" * 64)
    L.append(f"    {'total':<28}{n:>7}{100.0:>8.1f}%")
    return L
