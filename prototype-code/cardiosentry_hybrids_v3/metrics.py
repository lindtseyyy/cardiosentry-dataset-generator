"""
Multi-label evaluation for four independent sigmoid heads.

Design rules, all of which exist to stop a number looking better than it is:

  * Thresholds are tuned on VAL and then FROZEN for TEST. Every
    threshold-dependent metric below (sensitivity, specificity, precision, F1,
    FPR, accuracy) moves when the threshold moves, so where it was chosen is
    part of the result.
  * Macro AUPRC is the model-selection monitor - threshold-free, and unlike
    AUROC it does not flatter a model on the minority head.
  * Per-source metrics are always computed for test: all STEMI comes from one
    database and stage 5 measured residual source signal (0.515 vs 0.372).
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import config as C

# Metrics that depend on the decision threshold, as opposed to the ranking.
THRESHOLD_DEPENDENT = ("sensitivity", "specificity", "precision", "f1",
                       "fpr", "fnr", "accuracy", "npv", "balanced_accuracy")
RANKING_BASED = ("auroc", "auprc")


def _safe(fn, y, p):
    """AUROC/AUPRC are undefined when a split contains only one class."""
    return float(fn(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _counts(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def _div(a, b):
    return a / b if b else float("nan")


def binary_metrics(y: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    """Every requested metric for ONE head."""
    pred = (prob >= thr).astype(int)
    tp, fp, fn, tn = _counts(y, pred)

    sens = _div(tp, tp + fn)          # recall / true positive rate
    spec = _div(tn, tn + fp)          # true negative rate
    prec = _div(tp, tp + fp)          # positive predictive value
    return {
        "auroc": _safe(roc_auc_score, y, prob),
        "auprc": _safe(average_precision_score, y, prob),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "npv": _div(tn, tn + fn),
        "f1": _div(2 * prec * sens, prec + sens) if not (np.isnan(prec) or np.isnan(sens))
              and (prec + sens) > 0 else 0.0,
        "fpr": _div(fp, fp + tn),     # = 1 - specificity
        "fnr": _div(fn, fn + tp),     # = 1 - sensitivity
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": (sens + spec) / 2 if not (np.isnan(sens) or np.isnan(spec))
                             else float("nan"),
        "threshold": float(thr),
        "support": int(y.sum()),
        "n": int(len(y)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> list[float]:
    """Per-head threshold maximising F1. VALIDATION SET ONLY - never test."""
    out = []
    grid = np.linspace(0.05, 0.95, 91)
    for k in range(y_true.shape[1]):
        yk, pk = y_true[:, k], y_prob[:, k]
        if len(np.unique(yk)) < 2:
            out.append(0.5)
            continue
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            m = binary_metrics(yk, pk, t)
            if m["f1"] > best_f1:
                best_f1, best_t = m["f1"], t
        out.append(float(best_t))
    return out


def evaluate(y_true: np.ndarray, y_prob: np.ndarray,
             thresholds: list[float] | None = None) -> dict:
    """Per-label metrics + macro averages + multi-label whole-vector accuracy."""
    thresholds = thresholds or [0.5] * y_true.shape[1]
    per_label = {lab: binary_metrics(y_true[:, k], y_prob[:, k], thresholds[k])
                 for k, lab in enumerate(C.LABELS)}

    out = {"per_label": per_label}

    # macro = unweighted mean across the four heads, so a rare class counts as
    # much as a common one. NaN-safe: a head undefined on this split is skipped.
    for m in RANKING_BASED + THRESHOLD_DEPENDENT:
        vals = [per_label[l][m] for l in C.LABELS]
        out[f"macro_{m}"] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) \
                            else float("nan")

    # micro = pool the confusion matrices, so common classes dominate.
    tp = sum(per_label[l]["tp"] for l in C.LABELS)
    fp = sum(per_label[l]["fp"] for l in C.LABELS)
    fn = sum(per_label[l]["fn"] for l in C.LABELS)
    tn = sum(per_label[l]["tn"] for l in C.LABELS)
    mp, ms = _div(tp, tp + fp), _div(tp, tp + fn)
    out["micro_precision"] = mp
    out["micro_sensitivity"] = ms
    out["micro_f1"] = _div(2 * mp * ms, mp + ms) if (mp + ms) > 0 else 0.0
    out["micro_specificity"] = _div(tn, tn + fp)

    # Multi-label accuracy is ambiguous, so report both readings explicitly:
    pred = np.stack([(y_prob[:, k] >= thresholds[k]).astype(int)
                     for k in range(len(C.LABELS))], axis=1)
    #   hamming: fraction of individual label slots correct (this is what
    #            macro_accuracy averages per head)
    out["hamming_accuracy"] = float((pred == y_true).mean())
    #   exact match: ALL FOUR heads correct on the same ECG. Much stricter, and
    #            the honest number to quote as "accuracy" for a multi-label model.
    out["exact_match_accuracy"] = float((pred == y_true).all(axis=1).mean())
    out["n_samples"] = int(len(y_true))
    return out


def evaluate_per_source(y_true, y_prob, sources, thresholds) -> dict:
    """Same metrics split by originating database. A large gap is the signature
    of a model keying on acquisition artifacts rather than pathology."""
    out = {}
    sources = np.asarray(sources)
    for src in sorted(set(sources.tolist())):
        m = sources == src
        if not m.any():
            continue
        res = evaluate(y_true[m], y_prob[m], thresholds)
        out[src] = {
            "n": int(m.sum()),
            **{k: res[k] for k in ("macro_auroc", "macro_auprc", "macro_f1",
                                   "macro_sensitivity", "macro_specificity",
                                   "macro_precision", "exact_match_accuracy")},
            "per_label": {l: {k: res["per_label"][l][k]
                              for k in ("auroc", "auprc", "sensitivity",
                                        "specificity", "precision", "f1",
                                        "fpr", "accuracy", "support")}
                          for l in C.LABELS},
        }
    return out


# ------------------------------------------------------------------ formatting
_PL_COLS = [("AUROC", "auroc"), ("AUPRC", "auprc"), ("Sens", "sensitivity"),
            ("Spec", "specificity"), ("Prec", "precision"), ("F1", "f1"),
            ("FPR", "fpr"), ("Acc", "accuracy")]


def format_report(res: dict, title: str = "") -> str:
    L = []
    if title:
        L.append(title)
        L.append("-" * 92)
    L.append(f"  {'label':9}" + "".join(f"{h:>9}" for h, _ in _PL_COLS)
             + f"{'thr':>7}{'n+':>7}")
    for lab in C.LABELS:
        d = res["per_label"][lab]
        L.append(f"  {lab:9}" + "".join(f"{d[k]:9.4f}" for _, k in _PL_COLS)
                 + f"{d['threshold']:7.2f}{d['support']:7d}")
    L.append("  " + "-" * 90)
    L.append(f"  {'MACRO':9}" + "".join(f"{res['macro_'+k]:9.4f}" for _, k in _PL_COLS))
    L.append(f"  {'micro':9}{'':9}{'':9}{res['micro_sensitivity']:9.4f}"
             f"{res['micro_specificity']:9.4f}{res['micro_precision']:9.4f}"
             f"{res['micro_f1']:9.4f}")
    L.append("")
    L.append(f"  exact-match accuracy (all 4 heads correct): "
             f"{res['exact_match_accuracy']:.4f}")
    L.append(f"  hamming accuracy (per label slot):          "
             f"{res['hamming_accuracy']:.4f}     n = {res['n_samples']:,}")
    return "\n".join(L)


def format_per_source(per_src: dict) -> str:
    L = ["  per-source breakdown (source-confound check)",
         f"    {'source':10}{'n':>7}{'mAUROC':>9}{'mAUPRC':>9}{'mSens':>9}"
         f"{'mSpec':>9}{'mF1':>9}"]
    for src, d in per_src.items():
        L.append(f"    {src:10}{d['n']:7d}{d['macro_auroc']:9.4f}{d['macro_auprc']:9.4f}"
                 f"{d['macro_sensitivity']:9.4f}{d['macro_specificity']:9.4f}"
                 f"{d['macro_f1']:9.4f}")
    vals = [d["macro_auroc"] for d in per_src.values()]
    if len(vals) > 1:
        L.append(f"    {'spread':10}{'':7}{max(vals)-min(vals):9.4f}")
    return "\n".join(L)


def flat_rows(res: dict, split: str, model: str) -> list[dict]:
    """One row per (model, split, label) - for metrics.csv."""
    rows = []
    for lab in C.LABELS:
        rows.append({"model": model, "split": split, "label": lab,
                     **{k: v for k, v in res["per_label"][lab].items()}})
    macro = {"model": model, "split": split, "label": "MACRO"}
    for m in RANKING_BASED + THRESHOLD_DEPENDENT:
        macro[m] = res[f"macro_{m}"]
    macro["support"] = sum(res["per_label"][l]["support"] for l in C.LABELS)
    macro["n"] = res["n_samples"]
    rows.append(macro)
    return rows
