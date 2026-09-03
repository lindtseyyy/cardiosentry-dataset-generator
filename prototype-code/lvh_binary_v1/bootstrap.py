"""
Paired bootstrap confidence intervals for the binary-vs-four-head comparison.

WHY PAIRED, AND WHY IT MUST BE

Both models are evaluated on the SAME 938-image test split. Independent
bootstrap (resampling each model's images separately) would ignore that
pairing and inflate the variance of the DELTA - which is exactly the quantity
this round wants to put an interval around. Paired resampling - the same
bootstrap indices for both models on every draw - is the correct test, and it
is the same methodology the round-1 evaluation report used (1,200 paired
resamples from the saved test_predictions.csv files).

Threshold-dependent metrics (sensitivity, specificity, FPR, precision, F1) are
bootstrapped at the models' OWN val-tuned frozen thresholds, exactly as they
are reported: the operating point is part of the model and must not be
re-tuned inside the bootstrap.
"""
from __future__ import annotations

import numpy as np

# Same draw count as the round-1 evaluation report, so the two documents speak
# one language about uncertainty.
N_RESAMPLES = 1200
SEED = 42


def _auprc(y, p):
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def _auroc(y, p):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _at_threshold(metric: str):
    """Threshold metrics, computed from confusion counts at a fixed threshold."""
    def fn(y, p, thr):
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn_ = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        d = lambda a, b: a / b if b else float("nan")
        if metric == "sensitivity":
            return d(tp, tp + fn_)
        if metric == "specificity":
            return d(tn, tn + fp)
        if metric == "fpr":
            return d(fp, fp + tn)
        if metric == "precision":
            return d(tp, tp + fp)
        if metric == "f1":
            prec, sens = d(tp, tp + fp), d(tp, tp + fn_)
            return d(2 * prec * sens, prec + sens) if (prec + sens) > 0 else 0.0
        raise ValueError(metric)
    return fn


def _metric_fn(metric: str):
    if metric == "auprc":
        return lambda y, p, thr: _auprc(y, p)
    if metric == "auroc":
        return lambda y, p, thr: _auroc(y, p)
    return _at_threshold(metric)


def paired_delta(y_true: np.ndarray,
                 p_binary: np.ndarray,
                 p_shared: np.ndarray,
                 thr_binary: float,
                 thr_shared: float,
                 metric: str = "auprc",
                 n_resamples: int = N_RESAMPLES,
                 seed: int = SEED,
                 alpha: float = 0.05) -> dict:
    """Bootstrap the delta (binary - shared) for one metric, paired over images.

    Returns the point estimate delta, the percentile 95% CI, and
    P(binary beats shared), i.e. the fraction of resamples where delta > 0.
    """
    y = np.asarray(y_true).ravel()
    pb = np.asarray(p_binary).ravel()
    ps = np.asarray(p_shared).ravel()
    assert y.shape == pb.shape == ps.shape, \
        f"shape mismatch: y {y.shape}, binary {pb.shape}, shared {ps.shape}"

    fn = _metric_fn(metric)
    obs_b = fn(y, pb, thr_binary)
    obs_s = fn(y, ps, thr_shared)
    obs_delta = (obs_b - obs_s) if not (obs_b != obs_b or obs_s != obs_s) \
        else float("nan")

    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_resamples)
    for k in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        mb = fn(y[idx], pb[idx], thr_binary)
        ms = fn(y[idx], ps[idx], thr_shared)
        deltas[k] = (mb - ms) if not (mb != mb or ms != ms) else float("nan")

    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) == 0:
        return {"metric": metric, "binary": obs_b, "shared": obs_s,
                "delta": obs_delta, "ci_low": float("nan"), "ci_high": float("nan"),
                "p_binary_better": float("nan"), "n_used": 0}
    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return {"metric": metric, "binary": obs_b, "shared": obs_s,
            "delta": obs_delta, "ci_low": lo, "ci_high": hi,
            "p_binary_better": float((deltas > 0).mean()),
            "n_used": int(len(deltas))}


def bootstrap_table(y_true, p_binary, p_shared, thr_binary, thr_shared,
                    metrics=("auprc", "auroc", "sensitivity", "specificity",
                             "fpr", "precision", "f1"),
                    n_resamples: int = N_RESAMPLES) -> list[dict]:
    """One row per metric; the seven this round was asked to report."""
    return [paired_delta(y_true, p_binary, p_shared, thr_binary, thr_shared,
                         m, n_resamples=n_resamples) for m in metrics]
