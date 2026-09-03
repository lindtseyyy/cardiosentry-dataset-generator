"""
Bootstrap confidence intervals for the AF-vs-clean subgroup comparison.

WHY THE RESAMPLING IS INDEPENDENT PER GROUP, NOT PAIRED

The round-1 evaluation report and lvh_binary_v1 both used PAIRED bootstrap:
two models scored the SAME 938 images, so every draw resampled the same
indices for both models and the delta's variance came only from the images
the models disagreed on.

Two strata are different: `clean` and `af` are DISJOINT image sets (507 and
285 images in test). There are no shared indices to hold fixed. Each group is
therefore resampled independently - same 1,200 draws, same seed, same
percentile-95% rule - and the delta distribution is the per-draw difference
of the two independent resamples. This is the standard two-sample bootstrap;
it is written here, not hidden, because a reader who knows the project's
paired-bootstrap habit could otherwise assume pairing happened.

WHAT IS NOT RE-TUNED

Threshold-dependent metrics (sensitivity, specificity, FPR, precision, F1)
are bootstrapped at the FROZEN 0.47 threshold inside every draw. Re-tuning
inside the bootstrap would measure threshold-selection variance, not
subgroup performance, and would quietly move the operating point the
reference run reported.
"""
from __future__ import annotations

import numpy as np

import config as C
import metrics as M


def _draw_indices(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, n, size=(C.N_RESAMPLES, n))


def _percentiles(vals: np.ndarray, alpha: float = C.ALPHA):
    return float(np.percentile(vals, 100 * alpha / 2)), \
           float(np.percentile(vals, 100 * (1 - alpha / 2)))


def subgroup_comparison(y_clean, p_clean, y_af, p_af,
                        thr: float = C.LVH_THRESHOLD,
                        seed: int = C.SEED) -> list[dict]:
    """One row per metric: per-group point estimates with CIs, and the delta
    (af - clean) with its CI and P(af > clean).

    All draws come from ONE generator stream, consumed in a fixed metric
    order, so every re-run reproduces the exact intervals.
    """
    y_clean = np.asarray(y_clean).astype(int)
    p_clean = np.asarray(p_clean).astype(float)
    y_af = np.asarray(y_af).astype(int)
    p_af = np.asarray(p_af).astype(float)

    rng = np.random.default_rng(seed)
    rows = []
    for metric in ("auprc", "auroc", "sensitivity", "specificity", "fpr",
                   "precision", "f1", "sens_at_fpr"):
        iA = _draw_indices(rng, len(y_clean))
        iB = _draw_indices(rng, len(y_af))
        A = M.bootstrap_metric_values(metric, y_clean[iA], p_clean[iA], thr)
        B = M.bootstrap_metric_values(metric, y_af[iB], p_af[iB], thr)
        D = B - A

        obs_a = M.stratum_metrics(y_clean, p_clean, thr)[metric]
        obs_b = M.stratum_metrics(y_af, p_af, thr)[metric]
        obs_delta = obs_b - obs_a if not (obs_a != obs_a or obs_b != obs_b) \
            else float("nan")

        A = A[~np.isnan(A)]
        B = B[~np.isnan(B)]
        D = D[~np.isnan(D)]
        rows.append({
            "metric": metric,
            "clean": obs_a,
            "clean_ci_low": _percentiles(A)[0] if len(A) else float("nan"),
            "clean_ci_high": _percentiles(A)[1] if len(A) else float("nan"),
            "af": obs_b,
            "af_ci_low": _percentiles(B)[0] if len(B) else float("nan"),
            "af_ci_high": _percentiles(B)[1] if len(B) else float("nan"),
            "delta_af_minus_clean": obs_delta,
            "delta_ci_low": _percentiles(D)[0] if len(D) else float("nan"),
            "delta_ci_high": _percentiles(D)[1] if len(D) else float("nan"),
            "p_af_better": float((D > 0).mean()) if len(D) else float("nan"),
        })
    return rows


# ---------------------------------------------------------------- formatting
_METRIC_LABELS = {
    "auprc": "AUPRC", "auroc": "AUROC", "sensitivity": "Sens",
    "specificity": "Spec", "fpr": "FPR", "precision": "Prec",
    "f1": "F1", "sens_at_fpr": "s@.08",
}


def format_delta_table(rows: list[dict]) -> list[str]:
    L = []
    hdr = (f"{'metric':<7}{'clean':>9}{'95% CI':>17}{'AF':>9}{'95% CI':>17}"
           f"{'delta':>9}{'95% CI':>17}{'P(AF>clean)':>13}")
    L += [hdr, "-" * len(hdr)]
    for r in rows:
        lab = _METRIC_LABELS.get(r["metric"], r["metric"])
        ci_a = f"[{r['clean_ci_low']:.4f},{r['clean_ci_high']:.4f}]"
        ci_b = f"[{r['af_ci_low']:.4f},{r['af_ci_high']:.4f}]"
        ci_d = f"[{r['delta_ci_low']:+.4f},{r['delta_ci_high']:+.4f}]"
        L.append(f"{lab:<7}{r['clean']:>9.4f}{ci_a:>17}{r['af']:>9.4f}"
                 f"{ci_b:>17}{r['delta_af_minus_clean']:>+9.4f} {ci_d:>17}"
                 f"{r['p_af_better']:>13.3f}")
    return L
