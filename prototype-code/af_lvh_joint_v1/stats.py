"""
Exact tests and bootstrap CIs for the joint-head comparison.

NO scipy: McNemar and Fisher are implemented exactly (they are small
combinatorial sums here), so the notebook has zero installs. Both are
standard definitions:

  * McNemar (exact) - binomial test on the discordant pairs of the joint
    breakdown: given d = b + c discordant cases, under "the two heads miss
    equally often" each discordant case is AF-correct/LVH-missed with
    probability 1/2. Two-sided p = 2 * P(X <= min(b, c) | Bin(d, 1/2)).

  * Fisher exact (one-sided, 'less') - hypergeometric test on the 2x2 table
    [correct, missed] x [subgroup, comparand]: p = P(X <= a) with fixed
    margins, where a is the subgroup's correct count.

Bootstrap: 1,200 resamples, seed 42, percentile 95% CIs - the project
convention. Recall CIs resample each group's DECISIONS independently (the
groups are disjoint ECGs, so there is no pairing); the joint-category CIs
resample the 28 cases.
"""
from __future__ import annotations

import math

import numpy as np

import config as C


# ---------------------------------------------------------------- exact tests
def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p for discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(2.0 * tail / (2 ** n), 1.0)


def fisher_less_p(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p, alternative='less' (cell a smaller than
    expected), on [[a, b], [c, d]] with row/col sums (a+b, c+d) and (a+c, b+d)."""
    n1, n2 = a + b, c + d
    col1 = a + c
    n = n1 + n2
    num = 0
    for x in range(0, a + 1):
        if col1 >= x and n - col1 >= n1 - x:
            num += math.comb(col1, x) * math.comb(n - col1, n1 - x)
    return num / math.comb(n, n1)


# ---------------------------------------------------------------- bootstrap
def _ci(vals: np.ndarray) -> tuple[float, float]:
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return (float(np.percentile(vals, 100 * C.ALPHA / 2)),
            float(np.percentile(vals, 100 * (1 - C.ALPHA / 2))))


def recall_ci(decisions: np.ndarray, n_resamples: int = C.N_RESAMPLES,
              seed: int = C.SEED) -> dict:
    """Point estimate + CI of recall on one group's binary decisions."""
    dec = np.asarray(decisions).astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(dec), size=(n_resamples, len(dec)))
    draws = dec[idx].mean(axis=1)
    lo, hi = _ci(draws)
    return {"recall": float(dec.mean()), "ci_low": lo, "ci_high": hi,
            "n": int(len(dec))}


def recall_delta_ci(dec_a: np.ndarray, dec_b: np.ndarray,
                    n_resamples: int = C.N_RESAMPLES,
                    seed: int = C.SEED) -> dict:
    """Delta (a - b) of two groups' recalls, independent resampling per group,
    one generator stream so re-runs reproduce the exact intervals."""
    a = np.asarray(dec_a).astype(int)
    b = np.asarray(dec_b).astype(int)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_resamples, len(a)))
    ib = rng.integers(0, len(b), size=(n_resamples, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = _ci(d)
    return {"delta": float(a.mean() - b.mean()), "ci_low": lo, "ci_high": hi,
             "n_a": int(len(a)), "n_b": int(len(b)),
             "recall_a": float(a.mean()), "recall_b": float(b.mean())}


def joint_breakdown_ci(labels: np.ndarray, n_resamples: int = C.N_RESAMPLES,
                       seed: int = C.SEED) -> dict:
    """Point + CI for each of the 4 joint categories, from the 28 case labels.

    labels: array of category codes 0..3 for the subgroup cases. Multinomial
    bootstrap of the proportions.
    """
    lab = np.asarray(labels).astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(lab), size=(n_resamples, len(lab)))
    out = {}
    for k, name in enumerate(("af_ok_lvh_ok", "af_ok_lvh_miss",
                              "af_miss_lvh_ok", "af_miss_lvh_miss")):
        draws = (lab[idx] == k).mean(axis=1)
        lo, hi = _ci(draws)
        out[name] = {"count": int((lab == k).sum()),
                     "pct": float((lab == k).mean()), "ci_low": lo, "ci_high": hi}
    return out
