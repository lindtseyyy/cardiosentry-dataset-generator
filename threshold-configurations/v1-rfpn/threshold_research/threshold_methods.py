"""Pure per-label threshold optimisers.

Decision rule everywhere: positive iff score >= threshold (float32 comparison).
Every metric is piecewise constant in the threshold and changes only at observed
scores, so the candidate set is exactly the distinct observed scores. Thresholds
are always one of those scores, which keeps `>=` semantics exact under ties.

Statuses (guide section 9.3):
  ok, infeasible_constraint, insufficient_positive_support,
  insufficient_negative_support, no_score_separation
A non-ok result carries threshold=None; callers must not treat it as validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

Z95 = float(norm.ppf(0.975))


@dataclass(frozen=True)
class OperatingPoints:
    """Confusion counts at every distinct score, thresholds in descending order."""
    thresholds: np.ndarray  # float32, strictly descending
    tp: np.ndarray
    fp: np.ndarray
    n_pos: int
    n_neg: int

    @property
    def fn(self) -> np.ndarray:
        return self.n_pos - self.tp

    @property
    def tn(self) -> np.ndarray:
        return self.n_neg - self.fp

    @property
    def sensitivity(self) -> np.ndarray:
        return self.tp / self.n_pos

    @property
    def specificity(self) -> np.ndarray:
        return self.tn / self.n_neg


@dataclass(frozen=True)
class ThresholdResult:
    method: str
    status: str
    threshold: float | None
    params: dict = field(default_factory=dict)
    note: str = ""


def operating_points(y_true, scores) -> OperatingPoints:
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=np.float32)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y_true and scores must be 1-D arrays of equal length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must be 0/1")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order].astype(np.int64)
    # Last position of each run of equal scores: predicting `>= s_sorted[i]` selects rows 0..i.
    group_end = np.r_[np.flatnonzero(s_sorted[1:] != s_sorted[:-1]), len(s_sorted) - 1]
    tp = np.cumsum(y_sorted)[group_end]
    fp = (group_end + 1) - tp
    n_pos = int(y_sorted.sum())
    return OperatingPoints(s_sorted[group_end], tp, fp, n_pos, len(y_sorted) - n_pos)


def wilson_interval(k, n, z: float = Z95):
    """Wilson score interval; NaN where n == 0."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def _support_status(points: OperatingPoints) -> str | None:
    if points.n_pos == 0:
        return "insufficient_positive_support"
    if points.n_neg == 0:
        return "insufficient_negative_support"
    if len(points.thresholds) < 2:
        return "no_score_separation"
    return None


def _pick(points: OperatingPoints, mask: np.ndarray, keys: list[np.ndarray]) -> int:
    """Index maximising `keys` lexicographically (first key most important) within mask."""
    candidates = np.flatnonzero(mask)
    # np.lexsort sorts ascending by the LAST key first; take the final element.
    order = np.lexsort([key[candidates] for key in reversed(keys)])
    return int(candidates[order[-1]])


def max_fbeta(y_true, scores, beta: float = 1.0) -> ThresholdResult:
    """Maximise F-beta; ties resolve to the lowest threshold.

    F-beta is computed from integer counts, so exact ties are exact. Lowest-threshold
    among ties matches the notebook (`np.nanargmax` over sklearn's ascending thresholds)
    unless the notebook's float `2PR/(P+R)` rounding splits a tie; the baseline is
    verified against the saved thresholds instead of assumed.
    """
    method = "max_f1" if beta == 1 else f"max_f{beta:g}"
    points = operating_points(y_true, scores)
    status = _support_status(points)
    if status:
        return ThresholdResult(method, status, None, {"beta": beta})
    b2 = beta * beta
    fbeta = (1 + b2) * points.tp / ((1 + b2) * points.tp + b2 * points.fn + points.fp)
    best = fbeta.max()
    index = int(np.flatnonzero(fbeta == best)[-1])  # descending thresholds -> last = lowest
    return ThresholdResult(method, "ok", float(points.thresholds[index]), {"beta": beta})


def youden_j(y_true, scores) -> ThresholdResult:
    """Maximise sensitivity + specificity - 1; ties resolve to higher specificity."""
    points = operating_points(y_true, scores)
    status = _support_status(points)
    if status:
        return ThresholdResult("youden_j", status, None)
    # Exact integer form of J * n_pos * n_neg; same ranking, no float tie splitting.
    j = points.tp * points.n_neg - points.fp * points.n_pos
    index = _pick(points, j == j.max(), [points.specificity])
    return ThresholdResult("youden_j", "ok", float(points.thresholds[index]))


def sensitivity_constrained(y_true, scores, min_sensitivity: float,
                            min_specificity: float | None = None,
                            use_wilson_lower: bool = False) -> ThresholdResult:
    """Maximise specificity subject to sensitivity >= floor.

    `use_wilson_lower` requires the lower 95% Wilson bound of the validation
    sensitivity, not the point estimate, to reach the floor (guide section 9.4).
    Ties: higher specificity, then higher sensitivity, then higher threshold.
    """
    method = "sens_ci_constrained" if use_wilson_lower else "sens_constrained"
    params = {"min_sensitivity": min_sensitivity, "min_specificity": min_specificity,
              "use_wilson_lower": use_wilson_lower}
    points = operating_points(y_true, scores)
    status = _support_status(points)
    if status:
        return ThresholdResult(method, status, None, params)
    sensitivity = wilson_interval(points.tp, points.n_pos)[0] if use_wilson_lower else points.sensitivity
    feasible = sensitivity >= min_sensitivity
    if min_specificity is not None:
        feasible &= points.specificity >= min_specificity
    if not feasible.any():
        best = float(np.max(sensitivity))
        return ThresholdResult(method, "infeasible_constraint", None, params,
                               f"max achievable {'Wilson-lower ' if use_wilson_lower else ''}"
                               f"sensitivity {best:.4f} under the constraints")
    index = _pick(points, feasible, [points.specificity, points.sensitivity, points.thresholds])
    return ThresholdResult(method, "ok", float(points.thresholds[index]), params)


def specificity_constrained(y_true, scores, min_specificity: float,
                            use_wilson_lower: bool = False) -> ThresholdResult:
    """Maximise sensitivity subject to specificity >= floor.

    Ties: higher sensitivity, then higher specificity, then higher threshold.
    """
    method = "spec_ci_constrained" if use_wilson_lower else "spec_constrained"
    params = {"min_specificity": min_specificity, "use_wilson_lower": use_wilson_lower}
    points = operating_points(y_true, scores)
    status = _support_status(points)
    if status:
        return ThresholdResult(method, status, None, params)
    specificity = wilson_interval(points.tn, points.n_neg)[0] if use_wilson_lower else points.specificity
    feasible = specificity >= min_specificity
    if not feasible.any():
        return ThresholdResult(method, "infeasible_constraint", None, params,
                               f"max achievable specificity {float(np.max(specificity)):.4f}")
    index = _pick(points, feasible, [points.sensitivity, points.specificity, points.thresholds])
    return ThresholdResult(method, "ok", float(points.thresholds[index]), params)


def notebook_f1_thresholds(y_true: np.ndarray, scores: np.ndarray) -> list[float]:
    """Verbatim port of `tune_thresholds_on_validation` from the training notebook
    (prototype-code/.../convnext_tiny_v1_rfpn_clean12_colab_v1/make_notebook.py:2627),
    kept only to prove the baseline is reproduced."""
    from sklearn.metrics import precision_recall_curve

    thresholds = []
    for label_index in range(y_true.shape[1]):
        y_label = y_true[:, label_index]
        p_label = scores[:, label_index]
        if len(np.unique(y_label)) < 2:
            thresholds.append(0.5)
            continue
        precision, recall, candidates = precision_recall_curve(y_label, p_label)
        if len(candidates) == 0:
            thresholds.append(0.5)
            continue
        f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        thresholds.append(float(candidates[int(np.nanargmax(f1))]))
    return thresholds
