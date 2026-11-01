"""Behavioural checks for the threshold optimisers (guide section 10)."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (LABELS, LabelOrderError, SplitData, HeldOutSplitAccessError, predict,  # noqa: E402
                    require_development_split, validate_label_order)
from threshold_methods import (max_fbeta, notebook_f1_thresholds, operating_points,  # noqa: E402
                               sensitivity_constrained, specificity_constrained,
                               wilson_interval, youden_j)


def brute_force(y, s):
    """Every distinct score as threshold with explicit `>=` counting."""
    rows = []
    for t in np.unique(np.asarray(s, dtype=np.float32)):
        p = np.asarray(s, dtype=np.float32) >= t
        tp = int((p & (y == 1)).sum())
        fp = int((p & (y == 0)).sum())
        rows.append((float(t), tp, fp, tp / y.sum(), 1 - fp / (y == 0).sum()))
    return rows


def test_operating_points_count_ties_with_ge():
    y = np.array([1, 0, 1, 1, 0, 0])
    s = np.array([0.9, 0.7, 0.7, 0.4, 0.4, 0.1])
    points = operating_points(y, s)
    np.testing.assert_array_equal(points.thresholds, np.float32([0.9, 0.7, 0.4, 0.1]))
    # At t=0.7 both tied rows (one pos, one neg) are positive.
    np.testing.assert_array_equal(points.tp, [1, 2, 3, 3])
    np.testing.assert_array_equal(points.fp, [0, 1, 2, 3])


def test_score_equal_to_threshold_is_positive():
    scores = np.array([[0.3] * len(LABELS)], dtype=np.float32)
    assert predict(scores, [0.3] * len(LABELS)).all()
    assert not predict(scores, [np.nextafter(np.float32(0.3), np.float32(1))] * len(LABELS)).any()


def test_max_f1_matches_hand_computation_and_sklearn_port():
    y = np.array([1, 1, 0, 1, 0, 0, 0, 1])
    s = np.array([0.95, 0.8, 0.75, 0.6, 0.5, 0.3, 0.2, 0.1])
    # Hand F1 per threshold: t=0.6 -> tp3 fp1 fn1 -> 0.75 (best); t=0.8 -> tp2 fn2 -> 0.667.
    assert max_fbeta(y, s, 1).threshold == pytest.approx(0.6)
    port = notebook_f1_thresholds(np.tile(y[:, None], (1, 2)), np.tile(s[:, None], (1, 2)))
    assert port[0] == pytest.approx(0.6)


def test_max_f1_tie_resolves_to_lowest_threshold_like_notebook():
    # t=0.9 (tp1, fn1, fp0 -> 2/3) ties with t=0.5 (tp2, fp2 -> 2/3).
    y = np.array([1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.6, 0.5])
    assert max_fbeta(y, s, 1).threshold == pytest.approx(0.5)
    assert notebook_f1_thresholds(y[:, None], s[:, None])[0] == pytest.approx(0.5)


def test_f2_is_recall_weighted_but_not_a_sensitivity_guarantee():
    y = np.array([1, 0, 0, 0, 0, 1, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    f1, f2 = max_fbeta(y, s, 1).threshold, max_fbeta(y, s, 2).threshold
    assert f2 <= f1


@pytest.mark.parametrize("seed", range(25))
def test_sensitivity_constrained_is_feasible_and_maximally_specific(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 40)
    y[:2] = [0, 1]
    s = np.round(rng.random(40), 1)  # heavy ties
    for floor in (0.5, 0.8, 0.95, 1.0):
        result = sensitivity_constrained(y, s, floor)
        rows = brute_force(y, s)
        feasible = [r for r in rows if r[3] >= floor]
        best_spec = max(r[4] for r in feasible)
        chosen = next(r for r in rows if np.float32(r[0]) == np.float32(result.threshold))
        assert chosen[3] >= floor
        assert chosen[4] == best_spec
        # tie-break: highest sensitivity among maximally specific candidates
        assert chosen[3] == max(r[3] for r in feasible if r[4] == best_spec)


@pytest.mark.parametrize("seed", range(25))
def test_specificity_constrained_matches_brute_force(seed):
    rng = np.random.default_rng(100 + seed)
    y = rng.integers(0, 2, 30)
    y[:2] = [0, 1]
    s = np.round(rng.random(30), 2)
    result = specificity_constrained(y, s, 0.8)
    rows = brute_force(y, s)
    feasible = [r for r in rows if r[4] >= 0.8]
    chosen = next(r for r in rows if np.float32(r[0]) == np.float32(result.threshold))
    assert chosen[4] >= 0.8
    assert chosen[3] == max(r[3] for r in feasible)


def test_youden_matches_brute_force():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 60)
    s = y * 0.3 + rng.random(60) * 0.7
    rows = brute_force(y, s)
    best = max(r[3] + r[4] for r in rows)
    chosen = next(r for r in rows if np.float32(r[0]) == np.float32(youden_j(y, s).threshold))
    assert chosen[3] + chosen[4] == pytest.approx(best)


def test_wilson_constraint_needs_margin_and_reports_infeasible():
    y = np.array([1] * 20 + [0] * 20)
    s = np.r_[np.linspace(0.6, 1.0, 20), np.linspace(0.0, 0.5, 20)]
    # 20/20 detected -> Wilson lower ~0.839; a 0.90 floor on the bound is infeasible.
    assert wilson_interval(20, 20)[0] == pytest.approx(0.8389, abs=1e-3)
    result = sensitivity_constrained(y, s, 0.90, use_wilson_lower=True)
    assert result.status == "infeasible_constraint" and result.threshold is None
    assert sensitivity_constrained(y, s, 0.80, use_wilson_lower=True).status == "ok"


def test_joint_sensitivity_and_specificity_floors_can_be_infeasible():
    y = np.array([1, 0, 1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])  # interleaved, weak ranking
    result = sensitivity_constrained(y, s, 1.0, min_specificity=0.9)
    assert result.status == "infeasible_constraint" and result.threshold is None


def test_youden_exact_tie_prefers_higher_specificity():
    # (tp, fp) = (1, 1) and (7, 10) with n_pos=8, n_neg=12 give the same J = 1/24;
    # float J arithmetic splits this tie, the integer form must not.
    y = np.array([1, 0] + [1] * 6 + [0] * 9 + [1] + [0] * 2)
    s = np.array([0.9, 0.9] + [0.5] * 15 + [0.1] * 3)
    assert youden_j(y, s).threshold == pytest.approx(0.9)


@pytest.mark.parametrize("method", [
    lambda y, s: max_fbeta(y, s, 1), lambda y, s: max_fbeta(y, s, 2), youden_j,
    lambda y, s: sensitivity_constrained(y, s, 0.9), lambda y, s: specificity_constrained(y, s, 0.9),
])
def test_missing_class_support_is_explicit(method):
    s = np.array([0.1, 0.5, 0.9])
    assert method(np.zeros(3, int), s).status == "insufficient_positive_support"
    assert method(np.ones(3, int), s).status == "insufficient_negative_support"
    assert method(np.zeros(3, int), s).threshold is None


def test_all_tied_scores_are_deterministic_and_flagged():
    y = np.array([1, 0, 1, 0])
    s = np.full(4, 0.42)
    results = {max_fbeta(y, s).status, youden_j(y, s).status, sensitivity_constrained(y, s, 0.9).status}
    assert results == {"no_score_separation"}


def test_multilabel_predictions_keep_shape_and_allow_co_positives():
    scores = np.array([[0.9] * len(LABELS), [0.1] * len(LABELS)], dtype=np.float32)
    prediction = predict(scores, [0.5] * len(LABELS))
    assert prediction.shape == (2, len(LABELS))
    assert prediction[0].sum() == len(LABELS)


def test_permuted_label_order_is_rejected():
    permuted = list(LABELS)
    permuted[0], permuted[1] = permuted[1], permuted[0]
    with pytest.raises(LabelOrderError):
        validate_label_order(permuted)


def test_threshold_fitting_rejects_test_split():
    empty = np.zeros((0, len(LABELS)))
    split = SplitData("test", empty, empty, empty, empty.astype(np.int8), empty.astype(np.float32), LABELS)
    with pytest.raises(HeldOutSplitAccessError):
        require_development_split(split)


def test_methods_are_deterministic():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    s = rng.random(200).astype(np.float32)
    for method, kwargs in itertools.product(
            [sensitivity_constrained], [{"min_sensitivity": 0.9}, {"min_sensitivity": 0.95, "use_wilson_lower": True}]):
        assert method(y, s, **kwargs) == method(y, s, **kwargs)
