"""Per-label and multi-label evaluation with Wilson and patient-clustered bootstrap CIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import LABELS, NORMAL_LABEL, PATHOLOGY_LABELS, predict
from threshold_methods import wilson_interval

COUNT_NAMES = ("tp", "fp", "fn", "tn")


def _div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def rates_from_counts(tp, fp, fn, tn) -> dict:
    """Metrics from confusion counts (arrays broadcast). Undefined -> NaN, never 0."""
    n = np.asarray(tp) + fp + fn + tn
    return {
        "sensitivity": _div(tp, tp + fn),
        "specificity": _div(tn, tn + fp),
        "ppv": _div(tp, tp + fp),
        "npv": _div(tn, tn + fn),
        "f1": _div(2 * np.asarray(tp), 2 * np.asarray(tp) + fp + fn),
        "f2": _div(5 * np.asarray(tp), 5 * np.asarray(tp) + 4 * np.asarray(fn) + fp),
        "fnr": _div(fn, tp + fn),
        "fpr": _div(fp, tn + fp),
        "fp_per_1000": _div(1000 * np.asarray(fp), n),
        "fn_per_1000": _div(1000 * np.asarray(fn), n),
        "alerts_per_1000": _div(1000 * (np.asarray(tp) + fp), n),
    }


def confusion(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    """Per-label counts for (n, L) arrays -> dict of (L,) arrays."""
    y = y_true.astype(bool)
    p = prediction.astype(bool)
    return {"tp": (y & p).sum(0), "fp": (~y & p).sum(0), "fn": (y & ~p).sum(0), "tn": (~y & ~p).sum(0)}


def per_label_table(y_true, scores, thresholds, policy: str, split: str) -> pd.DataFrame:
    prediction = predict(scores, thresholds)
    counts = confusion(y_true, prediction)
    rates = rates_from_counts(*(counts[name] for name in COUNT_NAMES))
    sens_lo, sens_hi = wilson_interval(counts["tp"], counts["tp"] + counts["fn"])
    spec_lo, spec_hi = wilson_interval(counts["tn"], counts["tn"] + counts["fp"])
    rows = []
    for index, label in enumerate(LABELS):
        y = y_true[:, index]
        defined = 0 < y.sum() < len(y)
        rows.append({
            "split": split, "policy": policy, "label": label,
            "threshold": float(np.float32(thresholds[index])),
            "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
            **{name: int(counts[name][index]) for name in COUNT_NAMES},
            **{name: float(value[index]) for name, value in rates.items()},
            "sensitivity_ci_low": float(sens_lo[index]), "sensitivity_ci_high": float(sens_hi[index]),
            "specificity_ci_low": float(spec_lo[index]), "specificity_ci_high": float(spec_hi[index]),
            "auroc": float(roc_auc_score(y, scores[:, index])) if defined else np.nan,
            "average_precision": float(average_precision_score(y, scores[:, index])) if defined else np.nan,
        })
    return pd.DataFrame(rows)


def record_level(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    """Multi-label safety quantities, per record (boolean arrays of length n)."""
    normal = LABELS.index(NORMAL_LABEL)
    path_idx = [LABELS.index(label) for label in PATHOLOGY_LABELS]
    critical_idx = [LABELS.index("STEMI"), LABELS.index("3AVB")]
    true_abnormal = y_true[:, path_idx].any(1)
    pred_pathology = prediction[:, path_idx].astype(bool)
    pred_normal = prediction[:, normal].astype(bool)
    return {
        "true_abnormal": true_abnormal,
        "true_normal": y_true[:, normal].astype(bool),
        "pred_any_pathology": pred_pathology.any(1),
        "pred_normal": pred_normal,
        # Presentation rule evaluated separately: show "Normal" only with no pathology alert.
        "display_normal": pred_normal & ~pred_pathology.any(1),
        "normal_with_critical": pred_normal & prediction[:, critical_idx].astype(bool).any(1),
        "normal_with_any_pathology": pred_normal & pred_pathology.any(1),
        "pathology_false_positive_any": (pred_pathology & ~y_true[:, path_idx].astype(bool)).any(1),
    }


def aggregate_row(y_true, scores, thresholds, policy: str, split: str) -> dict:
    prediction = predict(scores, thresholds)
    counts = confusion(y_true, prediction)
    rates = rates_from_counts(*(counts[name] for name in COUNT_NAMES))
    path_idx = [LABELS.index(label) for label in PATHOLOGY_LABELS]
    pooled = {name: int(counts[name].sum()) for name in COUNT_NAMES}
    pooled_rates = rates_from_counts(*(pooled[name] for name in COUNT_NAMES))
    rec = record_level(y_true, prediction)
    n_abnormal = int(rec["true_abnormal"].sum())
    n_normal_true = int(rec["true_normal"].sum())
    af, afl, lbbb, stemi = (LABELS.index(label) for label in ("AF", "AFL", "LBBB", "STEMI"))
    true_af_any = y_true[:, [af, afl]].any(1)
    pred_af_any = prediction[:, [af, afl]].astype(bool).any(1)
    stemi_true = y_true[:, stemi] == 1
    stemi_pred = prediction[:, stemi] == 1
    in_lbbb = y_true[:, lbbb] == 1
    extras = {
        # AF and AFL share the anticoagulation pathway; confusing them is less harmful than missing both.
        "af_or_afl_sensitivity": float(_div((true_af_any & pred_af_any).sum(), true_af_any.sum())),
        "af_or_afl_specificity": float(_div((~true_af_any & ~pred_af_any).sum(), (~true_af_any).sum())),
        "stemi_n_pos_with_lbbb": int((stemi_true & in_lbbb).sum()),
        "stemi_sensitivity_with_lbbb": float(_div((stemi_true & stemi_pred & in_lbbb).sum(),
                                                  (stemi_true & in_lbbb).sum())),
        "stemi_sensitivity_without_lbbb": float(_div((stemi_true & stemi_pred & ~in_lbbb).sum(),
                                                     (stemi_true & ~in_lbbb).sum())),
    }
    return {
        "split": split, "policy": policy, "n_records": int(len(y_true)),
        "macro_sensitivity": float(np.nanmean(rates["sensitivity"])),
        "macro_specificity": float(np.nanmean(rates["specificity"])),
        "macro_ppv": float(np.nanmean(rates["ppv"])),
        "macro_f1": float(np.nanmean(rates["f1"])),
        "macro_f2": float(np.nanmean(rates["f2"])),
        "pathology_macro_sensitivity": float(np.nanmean(rates["sensitivity"][path_idx])),
        "pathology_macro_f1": float(np.nanmean(rates["f1"][path_idx])),
        "micro_sensitivity": float(pooled_rates["sensitivity"]),
        "micro_specificity": float(pooled_rates["specificity"]),
        "micro_ppv": float(pooled_rates["ppv"]),
        "micro_f1": float(pooled_rates["f1"]),
        "total_fn": pooled["fn"], "total_fp": pooled["fp"],
        "exact_match": float((prediction == y_true).all(1).mean()),
        "hamming_accuracy": float((prediction == y_true).mean()),
        "n_true_abnormal": n_abnormal,
        "any_abnormal_sensitivity": float(_div((rec["true_abnormal"] & rec["pred_any_pathology"]).sum(), n_abnormal)),
        "abnormal_with_no_pathology_alert": int((rec["true_abnormal"] & ~rec["pred_any_pathology"]).sum()),
        "abnormal_predicted_normal": int((rec["true_abnormal"] & rec["pred_normal"]).sum()),
        "abnormal_displayed_normal": int((rec["true_abnormal"] & rec["display_normal"]).sum()),
        "normal_records_with_pathology_alert": int((rec["true_normal"] & rec["pred_any_pathology"]).sum()),
        "normal_records_with_pathology_alert_rate": float(
            _div((rec["true_normal"] & rec["pred_any_pathology"]).sum(), n_normal_true)),
        "normal_with_critical_alert": int(rec["normal_with_critical"].sum()),
        "normal_pathology_contradictions": int(rec["normal_with_any_pathology"].sum()),
        "records_with_any_false_alert_per_1000": float(1000 * rec["pathology_false_positive_any"].mean()),
        **extras,
    }


# ---------------------------------------------------------------------------
# Patient-clustered bootstrap
# ---------------------------------------------------------------------------

def patient_bootstrap_weights(patient_keys: np.ndarray, rounds: int, seed: int):
    """(rounds, n_patients) resampling counts and record->patient index."""
    _, record_patient = np.unique(patient_keys, return_inverse=True)
    n_patients = int(record_patient.max()) + 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_patients, size=(rounds, n_patients))
    weights = np.zeros((rounds, n_patients), dtype=np.float32)
    rows = np.repeat(np.arange(rounds), n_patients)
    np.add.at(weights, (rows, draws.ravel()), 1.0)
    return weights, record_patient, n_patients


def bootstrap_counts(y_true, prediction, weights, record_patient, n_patients) -> dict:
    """Per-label confusion counts per bootstrap round: dict name -> (rounds, L)."""
    y = y_true.astype(bool)
    p = prediction.astype(bool)
    out = {}
    for name, cell in (("tp", y & p), ("fp", ~y & p), ("fn", y & ~p), ("tn", ~y & ~p)):
        per_patient = np.zeros((n_patients, y.shape[1]), dtype=np.float32)
        np.add.at(per_patient, record_patient, cell.astype(np.float32))
        out[name] = weights @ per_patient
    return out


def percentile_ci(draws: np.ndarray, confidence: float = 0.95):
    tail = (1 - confidence) / 2 * 100
    return np.nanpercentile(draws, tail, axis=0), np.nanpercentile(draws, 100 - tail, axis=0)
