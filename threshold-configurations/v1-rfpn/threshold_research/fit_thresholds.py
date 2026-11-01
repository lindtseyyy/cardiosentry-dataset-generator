"""Stages B-D: reproduce the F1 baseline, fit every policy on VALIDATION ONLY, freeze.

This script never opens the test predictions' contents. It records the test file's
sha256 in the freeze manifest so the later evaluation is pinned to a known file.

Run:  ../../../.venv/bin/python fit_thresholds.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (FROZEN_DIR, FROZEN_MANIFEST, LABELS, MODEL_DIR, OUTPUT_DIR, RESEARCH_DIR,  # noqa: E402
                    SEED, SplitData, load_predictions, predict, require_development_split, sha256_file)
from metrics import aggregate_row, per_label_table  # noqa: E402
from policies import BASELINE_POLICY, POLICIES, PRIMARY_POLICY, Rule  # noqa: E402
from threshold_methods import (ThresholdResult, max_fbeta, notebook_f1_thresholds,  # noqa: E402
                               operating_points, sensitivity_constrained, specificity_constrained,
                               youden_j)

STABILITY_ROUNDS = 300
CROSSFIT_REPEATS = 50
CALIBRATION_BINS = 10
FIGURE_LABELS = ("NORM", "AF", "AFL", "LQT", "3AVB", "2AVB", "STEMI", "STE")


def fit_rule(rule: Rule, y: np.ndarray, s: np.ndarray) -> ThresholdResult:
    if rule.method == "fixed":
        return ThresholdResult("fixed", "ok", float(rule.params["threshold"]), dict(rule.params))
    if rule.method == "max_fbeta":
        return max_fbeta(y, s, rule.params["beta"])
    if rule.method == "youden_j":
        return youden_j(y, s)
    if rule.method == "sens_constrained":
        return sensitivity_constrained(y, s, **rule.params)
    if rule.method == "spec_constrained":
        return specificity_constrained(y, s, **rule.params)
    raise ValueError(f"unknown method {rule.method}")


def fit_policy(rules: dict[str, Rule], split: SplitData) -> list[ThresholdResult]:
    require_development_split(split)
    return [fit_rule(rules[label], split.y_true[:, i], split.scores[:, i]) for i, label in enumerate(LABELS)]


def operational_thresholds(results: list[ThresholdResult]) -> list[float]:
    """Non-ok labels fall back to 0.5 for engineering compatibility only; never validated."""
    return [r.threshold if r.status == "ok" else 0.5 for r in results]


def reproduce_baseline(split: SplitData) -> dict:
    saved = json.loads((MODEL_DIR / "validation_tuned_thresholds.json").read_text())
    port = notebook_f1_thresholds(split.y_true, split.scores)
    ours = [max_fbeta(split.y_true[:, i], split.scores[:, i], 1.0).threshold for i in range(len(LABELS))]
    frame = pd.read_csv(MODEL_DIR / "validation_predictions.csv", usecols=lambda c: c.startswith("pred_"))
    stored = frame[[f"pred_validation_tuned_{label}" for label in LABELS]].to_numpy()
    replayed = predict(split.scores, [saved[label] for label in LABELS])
    stored_05 = frame[[f"pred_0_5_{label}" for label in LABELS]].to_numpy()
    csv_text_float64 = pd.read_csv(MODEL_DIR / "validation_predictions.csv",
                                   usecols=[f"prob_{label}" for label in LABELS])[[f"prob_{label}" for label in LABELS]]
    replayed_float64 = (csv_text_float64.to_numpy() >= np.array([saved[label] for label in LABELS])[None, :])
    report = {
        "saved_thresholds": saved,
        "notebook_port_matches_saved": all(np.float32(port[i]) == np.float32(saved[l]) for i, l in enumerate(LABELS)),
        "reimplementation_matches_saved": all(np.float32(ours[i]) == np.float32(saved[l]) for i, l in enumerate(LABELS)),
        "stored_tuned_prediction_mismatches_float32": int((replayed != stored).sum()),
        "stored_fixed_prediction_mismatches_float32": int((predict(split.scores, [0.5] * 12) != stored_05).sum()),
        "stored_tuned_prediction_mismatches_float64_csv_text": int((replayed_float64.astype(int) != stored).sum()),
    }
    if not (report["notebook_port_matches_saved"] and report["reimplementation_matches_saved"]
            and report["stored_tuned_prediction_mismatches_float32"] == 0):
        raise RuntimeError(f"baseline not reproduced: {report}")
    return report


def patient_groups(split: SplitData):
    _, index = np.unique(split.patient_keys, return_inverse=True)
    return index, int(index.max()) + 1


def threshold_stability(split: SplitData) -> pd.DataFrame:
    """Refit each (policy, label) rule on patient-bootstrap resamples of validation."""
    rng = np.random.default_rng(SEED)
    record_patient, n_patients = patient_groups(split)
    members = [np.flatnonzero(record_patient == p) for p in range(n_patients)]
    unique_rules = {}
    for policy, rules in POLICIES.items():
        for label, rule in rules.items():
            if rule.method != "fixed":
                unique_rules.setdefault((label, rule.method, json.dumps(rule.params, sort_keys=True)), rule)
    draws = {key: [] for key in unique_rules}
    for _ in range(STABILITY_ROUNDS):
        rows = np.concatenate([members[p] for p in rng.integers(0, n_patients, n_patients)])
        for (label, method, params), rule in unique_rules.items():
            i = LABELS.index(label)
            result = fit_rule(rule, split.y_true[rows, i], split.scores[rows, i])
            draws[(label, method, params)].append(np.nan if result.threshold is None else result.threshold)
    out = []
    for (label, method, params), values in draws.items():
        values = np.asarray(values, dtype=float)
        out.append({"label": label, "method": method, "params": params,
                    "threshold_p2_5": np.nanpercentile(values, 2.5), "threshold_median": np.nanmedian(values),
                    "threshold_p97_5": np.nanpercentile(values, 97.5),
                    "infeasible_rounds": int(np.isnan(values).sum()), "rounds": STABILITY_ROUNDS})
    return pd.DataFrame(out)


def crossfit_transport(split: SplitData) -> pd.DataFrame:
    """Repeated patient-level 2-fold: fit on one half, score the other.

    Estimates how much a validation-selected operating point degrades on unseen
    patients from the same distribution, before the test set is touched.
    """
    rng = np.random.default_rng(SEED + 1)
    record_patient, n_patients = patient_groups(split)
    records = []
    for repeat in range(CROSSFIT_REPEATS):
        half = rng.random(n_patients) < 0.5
        in_a = half[record_patient]
        for fit_mask, eval_mask in ((in_a, ~in_a), (~in_a, in_a)):
            for policy, rules in POLICIES.items():
                for i, label in enumerate(LABELS):
                    result = fit_rule(rules[label], split.y_true[fit_mask, i], split.scores[fit_mask, i])
                    if result.threshold is None:
                        records.append({"policy": policy, "label": label, "feasible": False})
                        continue
                    y = split.y_true[eval_mask, i]
                    p = split.scores[eval_mask, i] >= np.float32(result.threshold)
                    records.append({"policy": policy, "label": label, "feasible": True,
                                    "sensitivity": (p & (y == 1)).sum() / y.sum(),
                                    "specificity": (~p & (y == 0)).sum() / (y == 0).sum()})
    frame = pd.DataFrame(records)
    summary = frame.groupby(["policy", "label"], sort=False).agg(
        oof_sensitivity_mean=("sensitivity", "mean"),
        oof_sensitivity_p5=("sensitivity", lambda v: np.nanpercentile(v, 5)),
        oof_specificity_mean=("specificity", "mean"),
        oof_specificity_p5=("specificity", lambda v: np.nanpercentile(v, 5)),
        infeasible_fits=("feasible", lambda v: int((~v.astype(bool)).sum())),
        fits=("feasible", "size"),
    ).reset_index()
    return summary


def calibration_table(split: SplitData) -> pd.DataFrame:
    edges = np.linspace(0, 1, CALIBRATION_BINS + 1)
    rows = []
    for i, label in enumerate(LABELS):
        y = split.y_true[:, i].astype(float)
        s = split.scores[:, i].astype(float)
        bins = np.clip(np.digitize(s, edges[1:-1]), 0, CALIBRATION_BINS - 1)
        ece = sum(abs(y[bins == b].mean() - s[bins == b].mean()) * (bins == b).mean()
                  for b in range(CALIBRATION_BINS) if (bins == b).any())
        rows.append({"label": label, "prevalence": y.mean(), "mean_score": s.mean(),
                     "brier": float(np.mean((s - y) ** 2)), "ece_10_bins": float(ece)})
    return pd.DataFrame(rows)


def plot_validation_curves(split: SplitData, fitted: dict[str, list[ThresholdResult]], path):
    markers = {"max_f1": ("o", "F1 (baseline)"), "max_f2": ("s", "F2"), "youden_j": ("^", "Youden"),
               PRIMARY_POLICY: ("*", "clinical_v1 (primary)")}
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for ax, label in zip(axes.ravel(), FIGURE_LABELS):
        i = LABELS.index(label)
        points = operating_points(split.y_true[:, i], split.scores[:, i])
        ax.plot(1 - points.specificity, points.sensitivity, lw=1.2, color="0.3")
        for policy, (marker, name) in markers.items():
            t = fitted[policy][i].threshold
            if t is None:
                continue
            p = split.scores[:, i] >= np.float32(t)
            y = split.y_true[:, i] == 1
            ax.plot((p & ~y).sum() / (~y).sum(), (p & y).sum() / y.sum(), marker, ms=11 if marker == "*" else 7,
                    label=f"{name} t={t:.3f}")
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.6, 1.005)
        ax.set_title(f"{label} (validation, n_pos={int(points.n_pos)})")
        ax.set_xlabel("1 - specificity (symlog)")
        ax.set_ylabel("sensitivity")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def frozen_artifact(policy: str, results: list[ThresholdResult], checkpoint_hashes: dict) -> dict:
    rules = POLICIES[policy]
    return {
        "schema_version": 1,
        "policy_version": f"{policy}@research_v1",
        "model_checkpoint_sha256": checkpoint_hashes,
        "source_split": "validation",
        "label_order": list(LABELS),
        "score_definition": "post_sigmoid_float32_probability",
        "comparison_operator": ">=",
        "calibration_version": None,
        "selected_thresholds": [float(np.float32(t)) for t in operational_thresholds(results)],
        "selection_methods": [r.method for r in results],
        "selection_params": [r.params for r in results],
        "status_per_label": [r.status if r.status == "ok" else f"{r.status}:fallback_0_5_unvalidated"
                             for r in results],
        "notes_per_label": [r.note for r in results],
        "goal_category_per_label": [rules[label].goal_category for label in LABELS],
        "rationale_per_label": [rules[label].rationale for label in LABELS],
        "selection_evidence": "threshold-configurations/v1-rfpn/CardioSentry_Threshold_Study_Results_v1_rfpn.md",
    }


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=RESEARCH_DIR, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    if FROZEN_MANIFEST.exists():
        raise SystemExit(f"{FROZEN_MANIFEST} exists: thresholds are frozen. Delete the frozen/ "
                         "directory deliberately (and say so in the report) to refit.")
    OUTPUT_DIR.mkdir(exist_ok=True)
    validation = require_development_split(
        load_predictions(MODEL_DIR / "validation_predictions.csv", "validation"))

    baseline = reproduce_baseline(validation)
    (OUTPUT_DIR / "baseline_reproduction.json").write_text(json.dumps(baseline, indent=2))

    fitted = {policy: fit_policy(rules, validation) for policy, rules in POLICIES.items()}
    candidates, per_label, aggregate = [], [], []
    for policy, results in fitted.items():
        thresholds = operational_thresholds(results)
        for label, result, rule in zip(LABELS, results, POLICIES[policy].values()):
            candidates.append({"policy": policy, "label": label, "method": result.method,
                               "params": json.dumps(result.params, sort_keys=True), "status": result.status,
                               "threshold": result.threshold, "note": result.note,
                               "goal_category": rule.goal_category})
        per_label.append(per_label_table(validation.y_true, validation.scores, thresholds, policy, "validation"))
        aggregate.append(aggregate_row(validation.y_true, validation.scores, thresholds, policy, "validation"))
    pd.DataFrame(candidates).to_csv(OUTPUT_DIR / "threshold_candidates.csv", index=False)
    pd.concat(per_label).to_csv(OUTPUT_DIR / "results_validation_per_label.csv", index=False)
    pd.DataFrame(aggregate).to_csv(OUTPUT_DIR / "results_validation_aggregate.csv", index=False)

    threshold_stability(validation).to_csv(OUTPUT_DIR / "threshold_stability_validation.csv", index=False)
    crossfit_transport(validation).to_csv(OUTPUT_DIR / "crossfit_transport_validation.csv", index=False)
    calibration_table(validation).to_csv(OUTPUT_DIR / "calibration_validation.csv", index=False)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    plot_validation_curves(validation, fitted, OUTPUT_DIR / "figures" / "validation_operating_points.png")

    # Stage D: freeze.
    FROZEN_DIR.mkdir(parents=True)
    checkpoint_hashes = {name: sha256_file(MODEL_DIR / name) for name in ("best.pt", "inference.pt")}
    frozen_files = {}
    for policy, results in fitted.items():
        path = FROZEN_DIR / f"thresholds_{policy}.json"
        path.write_text(json.dumps(frozen_artifact(policy, results, checkpoint_hashes), indent=2))
        frozen_files[path.name] = sha256_file(path)
    code_files = sorted(p.name for p in RESEARCH_DIR.glob("*.py"))
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "primary_policy": PRIMARY_POLICY,
        "baseline_policy": BASELINE_POLICY,
        "policies": list(POLICIES),
        "label_order": list(LABELS),
        "comparison_operator": ">=",
        "score_dtype": "float32",
        "checkpoint_sha256": checkpoint_hashes,
        "validation_predictions_sha256": sha256_file(MODEL_DIR / "validation_predictions.csv"),
        "test_predictions_sha256_pinned_unopened": sha256_file(MODEL_DIR / "test_predictions.csv"),
        "frozen_files_sha256": frozen_files,
        "code_sha256": {name: sha256_file(RESEARCH_DIR / name) for name in code_files},
        "seed": SEED,
        "planned_test_analyses": [
            "per-label TP/FP/FN/TN, sensitivity, specificity, PPV, NPV, F1, F2, FP/FN per 1000",
            "Wilson 95% CI for sensitivity/specificity; patient-clustered percentile bootstrap "
            "(2000 rounds) for PPV/NPV/F1",
            "paired patient-clustered bootstrap differences vs max_f1 (sensitivity, specificity, PPV, F1)",
            "macro/micro aggregates; any-abnormal sensitivity; abnormal predicted/displayed normal",
            "per-source (I0001/I0006) breakdown for the primary policy",
            "two-tier review band (clinical threshold <= score < F1 threshold) for STEMI/3AVB",
            "PPV at hypothetical intended-use prevalences (not observed)",
            "goal attainment: test sensitivity vs the 0.90 STEMI/3AVB goal, reported even if missed",
        ],
    }
    FROZEN_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({policy: dict(zip(LABELS, operational_thresholds(r))) for policy, r in fitted.items()},
                     indent=1))


if __name__ == "__main__":
    main()
