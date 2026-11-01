"""Stage E: one independent evaluation of the FROZEN policies on the test split.

Refuses to run unless every frozen artifact, the checkpoint and the test file
match the hashes recorded at freeze time. No threshold is fitted here.

Run:  ../../../.venv/bin/python evaluate_test.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (FROZEN_DIR, FROZEN_MANIFEST, LABELS, MODEL_DIR, OUTPUT_DIR, SEED,  # noqa: E402
                    load_predictions, predict, sha256_file, validate_label_order)
from metrics import (aggregate_row, bootstrap_counts, patient_bootstrap_weights,  # noqa: E402
                     per_label_table, percentile_ci, rates_from_counts)

BOOTSTRAP_ROUNDS = 2000
CRITICAL_LABELS = ("STEMI", "3AVB")
# Approximate share of ECGs carrying the statement in the HEEDB I0001 hospital archive
# (12SL v24 machine statements; dataset-files/heedb-review/HEEDB_conditions.md). Used only
# to illustrate how PPV would fall at hospital-archive prevalence; NOT an observed result.
ARCHIVE_PREVALENCE = {"NORM": 0.155, "SB": 0.109, "STACH": 0.086, "LBBB": 0.022, "LQT": 0.024,
                      "AF": 0.073, "AFL": 0.012, "3AVB": 0.0008, "2AVB": 0.003, "1AVB": 0.055,
                      "STEMI": 0.015, "STE": 0.009}


def load_frozen() -> tuple[dict, dict[str, dict]]:
    manifest = json.loads(FROZEN_MANIFEST.read_text())
    validate_label_order(manifest["label_order"])
    for name, digest in manifest["frozen_files_sha256"].items():
        if sha256_file(FROZEN_DIR / name) != digest:
            raise RuntimeError(f"frozen artifact {name} changed after freezing")
    for name, digest in manifest["checkpoint_sha256"].items():
        if sha256_file(MODEL_DIR / name) != digest:
            raise RuntimeError(f"checkpoint {name} differs from the one thresholds were fitted for")
    if sha256_file(MODEL_DIR / "test_predictions.csv") != manifest["test_predictions_sha256_pinned_unopened"]:
        raise RuntimeError("test_predictions.csv differs from the file pinned at freeze time")
    policies = {}
    for policy in manifest["policies"]:
        artifact = json.loads((FROZEN_DIR / f"thresholds_{policy}.json").read_text())
        validate_label_order(artifact["label_order"])
        if artifact["comparison_operator"] != ">=" or artifact["source_split"] != "validation":
            raise RuntimeError(f"{policy}: unexpected artifact semantics")
        policies[policy] = artifact
    for policy in (manifest["primary_policy"], manifest["baseline_policy"]):
        bad = [s for s in policies[policy]["status_per_label"] if s != "ok"]
        if bad:
            raise RuntimeError(f"{policy} has unvalidated fallback thresholds {bad}; not reportable as validated")
    return manifest, policies


def bootstrap_tables(test, policies, baseline: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights, record_patient, n_patients = patient_bootstrap_weights(test.patient_keys, BOOTSTRAP_ROUNDS, SEED)
    rates = {}
    for policy, artifact in policies.items():
        counts = bootstrap_counts(test.y_true, predict(test.scores, artifact["selected_thresholds"]),
                                  weights, record_patient, n_patients)
        rates[policy] = rates_from_counts(counts["tp"], counts["fp"], counts["fn"], counts["tn"])
    ci_rows, diff_rows, macro_rows = [], [], []
    for policy, policy_rates in rates.items():
        for metric in ("sensitivity", "specificity", "ppv", "npv", "f1"):
            low, high = percentile_ci(policy_rates[metric])
            for i, label in enumerate(LABELS):
                ci_rows.append({"policy": policy, "label": label, "metric": metric,
                                "boot_ci_low": low[i], "boot_ci_high": high[i]})
            if metric in ("sensitivity", "specificity", "ppv", "f1"):
                diff = policy_rates[metric] - rates[baseline][metric]
                low, high = percentile_ci(diff)
                for i, label in enumerate(LABELS):
                    diff_rows.append({"policy": policy, "baseline": baseline, "label": label, "metric": metric,
                                      "diff_mean": float(np.nanmean(diff[:, i])),
                                      "diff_ci_low": low[i], "diff_ci_high": high[i]})
        macro_f1 = np.nanmean(policy_rates["f1"], axis=1)
        macro_sens = np.nanmean(policy_rates["sensitivity"], axis=1)
        for name, draws, base in (("macro_f1", macro_f1, np.nanmean(rates[baseline]["f1"], axis=1)),
                                  ("macro_sensitivity", macro_sens,
                                   np.nanmean(rates[baseline]["sensitivity"], axis=1))):
            low, high = percentile_ci(draws - base)
            macro_rows.append({"policy": policy, "metric": name, "diff_vs_baseline_ci_low": float(low),
                               "diff_vs_baseline_ci_high": float(high)})
    return pd.DataFrame(ci_rows), pd.DataFrame(diff_rows), pd.DataFrame(macro_rows)


def review_band(test, primary: dict, baseline: dict) -> pd.DataFrame:
    """Records between the clinical (lower) and F1 (higher) thresholds: a 'review' tier."""
    rows = []
    for label in CRITICAL_LABELS:
        i = LABELS.index(label)
        low, high = np.float32(primary["selected_thresholds"][i]), np.float32(baseline["selected_thresholds"][i])
        if low > high:
            raise RuntimeError(f"{label}: review threshold {low} above alert threshold {high}")
        s, y = test.scores[:, i], test.y_true[:, i] == 1
        band = (s >= low) & (s < high)
        rows.append({"label": label, "review_threshold": float(low), "alert_threshold": float(high),
                     "records_in_band": int(band.sum()), "band_per_1000": 1000 * band.mean(),
                     "true_positives_in_band": int((band & y).sum()),
                     "band_ppv": float((band & y).sum() / band.sum()) if band.any() else np.nan,
                     "high_tier_alerts": int((s >= high).sum()), "missed_below_review": int((y & (s < low)).sum())})
    return pd.DataFrame(rows)


def prevalence_shift(per_label: pd.DataFrame, policies: list[str]) -> pd.DataFrame:
    rows = []
    for _, r in per_label[per_label["policy"].isin(policies)].iterrows():
        prevalence = ARCHIVE_PREVALENCE[r["label"]]
        sens, spec = r["sensitivity"], r["specificity"]
        ppv = sens * prevalence / (sens * prevalence + (1 - spec) * (1 - prevalence))
        rows.append({"policy": r["policy"], "label": r["label"], "observed_test_prevalence": r["n_pos"] / (r["n_pos"] + r["n_neg"]),
                     "archive_prevalence_assumed": prevalence, "observed_test_ppv": r["ppv"],
                     "ppv_at_archive_prevalence": ppv,
                     "false_alerts_per_1000_at_archive_prevalence": 1000 * (1 - spec) * (1 - prevalence)})
    return pd.DataFrame(rows)


def plot_test_tradeoff(per_label: pd.DataFrame, primary: str, baseline: str, path) -> None:
    labels = list(LABELS)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, metric, title in ((axes[0], "sensitivity", "Test sensitivity (Wilson 95% CI)"),
                              (axes[1], "specificity", "Test specificity (Wilson 95% CI)")):
        for offset, policy in ((-0.18, baseline), (0.18, primary)):
            frame = per_label[per_label["policy"] == policy].set_index("label").loc[labels]
            value = frame[metric].to_numpy()
            err = np.vstack([value - frame[f"{metric}_ci_low"], frame[f"{metric}_ci_high"] - value])
            ax.errorbar(x + offset, value, yerr=err, fmt="o", capsize=3, label=policy)
        ax.set_xticks(x, labels, rotation=45)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].axhline(0.90, color="red", lw=0.8, ls="--")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    manifest, policies = load_frozen()
    test = load_predictions(MODEL_DIR / "test_predictions.csv", "test")
    primary, baseline = manifest["primary_policy"], manifest["baseline_policy"]

    per_label = pd.concat([per_label_table(test.y_true, test.scores, a["selected_thresholds"], p, "test")
                           for p, a in policies.items()])
    per_label["status"] = [policies[p]["status_per_label"][LABELS.index(label)]
                           for p, label in zip(per_label["policy"], per_label["label"])]
    aggregate = pd.DataFrame([aggregate_row(test.y_true, test.scores, a["selected_thresholds"], p, "test")
                              for p, a in policies.items()])
    boot_ci, paired, macro = bootstrap_tables(test, policies, baseline)
    per_label = per_label.merge(
        boot_ci.pivot_table(index=["policy", "label"], columns="metric",
                            values=["boot_ci_low", "boot_ci_high"]).pipe(
            lambda f: f.set_axis([f"{m}_boot_{side.split('_')[-1]}" for side, m in f.columns], axis=1)).reset_index(),
        on=["policy", "label"])

    by_source = []
    for source in sorted(np.unique(test.sources)):
        mask = test.sources == source
        for policy in (baseline, primary):
            frame = per_label_table(test.y_true[mask], test.scores[mask],
                                    policies[policy]["selected_thresholds"], policy, "test")
            frame.insert(0, "source", source)
            by_source.append(frame)

    per_label.to_csv(OUTPUT_DIR / "results_test_per_label.csv", index=False)
    aggregate.to_csv(OUTPUT_DIR / "results_test_aggregate.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "results_test_paired_vs_baseline.csv", index=False)
    macro.to_csv(OUTPUT_DIR / "results_test_macro_paired_vs_baseline.csv", index=False)
    pd.concat(by_source).to_csv(OUTPUT_DIR / "results_test_by_source.csv", index=False)
    review_band(test, policies[primary], policies[baseline]).to_csv(OUTPUT_DIR / "results_test_review_band.csv", index=False)
    prevalence_shift(per_label, [baseline, primary]).to_csv(OUTPUT_DIR / "results_test_prevalence_shift.csv", index=False)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    plot_test_tradeoff(per_label, primary, baseline, OUTPUT_DIR / "figures" / "test_sensitivity_specificity.png")
    (OUTPUT_DIR / "test_evaluation_log.json").write_text(json.dumps({
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_manifest_sha256": sha256_file(FROZEN_MANIFEST),
        "frozen_at_utc": manifest["frozen_at_utc"],
        "bootstrap_rounds": BOOTSTRAP_ROUNDS, "seed": SEED,
        "test_records": int(len(test.y_true)), "test_patients": int(len(np.unique(test.patient_keys))),
    }, indent=2))
    print(aggregate.set_index("policy")[["macro_f1", "macro_sensitivity", "total_fn", "total_fp"]].round(4))


if __name__ == "__main__":
    main()
