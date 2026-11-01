"""Render markdown tables from the outputs/ CSVs for the results report.

Run after evaluate_test.py:  ../../../.venv/bin/python make_report_tables.py
Writes outputs/report_tables.md.
"""

import json

import pandas as pd

from common import LABELS, OUTPUT_DIR
from policies import BASELINE_POLICY, POLICIES, PRIMARY_POLICY


def pct(value, digits=1):
    return "n/a" if pd.isna(value) else f"{100 * value:.{digits}f}"


def ci(low, high, digits=1):
    return f"[{pct(low, digits)}–{pct(high, digits)}]"


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    val = pd.read_csv(OUTPUT_DIR / "results_validation_per_label.csv")
    test = pd.read_csv(OUTPUT_DIR / "results_test_per_label.csv")
    val_agg = pd.read_csv(OUTPUT_DIR / "results_validation_aggregate.csv").set_index("policy")
    test_agg = pd.read_csv(OUTPUT_DIR / "results_test_aggregate.csv").set_index("policy")
    paired = pd.read_csv(OUTPUT_DIR / "results_test_paired_vs_baseline.csv")
    macro = pd.read_csv(OUTPUT_DIR / "results_test_macro_paired_vs_baseline.csv")
    crossfit = pd.read_csv(OUTPUT_DIR / "crossfit_transport_validation.csv")
    band = pd.read_csv(OUTPUT_DIR / "results_test_review_band.csv")
    source = pd.read_csv(OUTPUT_DIR / "results_test_by_source.csv")
    shift = pd.read_csv(OUTPUT_DIR / "results_test_prevalence_shift.csv")
    stability = pd.read_csv(OUTPUT_DIR / "threshold_stability_validation.csv")
    candidates = pd.read_csv(OUTPUT_DIR / "threshold_candidates.csv")

    def row(frame, policy, label):
        return frame[(frame.policy == policy) & (frame.label == label)].iloc[0]

    out = []

    # 1. Thresholds before/after
    rows = []
    for label in LABELS:
        b, p = row(candidates, BASELINE_POLICY, label), row(candidates, PRIMARY_POLICY, label)
        rule = POLICIES[PRIMARY_POLICY][label]
        method = rule.method + ("" if not rule.params else " " + ", ".join(f"{k}={v}" for k, v in rule.params.items()))
        st = stability[(stability.label == label) & (stability.method == rule.method)
                       & (stability.params == json.dumps(rule.params, sort_keys=True))]
        band_txt = (f"{st.threshold_p2_5.iloc[0]:.3f}–{st.threshold_p97_5.iloc[0]:.3f}" if len(st) else "n/a")
        rows.append([label, f"{b.threshold:.4f}", f"**{p.threshold:.4f}**", method, p.status, band_txt])
    out.append("## T1 Thresholds\n\n" + table(
        ["Label", "Before: max-F1", "After: clinical_v1", "Rule", "Status", "Bootstrap 95% range of refitted threshold"], rows))

    # 2. Validation before/after
    rows = []
    for label in LABELS:
        b, p = row(val, BASELINE_POLICY, label), row(val, PRIMARY_POLICY, label)
        rows.append([label, b.n_pos, f"{pct(b.sensitivity)} → {pct(p.sensitivity)}",
                     f"{pct(b.specificity, 2)} → {pct(p.specificity, 2)}", f"{pct(b.ppv)} → {pct(p.ppv)}",
                     f"{pct(b.f1)} → {pct(p.f1)}", f"{b.fn} → {p.fn}", f"{b.fp} → {p.fp}"])
    out.append("## T2 Validation (threshold-selection data), before → after\n\n" + table(
        ["Label", "n_pos", "Sensitivity %", "Specificity %", "PPV %", "F1 %", "FN", "FP"], rows))

    # 3. Test before/after with CIs
    rows = []
    for label in LABELS:
        b, p = row(test, BASELINE_POLICY, label), row(test, PRIMARY_POLICY, label)
        rows.append([label, b.n_pos,
                     f"{pct(b.sensitivity)} {ci(b.sensitivity_ci_low, b.sensitivity_ci_high)}",
                     f"**{pct(p.sensitivity)}** {ci(p.sensitivity_ci_low, p.sensitivity_ci_high)}",
                     f"{pct(b.specificity, 2)} → **{pct(p.specificity, 2)}**",
                     f"{pct(b.ppv)} → **{pct(p.ppv)}** {ci(p.ppv_boot_low, p.ppv_boot_high)}",
                     f"{pct(b.npv, 2)} → {pct(p.npv, 2)}",
                     f"{pct(b.f1)} → {pct(p.f1)}", f"{b.fn} → {p.fn}", f"{b.fp} → {p.fp}"])
    out.append("## T3 Test (untouched), before (max-F1) → after (clinical_v1)\n\n" + table(
        ["Label", "n_pos", "Sens before [95% CI]", "Sens after [95% CI]", "Spec %", "PPV % (after CI)",
         "NPV %", "F1 %", "FN", "FP"], rows))

    # 4. Paired differences on test
    rows = []
    for label in LABELS:
        cells = [label]
        for metric in ("sensitivity", "specificity", "ppv", "f1"):
            d = paired[(paired.policy == PRIMARY_POLICY) & (paired.label == label) & (paired.metric == metric)].iloc[0]
            cells.append("0" if d.diff_mean == 0 and d.diff_ci_low == 0 and d.diff_ci_high == 0 else
                         f"{100 * d.diff_mean:+.1f} [{100 * d.diff_ci_low:+.1f}, {100 * d.diff_ci_high:+.1f}]")
        rows.append(cells)
    out.append("## T4 Test paired difference clinical_v1 − max-F1 (percentage points, patient-clustered bootstrap 95% CI)\n\n"
               + table(["Label", "ΔSensitivity", "ΔSpecificity", "ΔPPV", "ΔF1"], rows))

    # 5. Aggregates
    cols = [("macro_sensitivity", "Macro sens"), ("macro_specificity", "Macro spec"), ("macro_ppv", "Macro PPV"),
            ("macro_f1", "Macro F1"), ("micro_f1", "Micro F1"), ("total_fn", "Total FN"), ("total_fp", "Total FP"),
            ("exact_match", "Exact match"), ("any_abnormal_sensitivity", "Any-abnormal sens"),
            ("abnormal_predicted_normal", "Abnormal→NORM+"), ("abnormal_displayed_normal", "Abnormal shown Normal"),
            ("abnormal_with_no_pathology_alert", "Abnormal, no alert"),
            ("normal_records_with_pathology_alert", "Normal with alert"),
            ("records_with_any_false_alert_per_1000", "Records w/ ≥1 false alert /1000"),
            ("af_or_afl_sensitivity", "AF∪AFL sens"), ("af_or_afl_specificity", "AF∪AFL spec")]
    for name, agg in (("Validation", val_agg), ("Test", test_agg)):
        rows = []
        for policy in POLICIES:
            r = agg.loc[policy]
            rows.append([f"**{policy}**" if policy in (PRIMARY_POLICY, BASELINE_POLICY) else policy] +
                        [pct(r[c]) if isinstance(r[c], float) and r[c] <= 1 and "per_1000" not in c else
                         (f"{r[c]:.1f}" if isinstance(r[c], float) else int(r[c])) for c, _ in cols])
        out.append(f"## T5 {name} aggregates, all policies (% unless count)\n\n" + table(["Policy"] + [h for _, h in cols], rows))
    rows = [[m.policy, m.metric, f"[{100 * m.diff_vs_baseline_ci_low:+.2f}, {100 * m.diff_vs_baseline_ci_high:+.2f}]"]
            for m in macro.itertuples() if m.policy != BASELINE_POLICY]
    out.append("## T6 Test macro paired difference vs max-F1 (pp, 95% CI)\n\n" + table(["Policy", "Metric", "95% CI"], rows))

    # 7. Sensitivity-analysis arms per label, test
    arms = list(POLICIES)
    for name, frame in (("Validation", val), ("Test", test)):
        rows = []
        for label in LABELS:
            rows.append([label] + [f"{pct(row(frame, a, label).sensitivity)} / {pct(row(frame, a, label).specificity)}"
                                   for a in arms])
        out.append(f"## T7 {name} sensitivity / specificity (%) for every frozen arm\n\n" + table(["Label"] + arms, rows))

    # 8. Transport: cross-fit prediction vs test
    rows = []
    for label in LABELS:
        c = crossfit[(crossfit.policy == PRIMARY_POLICY) & (crossfit.label == label)].iloc[0]
        v, t = row(val, PRIMARY_POLICY, label), row(test, PRIMARY_POLICY, label)
        rows.append([label, pct(v.sensitivity), f"{pct(c.oof_sensitivity_mean)} (p5 {pct(c.oof_sensitivity_p5)})",
                     pct(t.sensitivity), pct(v.specificity, 2), pct(c.oof_specificity_mean, 2), pct(t.specificity, 2)])
    out.append("## T8 clinical_v1 transport: validation fit → cross-fit estimate → test\n\n" + table(
        ["Label", "Val sens", "Cross-fit OOF sens (5th pct)", "Test sens", "Val spec", "OOF spec", "Test spec"], rows))

    # 9. Review band
    rows = [[b.label, f"{b.review_threshold:.4f}", f"{b.alert_threshold:.4f}", b.high_tier_alerts, b.records_in_band,
             f"{b.band_per_1000:.1f}", b.true_positives_in_band, pct(b.band_ppv), b.missed_below_review]
            for b in band.itertuples()]
    out.append("## T9 Two-tier view on test (review tier = clinical ≤ score < F1)\n\n" + table(
        ["Label", "Review threshold", "Alert threshold", "High-tier alerts", "Review-tier records", "Review /1000",
         "True positives in review tier", "Review-tier PPV %", "Missed (below review)"], rows))

    # 10. By source
    rows = []
    for label in LABELS:
        cells = [label]
        for src in sorted(source.source.unique()):
            for policy in (BASELINE_POLICY, PRIMARY_POLICY):
                r = source[(source.source == src) & (source.policy == policy) & (source.label == label)].iloc[0]
                cells.append(f"{pct(r.sensitivity)}/{pct(r.specificity)} (n={r.n_pos})")
        rows.append(cells)
    hdr = ["Label"] + [f"{s} {p}" for s in sorted(source.source.unique()) for p in (BASELINE_POLICY, PRIMARY_POLICY)]
    out.append("## T10 Test sens/spec % by source site\n\n" + table(hdr, rows))

    # 11. Prevalence shift
    rows = []
    for label in LABELS:
        b = shift[(shift.policy == BASELINE_POLICY) & (shift.label == label)].iloc[0]
        p = shift[(shift.policy == PRIMARY_POLICY) & (shift.label == label)].iloc[0]
        rows.append([label, pct(b.observed_test_prevalence), pct(b.archive_prevalence_assumed, 2),
                     f"{pct(b.observed_test_ppv)} → {pct(p.observed_test_ppv)}",
                     f"{pct(b.ppv_at_archive_prevalence)} → {pct(p.ppv_at_archive_prevalence)}",
                     f"{b.false_alerts_per_1000_at_archive_prevalence:.1f} → {p.false_alerts_per_1000_at_archive_prevalence:.1f}"])
    out.append("## T11 PPV re-weighted to approximate hospital-archive prevalence (illustrative, not observed)\n\n" + table(
        ["Label", "Test prevalence %", "Archive prevalence %", "Observed PPV % before → after",
         "PPV % at archive prevalence before → after", "False alerts /1000 ECGs before → after"], rows))

    (OUTPUT_DIR / "report_tables.md").write_text("\n\n".join(out) + "\n")
    print((OUTPUT_DIR / "report_tables.md").read_text()[:1500])


if __name__ == "__main__":
    main()
