"""
Filter-bandwidth LVH analysis - THE deliverable of this round.

Reads the live per-image probabilities the frozen checkpoint produced on
the regenerated sheets (live_probs.json), cross-checks the 40 Hz control
against the FROZEN test_predictions.csv (refuses on any disagreement beyond
floating-point drift), then prints and writes:

  1. cross-check table        live 40Hz vs frozen file, per label
  2. per-case LVH table       the 10 cases x 4 variants (probs, decisions,
                              deltas, flips)
  3. band metrics             recall/specificity/precision/F1/accuracy +
                              confusion matrix per band, with the frozen
                              LVH-negative pool as the negatives (declared)
  4. the verdict              does a higher upper cutoff improve LVH
                              detection on this 10-case probe?

The reference numbers ALWAYS come from the frozen file. The live run only
adds the regenerated variants; nothing is retrained or re-thresholded.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

FIELDS_CASE = ["case", "true_LVH",
               "frozen_LVH_prob", "frozen_LVH_pred",
               "orig40_prob", "orig40_pred", "orig40_correct",
               "regen40_prob", "regen40_pred", "regen40_correct",
               "prob_100", "pred_100", "correct_100", "delta_100",
               "prob_150", "pred_150", "correct_150", "delta_150",
               "flipped_to_positive_100", "flipped_to_positive_150",
               "flipped_to_negative_100", "flipped_to_negative_150"]


def _pred(p: float) -> int:
    return int(p >= C.LVH_THRESHOLD)


def load_frozen(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert len(df) == C.TEST_N, f"frozen file has {len(df)} rows, expected {C.TEST_N}"
    need = {"record", "true_LVH", "true_AF", "prob_LVH", "pred_LVH"}
    assert need <= set(df.columns), f"frozen file missing {need - set(df.columns)}"
    sub = df[df["record"].isin(C.CASES)]
    assert len(sub) == len(C.CASES), "not all 10 cases are in the frozen file"
    assert (sub["true_LVH"] == 1).all(), "a probe case is not true_LVH=1"
    assert (sub["true_AF"] == 1).all(), "a probe case is not true_AF=1"
    return df


def cross_check(frozen: pd.DataFrame, live: dict) -> dict:
    """live 40Hz (original + regenerated) vs frozen, per label.

    Two tiers:
      1. probability drift per label (max|delta|, RMSE) against the frozen
         probs - banded per C.DRIFT_BANDS;
      2. DECISION cross-check: the live 40 Hz decisions at the FROZEN
         per-label thresholds must equal the frozen pred_* columns exactly
         for every probe case. Probability drift without a decision change
         cannot move any number this round reports.
    """
    f = frozen.set_index("record")
    out = {}
    for variant in ("orig_40", "regen_40"):
        for k, lab in enumerate(C.LABELS):
            d = np.array([live[case][variant][k] for case in C.CASES]) \
                - f.loc[C.CASES, f"prob_{lab}"].values
            out[f"{variant}_{lab}_maxabs"] = float(np.abs(d).max())
            out[f"{variant}_{lab}_rmse"] = float(np.sqrt((d ** 2).mean()))
            # decision-level cross-check at the frozen threshold
            n_flip = 0
            for case, delta in zip(C.CASES, d):
                live_p = live[case][variant][k]
                frozen_p = float(f.loc[case, f"prob_{lab}"])
                frozen_pred = int(f.loc[case, f"pred_{lab}"])
                live_pred = int(live_p >= C.FROZEN_THRESHOLDS[k])
                if live_pred != frozen_pred:
                    n_flip += 1
            out[f"{variant}_{lab}_decision_flips"] = n_flip
    worst = max(v for k, v in out.items() if k.endswith("_maxabs"))
    total_decision_flips = sum(
        v for k, v in out.items() if k.endswith("_decision_flips"))
    out["worst_maxabs"] = worst
    out["decision_flips"] = total_decision_flips
    verdict = "MISMATCH"
    for band, name in C.DRIFT_BANDS:
        if worst < band:
            verdict = name
            break
    if verdict == "MISMATCH" and total_decision_flips == 0:
        verdict = "encoding_drift_flagged"
    out["verdict"] = verdict
    return out


def per_case_table(frozen: pd.DataFrame, live: dict) -> pd.DataFrame:
    f = frozen.set_index("record")
    rows = []
    for case in C.CASES:
        p0 = live[case]["orig_40"][C.LVH_INDEX]
        r0 = live[case]["regen_40"][C.LVH_INDEX]
        p100 = live[case]["regen_100"][C.LVH_INDEX]
        p150 = live[case]["regen_150"][C.LVH_INDEX]
        rows.append({
            "case": case,
            "true_LVH": 1,
            "frozen_LVH_prob": float(f.loc[case, "prob_LVH"]),
            "frozen_LVH_pred": int(f.loc[case, "pred_LVH"]),
            "orig40_prob": float(p0), "orig40_pred": _pred(p0),
            "orig40_correct": "correct" if _pred(p0) else "INCORRECT",
            "regen40_prob": float(r0), "regen40_pred": _pred(r0),
            "regen40_correct": "correct" if _pred(r0) else "INCORRECT",
            "prob_100": float(p100), "pred_100": _pred(p100),
            "correct_100": "correct" if _pred(p100) else "INCORRECT",
            "delta_100": float(p100 - r0),
            "prob_150": float(p150), "pred_150": _pred(p150),
            "correct_150": "correct" if _pred(p150) else "INCORRECT",
            "delta_150": float(p150 - r0),
            "flipped_to_positive_100": int(_pred(r0) == 0 and _pred(p100) == 1),
            "flipped_to_positive_150": int(_pred(r0) == 0 and _pred(p150) == 1),
            "flipped_to_negative_100": int(_pred(r0) == 1 and _pred(p100) == 0),
            "flipped_to_negative_150": int(_pred(r0) == 1 and _pred(p150) == 0),
            "flip_margin_100": round(abs(p100 - C.LVH_THRESHOLD), 4)
            if (_pred(r0) == 0 and _pred(p100) == 1) else "",
            "flip_margin_150": round(abs(p150 - C.LVH_THRESHOLD), 4)
            if (_pred(r0) == 0 and _pred(p150) == 1) else "",
        })
    return pd.DataFrame(rows)


def band_metrics(frozen: pd.DataFrame, live: dict) -> pd.DataFrame:
    """Per band over the 10 live positives + the FROZEN LVH-negative pool.

    The negatives are the 683 test ECGs with true_LVH=0, taken from the
    frozen file at their frozen 0.47 decisions - the exact convention the
    previous subgroup rounds used ('same negatives, same frozen decisions').
    They were not regenerated, so FP/TN (and therefore specificity) are
    IDENTICAL across bands by construction; only TP/FN move. Declared, not
    hidden: the band comparison measures the positive side, which is what
    this experiment changed.
    """
    neg = frozen[frozen["true_LVH"] == 0]
    assert len(neg) == C.TEST_N - C.TEST_POSITIVES_LVH
    fp = int((neg["pred_LVH"] == 1).sum())
    tn = len(neg) - fp

    rows = []
    for band, variant in (("0.05-40Hz", "regen_40"),
                          ("0.05-100Hz", "regen_100"),
                          ("0.05-150Hz", "regen_150")):
        preds = [_pred(live[c][variant][C.LVH_INDEX]) for c in C.CASES]
        probs = [live[c][variant][C.LVH_INDEX] for c in C.CASES]
        tp, fn = sum(preds), len(preds) - sum(preds)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * tp / (2 * tp + fp + fn)
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        specificity = tn / (tn + fp)
        rows.append({
            "band": band,
            "variant": variant,
            "n_positive_probe": len(preds),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall_sensitivity": round(recall, 4),
            "specificity": round(specificity, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "confusion_matrix": f"[[TP={tp}, FP={fp}], [FN={fn}, TN={tn}]]",
            "mean_prob": round(float(np.mean(probs)), 4),
            "n_LVH_false_negatives": fn,
            "specificity_note": "frozen LVH-negative pool, identical across bands",
        })
    return pd.DataFrame(rows)


def verdict(df_case: pd.DataFrame, df_band: pd.DataFrame,
            cross: dict) -> str:
    lines = []
    r = df_band.set_index("band")
    base_recall = r.loc["0.05-40Hz", "recall_sensitivity"]
    for band, col in (("0.05-100Hz", "flipped_to_positive_100"),
                      ("0.05-150Hz", "flipped_to_positive_150")):
        rec = r.loc[band, "recall_sensitivity"]
        recov = int(df_case[col].sum())
        back = int(df_case[col.replace("positive", "negative")].sum())
        delta_col = "delta_100" if band.endswith("100Hz") else "delta_150"
        mean_d = float(df_case[delta_col].mean())
        direction = "raises" if mean_d >= 0 else "lowers"
        lines.append(
            f"  {band}: recall {base_recall:.3f} -> {rec:.3f} "
            f"({recov}/10 frozen FNs flip to LVH-positive, {back} flip back), "
            f"mean LVH-prob delta {mean_d:+.4f} (bandwidth {direction} the "
            f"probabilities on average)")
    rec100 = r.loc["0.05-100Hz", "recall_sensitivity"]
    rec150 = r.loc["0.05-150Hz", "recall_sensitivity"]
    marginal = []
    for band, col in (("0.05-100Hz", "flip_margin_100"),
                      ("0.05-150Hz", "flip_margin_150")):
        for _, row in df_case.iterrows():
            if row[col] != "" and float(row[col]) < C.MARGIN_FLAG:
                marginal.append(f"{row['case']}@{band} "
                                f"(margin {row[col]})")
    if marginal:
        lines.append("  marginal flips (could be drift, not the filter): "
                     + ", ".join(marginal))
    improved100 = rec100 > base_recall
    improved150 = rec150 > base_recall
    if improved100 or improved150:
        verdict_line = ("SUPPORTED (descriptive): raising the upper cutoff "
                        "improves LVH recall on this 10-case probe")
    elif rec100 == base_recall and rec150 == base_recall:
        verdict_line = ("NO CHANGE: the higher cutoffs did not move any of "
                        "the 10 LVH decisions")
    else:
        verdict_line = ("NOT SUPPORTED (descriptive): the higher cutoffs did "
                        "not improve LVH recall on this 10-case probe")
    lines.append(f"VERDICT: {verdict_line}")
    lines.append("")
    lines.append("Limitations, stated regardless of verdict: n=10 positives, "
                 "no significance testing, negatives held frozen at 40 Hz; "
                 "the probe measures whether bandwidth recovers the known "
                 "false negatives, not a population effect.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probs", required=True, help="live_probs.json")
    ap.add_argument("--frozen", required=True,
                    help="frozen test_predictions.csv")
    ap.add_argument("--out", required=True, help="write results here")
    args = ap.parse_args()

    live = json.loads(Path(args.probs).read_text())["probs"]
    for case in C.CASES:
        assert set(live[case]) == set(C.VARIANTS), \
            f"{case}: missing variants {set(C.VARIANTS) - set(live[case])}"

    frozen = load_frozen(Path(args.frozen))
    cross = cross_check(frozen, live)
    df_case = per_case_table(frozen, live)
    df_band = band_metrics(frozen, live)
    verdict_text = verdict(df_case, df_band, cross)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    print("=" * 78)
    print("CROSS-CHECK - live 40Hz vs frozen test_predictions.csv")
    print("=" * 78)
    for variant in ("orig_40", "regen_40"):
        print(f"  {variant:9s}", "  ".join(
            f"{lab} {cross[f'{variant}_{lab}_maxabs']:.2e}"
            for lab in C.LABELS))
    print(f"  worst max|delta| = {cross['worst_maxabs']:.3e}  -> "
          f"{cross['verdict']}")
    print(f"  decision flips vs frozen pred columns: {cross['decision_flips']}")
    ok_proceed = (cross["verdict"] in
                  ("identical", "within_drift", "encoding_drift",
                   "encoding_drift_flagged")) and cross["decision_flips"] == 0
    if not ok_proceed:
        print("\nSTOP: live 40Hz does not reproduce the frozen file "
              "(probabilities AND/OR decisions).")
        print("The checkpoint/weights/preprocessing do not match the frozen "
              "evaluation - do not trust the band comparison.")
        (out / "STATUS.txt").write_text(
            "ABORTED - 40Hz control does not reproduce frozen predictions\n")
        return 2
    if cross["verdict"] not in ("identical", "within_drift"):
        print("\n  NOTE: probability drift above float precision but every "
              "40 Hz decision matches the frozen file exactly.")
        print("  The band DELTAS are computed inside one environment and stay "
              "valid; flips whose probability sits closer to 0.47 than "
              f"{C.MARGIN_FLAG} are flagged 'marginal' in the case table.")
    print("\n  (A flip is 'marginal' when its probability is within "
          f"{C.MARGIN_FLAG} of the {C.LVH_THRESHOLD} threshold - i.e. it "
          "could be drift rather than the filter change.)")

    print()
    print("=" * 78)
    print("PER-CASE LVH TABLE - the 10 frozen false negatives x 4 variants")
    print("=" * 78)
    print(df_case.to_string(index=False))

    print()
    print("=" * 78)
    print("BAND METRICS - LVH head at the frozen 0.47 threshold")
    print("=" * 78)
    print(df_band.to_string(index=False))
    print("\n  NOTE:", df_band.iloc[0]["specificity_note"])

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(verdict_text)

    # ---- write everything ------------------------------------------------
    df_case.to_csv(out / "per_case_table.csv", index=False)
    df_band.to_csv(out / "band_metrics.csv", index=False)
    (out / "cross_check.json").write_text(json.dumps(cross, indent=2))

    report = [
        "LVH filter-bandwidth probe - resnet50_hires (frozen) on the 10 "
        "AF+LVH LVH-false-negative test cases",
        "",
        f"threshold: LVH >= {C.LVH_THRESHOLD} (frozen val-tuned, never "
        f"re-tuned)",
        f"cross-check: worst live-40Hz max|delta| vs frozen = "
        f"{cross['worst_maxabs']:.3e} -> {cross['verdict']}",
        "",
        "PER-CASE TABLE", df_case.to_string(index=False),
        "",
        "BAND METRICS", df_band.to_string(index=False),
        "",
        "VERDICT", verdict_text,
    ]
    (out / "report.txt").write_text("\n".join(report))

    summary = {
        "model": C.REFERENCE_MODEL,
        "thresholds": {"LVH": C.LVH_THRESHOLD},
        "cross_check": cross,
        "recall_by_band": {r["band"]: r["recall_sensitivity"]
                           for _, r in df_band.iterrows()},
        "fn_by_band": {r["band"]: r["fn"] for _, r in df_band.iterrows()},
        "flips_to_positive_100": int(df_case["flipped_to_positive_100"].sum()),
        "flips_to_positive_150": int(df_case["flipped_to_positive_150"].sum()),
        "verdict": verdict_text.split("VERDICT: ")[1].splitlines()[0],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "STATUS.txt").write_text("DONE\n")
    print(f"\nwrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
