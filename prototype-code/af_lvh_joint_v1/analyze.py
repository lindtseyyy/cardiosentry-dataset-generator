"""
The AF x LVH joint-head analysis of the frozen four-head reference.

RUN ORDER (all before any number is printed)

  1. load the SAME frozen test predictions the previous round used
     (hybrid_v1/resnet50_hires/predictions/test_predictions.csv);
  2. prove the decisions are the frozen ones: thresholded probabilities must
     reproduce the saved pred_ columns exactly (0 mismatches), the pooled AF
     and LVH rows must equal the frozen transcription in baseline.py, the
     live result.json must agree, and the subgroup sizes must match;
  3. evaluate each head per subgroup (positives = subgroup, negatives = the
     head's frozen negative pool) at the frozen thresholds;
  4. build the joint breakdown of the 28 AF+LVH cases and its bootstrap CIs;
  5. run the three reads of the verdict rule - McNemar on the joint
     breakdown, Fisher on AF recall (AF+LVH vs AF-only), Fisher on LVH
     recall (AF+LVH vs LVH-only) - with bootstrap CIs beside them;
  6. apply the verdict rule and print the sample-size limitation with it.

Nothing is retrained, re-thresholded, re-predicted or re-split. The frozen
results folders are read-only for this round.

USAGE

    python analyze.py --predictions <test_predictions.csv> \
                      [--result-json <result.json>] \
                      --out <run_dir>
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import baseline
import config as C
import metrics as M
import stats

VERDICT_RULE = """
THE VERDICT RULE (frozen - stated here, applied below, never adjusted to fit)

  HYPOTHESIS: "When AF and LVH coexist, the model correctly detects AF but
  frequently misses LVH."

  R1  WITHIN-CASE ASYMMETRY - McNemar exact on the joint breakdown of the 28
      cases (discordant pairs "AF correct + LVH missed" vs "AF missed + LVH
      correct"). Significant at p < 0.05 in the claimed direction = the two
      heads behave differently ON THE SAME ECGs.
  R2  AF NOT DEGRADED - AF recall on AF+LVH vs AF-only (Fisher exact,
      one-sided). The "correctly detects AF" half is credible if AF recall
      on AF+LVH is not significantly below AF-only recall.
  R3  LVH DEGRADED - LVH recall on AF+LVH vs LVH-only (Fisher exact,
      one-sided). The "misses LVH" half is a subgroup deficit only if LVH
      recall on AF+LVH is significantly below LVH-only recall.

      SUPPORTED       R1 significant in the claimed direction, R2 not
                      significant, R3 significant.
      PARTIALLY       R1 significant but R2 significant or R3 not significant.
      NOT SEPARATED   R1 not significant (directional evidence only).
      CONTRADICTED    R1 significant in the opposite direction.

  SAMPLE-SIZE LIMITATION (stated regardless of verdict): the subgroup has 28
  ECGs, so the recall CIs are wide and R2/R3 are low-power. The verdict
  grades the evidence for the ASSOCIATION on this test split; no causal
  claim is made - "AF causes the miss" is not decidable from these numbers.
"""


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"MISSING predictions file: {path}\n"
                         f"  expected at hybrid_v1/resnet50_hires/predictions/"
                         f"test_predictions.csv on Drive.")
    df = pd.read_csv(path)
    need = {"record", "rel_path", "source", "split"}
    for lab in C.LABELS:
        need |= {f"true_{lab}", f"prob_{lab}", f"pred_{lab}"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"predictions file is missing columns: {sorted(missing)}")
    if (df["split"] != "test").any() or len(df) != C.TEST_N:
        raise SystemExit(f"predictions file has {len(df)} rows / non-test rows - "
                         f"expected exactly {C.TEST_N} test rows")
    return df


def cross_check(df: pd.DataFrame, result_json: dict | None) -> list[str]:
    """Prove the frozen predictions are exactly the reference evaluation."""
    problems = []

    # 1. the saved decisions are the frozen thresholds applied to the saved
    #    probabilities - i.e. this file's decisions are the reported ones
    for lab in ("AF", "LVH"):
        canon = (df[f"prob_{lab}"] >= C.THRESHOLDS[lab]).astype(int)
        mm = int((canon != df[f"pred_{lab}"]).sum())
        if mm:
            problems.append(f"  {lab}: {mm} pred_ entries disagree with "
                            f"thresholding prob_ at {C.THRESHOLDS[lab]}")

    # 2. subgroup sizes re-derived from the ground truth
    sizes = {k: int(M.subgroup_mask(df, k).sum()) for k in C.SUBGROUPS}
    for k, want in C.SUBGROUP_N.items():
        if sizes[k] != want:
            problems.append(f"  subgroup {k}: {sizes[k]} != frozen {want}")

    # 3. pooled rows equal the frozen transcription
    for lab, ref in (("AF", baseline.REFERENCE_AF),
                     ("LVH", baseline.REFERENCE_LVH)):
        row = M.head_row(df, lab, "all_af" if lab == "AF" else "all_lvh")
        for k_int in ("tp", "fp", "fn", "tn"):
            if row[k_int] != ref[k_int]:
                problems.append(f"  {lab} pooled {k_int}={row[k_int]} != "
                                f"frozen {ref[k_int]}")
        for k in ("sensitivity", "specificity", "precision", "f1", "fpr",
                  "auprc", "auroc"):
            if abs(row[k] - ref[k]) > 1e-6:
                problems.append(f"  {lab} pooled {k}={row[k]:.8f} != frozen "
                                f"{ref[k]:.8f}")
        if abs(row["threshold"] - ref["threshold"]) > 1e-9:
            problems.append(f"  {lab} threshold {row['threshold']} != frozen "
                            f"{ref['threshold']}")

    # 4. live result.json agrees with the transcription
    if result_json is not None:
        if result_json.get("protocol_fingerprint") != C.REFERENCE_PROTOCOL_FP:
            problems.append(f"  result.json protocol fp "
                            f"{result_json.get('protocol_fingerprint')} != "
                            f"frozen {C.REFERENCE_PROTOCOL_FP}")
        for lab, ref in (("AF", baseline.REFERENCE_AF),
                         ("LVH", baseline.REFERENCE_LVH)):
            try:
                r = result_json["test"]["per_label"][lab]
                for k_int in ("tp", "fp", "fn", "tn"):
                    if int(r.get(k_int, -1)) != ref[k_int]:
                        problems.append(f"  result.json {lab} {k_int}="
                                        f"{r.get(k_int)} != frozen {ref[k_int]}")
            except (KeyError, TypeError) as exc:
                problems.append(f"  result.json unreadable at {lab}: {exc}")

    if problems:
        raise SystemExit(
            "FROZEN-REFERENCE CROSS-CHECK FAILED - the predictions file does "
            "not agree with the frozen reference. The analysis would not be "
            "about the reported model. STOP.\n" + "\n".join(problems))

    return [
        f"  frozen decisions verified: thresholding prob_ at the frozen "
        f"thresholds reproduces every saved pred_ entry (0 mismatches)",
        f"  subgroup sizes re-derived from ground truth match the frozen "
        f"counts: AF+LVH {sizes['af_lvh']}, AF-only {sizes['af_only']}, "
        f"LVH-only {sizes['lvh_only']}",
        f"  pooled AF row reproduces the frozen transcription exactly "
        f"(TP 268 FP 15 FN 17 TN 638, AUPRC {baseline.REFERENCE_AF['auprc']:.4f})",
        f"  pooled LVH row reproduces the frozen transcription exactly "
        f"(TP 216 FP 34 FN 39 TN 649, AUPRC {baseline.REFERENCE_LVH['auprc']:.4f})",
    ] + (["  live result.json checked: protocol fp + AF/LVH counts match"]
         if result_json is not None else [])


def build_report(df: pd.DataFrame, rows_by_head: dict, joint: dict,
                 recalls: dict, deltas: dict, tests: dict, verdict: str,
                 checks: list[str]) -> str:
    L = []
    L.append("=" * 100)
    L.append(f"CardioSentry - AF x LVH joint-head analysis  |  "
             f"{baseline.REFERENCE_MODEL}")
    L.append("=" * 100)
    L.append("question      when AF and LVH coexist, does the model correctly")
    L.append("              detect AF but miss LVH?")
    L.append(f"numbers from  the SAME frozen test predictions as "
             f"af_lvh_subgroup_v1 ({len(df)} images)")
    L.append(f"thresholds    AF {C.THRESHOLDS['AF']:.2f} / LVH "
             f"{C.THRESHOLDS['LVH']:.2f} - val-tuned, frozen, never re-tuned")
    L.append(f"ran at        {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    L.append("")
    L.append("CROSS-CHECKS BEFORE ANY NUMBER")
    L.append("-" * 100)
    L += checks
    L.append("")
    L.append("PER-HEAD EVALUATION (test split, frozen thresholds)")
    L.append("-" * 100)
    L += M.format_head_table(rows_by_head["AF"], "AF")
    L.append("")
    L += M.format_head_table(rows_by_head["LVH"], "LVH")
    L.append("")
    L.append("WITHIN-SUBGROUP COUNTS (the 28 AF+LVH cases - a subgroup of")
    L.append("positives only, so FP/TN/specificity do not exist inside it)")
    L.append("-" * 100)
    af = rows_by_head["AF"][0]
    lvh = rows_by_head["LVH"][0]
    L.append(f"  AF  head: TP {af['tp']:>2}   FN {af['fn']:>2}   "
             f"(recall {af['sensitivity']:.4f})")
    L.append(f"  LVH head: TP {lvh['tp']:>2}   FN {lvh['fn']:>2}   "
             f"(recall {lvh['sensitivity']:.4f})")
    L.append("")
    L.append("THE JOINT BREAKDOWN")
    L.append("-" * 100)
    L += M.format_joint(joint)
    L.append("")
    L.append("COMPARISONS - exact tests and bootstrap CIs "
             f"({C.N_RESAMPLES:,} resamples, seed {C.SEED})")
    L.append("-" * 100)
    r = recalls
    L.append(f"  AF recall on AF+LVH   {r['af_af_lvh']['recall']:.4f}  "
             f"95% CI [{r['af_af_lvh']['ci_low']:.4f}, {r['af_af_lvh']['ci_high']:.4f}]"
             f"  (n={r['af_af_lvh']['n']})")
    L.append(f"  AF recall on AF-only  {r['af_af_only']['recall']:.4f}  "
             f"95% CI [{r['af_af_only']['ci_low']:.4f}, {r['af_af_only']['ci_high']:.4f}]"
             f"  (n={r['af_af_only']['n']})")
    L.append(f"  delta (AF+LVH - AF-only) "
             f"{deltas['af']['delta']:+.4f}  "
             f"95% CI [{deltas['af']['ci_low']:+.4f}, {deltas['af']['ci_high']:+.4f}]"
             f"   Fisher one-sided (AF+LVH worse) p = {tests['fisher_af']:.4f}")
    L.append("")
    L.append(f"  LVH recall on AF+LVH  {r['lvh_af_lvh']['recall']:.4f}  "
             f"95% CI [{r['lvh_af_lvh']['ci_low']:.4f}, {r['lvh_af_lvh']['ci_high']:.4f}]"
             f"  (n={r['lvh_af_lvh']['n']})")
    L.append(f"  LVH recall on LVH-only {r['lvh_lvh_only']['recall']:.4f}  "
             f"95% CI [{r['lvh_lvh_only']['ci_low']:.4f}, {r['lvh_lvh_only']['ci_high']:.4f}]"
             f"  (n={r['lvh_lvh_only']['n']})")
    L.append(f"  delta (AF+LVH - LVH-only) "
             f"{deltas['lvh']['delta']:+.4f}  "
             f"95% CI [{deltas['lvh']['ci_low']:+.4f}, {deltas['lvh']['ci_high']:+.4f}]"
             f"   Fisher one-sided (AF+LVH worse) p = {tests['fisher_lvh']:.4f}")
    L.append("")
    L.append(f"  McNemar exact on the 28 cases: discordant pairs "
             f"{tests['b']} 'AF correct + LVH missed' vs {tests['c']} "
             f"'AF missed + LVH correct', two-sided p = {tests['mcnemar']:.4f}")
    L.append("")
    L.append(VERDICT_RULE)
    L.append("")
    L.append("=" * 100)
    L.append(f"VERDICT: {verdict}")
    L.append("=" * 100)
    L.append("  LIMITATION (stated regardless of verdict): the subgroup has 28")
    L.append("  ECGs (the previous round counted 25 distinct patients), so the")
    L.append("  recall CIs are wide and the cross-subgroup Fisher tests are")
    L.append("  low-power. The verdict is graded evidence for the ASSOCIATION")
    L.append("  on this frozen test split - no causal claim is made, and the")
    L.append("  precision/AUPRC columns of the subgroup rows are small-sample")
    L.append("  reads, not operating characteristics of the heads.")
    return "\n".join(L)


def write_outputs(run_dir: Path, rows_by_head: dict, joint: dict,
                  recalls: dict, deltas: dict, tests: dict, report: str,
                  verdict: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for head in ("AF", "LVH"):
        pd.DataFrame(rows_by_head[head]).to_csv(
            run_dir / f"{head.lower()}_head_rows.csv", index=False)
    pd.DataFrame([{**{"category": M.JOINT_NAMES[k]}, **v}
                  for k, v in joint.items()]).to_csv(
        run_dir / "joint_breakdown.csv", index=False)
    pd.DataFrame([{**{"comparison": k}, **v}
                  for k, v in recalls.items()]).to_csv(
        run_dir / "recalls.csv", index=False)
    pd.DataFrame([{**{"comparison": k}, **v}
                  for k, v in deltas.items()]).to_csv(
        run_dir / "recall_deltas.csv", index=False)
    (run_dir / "report.txt").write_text(report)
    (run_dir / "STATUS.txt").write_text("DONE\n")
    summary = {
        "model": baseline.REFERENCE_MODEL,
        "protocol_fingerprint": baseline.REFERENCE_PROTOCOL_FP,
        "thresholds": {k: v for k, v in C.THRESHOLDS.items()},
        "af_lvh_n": baseline.AF_LVH_N,
        "verdict": verdict,
        "tests": {k: (float(v) if isinstance(v, float) else v)
                  for k, v in tests.items()},
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if not args.no_write:
        out.mkdir(parents=True, exist_ok=True)
        (out / "STATUS.txt").write_text("RUNNING\n")

    df = load_predictions(Path(args.predictions))
    result_json = None
    if args.result_json and Path(args.result_json).exists():
        result_json = json.loads(Path(args.result_json).read_text())
    checks = cross_check(df, result_json)

    # per-head rows
    rows_by_head = {
        head: [M.head_row(df, head, key) for key in C.HEAD_ROWS[head]]
        for head in ("AF", "LVH")}

    # joint breakdown + CIs
    codes, table = M.joint_breakdown(df)
    joint = stats.joint_breakdown_ci(codes)
    for k, v in table.items():
        assert joint[k]["count"] == v, "joint table vs CI counts disagree"

    # recalls + deltas + exact tests
    thr = C.THRESHOLDS
    af_af_lvh = (df.loc[M.subgroup_mask(df, "af_lvh"), "prob_AF"] >= thr["AF"])
    af_af_only = (df.loc[M.subgroup_mask(df, "af_only"), "prob_AF"] >= thr["AF"])
    lvh_af_lvh = (df.loc[M.subgroup_mask(df, "af_lvh"), "prob_LVH"] >= thr["LVH"])
    lvh_lvh_only = (df.loc[M.subgroup_mask(df, "lvh_only"), "prob_LVH"] >= thr["LVH"])
    recalls = {
        "af_af_lvh": stats.recall_ci(af_af_lvh.values),
        "af_af_only": stats.recall_ci(af_af_only.values),
        "lvh_af_lvh": stats.recall_ci(lvh_af_lvh.values),
        "lvh_lvh_only": stats.recall_ci(lvh_lvh_only.values),
    }
    deltas = {
        "af": stats.recall_delta_ci(af_af_lvh.values, af_af_only.values),
        "lvh": stats.recall_delta_ci(lvh_af_lvh.values, lvh_lvh_only.values),
    }

    a = int(af_af_lvh.sum()); b0 = int((~af_af_lvh).sum())
    c = int(af_af_only.sum()); d0 = int((~af_af_only).sum())
    fisher_af = stats.fisher_less_p(a, b0, c, d0)
    a2 = int(lvh_af_lvh.sum()); b2 = int((~lvh_af_lvh).sum())
    c2 = int(lvh_lvh_only.sum()); d2 = int((~lvh_lvh_only).sum())
    fisher_lvh = stats.fisher_less_p(a2, b2, c2, d2)

    b_disc = table["af_ok_lvh_miss"]
    c_disc = table["af_miss_lvh_ok"]
    mcnemar = stats.mcnemar_p(b_disc, c_disc)
    tests = {"fisher_af": fisher_af, "fisher_lvh": fisher_lvh,
             "mcnemar": mcnemar, "b": b_disc, "c": c_disc,
             "discordant_total": b_disc + c_disc}

    # the verdict rule, applied as stated
    r1_sig = mcnemar < 0.05 and b_disc > c_disc
    r1_opposite = mcnemar < 0.05 and b_disc < c_disc
    r2_sig = fisher_af < 0.05
    r3_sig = fisher_lvh < 0.05
    if r1_opposite:
        verdict = ("CONTRADICTED - the within-case asymmetry is significant in "
                   "the OPPOSITE direction (LVH detected more often than AF on "
                   "the same ECGs)")
    elif not r1_sig:
        verdict = ("NOT SEPARATED - the within-case asymmetry is not "
                   "significant at p < 0.05 (directional evidence only)")
    elif r1_sig and r2_sig:
        verdict = ("PARTIALLY - AF is detected more often than LVH on the same "
                   "ECGs, but AF recall is ALSO significantly lower on AF+LVH "
                   "than on AF-only")
    elif r1_sig and not r3_sig:
        verdict = ("PARTIALLY - AF is detected more often than LVH on the same "
                   "ECGs, but the LVH recall deficit vs LVH-only is not "
                   "significant")
    else:
        verdict = ("SUPPORTED - the model correctly detects AF and "
                   "disproportionately misses LVH on the AF+LVH cases: "
                   "within-case asymmetry significant (McNemar p < 0.05), no "
                   "significant AF degradation vs AF-only, and LVH recall "
                   "significantly below LVH-only")

    report = build_report(df, rows_by_head, joint, recalls, deltas, tests,
                          verdict, checks)
    if not args.no_write:
        write_outputs(out, rows_by_head, joint, recalls, deltas, tests,
                      report, verdict)
    print(report)
    if not args.no_write:
        print(f"\nwrote to {out}: report.txt, af_head_rows.csv, "
              f"lvh_head_rows.csv, joint_breakdown.csv, recalls.csv, "
              f"recall_deltas.csv, summary.json")


if __name__ == "__main__":
    main()
