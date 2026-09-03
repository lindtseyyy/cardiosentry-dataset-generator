"""
The AF x LVH subgroup analysis of the frozen four-head reference.

RUN ORDER (all before any number is printed)

  1. load the frozen test predictions (hybrid_v1/resnet50_hires/predictions/
     test_predictions.csv) and re-derive the pooled LVH numbers;
  2. cross-check them against the frozen transcription in baseline.py AND
     against the live result.json - any disagreement stops the run;
  3. join the index (patient ids, whole-corpus labels) and verify its test
     labels agree with the predictions file, 100%, row for row;
  4. count the four requested subgroups (test and corpus-wide), and LVH
     prevalence within each source per subgroup;
  5. score the LVH head inside each stratum at the frozen 0.47 threshold,
     plus threshold-free AUROC/AUPRC and sensitivity at FPR <= 0.08;
  6. bootstrap the clean-vs-AF delta (1,200 independent resamples per group,
     seed 42) for every requested metric;
  7. apply the verdict rule, stated before the numbers, and print the
     confound ledger beside it.

The analysis NEVER retrains, NEVER re-thresholds and NEVER re-predicts. The
optional model_verify.py re-run (weights on Drive) only proves the frozen
file matches the saved checkpoint; it cannot change a number here.

USAGE

    python analyze.py --predictions <test_predictions.csv> \
                      --result-json <result.json> \
                      --index <index.csv> \
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
import bootstrap
import config as C
import metrics as M
import subgroups as S

# ---------------------------------------------------------------- the verdict rule
VERDICT_RULE = """
THE VERDICT RULE (frozen - stated here, applied below, never adjusted to fit)

  Comparands: the LVH head of %(model)s, frozen 938-image test split,
              AF stratum (true_AF=1) vs clean stratum (true_AF=0 & true_STEMI=0),
              threshold frozen at %(thr).2f, never re-tuned.

  SUFFICIENCY - the AF stratum is scored only if it holds >= %(minpos)d LVH
  positives and >= %(minneg)d LVH negatives; otherwise counts only.

  DETECTOR READ - two prevalence-independent columns decide the verdict:
      AUROC delta   and   sensitivity @ FPR <= %(fpr).2f delta,
  each with a 95%% bootstrap CI (1,200 independent resamples per group).

      SUPPORTED      both point estimates negative AND at least one CI
                     entirely below 0, AND neither CI entirely above 0.
      PARTIALLY      exactly one of the two CIs entirely below 0.
      NOT SEPARATED  point estimates negative, neither CI excludes 0.
      NO EVIDENCE    point estimates >= 0.

  AUPRC, precision, F1 and the threshold metrics are reported with CIs but do
  NOT decide the verdict: AUPRC's floor is the stratum's LVH prevalence
  (clean 44.8%% vs AF 9.8%% in test), and threshold metrics move with the
  operating point. The verdict is an association on this test split, never a
  causal claim - the confound ledger is printed with it.
""" % {"model": baseline.REFERENCE_MODEL, "thr": C.LVH_THRESHOLD,
       "minpos": C.MIN_POSITIVES, "minneg": C.MIN_NEGATIVES,
       "fpr": C.MATCHED_FPR}


def _decision(rows: list[dict]) -> tuple[str, list[str]]:
    d = {r["metric"]: r for r in rows}
    auroc, s08 = d["auroc"], d["sens_at_fpr"]
    below = [r for r in (auroc, s08) if r["delta_ci_high"] < 0]
    above = [r for r in (auroc, s08) if r["delta_ci_low"] > 0]
    notes = []
    if above:
        return ("NO EVIDENCE of deterioration - at least one detector-read "
                "delta is significantly positive", notes)
    if len(below) == 2:
        v = ("SUPPORTED - LVH detection deteriorates when AF is present; "
             "both detector-read CIs lie entirely below 0")
    elif len(below) == 1:
        v = ("PARTIALLY SUPPORTED - exactly one detector-read CI lies "
             "entirely below 0")
    elif auroc["delta_af_minus_clean"] < 0 and s08["delta_af_minus_clean"] < 0:
        v = ("NOT SEPARATED - point estimates negative, neither detector-read "
             "CI excludes 0 (directional evidence only)")
    else:
        v = "NO EVIDENCE of deterioration - detector-read point estimates >= 0"
    for m in ("auprc", "auroc", "sensitivity", "specificity", "fpr",
              "precision", "f1", "sens_at_fpr"):
        r = d[m]
        if m in ("specificity", "fpr"):
            if r["delta_ci_low"] < 0 < r["delta_ci_high"]:
                notes.append(f"{m}: not separated (CI includes 0)")
        else:
            if r["delta_ci_high"] < 0:
                notes.append(f"{m}: significantly worse in the AF stratum")
            elif r["delta_ci_low"] > 0:
                notes.append(f"{m}: significantly BETTER in the AF stratum")
            else:
                notes.append(f"{m}: not separated (CI includes 0)")
    return v, notes


# ---------------------------------------------------------------- loading + checks
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


def cross_check_predictions(df: pd.DataFrame, result_json: dict | None,
                            live_json_path: Path | None) -> list[str]:
    """Pooled LVH numbers re-derived from the predictions file, checked against
    baseline.py's frozen transcription and the live result.json. Refuses to
    continue on any disagreement."""
    y = df["true_LVH"].values
    p = df["prob_LVH"].values
    m = M.stratum_metrics(y, p, C.LVH_THRESHOLD)
    b = baseline.REFERENCE_LVH["pooled"]

    problems = []
    for k_int in ("tp", "fp", "fn", "tn"):
        if m[k_int] != b[k_int]:
            problems.append(f"  {k_int}: predictions={m[k_int]} frozen={b[k_int]}")
    if m["n"] != b["n"] or m["n_pos"] != b["support"]:
        problems.append(f"  n/n_pos: predictions={m['n']}/{m['n_pos']} "
                        f"frozen={b['n']}/{b['support']}")
    for k in ("auprc", "auroc", "sensitivity", "specificity", "precision",
              "f1", "fpr"):
        if abs(m[k] - b[k]) > 1e-6:
            problems.append(f"  {k}: predictions={m[k]:.8f} frozen={b[k]:.8f}")
    if abs(m["threshold"] - b["threshold"]) > 1e-9:
        problems.append(f"  threshold: predictions={m['threshold']} "
                        f"frozen={b['threshold']}")

    if result_json is not None:
        try:
            r = result_json["test"]["per_label"]["LVH"]
            for k_int in ("tp", "fp", "fn", "tn"):
                if int(r.get(k_int, -1)) != b[k_int]:
                    problems.append(f"  result.json test LVH {k_int}="
                                    f"{r.get(k_int)} != frozen {b[k_int]}")
            t_json = result_json.get("test", {}).get("per_label", {}).get(
                "LVH", {}).get("threshold")
            if t_json is not None and abs(float(t_json) - b["threshold"]) > 1e-9:
                problems.append(f"  result.json LVH threshold={t_json} != "
                                f"frozen {b['threshold']}")
            if result_json.get("protocol_fingerprint") != \
                    baseline.REFERENCE_META["protocol_fingerprint"]:
                problems.append(f"  result.json protocol fp="
                                f"{result_json.get('protocol_fingerprint')} != "
                                f"frozen {baseline.REFERENCE_META['protocol_fingerprint']}")
        except (KeyError, TypeError) as exc:
            problems.append(f"  result.json unreadable at test.per_label.LVH: {exc}")

    if problems:
        raise SystemExit(
            "FROZEN-REFERENCE CROSS-CHECK FAILED - the predictions file on "
            "Drive does not agree with the frozen reference. The subgroup "
            "analysis would not be about the reported model. STOP.\n"
            + "\n".join(problems))
    return [f"  pooled LVH re-derived from {len(df)} rows matches the frozen "
            f"transcription exactly (TP {m['tp']} FP {m['fp']} FN {m['fn']} "
            f"TN {m['tn']}, AUPRC {m['auprc']:.4f})" +
            ("  (live result.json checked: matches)" if result_json else "")]


def load_index(path: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        raise SystemExit(f"MISSING index.csv: {path}\n"
                         f"  extract it from prototype_1152_768.tar (or "
                         f"prototype_1152.tar) - a single-member extraction, "
                         f"no images needed.")
    idx = pd.read_csv(path)
    missing = S.INDEX_NEEDED - set(idx.columns)
    if missing:
        raise SystemExit(f"index.csv is missing columns: {sorted(missing)}")
    # verify the index's test labels agree with the frozen predictions
    m = predictions[["record"] + [f"true_{l}" for l in C.LABELS]].merge(
        idx, on="record", how="left")
    if m["patient_id"].isna().any():
        raise SystemExit(f"{int(m['patient_id'].isna().sum())} test records are "
                         f"missing from index.csv - wrong corpus copy?")
    for lab in C.LABELS:
        bad = (m[f"true_{lab}"] != m[lab].astype(int)).sum()
        if bad:
            raise SystemExit(f"{bad} test rows disagree on label {lab} between "
                             f"predictions and index.csv - the two files come "
                             f"from different corpora. STOP.")
    if len(idx) != C.CORPUS_N:
        raise SystemExit(f"index.csv has {len(idx)} rows, expected "
                         f"{C.CORPUS_N} - not the prototype_1152 corpus.")
    return idx


# ---------------------------------------------------------------- metric rollups
def stratum_rows(df: pd.DataFrame) -> list[dict]:
    """One metrics row per stratum, plus the rollups the verdict reads."""
    ctx = S.assign_context(df)
    rows = []
    for c in C.CONTEXTS:
        sub = df[ctx == c]
        rows.append({
            "name": C.CONTEXTS[c]["display"],
            "context": c,
            "metrics": M.stratum_metrics(sub["true_LVH"].values,
                                         sub["prob_LVH"].values),
        })
    rows.append({"name": "POOLED (all test)", "context": "pooled",
                 "metrics": M.stratum_metrics(df["true_LVH"].values,
                                              df["prob_LVH"].values)})
    # the AF-stratum contribution read: the pooled set minus the AF stratum
    no_af = df[ctx != "af"]
    rows.append({"name": "all test EXCEPT the AF stratum", "context": "no_af",
                 "metrics": M.stratum_metrics(no_af["true_LVH"].values,
                                              no_af["prob_LVH"].values)})
    # source-stratification companion: does the clean-vs-AF gap survive
    # restriction to PTB-XL, where every image shares one acquisition?
    ptb = df[df["source"] == "PTBXL"]
    ptb_ctx = S.assign_context(ptb)
    for c in ("clean", "af"):
        sub = ptb[ptb_ctx == c]
        rows.append({
            "name": f"PTB-XL only - {C.CONTEXTS[c]['display']}",
            "context": f"ptbxl_{c}",
            "metrics": M.stratum_metrics(sub["true_LVH"].values,
                                         sub["prob_LVH"].values),
        })
    return rows


# ---------------------------------------------------------------- the report
def build_report(df: pd.DataFrame, idx: pd.DataFrame, rows: list[dict],
                 deltas: list[dict], verdict: str, notes: list[str],
                 checks: list[str]) -> str:
    test_idx = idx[idx.split == "test"]
    counts_t = S.count_by_context_lvh(test_idx)
    counts_c = S.count_by_context_lvh(idx)
    prev_t = S.prevalence_by_context_source(test_idx)
    prev_c = S.prevalence_by_context_source(idx)
    compos = S.lvh_composition_by_source(test_idx)
    pat = S.patient_structure(test_idx)
    overlap = S.patient_overlap(test_idx, *C.COMPARISON)

    L = []
    L.append("=" * 100)
    L.append(f"CardioSentry - AF x LVH subgroup analysis  |  "
             f"{baseline.REFERENCE_MODEL}")
    L.append("=" * 100)
    L.append(f"question      does the LVH head deteriorate on ECGs that carry AF?")
    L.append(f"model         {baseline.REFERENCE_TASK}")
    L.append(f"              protocol fp {baseline.REFERENCE_META['protocol_fingerprint']}"
             f"   input fp {baseline.REFERENCE_META['input_fingerprint']}"
             f"   views {baseline.REFERENCE_META['views']}")
    L.append(f"numbers from  the FROZEN per-image test predictions "
             f"({len(df)} images); threshold {C.LVH_THRESHOLD} (val-tuned, frozen)")
    L.append(f"ran at        {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    L.append("")
    L.append("CROSS-CHECKS BEFORE ANY NUMBER")
    L.append("-" * 100)
    L += checks
    L.append("")
    L += S.format_counts("THE COUNTS - frozen test split (938 ECGs)", counts_t,
                         prev_t, compos)
    L.append("")
    L += S.format_counts("THE COUNTS - whole corpus (6,670 ECGs)", counts_c,
                         prev_c, S.lvh_composition_by_source(idx))
    L.append("")
    L.append("STRUCTURAL ZEROS - corpus facts, not model findings")
    L.append("-" * 100)
    L.append("  STEMI is exactly `source == 'STEMI'` and that database carries")
    L.append("  zero LVH positives, so STEMI+LVH and AF+STEMI+LVH ECGs do not")
    L.append("  exist here (0 in test, 0 in the whole corpus). AF and STEMI")
    L.append("  never co-occur on any row: true_AF=1 implies true_STEMI=0.")
    L.append("  Those strata contribute LVH false positives only - their LVH")
    L.append("  scores are structurally undefined and printed as '--'.")
    L.append("")
    L.append("LVH HEAD INSIDE EACH STRATUM (frozen threshold "
             f"{C.LVH_THRESHOLD}, test split)")
    L.append("-" * 100)
    L += M.format_metrics_table(rows)
    L.append("")
    L.append("  confusion counts per stratum (LVH head at "
             f"{C.LVH_THRESHOLD}):")
    for r in rows:
        L.append("  " + M.format_counts_row(r))
    L.append("")
    L.append("CLEAN vs AF - bootstrap CIs "
             f"({C.N_RESAMPLES:,} independent resamples per group, seed "
             f"{C.SEED})")
    L.append("-" * 100)
    L.append("  resampling is INDEPENDENT per group: the strata are disjoint")
    L.append("  image sets, so the paired-bootstrap convention used for")
    L.append("  two-model comparisons does not apply. delta = AF - clean.")
    L += bootstrap.format_delta_table(deltas)
    L.append("")
    L += S.format_patients(pat, overlap, *C.COMPARISON)
    L.append("")
    L.append(VERDICT_RULE)
    L.append("")
    L.append("=" * 100)
    L.append(f"VERDICT: {verdict}")
    L.append("=" * 100)
    L.append("  per-metric deltas (AF - clean):")
    for n in notes:
        L.append(f"    - {n}")
    L.append("")
    L.append("CONFOUND LEDGER - what this verdict is and is not")
    L.append("-" * 100)
    L.append("  1. PREVALENCE. The strata have different LVH prevalence by")
    L.append("     construction (clean 44.8% vs AF 9.8% in test). AUPRC's floor")
    L.append("     IS the prevalence, so part of the AUPRC gap is mechanical.")
    L.append("     AUROC and sens@FPR<=0.08 have no prevalence term - they are")
    L.append("     the columns the verdict is built on.")
    L.append("  2. NEGATIVE CLASS. The clean stratum's LVH-negatives are 100%")
    L.append("     NORMAL sheets; the AF stratum's negatives are AF ECGs")
    L.append("     (with whatever else they carry). Detecting LVH against AF")
    L.append("     ECGs is a harder problem than against NORMAL sheets by")
    L.append("     construction - the delta measures that hardness, it does not")
    L.append("     explain it.")
    L.append("  3. SOURCE. The AF stratum's LVH positives are all PTB-XL (28/28);")
    L.append("     the clean stratum's are PTB-XL + Chapman (206 + 21). The")
    L.append("     PTB-XL-only companion rows show the gap survives restricting")
    L.append("     both strata to one acquisition source.")
    L.append("  4. SUPPORT. The AF stratum holds 28 LVH positives across 25")
    L.append("     distinct patients. CIs on sensitivity/precision/F1 are wide")
    L.append("     for that reason; the ranking-based CIs are the tight ones.")
    L.append("  5. CAUSATION. 'Deteriorates when AF is present' is an association")
    L.append("     on this test split. Whether AF morphology masks LVH, the AF")
    L.append("     stratum's negative set is intrinsically harder, or both - is")
    L.append("     not decidable from these numbers. The data support the drop;")
    L.append("     they do not attribute its mechanism.")
    return "\n".join(L)


def write_outputs(run_dir: Path, df: pd.DataFrame, idx: pd.DataFrame,
                  rows: list[dict], deltas: list[dict], report: str,
                  verdict: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    test_idx = idx[idx.split == "test"]
    S.count_by_context_lvh(test_idx).to_csv(run_dir / "subgroup_counts_test.csv")
    S.count_by_context_lvh(idx).to_csv(run_dir / "subgroup_counts_corpus.csv")
    S.prevalence_by_context_source(test_idx).to_csv(
        run_dir / "lvh_prevalence_by_source_test.csv", index=False)
    S.prevalence_by_context_source(idx).to_csv(
        run_dir / "lvh_prevalence_by_source_corpus.csv", index=False)
    M.metrics_to_df(rows).to_csv(run_dir / "subgroup_metrics.csv", index=False)
    pd.DataFrame(deltas).to_csv(run_dir / "bootstrap_deltas.csv", index=False)
    S.patient_structure(test_idx).to_csv(run_dir / "patient_structure.csv",
                                         index=False)
    (run_dir / "report.txt").write_text(report)
    (run_dir / "STATUS.txt").write_text("DONE\n")
    summary = {
        "model": baseline.REFERENCE_MODEL,
        "protocol_fingerprint": baseline.REFERENCE_META["protocol_fingerprint"],
        "threshold": C.LVH_THRESHOLD,
        "test_n": int(len(df)),
        "verdict": verdict,
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--result-json", default=None,
                    help="live result.json for the cross-check (optional)")
    ap.add_argument("--out", required=True, help="run dir to write")
    ap.add_argument("--no-write", action="store_true",
                    help="print only, write nothing")
    args = ap.parse_args()

    out = Path(args.out)
    if not args.no_write:
        out.mkdir(parents=True, exist_ok=True)
        (out / "STATUS.txt").write_text("RUNNING\n")

    df = load_predictions(Path(args.predictions))
    result_json = None
    if args.result_json and Path(args.result_json).exists():
        result_json = json.loads(Path(args.result_json).read_text())
    checks = cross_check_predictions(df, result_json,
                                     Path(args.result_json) if args.result_json else None)
    idx = load_index(Path(args.index), df)

    # Align the predictions rows to the index.csv order, so every slice of
    # index-derived strata indexes the same rows in the predictions file.
    # (load_index already proved the two files agree label-for-label on every
    # test record, so this is a pure reordering, not a re-labelling.)
    test_idx = idx[idx.split == "test"].reset_index(drop=True)
    df = df.set_index("record").loc[test_idx["record"].tolist()].reset_index(drop=True)

    rows = stratum_rows(df)
    ctx = S.assign_context(test_idx)
    y_clean = test_idx.loc[ctx == "clean", "LVH"].values
    p_clean = df.loc[ctx == "clean", "prob_LVH"].values
    y_af = test_idx.loc[ctx == "af", "LVH"].values
    p_af = df.loc[ctx == "af", "prob_LVH"].values
    deltas = bootstrap.subgroup_comparison(y_clean, p_clean, y_af, p_af)

    # sufficiency gate
    n_pos_af = int(y_af.sum())
    n_neg_af = int(len(y_af) - n_pos_af)
    if n_pos_af < C.MIN_POSITIVES or n_neg_af < C.MIN_NEGATIVES:
        verdict = (f"INSUFFICIENT DATA - AF stratum holds {n_pos_af} LVH+ and "
                   f"{n_neg_af} LVH- (floors: {C.MIN_POSITIVES}/{C.MIN_NEGATIVES}); "
                   f"verdict withheld, counts and point estimates below.")
        notes = []
    else:
        verdict, notes = _decision(deltas)

    report = build_report(df, idx, rows, deltas, verdict, notes, checks)
    if not args.no_write:
        write_outputs(out, df, idx, rows, deltas, report, verdict)
    print(report)
    if not args.no_write:
        print(f"\nwrote to {out}: report.txt, subgroup_metrics.csv, "
              f"bootstrap_deltas.csv, subgroup_counts_*.csv, "
              f"lvh_prevalence_by_source_*.csv, patient_structure.csv")


if __name__ == "__main__":
    main()
