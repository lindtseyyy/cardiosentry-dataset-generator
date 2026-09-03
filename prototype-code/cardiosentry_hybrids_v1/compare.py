#!/usr/bin/env python3
"""
Tabulate finished runs - and first PROVE they share the training protocol.

    python compare.py --runs runs/ --csv runs/comparison.csv

TWO-TIER COMPARABILITY, WHICH IS THE WHOLE DIFFERENCE FROM ROUND 1

Round 1 hashed input resolution into the one fingerprint and refused to compare
runs that disagreed. This round has resolution AS the experimental variable, so:

  PROTOCOL must match      - seed, optimiser, schedule, epochs, augmentation,
                             monitor, threshold rule. Mismatch = refuse to rank.
  INPUT is expected to vary - printed as a column, never enforced.

Everything else the round-1 script refused on still applies: different datasets,
different row counts and smoke runs all block the table.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import config as C
import subgroup
import af_lvh

FOC = C.FOCUS_LABEL

import baseline

ROUND1 = baseline.ROUND1
R1_REF = baseline.REFERENCE
P_SRC = subgroup.PRIMARY_SOURCE


def _primary_focus_auprc(run: dict) -> float:
    """Focus-label AUPRC inside the primary source, straight from result.json.

    train.py already computes and stores per-source per-label metrics, so the
    ladder does not need the prediction CSVs. NaN when a run predates that.
    """
    try:
        return run["test_per_source"][P_SRC]["per_label"][FOC]["auprc"]
    except (KeyError, TypeError):
        return float("nan")


def focus_by_source_section(runs: list[dict], views_of) -> list[str]:
    """The required read for this round - see subgroup.py for why.

    Pooled focus-label metrics on this corpus are inflated by database
    recognition: the STEMI database contributes 146 images, every one of them
    focus-label-NEGATIVE, so a model that learns to spot that database cleans up
    its pooled false positives without reading a single QRS complex. PTB-XL
    carries 234 of the 255 positives and is internally single-source, so it is
    the subgroup the shortcut cannot reach.
    """
    P = subgroup.PRIMARY_SOURCE
    L = ["", "=" * 108,
         f"{FOC} BY SOURCE - is the gain ECG information, or dataset recognition?",
         "=" * 108]

    rows, missing = [], []
    for r in runs:
        df = subgroup.load_test_predictions(r["_dir"])
        if df is None:
            missing.append(r["model"])
            continue
        by = subgroup.focus_by_source(df, FOC)
        if P not in by:
            missing.append(r["model"])
            continue
        rows.append({"model": r["model"], "views": views_of(r),
                     "pooled": by["POOLED"], "primary": by[P],
                     "stemi_db": by.get("STEMI"), "by": by,
                     "coupled": subgroup.label_source_coupling(df)})

    if not rows:
        L += ["  NO PER-IMAGE PREDICTIONS FOUND under */predictions/.",
              "  This section is REQUIRED for every model in this round. Without it",
              f"  a pooled {FOC} number cannot be told apart from a database",
              "  classifier. Re-run without --smoke (smoke runs skip predictions).",
              ""]
        return L

    n_pos = rows[0]["primary"]["n_pos"]
    n_tot = rows[0]["pooled"]["n_pos"]
    L += [f"  {FOC} positives in the test split: {n_pos} of {n_tot} are {P}.",
          "  The STEMI database contributes 146 images and ZERO positives, so every",
          f"  {FOC} call it receives is a false positive that pooled AUPRC punishes",
          "  and that better database recognition - not better ECG reading - fixes.",
          ""]

    coupled = rows[0]["coupled"]
    if coupled:
        for lab, src in coupled.items():
            L += [f"  WARNING: the {lab} label is EXACTLY 'source == {src}' on this",
                  f"  split. Any {lab} metric here is indistinguishable from database",
                  "  identification and must not be reported as clinical performance.",
                  ""]

    # Control preference: the 384 single-stream run if it is in the folder, else
    # the lowest-resolution run present. Deltas need a fixed anchor or the
    # "shortcut share" column silently changes meaning between invocations.
    ctrl = next((x for x in rows if x["model"] == "resnet50_base"), None)
    if ctrl is None:
        ctrl = min(rows, key=lambda x: x["primary"]["auprc"])
    for x in rows:
        x["share"] = subgroup.shortcut_share(
            x["pooled"]["auprc"] - ctrl["pooled"]["auprc"],
            x["primary"]["auprc"] - ctrl["primary"]["auprc"])

    rows.sort(key=lambda x: -x["primary"]["auprc"])
    L += subgroup.format_table(rows, ctrl["model"])
    L += ["", subgroup.MATCHED_FPR_NOTE, ""]
    L += subgroup.format_deltas(rows, ctrl)

    # The bar, stated as a number rather than left to the reader to find in the
    # table. Frozen in baseline.py so a re-run of notebook A cannot move it.
    bar = baseline.ROUNDA[baseline.ROUNDA_BAR]
    live = next((x for x in rows if x["model"] == baseline.ROUNDA_BAR), None)
    L += ["", f"  THE BAR: notebook A's {baseline.ROUNDA_BAR} scored "
              f"{bar['ptbxl_auprc']:.4f} {P} {FOC} AUPRC",
          f"  ({bar['pooled_auprc']:.4f} pooled) at {bar['params_m']:.1f} M params and "
          f"{bar['train_min']:.0f} min of training,"]
    if live is None:
        L += ["  from a single 768px stream. It is not in this folder - the numbers",
              "  above are frozen from prototype-results/hybrid_v1/."]
    else:
        L += ["  from a single 768px stream, with no second backbone."]
    L += ["  A dual-resolution model has to beat THAT number, not round 1's 0.8805.",
          "",
          f"  A hybrid that raises pooled {FOC} AUPRC while its {P} column stands",
          f"  still has not learned to read {FOC}. It has learned to read the corpus.",
          f"  {P} AUPRC is the number that goes in the paper's headline claim;",
          "  the pooled number goes next to it, with this table underneath."]
    if missing:
        L += ["", f"  MISSING per-image predictions (excluded above): "
                  f"{', '.join(missing)}"]
    return L


def load_runs(root: Path) -> list[dict]:
    out = []
    for f in sorted(root.glob("*/result.json")):
        try:
            r = json.loads(f.read_text())
            # The run directory, so the source-stratified section can reopen the
            # per-image predictions. Underscored: it is not part of the result
            # schema and must never be written back into a result.json.
            r["_dir"] = f.parent
            out.append(r)
        except Exception as exc:
            print(f"  skipping unreadable {f}: {exc}")
    return out


def check_comparable(runs: list[dict]) -> tuple[bool, list[str]]:
    notes, ok = [], True

    fps = {r["model"]: r.get("protocol_fingerprint") for r in runs}
    if len(set(fps.values())) > 1:
        ok = False
        notes.append("PROTOCOL FINGERPRINTS DIFFER - these runs are NOT comparable:")
        for m, fp in fps.items():
            notes.append(f"    {m:22} {fp}")
        notes.append("  config.py changed between runs. Re-run the stale ones.")

    roots = {r["model"]: r["data"]["root"] for r in runs}
    if len(set(roots.values())) > 1:
        ok = False
        notes.append("DIFFERENT DATASETS used:")
        for m, d in roots.items():
            notes.append(f"    {m:22} {d}")

    rows = {r["model"]: r["data"].get("rows") for r in runs}
    if len(set(rows.values())) > 1:
        ok = False
        notes.append(f"DIFFERENT ROW COUNTS: {rows}")

    smoke = [r["model"] for r in runs if r.get("smoke")]
    if smoke:
        ok = False
        notes.append(f"SMOKE RUNS PRESENT (metrics meaningless): {smoke}")

    lrs = {r["model"]: r.get("lr_used") for r in runs}
    if len(set(lrs.values())) > 1 and not all(r.get("sweep") for r in runs):
        notes.append(f"  NOTE: learning rates differ {lrs} and not every run used "
                     f"--sweep. Either use one LR everywhere, or --sweep everywhere.")

    # BatchNorm normalises over the micro-batch, and gradient accumulation does
    # not touch that. Every backbone here contains BN, so runs that disagree on
    # micro_batch differ in BN statistics as well as in whatever they were meant
    # to differ in. This is a WARNING rather than a refusal because on a 16 GB
    # card a uniform value is not always reachable - but it belongs in the
    # write-up whenever it fires.
    mbs = {r["model"]: r.get("micro_batch") for r in runs}
    if len([v for v in mbs.values() if v is not None]) and len(set(mbs.values())) > 1:
        notes.append("  MICRO-BATCH DIFFERS ACROSS RUNS - a real confound, not a "
                     "VRAM detail:")
        for m, v in mbs.items():
            notes.append(f"    {m:22} {v}")
        notes.append("    BatchNorm normalises over the micro-batch, so these runs")
        notes.append("    differ in BN statistics too. Gradient accumulation keeps")
        notes.append("    the effective batch at 32 but does NOT fix this.")
        if {"resnet50_base", "resnet50_hires"} <= set(mbs):
            if mbs["resnet50_base"] != mbs["resnet50_hires"]:
                notes.append("    ** the control pair resnet50_base vs resnet50_hires")
                notes.append("       is affected: 'only the resolution differs' is")
                notes.append("       NOT true of these two runs as they stand.")
        notes.append("    Re-run with a uniform value on a 24 GB card:")
        notes.append("        MICRO_BATCH=8 ./run_hybrids.sh <data> <out>")

    # Expected to differ. Say so out loud so nobody reads it as a fault.
    ifs = {r["model"]: "+".join(map(str, r["model_info"].get("views", [])))
           for r in runs}
    if len(set(ifs.values())) > 1:
        notes.append("  input views differ across runs - THIS IS THE EXPERIMENT, "
                     "not a problem:")
        for m, v in ifs.items():
            notes.append(f"    {m:22} {v}")

    pre = {r["model"]: r["model_info"].get("pretrain") for r in runs}
    if len(set(pre.values())) > 1:
        notes.append("  NOTE: pretraining corpora differ - a real asymmetry, not a bug:")
        for m, p in pre.items():
            notes.append(f"    {m:22} {p}")
    return ok, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", type=Path)
    ap.add_argument("--sort", default=f"{FOC.lower()}_auprc",
                    choices=["macro_auprc", "macro_auroc", "macro_f1",
                             "macro_sensitivity", f"{FOC.lower()}_auprc",
                             f"{FOC.lower()}_f1", "stemi_auprc"],
                    help=f"default {FOC.lower()}_auprc - this round is about {FOC}")
    ap.add_argument("--csv", type=Path, help="also write the table as CSV")
    ap.add_argument("--force", action="store_true",
                    help="print the table even if the runs are not comparable")
    a = ap.parse_args()

    runs = load_runs(a.runs)
    if not runs:
        print(f"no result.json under {a.runs}/ - train something first")
        return 1

    print("=" * 108)
    print(f"CardioSentry hybrids - comparison   ({len(runs)} runs)")
    print("=" * 108)

    ok, notes = check_comparable(runs)
    for n in notes:
        print(n)
    if not ok and not a.force:
        print("\nREFUSING to rank runs that did not share the protocol. "
              "Fix the above, or pass --force.")
        return 2
    if ok:
        print(f"comparability: OK - all runs share protocol fingerprint "
              f"{runs[0]['protocol_fingerprint']}")
        d = runs[0]["data"]
        print(f"dataset: {d['rows']:,} images, {d['records']:,} records, "
              f"{d['patients']:,} patients")

    key = {"macro_auprc": lambda r: r["test"]["macro_auprc"],
           "macro_auroc": lambda r: r["test"]["macro_auroc"],
           "macro_f1": lambda r: r["test"]["macro_f1"],
           "macro_sensitivity": lambda r: r["test"]["macro_sensitivity"],
           f"{FOC.lower()}_auprc": lambda r: r["test"]["per_label"][FOC]["auprc"],
           f"{FOC.lower()}_f1": lambda r: r["test"]["per_label"][FOC]["f1"],
           "stemi_auprc": lambda r: r["test"]["per_label"]["STEMI"]["auprc"],
           }[a.sort]
    runs.sort(key=key, reverse=True)

    out_lines = []
    def emit(line=""):
        print(line)
        out_lines.append(line)

    def views_of(r):
        return "+".join(map(str, r["model_info"].get("views", []))) or "?"

    # ---- the question this round asks -------------------------------------
    R = lambda met: baseline.reference(met, FOC)
    bm, bv = baseline.best("auprc", FOC)
    emit()
    emit("=" * 108)
    emit(f"{FOC} - THE HEAD THIS ROUND EXISTS TO IMPROVE")
    emit("=" * 108)
    emit(f"  deltas are against round 1's {R1_REF}: {FOC} AUPRC "
         f"{R('auprc'):.4f}, sens {R('sensitivity'):.4f}, FPR {R('fpr'):.4f}")
    emit(f"  best {FOC} AUPRC anywhere in round 1 was {bv:.4f} ({bm}).")
    emit()
    hdr = (f"{'model':<22}{'views':>10}{'AUPRC':>9}{'d':>8}{'AUROC':>9}"
           f"{'Sens':>8}{'d':>8}{'FPR':>8}{'d':>8}{'F1':>8}{'TP':>6}{'FP':>6}{'FN':>6}")
    emit(hdr); emit("-" * len(hdr))
    for r in runs:
        f = r["test"]["per_label"][FOC]
        emit(f"{r['model']:<22}{views_of(r):>10}{f['auprc']:9.4f}"
             f"{f['auprc']-R('auprc'):+8.4f}{f['auroc']:9.4f}"
             f"{f['sensitivity']:8.4f}{f['sensitivity']-R('sensitivity'):+8.4f}"
             f"{f['fpr']:8.4f}{f['fpr']-R('fpr'):+8.4f}{f['f1']:8.4f}"
             f"{f['tp']:6d}{f['fp']:6d}{f['fn']:6d}")
    emit()
    emit(f"  Round 1's {FOC} failure was FALSE-POSITIVE dominated: 333 FP against")
    emit("  229 FN summed over five backbones, 2.20 errors per positive versus 1.11")
    emit("  for AF. Read the FPR delta before the AUPRC delta. A model that gains")
    emit("  sensitivity and gains FPR has moved its operating point, not improved.")
    emit()
    emit(f"  ^ THE TABLE ABOVE IS POOLED ACROSS DATABASES AND IS NOT THE PRIMARY")
    emit(f"    READ. Notebook A showed four fifths of the 384->768 pooled {FOC} gain")
    emit(f"    did not reproduce inside PTB-XL. Read the next section first.")

    # ---- source-stratified focus label ------------------------------------
    for line in focus_by_source_section(runs, views_of):
        emit(line)

    # ---- per-condition regression check -----------------------------------
    #
    # THE REASON THIS SECTION EXISTS.
    #
    # Macro averages dilute. A model that gains 0.04 on LVH and loses 0.04 on
    # STEMI has an unchanged macro AUPRC and looks like a clean win in every
    # headline table. It is not one: an over-called LVH earns a follow-up echo,
    # a missed STEMI does not. The asymmetry is clinical, so the check has to be
    # per condition and it has to be impossible to skip.
    emit()
    emit("=" * 108)
    emit("PER-CONDITION REGRESSION CHECK - did improving LVH cost anything else?")
    emit("=" * 108)
    emit(f"  deltas vs round 1's {R1_REF} on the same frozen test split.")
    emit("  macro dilutes a single-head regression by 4x, so read this before the")
    emit("  macro table, not after it.")

    for metric, nice, lower_better in (("auprc", "AUPRC", False),
                                       ("sensitivity", "SENSITIVITY", False),
                                       ("fpr", "FALSE POSITIVE RATE", True)):
        emit()
        emit(f"  delta {nice}   ({'lower' if lower_better else 'higher'} is better)")
        hdr = (f"  {'model':<22}" + "".join(f"{l:>10}" for l in C.LABELS)
               + f"{'macro':>10}   regressions")
        emit(hdr); emit("  " + "-" * (len(hdr) - 2))
        for r in runs:
            t = r["test"]
            cells, worse = [], []
            for l in C.LABELS:
                d = t["per_label"][l][metric] - baseline.reference(metric, l)
                if lower_better:
                    d = -d          # show improvement as positive everywhere
                cells.append(f"{d:+10.4f}")
                if d < -0.005:      # below run-to-run noise on 938 samples
                    worse.append(f"{l}{d:+.3f}")
            dm = t[f"macro_{metric}"] - baseline.reference(metric, "macro")
            if lower_better:
                dm = -dm
            emit(f"  {r['model']:<22}" + "".join(cells) + f"{dm:+10.4f}   "
                 + (", ".join(worse) if worse else "none"))
        if lower_better:
            emit("  (FPR deltas are sign-flipped so that positive = fewer false "
                 "positives,")
            emit("   consistent with the two tables above.)")

    # ---- the flags that should stop you shipping ---------------------------
    #
    # Thresholds below are judgement calls, stated explicitly so they can be
    # argued with rather than buried. 0.005 is roughly the run-to-run noise
    # floor on a 938-image test split; the STEMI sensitivity flag is tighter
    # because that is the failure that harms a patient.
    emit()
    emit("  " + "-" * 104)
    emit("  FLAGS")
    emit("  " + "-" * 104)
    any_flag = False
    for r in runs:
        t = r["test"]
        msgs = []
        ss = t["per_label"]["STEMI"]["sensitivity"]
        d_ss = ss - baseline.reference("sensitivity", "STEMI")
        if d_ss < -0.01:
            n = t["per_label"]["STEMI"]["fn"]
            msgs.append(f"STEMI sensitivity {ss:.4f} is {abs(d_ss):.4f} BELOW "
                        f"round 1 - {n} missed infarct{'' if n == 1 else 's'} "
                        f"against round 1's {baseline.COUNTS['STEMI']['fn']}")
        for l in C.LABELS:
            if l == C.FOCUS_LABEL:
                continue
            d = t["per_label"][l]["auprc"] - baseline.reference("auprc", l)
            if d < -0.02:
                msgs.append(f"{l} AUPRC down {abs(d):.4f}")
        dfoc = t["per_label"][C.FOCUS_LABEL]["auprc"] - baseline.reference(
            "auprc", C.FOCUS_LABEL)
        if dfoc < 0.005:
            msgs.append(f"{C.FOCUS_LABEL} AUPRC not improved ({dfoc:+.4f}) - this "
                        f"model did not do what the round asked")
        if msgs:
            any_flag = True
            emit(f"  {r['model']}")
            for m in msgs:
                emit(f"      - {m}")
    if not any_flag:
        emit("  none - no run lost more than 0.02 AUPRC on any head, and none lost")
        emit("  more than 0.01 STEMI sensitivity.")
    emit()
    emit("  A model that improves LVH and trips a STEMI flag is not a better model.")
    emit(f"  Round 1's {R1_REF} missed {baseline.COUNTS['STEMI']['fn']} of "
         f"{baseline.TEST_POSITIVES['STEMI']} STEMIs, and over-called LVH "
         f"{baseline.COUNTS['LVH']['fp']} times.")
    emit("  Keep both numbers in view: they are what 'better' has to beat.")

    # ---- AF x LVH subgroup -------------------------------------------------
    #
    # The question notebook C exists to answer. The frozen joint-head analysis
    # scored resnet50_hires (768) and found it detects AF but misses LVH when
    # the two coexist (recall 0.6429 on AF+LVH vs 0.8722 on LVH-only, Fisher
    # one-sided p = 0.0038). Every run in this folder is scored the same way,
    # from its own frozen decisions, so the table shows whether any
    # representation preserves LVH information under AF. See af_lvh.py.
    af_rows, af_missing = [], []
    for r in runs:
        row = af_lvh.score_run(r["_dir"], r["model"])
        if row is None:
            af_missing.append(r["model"])
            continue
        row["views"] = views_of(r)
        af_rows.append(row)
    for line in af_lvh.format_section(af_rows, af_missing):
        emit(line)

    # ---- resolution, isolated ---------------------------------------------
    by_name = {r["model"]: r for r in runs}
    if "resnet50_base" in by_name and "resnet50_hires" in by_name:
        b, h = by_name["resnet50_base"], by_name["resnet50_hires"]
        emit()
        emit("=" * 108)
        emit("THE CONTROL - does resolution alone explain it?")
        emit("=" * 108)
        emit("  resnet50_base and resnet50_hires are the SAME model, the same head,")
        emit("  the same protocol and the same source files. The only difference is")
        emit(f"  {views_of(b)}px versus {views_of(h)}px input.")
        bv, hv = views_of(b), views_of(h)
        emit()
        emit(f"  {'metric':<24}{bv:>10}{hv:>10}{'delta':>10}")
        emit("  " + "-" * 54)
        for label, path in ((f"{FOC} AUPRC", ("per_label", FOC, "auprc")),
                            (f"{FOC} sensitivity", ("per_label", FOC, "sensitivity")),
                            (f"{FOC} FPR", ("per_label", FOC, "fpr")),
                            ("macro AUPRC", ("macro_auprc",)),
                            ("macro AUROC", ("macro_auroc",))):
            def dig(r):
                v = r["test"]
                for k in path:
                    v = v[k]
                return v
            emit(f"  {label:<24}{dig(b):10.4f}{dig(h):10.4f}{dig(h)-dig(b):+10.4f}")
        emit()
        emit("  If this delta is where the round's gain lives, the finding is")
        emit("  'preprocess at higher resolution', which is cheaper to deploy and")
        emit("  easier to defend than any hybrid below it.")

    # ---- 384 vs 768, per architecture --------------------------------------
    #
    # Round A tested CBAM and FPN at 384 only, and both lost to a plain ResNet-50
    # at 768. That compared an architecture change against a resolution change
    # and concluded the architectures were worthless - which the experiment could
    # not support, because attention cannot reweight detail the resampling has
    # already deleted. These pairs separate the two: each row holds the
    # architecture fixed and moves ONLY the input size.
    PAIRS = [("resnet50_base", "resnet50_hires", "plain ResNet-50"),
             ("resnet50_cbam", "resnet50_cbam_hires", "+ CBAM attention"),
             ("resnet50_fpn", "resnet50_fpn_hires", "+ FPN multi-scale")]
    have = [(lo, hi, lab) for lo, hi, lab in PAIRS
            if lo in by_name and hi in by_name]
    if have:
        emit()
        emit("=" * 108)
        emit(f"384 vs 768, ARCHITECTURE HELD FIXED - who actually needs the pixels?")
        emit("=" * 108)
        emit("  each row is ONE architecture at two input sizes. Same head, same")
        emit("  protocol, same source files, same micro-batch - only the resample.")
        emit(f"  the deciding column is {P_SRC} {FOC} AUPRC, not the pooled one.")
        emit()
        hdr = (f"{'architecture':<20}{'384 pooled':>12}{'768 pooled':>12}{'d':>9}"
               f"{'384 '+P_SRC:>13}{'768 '+P_SRC:>13}{'d':>9}")
        emit(hdr); emit("-" * len(hdr))
        for lo, hi, lab in have:
            a_, b_ = by_name[lo], by_name[hi]
            ap_ = a_["test"]["per_label"][FOC]["auprc"]
            bp_ = b_["test"]["per_label"][FOC]["auprc"]
            aq_, bq_ = _primary_focus_auprc(a_), _primary_focus_auprc(b_)
            emit(f"{lab:<20}{ap_:12.4f}{bp_:12.4f}{bp_-ap_:+9.4f}"
                 + (f"{'':>13}{'':>13}{'':>9}" if aq_ != aq_ or bq_ != bq_
                    else f"{aq_:13.4f}{bq_:13.4f}{bq_-aq_:+9.4f}"))
        emit()
        emit("  TWO QUESTIONS, and the rows answer them in different directions:")
        emit("    down a column - does any architecture beat plain ResNet-50 at the")
        emit("                    SAME resolution? That is the architecture's value.")
        emit("    along a row   - how much does that architecture gain from 768?")
        emit("                    A large gain means it was starved of detail at 384")
        emit("                    and round A's verdict on it was premature.")
        emit()
        emit("  At 384 the sheet is 1.73 px/mm vertically: a 0.5 mV LVH voltage")
        emit("  difference spans under 9 px. At 768 it is 3.46 px/mm. Attention that")
        emit("  looked useless in the first column may simply have had nothing left")
        emit("  to attend to.")
        missing_hi = [hi for lo, hi, _ in PAIRS
                      if lo in by_name and hi not in by_name]
        if missing_hi:
            emit()
            emit(f"  NOT YET RUN: {', '.join(missing_hi)} - the 384 verdict on those")
            emit("  architectures stays provisional until they are.")

    # ---- the resolution ladder --------------------------------------------
    #
    # THE QUESTION NOTEBOOK B EXISTS TO ANSWER, and the only place it is asked
    # as one question: 384 -> 768 -> both. Each rung has to earn its cost over
    # the rung below it, not over round 1. Comparing resnet50_dual against
    # round 1's resnet50 measures three changes at once and credits all of them
    # to the second branch.
    ladder = [m for m in ("resnet50_base", "resnet50_hires", "resnet50_dual",
                          "resnet50_effb3_dual") if m in by_name]
    if "resnet50_dual" in by_name and len(ladder) >= 3:
        emit()
        emit("=" * 108)
        emit("THE RESOLUTION LADDER - was the second branch necessary?")
        emit("=" * 108)
        emit("  each rung is scored against THE RUNG BELOW IT, not against round 1.")
        emit(f"  the column that decides this is {P_SRC} {FOC} AUPRC - see the")
        emit(f"  '{FOC} BY SOURCE' section for why the pooled one cannot.")
        emit()
        hdr = (f"{'rung':<22}{'views':>9}{'pooled':>9}{P_SRC:>9}{'d vs prev':>11}"
               f"{'params M':>10}{'ms/img':>9}{'train min':>11}")
        emit(hdr); emit("-" * len(hdr))
        prev_q = None
        for m in ladder:
            r = by_name[m]
            q = _primary_focus_auprc(r)
            d = "" if prev_q is None or q != q else f"{q - prev_q:+11.4f}"
            emit(f"{m:<22}{views_of(r):>9}"
                 f"{r['test']['per_label'][FOC]['auprc']:9.4f}"
                 + (f"{'':>9}" if q != q else f"{q:9.4f}") + f"{d:>11}"
                 + f"{r['complexity']['params_total_m']:10.1f}"
                 f"{r['inference']['latency_ms_b1_mean']:9.2f}"
                 f"{r['train_seconds']/60:11.1f}")
            if q == q:                       # NaN never becomes the anchor, or
                prev_q = q                   # every later rung reads "d vs nan"
        emit()
        emit("  READ IT AS A DECISION, not a ranking:")
        emit(f"    dual's {P_SRC} gain over resnet50_hires is what the second")
        emit("    backbone bought. Everything below that line - 2x the parameters,")
        emit("    a second forward pass per image, a longer train - is what it cost.")
        emit("    A gain under ~0.01 there does not pay for any of it, and the")
        emit("    round's finding becomes 'higher-resolution preprocessing was")
        emit("    sufficient; the dual-branch complexity was unnecessary'.")
        emit("    Confirm that reading against the stream ablation below: if")
        emit("    zeroing 'hi' barely moves the score, the model never used 768 at")
        emit("    all and the two branches were an ensemble.")

    # ---- stream ablation ---------------------------------------------------
    abl = [r for r in runs if r.get("test_stream_ablation")]
    if abl:
        emit()
        emit("=" * 108)
        emit("STREAM ABLATION - what each branch was actually worth")
        emit("=" * 108)
        emit("  each row zeroes one feature group after its own LayerNorm and")
        emit("  re-scores TEST with the SAME val-frozen thresholds.")
        emit()
        hdr = (f"{'model':<22}{'group zeroed':<18}{'mAUPRC':>10}{'d':>9}"
               f"{FOC+' AUPRC':>13}{'d':>9}")
        emit(hdr); emit("-" * len(hdr))
        for r in abl:
            t = r["test"]
            fa, ff = t["macro_auprc"], t["per_label"][FOC]["auprc"]
            emit(f"{r['model']:<22}{'(none)':<18}{fa:10.4f}{'':>9}{ff:13.4f}{'':>9}")
            for k, res in r["test_stream_ablation"].items():
                m, f = res["macro_auprc"], res["per_label"][FOC]["auprc"]
                emit(f"{'':<22}{k:<18}{m:10.4f}{m-fa:+9.4f}{f:13.4f}{f-ff:+9.4f}")
        emit()
        emit("  A dual-resolution model whose score barely moves when the 768 group")
        emit("  is zeroed did not use the extra pixels - it ensembled two backbones.")
        emit("  For resnet50_fpn the groups are pyramid levels, so this table says")
        emit("  which SCALE the diagnosis was read at.")

    # ---- headline ---------------------------------------------------------
    emit()
    emit("=" * 108)
    emit("TEST SET  (thresholds tuned on val, frozen before test)")
    emit("=" * 108)
    emit()
    hdr = (f"{'model':<22}{'views':>8}{'mAUROC':>9}{'mAUPRC':>9}{'mSens':>8}"
           f"{'mSpec':>8}{'mPrec':>8}{'mF1':>8}{'mFPR':>8}{'ExactAcc':>10}")
    emit(hdr); emit("-" * len(hdr))
    for r in runs:
        t = r["test"]
        emit(f"{r['model']:<22}{views_of(r):>8}{t['macro_auroc']:9.4f}"
             f"{t['macro_auprc']:9.4f}{t['macro_sensitivity']:8.4f}"
             f"{t['macro_specificity']:8.4f}{t['macro_precision']:8.4f}"
             f"{t['macro_f1']:8.4f}{t['macro_fpr']:8.4f}"
             f"{t['exact_match_accuracy']:10.4f}")
    emit()
    emit("round 1, same splits and same protocol, single-Linear head:")
    for m in sorted(ROUND1, key=lambda k: -ROUND1[k]["auprc"]["macro"]):
        v = ROUND1[m]
        emit(f"  {m:<22}{'384':>8}{v['auroc']['macro']:9.4f}{v['auprc']['macro']:9.4f}")

    # ---- cost -------------------------------------------------------------
    emit()
    emit("COST  (accuracy is only half the comparison)")
    emit()
    hdr = (f"{'model':<22}{'params M':>10}{'size MB':>10}{'ms/img b1':>11}"
           f"{'img/s batch':>13}{'train min':>11}{'epochs':>8}")
    emit(hdr); emit("-" * len(hdr))
    for r in runs:
        c = r.get("complexity", {}); b = r.get("inference", {})
        emit(f"{r['model']:<22}{c.get('params_total_m', float('nan')):10.1f}"
             f"{c.get('size_mb', float('nan')):10.1f}"
             f"{b.get('latency_ms_b1_mean', float('nan')):11.2f}"
             f"{b.get('throughput_ips_batched', float('nan')):13.1f}"
             f"{r['train_seconds']/60:11.1f}{r.get('epochs_run', 0):8d}")
    emit()
    emit("  round 1's resnet50: 23.5 M, 89.9 MB, 12.21 ms/img, 288.4 img/s, 22.3 min.")

    # ---- per label --------------------------------------------------------
    for metric, nice in (("auroc", "AUROC"), ("auprc", "AUPRC"),
                         ("sensitivity", "SENSITIVITY (recall)"),
                         ("specificity", "SPECIFICITY"),
                         ("precision", "PRECISION"), ("f1", "F1"),
                         ("fpr", "FALSE POSITIVE RATE"), ("accuracy", "ACCURACY")):
        emit()
        emit(f"per-label {nice} (test)")
        emit()
        hdr = f"{'model':<22}" + "".join(f"{l:>10}" for l in C.LABELS) + f"{'MACRO':>10}"
        emit(hdr); emit("-" * len(hdr))
        for r in runs:
            emit(f"{r['model']:<22}"
                 + "".join(f"{r['test']['per_label'][l][metric]:10.4f}" for l in C.LABELS)
                 + f"{r['test']['macro_'+metric]:10.4f}")

    # ---- source confound --------------------------------------------------
    if any(r.get("test_per_source") for r in runs):
        emit()
        emit("per-source macro AUROC (test) - large spread = source confound")
        emit()
        srcs = sorted({s for r in runs for s in r.get("test_per_source", {})})
        hdr = f"{'model':<22}" + "".join(f"{s:>11}" for s in srcs) + f"{'spread':>10}"
        emit(hdr); emit("-" * len(hdr))
        for r in runs:
            ps = r.get("test_per_source", {})
            vals = [ps[s]["macro_auroc"] for s in srcs if s in ps]
            row = "".join(f"{ps[s]['macro_auroc']:11.4f}" if s in ps else f"{'-':>11}"
                          for s in srcs)
            spread = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")
            emit(f"{r['model']:<22}{row}{spread:10.4f}")
        emit()
        emit("  All STEMI comes from one source, and stage 5 measured residual source")
        emit("  signal (0.515 vs 0.372 chance). Round 1 spreads: resnet50 0.0225,")
        emit("  convnext 0.0287, vgg16 0.0188, vit 0.0096, effnet 0.0347.")
        emit("  A HIGHER spread this round is the thing to be suspicious of: the 768")
        emit("  and FPN models can see grid colour and JPEG texture that the 384")
        emit("  models could not, and those identify the source database.")

    txt = a.runs / "comparison.txt"
    txt.write_text("\n".join(out_lines))
    print(f"\ntext report -> {txt}")

    if a.csv:
        import csv as _csv
        af_by = {row["model"]: row for row in af_rows}
        # CSV column name -> key inside af_lvh.score_run's row dict.
        AF_LVH_KEYMAP = {
            "af_lvh_n": "af_lvh_n",
            "af_lvh_lvh_recall": "lvh_recall_af_lvh",
            "af_lvh_lvh_recall_comparand": "lvh_recall_comparand",
            "af_lvh_lvh_delta": "lvh_delta",
            "af_lvh_lvh_delta_ci_low": "lvh_delta_ci_low",
            "af_lvh_lvh_delta_ci_high": "lvh_delta_ci_high",
            "af_lvh_lvh_fisher_p": "lvh_fisher_p",
            "af_lvh_lvh_auprc": "lvh_auprc_af_lvh",
            "af_lvh_lvh_tp": "lvh_tp_af_lvh",
            "af_lvh_lvh_fn": "lvh_fn_af_lvh",
            "af_lvh_af_recall": "af_recall_af_lvh",
            "af_lvh_af_recall_comparand": "af_recall_comparand",
            "af_lvh_af_delta": "af_delta",
            "af_lvh_af_fisher_p": "af_fisher_p",
            "af_lvh_af_auprc": "af_auprc_af_lvh",
            "af_lvh_af_tp": "af_tp_af_lvh",
            "af_lvh_af_fn": "af_fn_af_lvh",
        }
        AF_LVH_CSV = list(AF_LVH_KEYMAP)
        with a.csv.open("w", newline="") as f:
            w = _csv.writer(f)
            MAC = ["auroc", "auprc", "sensitivity", "specificity", "precision",
                   "f1", "fpr", "accuracy"]
            PL = ["auroc", "auprc", "sensitivity", "specificity", "precision",
                  "f1", "fpr", "accuracy", "threshold", "support", "tp", "fp", "fn", "tn"]
            w.writerow(["model", "views"] + [f"macro_{m}" for m in MAC]
                       + ["exact_match_accuracy", "hamming_accuracy",
                          "micro_f1", "micro_precision", "micro_sensitivity"]
                       + [f"{l}_{m}" for l in C.LABELS for m in PL]
                       + ["params_total", "params_total_m", "size_mb",
                          "latency_ms_b1", "latency_p95_b1", "throughput_ips_batched",
                          "train_min", "epochs", "lr", "pretrain", "tags",
                          "protocol_fingerprint", "input_fingerprint",
                          "source_spread"]
                       # Source-stratified focus label. In the CSV as well as the
                       # text report, because the pooled columns above are the
                       # ones a spreadsheet will sort by and they are the ones
                       # the database shortcut inflates.
                       + [f"{P_SRC}_{FOC}_{m}" for m in
                          ("auprc", "auroc", "sensitivity", "fpr", "support")]
                       + [f"ablate_{g}_macro_auprc" for g in ("lo", "hi", "cnn", "vit",
                                                              "P2", "P3", "P4", "P5")]
                       + [f"ablate_{g}_{FOC}_auprc" for g in ("lo", "hi", "cnn", "vit",
                                                              "P2", "P3", "P4", "P5")]
                       + AF_LVH_CSV)
            for r in runs:
                t = r["test"]; c = r.get("complexity", {}); b = r.get("inference", {})
                ps = r.get("test_per_source", {})
                sv = [d["macro_auroc"] for d in ps.values()]
                spread = (max(sv) - min(sv)) if len(sv) > 1 else ""
                ab = r.get("test_stream_ablation", {})
                w.writerow([r["model"], views_of(r)] + [t[f"macro_{m}"] for m in MAC]
                           + [t["exact_match_accuracy"], t["hamming_accuracy"],
                              t["micro_f1"], t["micro_precision"], t["micro_sensitivity"]]
                           + [t["per_label"][l][m] for l in C.LABELS for m in PL]
                           + [c.get("params_total"), c.get("params_total_m"),
                              c.get("size_mb"), b.get("latency_ms_b1_mean"),
                              b.get("latency_ms_b1_p95"), b.get("throughput_ips_batched"),
                              round(r["train_seconds"] / 60, 1), r.get("epochs_run"),
                              r.get("lr_used"), r["model_info"].get("pretrain"),
                              "|".join(r["model_info"].get("tags", [])),
                              r.get("protocol_fingerprint"), r.get("input_fingerprint"),
                              spread]
                           + [ps.get(P_SRC, {}).get("per_label", {})
                                .get(FOC, {}).get(m, "")
                              for m in ("auprc", "auroc", "sensitivity", "fpr",
                                        "support")]
                           + [ab.get(f"without_{g}", {}).get("macro_auprc", "")
                              for g in ("lo", "hi", "cnn", "vit", "P2", "P3", "P4", "P5")]
                           + [ab.get(f"without_{g}", {}).get("per_label", {})
                                .get(FOC, {}).get("auprc", "")
                              for g in ("lo", "hi", "cnn", "vit", "P2", "P3", "P4", "P5")]
                           + [af_by.get(r["model"], {}).get(AF_LVH_KEYMAP[k], "")
                              for k in AF_LVH_CSV])
        print(f"csv -> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
