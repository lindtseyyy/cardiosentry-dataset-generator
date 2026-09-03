"""
Source-stratified analysis of the focus label. THE REQUIRED READ FOR THIS ROUND.

WHY THIS MODULE EXISTS

Notebook A produced a clean-looking result: 768px input lifted pooled LVH AUPRC
from 0.9036 (384) to 0.9248, and beat both attention hybrids. Then the test
split was broken down by originating database, and most of that gain turned out
not to be about LVH at all.

    LVH positives in the 938-image test split
        PTB-XL     234 of 255
        Chapman     21 of 255
        STEMI db     0 of 255   <- all 146 of its images are LVH-negative

    LVH AUPRC          pooled     PTB-XL only
        resnet50_base  0.9036     0.9270
        resnet50_hires 0.9248     0.9315
        delta          +0.0212    +0.0045   (95% CI [-0.0124, +0.0208])

Within PTB-XL - where 92% of the positives live, and where the source shortcut
is unavailable because every image comes from the same database - four fifths
of the pooled gain evaporates. What the 768 model actually did better was stop
calling LVH on images from the STEMI database: 11 such false positives at 384,
1 at 768. That is a database-recognition improvement wearing an LVH costume.

The same test split makes it worse than it sounds: the STEMI label is EXACTLY
`source == "STEMI"` - 146 of 146, no exceptions - so any model that learns to
recognise the STEMI database gets a free, perfect STEMI head AND a cleaner
pooled LVH number. resnet50_hires scored 0.9996 STEMI AUROC. That is the
signature to be suspicious of, not the result to report.

WHAT TO READ, IN ORDER

  1. PTB-XL LVH AUPRC        the shortcut cannot reach it. This is the number.
  2. PTB-XL sens @ matched FPR   threshold-free, and comparable across models
                             because it fixes specificity rather than trusting
                             a globally-tuned threshold on a subgroup.
  3. shortcut share          how much of the pooled gain does NOT reproduce
                             inside PTB-XL. High = the model got better at
                             telling the databases apart.
  4. LVH FP on STEMI-db images   the concrete count behind (3).

Only then the pooled number, and only as the number a reader of the paper will
see first.
"""
from __future__ import annotations

from pathlib import Path

import config as C

# The database that carries the overwhelming majority of focus-label positives,
# and therefore the only subgroup large enough to score the focus label inside.
# Not a free parameter: it is a property of this corpus, asserted at load time.
PRIMARY_SOURCE = "PTBXL"

# Reference operating point for the matched-FPR column. Notebook A's 384 control
# sits at 0.0779 FPR on PTB-XL (30 false positives of 385 negatives); this is
# rounded UP to 0.08 so that the control's own operating point is inside the
# constraint rather than balanced on it. Landing exactly on a control's ROC grid
# point makes the column flip by a whole step on a rounding difference.
#
# Frozen for the same reason baseline.py is frozen: if a later run could move the
# reference, "sensitivity at a fixed specificity" would stop meaning anything.
MATCHED_FPR = 0.08

MATCHED_FPR_NOTE = (
    f"  sens@.08 is PTB-XL sensitivity capped at a false-positive rate of\n"
    f"  {MATCHED_FPR:.2f} - just above notebook A's 384 control (0.0779). It is the only\n"
    f"  sensitivity column that compares DETECTORS rather than operating points:\n"
    f"  thresholds are tuned on the whole validation set, so a model can look\n"
    f"  more sensitive on a subgroup purely by sitting at a laxer threshold there."
)


def load_test_predictions(run_dir: Path):
    """Per-image test probabilities written by artifacts.save_predictions.

    Returns None rather than raising: a smoke run writes no predictions, and a
    half-finished batch should still tabulate.
    """
    p = Path(run_dir) / "predictions" / "test_predictions.csv"
    if not p.exists():
        return None
    import pandas as pd
    df = pd.read_csv(p)
    need = {"source", f"true_{C.FOCUS_LABEL}", f"prob_{C.FOCUS_LABEL}",
            f"pred_{C.FOCUS_LABEL}"}
    return df if need <= set(df.columns) else None


def sens_at_fpr(y, prob, target_fpr: float) -> float:
    """Sensitivity at the highest achievable FPR that does not exceed target.

    Threshold-free. The point of the column: comparing two models' sensitivity
    at THEIR OWN tuned thresholds compares operating points, not detectors.
    """
    import numpy as np
    from sklearn.metrics import roc_curve
    y = np.asarray(y)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, prob)
    i = int(np.searchsorted(fpr, target_fpr, side="right")) - 1
    return float(tpr[max(i, 0)])


def _scores(y, prob, pred) -> dict:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, prob, pred = np.asarray(y), np.asarray(prob), np.asarray(pred)
    pos, neg = y == 1, y == 0
    out = {
        "n": int(len(y)),
        "n_pos": int(pos.sum()),
        "tp": int((pred[pos] == 1).sum()) if pos.any() else 0,
        "fn": int((pred[pos] == 0).sum()) if pos.any() else 0,
        "fp": int((pred[neg] == 1).sum()) if neg.any() else 0,
        "sensitivity": float(pred[pos].mean()) if pos.any() else float("nan"),
        "fpr": float(pred[neg].mean()) if neg.any() else float("nan"),
    }
    if pos.any() and neg.any():
        out["auprc"] = float(average_precision_score(y, prob))
        out["auroc"] = float(roc_auc_score(y, prob))
        out["sens_at_matched_fpr"] = sens_at_fpr(y, prob, MATCHED_FPR)
    else:
        # A source with no positives cannot be ranked. It still contributes
        # false positives, and those are the whole point for the STEMI database.
        out["auprc"] = out["auroc"] = out["sens_at_matched_fpr"] = float("nan")
    return out


def focus_by_source(df, focus: str = None) -> dict:
    """{source -> scores} plus 'POOLED' and 'ALL_BUT_PRIMARY' rollups."""
    focus = focus or C.FOCUS_LABEL
    y, p, k = f"true_{focus}", f"prob_{focus}", f"pred_{focus}"
    out = {"POOLED": _scores(df[y], df[p], df[k])}
    for src in sorted(df["source"].unique()):
        s = df[df["source"] == src]
        out[src] = _scores(s[y], s[p], s[k])
    return out


def label_source_coupling(df) -> dict:
    """{label -> source} for any label that is EXACTLY 'came from this database'.

    On this corpus STEMI is such a label. A head whose target is identical to
    its source cannot be distinguished from a database classifier, so its
    metrics are uninterpretable as clinical performance no matter how good they
    look - and they look excellent, which is the problem.
    """
    coupled = {}
    for lab in C.LABELS:
        col = df[f"true_{lab}"]
        for src in df["source"].unique():
            if ((df["source"] == src).astype(int) == col).all():
                coupled[lab] = src
    return coupled


def shortcut_share(pooled_delta: float, primary_delta: float) -> float:
    """Fraction of a pooled gain that does NOT reproduce inside the primary source.

    1.0 = the whole pooled improvement came from somewhere other than the
    subgroup holding the positives, i.e. from separating the databases.
    NaN when the pooled delta is too small to apportion - a ratio with a near
    zero denominator is noise with a percent sign on it.
    """
    if not (pooled_delta > 0.005):
        return float("nan")
    return float(1.0 - primary_delta / pooled_delta)


# ---------------------------------------------------------------- formatting
def format_table(rows: list[dict], control: str | None) -> list[str]:
    """rows: [{model, views, pooled, primary, ...}] already in display order."""
    def f(v, w=9, p=4):
        return f"{'':>{w}}" if v != v else f"{v:{w}.{p}f}"

    L = []
    hdr = (f"{'model':<22}{'views':>9}{'overall':>9}{'PTB-XL':>9}{'PTB-XL':>9}"
           f"{'PTB-XL':>9}{'PTB-XL':>10}{'STEMI-db':>10}{'shortcut':>10}")
    sub = (f"{'':<22}{'':>9}{'AUPRC':>9}{'AUPRC':>9}{'Sens':>9}"
           f"{'FPR':>9}{'sens@.08':>10}{'false pos':>10}{'share':>10}")
    L += [hdr, sub, "-" * len(hdr)]
    for r in rows:
        pr, st = r["primary"], r.get("stemi_db") or {}
        share = r.get("share", float("nan"))
        L.append(
            f"{r['model']:<22}{r['views']:>9}"
            f"{f(r['pooled']['auprc'])}{f(pr['auprc'])}"
            f"{f(pr['sensitivity'])}{f(pr['fpr'])}"
            f"{f(pr['sens_at_matched_fpr'], 10)}"
            + (f"{st['fp']:>10d}" if st else f"{'':>10}")
            + (f"{'':>10}" if share != share else f"{share*100:>9.0f}%"))
    if control:
        L.append("")
        L.append(f"  deltas below are against {control}; 'shortcut share' is the "
                 f"fraction of each")
        L.append("  model's POOLED gain over it that does NOT reproduce inside "
                 "PTB-XL.")
    return L


def format_deltas(rows: list[dict], control_row: dict) -> list[str]:
    """The pooled-vs-PTB-XL gap, which is the finding, stated as two deltas."""
    c = control_row
    L = [f"{'model':<22}{'pooled d':>11}{'PTB-XL d':>11}{'gap':>11}"
         f"{'  verdict':<40}"]
    L.append("-" * 95)
    for r in rows:
        if r["model"] == c["model"]:
            continue
        dp = r["pooled"]["auprc"] - c["pooled"]["auprc"]
        dq = r["primary"]["auprc"] - c["primary"]["auprc"]
        if dp <= 0.005:
            v = "no pooled gain to explain"
        elif dq >= 0.8 * dp:
            v = "gain survives stratification - real"
        elif dq >= 0.3 * dp:
            v = "PARTLY source-driven - report both"
        else:
            v = "MOSTLY source-driven - do not report pooled"
        L.append(f"{r['model']:<22}{dp:+11.4f}{dq:+11.4f}{dp-dq:+11.4f}  {v:<40}")
    return L
