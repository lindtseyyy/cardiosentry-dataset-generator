"""
Stratum definitions and the counting tables this round exists to produce.

THE STRATA

Every ECG in the frozen index falls into exactly one stratum, by GROUND-TRUTH
labels only (never by model predictions - see config.py for why):

    clean     true_AF=0 and true_STEMI=0     "LVH only"
    af        true_AF=1                      "AF + LVH"
    stemi     true_STEMI=1                   "STEMI + LVH"
    af_stemi  true_AF=1 and true_STEMI=1     "AF + STEMI + LVH"

A stratum is not the same as the LVH-positive subset inside it. Each stratum
holds LVH-negatives as well, and those negatives are what make specificity,
FPR, precision and AUPRC computable INSIDE the stratum - which is the entire
point: "how well does the LVH head work when AF is present" needs both LVH
positives and LVH negatives WITH AF.

THE STRUCTURAL ZEROS - a corpus fact, not a model finding

Two of the four requested counts are zero by construction and stay zero for
every model ever trained on this corpus:

  * STEMI is EXACTLY `source == "STEMI"` (146 of 146 test images, 1442 of
    6670 corpus-wide), and that database carries zero LVH positives - so
    "STEMI + LVH" and "AF + STEMI + LVH" ECGs do not exist here.
  * AF and STEMI never co-occur at all: true_AF=1 implies true_STEMI=0 on
    every row of the corpus. There are no "AF + STEMI" ECGs of ANY LVH status.

Those strata therefore contribute LVH false-positive counts (and nothing
else) to the evaluation, and their LVH scores are structurally undefined -
the tables mark them `--` rather than printing a number that would read like
a measurement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

# Columns the per-row stratum assignment needs from the index.
INDEX_NEEDED = {"record", "split", "source", "patient_id", "LVH", "AF", "STEMI"}


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Index files use 'AF'/'STEMI'/'LVH'; the predictions file uses the
    'true_AF' spelling. Both are legal inputs everywhere in this module."""
    if name in df.columns:
        return df[name]
    if f"true_{name}" in df.columns:
        return df[f"true_{name}"]
    raise KeyError(f"neither '{name}' nor 'true_{name}' is a column")


def assign_context(df: pd.DataFrame) -> pd.Series:
    """Ground-truth stratum per row: 'clean' | 'af' | 'stemi' | 'af_stemi'."""
    af = _col(df, "AF").astype(int)
    st = _col(df, "STEMI").astype(int)
    return pd.Series(np.where((af == 0) & (st == 0), "clean",
                     np.where((af == 1) & (st == 0), "af",
                     np.where((af == 0) & (st == 1), "stemi", "af_stemi"))),
                     index=df.index)


def _lvh_positive(df: pd.DataFrame) -> pd.Series:
    return _col(df, "LVH").astype(int) == 1


# ---------------------------------------------------------------- count tables
def count_by_context_lvh(df: pd.DataFrame) -> pd.DataFrame:
    """n, LVH+ and LVH- per stratum (any split). The four headline counts come
    from the LVH+ column of the test-split instance of this table."""
    ctx = assign_context(df)
    lvh = _lvh_positive(df)
    t = pd.DataFrame({"context": ctx, "lvh": lvh})
    out = (t.groupby(["context", "lvh"]).size().unstack(fill_value=0)
            .rename(columns={0: "lvh_neg", 1: "lvh_pos"}))
    for c in C.CONTEXTS:
        if c not in out.index:
            out.loc[c] = [0, 0]
    out["n"] = out["lvh_pos"] + out["lvh_neg"]
    out = out.loc[list(C.CONTEXTS), ["n", "lvh_pos", "lvh_neg"]]
    out.index.name = "context"
    return out


def prevalence_by_context_source(df: pd.DataFrame) -> pd.DataFrame:
    """LVH prevalence within each source, per stratum: n, LVH+, prevalence.

    This is the 'LVH prevalence within each source for these subgroups' table
    the task asks for. It answers, e.g., 'of the PTB-XL ECGs that carry AF,
    what fraction also carry LVH' - the co-occurrence structure that decides
    how much weight each stratum has in the pooled LVH number.
    """
    ctx = assign_context(df)
    lvh = _lvh_positive(df)
    t = pd.DataFrame({"context": ctx, "source": df["source"], "lvh": lvh})
    g = (t.groupby(["context", "source"])["lvh"]
          .agg(n="size", lvh_pos="sum").reset_index())
    g["lvh_prev"] = g["lvh_pos"] / g["n"]
    g["context"] = pd.Categorical(g["context"], list(C.CONTEXTS))
    g["source"] = pd.Categorical(g["source"], sorted(df["source"].unique()))
    return (g.sort_values(["context", "source"])
             .reset_index(drop=True))


def lvh_composition_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """What fraction of each source's LVH positives live in each stratum.

    The mirror read of prevalence_by_context_source: instead of 'how much LVH
    inside each stratum', this says 'of PTB-XL's LVH, how much is AF+LVH'.
    A source whose LVH positives are mostly AF+LVH is a source where the AF
    stratum, not the clean one, decides the source's LVH number.
    """
    lvh = df[_lvh_positive(df)]
    if len(lvh) == 0:
        return pd.DataFrame(columns=["source", "context", "lvh_pos", "share"])
    ctx = assign_context(lvh)
    t = pd.DataFrame({"source": lvh["source"], "context": ctx})
    g = t.groupby(["source", "context"]).size().rename("lvh_pos").reset_index()
    g["share"] = g["lvh_pos"] / g.groupby("source")["lvh_pos"].transform("sum")
    g["context"] = pd.Categorical(g["context"], list(C.CONTEXTS))
    return g.sort_values(["source", "context"]).reset_index(drop=True)


# ---------------------------------------------------------------- patient reads
def patient_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Per stratum: images, distinct patients, LVH+ images, LVH+ patients.

    The confound ledger for the verdict: if the AF stratum's LVH positives
    were 28 images of ONE patient, a per-image CI would overstate the
    evidence. These counts say how concentrated each stratum actually is.
    """
    ctx = assign_context(df)
    lvh = _lvh_positive(df)
    t = pd.DataFrame({"context": ctx, "patient": df["patient_id"], "lvh": lvh})
    rows = []
    for c in C.CONTEXTS:
        s = t[t.context == c]
        rows.append({
            "context": c,
            "n_imgs": int(len(s)),
            "n_patients": int(s.patient.nunique()) if len(s) else 0,
            "lvh_pos_imgs": int(s.lvh.sum()),
            "lvh_pos_patients": int(s.loc[s.lvh, "patient"].nunique())
            if s.lvh.any() else 0,
        })
    return pd.DataFrame(rows)


def patient_overlap(df: pd.DataFrame, a: str = "clean", b: str = "af") -> dict:
    """Distinct-patient overlap between two strata, overall and among LVH+."""
    ctx = assign_context(df)
    lvh = _lvh_positive(df)
    pa = set(df.loc[ctx == a, "patient_id"])
    pb = set(df.loc[ctx == b, "patient_id"])
    lvha = set(df.loc[(ctx == a) & lvh, "patient_id"])
    lvhb = set(df.loc[(ctx == b) & lvh, "patient_id"])
    return {
        f"patients_{a}": len(pa), f"patients_{b}": len(pb),
        "patient_overlap": len(pa & pb),
        f"lvh_patients_{a}": len(lvha), f"lvh_patients_{b}": len(lvhb),
        "lvh_patient_overlap": len(lvha & lvhb),
    }


# ---------------------------------------------------------------- formatting
def _pct(x: float) -> str:
    return f"{x*100:6.1f}%" if x == x else f"{'--':>6}"


def format_counts(title: str, counts: pd.DataFrame,
                  prev: pd.DataFrame, compos: pd.DataFrame) -> list[str]:
    L = [title, "-" * 88]
    L.append(f"  {'subgroup':<26}{'n':>6}{'LVH+':>7}{'LVH-':>7}")
    for c in C.CONTEXTS:
        r = counts.loc[c]
        L.append(f"  {C.CONTEXTS[c]['display']:<26}{int(r['n']):>6}"
                 f"{int(r['lvh_pos']):>7}{int(r['lvh_neg']):>7}")
    L.append("")
    L.append("  LVH prevalence within each source, per subgroup:")
    L.append(f"    {'subgroup':<26}{'source':>9}{'n':>7}{'LVH+':>7}{'prevalence':>11}")
    for _, r in prev.iterrows():
        L.append(f"    {C.CONTEXTS[r['context']]['display']:<26}{r['source']:>9}"
                 f"{int(r['n']):>7}{int(r['lvh_pos']):>7}"
                 f"{_pct(r['lvh_prev']):>11}")
    L.append("")
    L.append("  share of each source's LVH positives that carry the comorbidity:")
    shown = [c for c in C.CONTEXTS if (compos.context == c).any()]
    L.append(f"    {'source':>9}{'':<26}" +
             "".join(f"{C.CONTEXTS[c]['display']:<28}" for c in shown))
    for src, g in compos.groupby("source"):
        cells = "".join(
            f"{_pct(float(g.loc[g.context == c, 'share'].iloc[0])):<28}"
            if (g.context == c).any() else f"{'--':<28}"
            for c in shown)
        L.append(f"    {src:>9}{'':<26}{cells}")
    return L


def format_patients(rows: pd.DataFrame, overlap: dict, a: str, b: str) -> list[str]:
    L = ["  patient-level structure of the test strata",
         f"    {'subgroup':<26}{'images':>7}{'patients':>9}{'LVH+ imgs':>10}"
         f"{'LVH+ pts':>10}"]
    for _, r in rows.iterrows():
        L.append(f"    {C.CONTEXTS[r['context']]['display']:<26}"
                 f"{int(r['n_imgs']):>7}{int(r['n_patients']):>9}"
                 f"{int(r['lvh_pos_imgs']):>10}{int(r['lvh_pos_patients']):>10}")
    L.append("")
    L.append(f"  patient overlap between '{C.CONTEXTS[a]['display']}' and "
             f"'{C.CONTEXTS[b]['display']}':")
    L.append(f"    patients:            {overlap[f'patients_{a}']} vs "
             f"{overlap[f'patients_{b}']}, shared {overlap['patient_overlap']}")
    L.append(f"    LVH+ patients:       {overlap[f'lvh_patients_{a}']} vs "
             f"{overlap[f'lvh_patients_{b}']}, shared {overlap['lvh_patient_overlap']}")
    L.append("    near-zero overlap = the strata are ~disjoint patient groups,")
    L.append("    so the AF-vs-clean delta is BETWEEN patients, not within them.")
    return L
