#!/usr/bin/env python3
"""
Build the SELF-CONTAINED Colab notebook for the AF x LVH joint-head analysis.

    python make_notebook.py

The harness is embedded in cell 1 as a base64 tar.gz (the project convention),
so one file travels to Colab with everything it needs. The analysis is pure
pandas/numpy + exact combinatorial tests over 938 rows - no pip installs, no
GPU, runs in seconds on any Colab runtime. It reads the SAME frozen
predictions the previous round used (hybrid_v1/resnet50_hires/) and writes
only to the NEW folder:

    cardiosentry_runs/af_lvh_joint_v1/

The frozen results folders are read-only for this round.

Re-run this script after editing any module to refresh the embedded copy.
"""
from __future__ import annotations
import base64, io, json, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = [HERE / n for n in ("config.py", "baseline.py", "metrics.py",
                            "stats.py", "analyze.py")]

DRIVE_ROOT = "/content/drive/MyDrive/CardioSentry"
RUN_NAME = "af_lvh_joint_v1"

RUNS = f"{DRIVE_ROOT}/cardiosentry_runs/{RUN_NAME}"
HYBRID_RUNS = f"{DRIVE_ROOT}/cardiosentry_runs/hybrid_v1"
REFERENCE = f"{HYBRID_RUNS}/resnet50_hires"


def payload() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in FILES:
            tar.add(f, arcname=f.name)
    return base64.b64encode(buf.getvalue()).decode()


def _src(text: str) -> list[str]:
    """nbformat wants a list of lines WITH their trailing newlines."""
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": _src(s.strip())}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": _src(s.strip("\n"))}

B64 = payload()

UNPACK = f'''
import base64, io, tarfile, os
os.makedirs("/content/code", exist_ok=True)
_PKG = "{B64}"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(_PKG)), mode="r:gz").extractall("/content/code")
print("unpacked to /content/code:")
for f in sorted(os.listdir("/content/code")):
    print("   ", f)
print("\\nno pip installs needed: the exact tests (McNemar, Fisher) are")
print("implemented in stats.py, and pandas/sklearn/numpy are preinstalled.")
'''

MOUNT = f'''
import os
from google.colab import drive
drive.mount('/content/drive')

# ---- Drive layout. Edit DRIVE_ROOT only; everything else follows. ----------
DRIVE_ROOT = "{DRIVE_ROOT}"
RUN_NAME   = "{RUN_NAME}"

HYBRID_RUNS = f"{{DRIVE_ROOT}}/cardiosentry_runs/hybrid_v1"
REFERENCE   = f"{{HYBRID_RUNS}}/resnet50_hires"         # the four-head model, READ-ONLY
RUNS        = f"{{DRIVE_ROOT}}/cardiosentry_runs/{{RUN_NAME}}"   # THIS round writes here

os.makedirs(RUNS, exist_ok=True)

print("Drive layout check:")
for _p, _label in (
        (REFERENCE + "/predictions/test_predictions.csv", "frozen test predictions"),
        (REFERENCE + "/result.json",                     "reference result.json (optional)")):
    print(f"  {{'FOUND ' if os.path.exists(_p) else 'missing'}}  {{_label:<32}} {{_p}}")

if not os.path.exists(REFERENCE + "/predictions/test_predictions.csv"):
    raise SystemExit(
        f"\\nMISSING the frozen predictions file:\\n  "
        f"{{REFERENCE}}/predictions/test_predictions.csv\\n"
        f"This round reads the EXACT file the reference numbers and the\\n"
        f"previous subgroup analysis were computed from. Copy the\\n"
        f"hybrid_v1/resnet50_hires/ run folder back to Drive if it is missing.")
'''

ANALYSIS = '''
%cd /content/code
!python analyze.py \
    --predictions "$REFERENCE/predictions/test_predictions.csv" \
    --result-json  "$REFERENCE/result.json" \
    --out          "$RUNS"
'''

VERDICT = '''
%cd /content/code
_report = open(f"{RUNS}/report.txt").read()
_i = _report.find("VERDICT:")
print(_report[_i:] if _i >= 0 else _report[-4000:])
'''

MORNING = '''
!cat "$RUNS/STATUS.txt"
!ls "$RUNS"
'''


def header() -> dict:
    names = ", ".join(f.name for f in FILES)
    return md(f"""
# AF × LVH joint-head analysis — "detects AF, misses LVH"?

**The question:** the previous round (af_lvh_subgroup_v1) showed the LVH head
deteriorates inside the AF stratum of the frozen test split. This round zooms
into the **28 ECGs where AF and LVH co-occur** and asks head by head:

> **"When AF and LVH coexist, does the model correctly detect AF but
> frequently miss LVH?"**

**Same frozen everything** — the SAME `test_predictions.csv` of
`resnet50_hires`, the SAME frozen val-tuned thresholds (**AF 0.81, LVH
0.47**), the SAME ground-truth subgroup definition. Nothing is retrained,
re-thresholded, re-predicted or re-split; the frozen results folders are
read-only for this round.

**What the notebook produces** (all written to
`cardiosentry_runs/af_lvh_joint_v1/` on Drive):

1. cross-checks: the frozen decisions reproduce the saved predictions
   exactly, the pooled AF/LVH rows equal the frozen reference numbers, and
   the subgroup sizes match (refuses on any disagreement);
2. per-head evaluation of **AF+LVH**, **AF-only** and **LVH-only** —
   sensitivity, specificity, precision, F1, AUPRC, AUROC and TP/FP/FN/TN —
   where the subgroup is the positive class and the negatives are the frozen
   pool of ECGs that are true-negative for that head;
3. the **joint breakdown** of the 28 cases with bootstrap CIs;
4. the comparison with exact tests — **McNemar** (AF vs LVH on the same 28
   ECGs), **Fisher** (AF recall AF+LVH vs AF-only; LVH recall AF+LVH vs
   LVH-only) — plus recall CIs;
5. the **verdict**, applying the rule stated in the header of this notebook
   before any number, and the sample-size limitation with it.

**The verdict rule (frozen):** R1 McNemar on the joint breakdown (p < 0.05
in the claimed direction), R2 AF recall on AF+LVH not significantly below
AF-only (Fisher), R3 LVH recall on AF+LVH significantly below LVH-only
(Fisher). SUPPORTED needs R1 ✓ + R2 not significant + R3 significant.

**Limitation, stated regardless of verdict:** the subgroup has **28 ECGs**
(25 distinct patients per the previous round) — recall CIs are wide, the
cross-subgroup tests are low-power, and the verdict grades the evidence for
the **association** on this test split. No causal claim is made.

**Runtime:** pure pandas/numpy + exact combinatorial tests over 938 rows —
any Colab runtime, ~10 s, no GPU, no pip installs.

The whole harness ({names}) is embedded in cell 1 — no folder upload needed.
""")

def notebook() -> dict:
    cells = [
        header(),

        md("""
## The reference-pool design — why "AF specificity" exists for a subgroup

All 28 AF+LVH ECGs are positive for both heads, so sensitivity is computable
inside the subgroup but specificity, precision, AUPRC and AUROC are not (no
negatives inside it). Each head is therefore evaluated with the subgroup as
the positive class and the frozen pool of test ECGs that are true-negative
for that head as the negatives (**653** non-AF for the AF head, **683**
non-LVH for the LVH head). Consequences, declared in the tables:

- specificity and FP/TN are **identical across the rows of one head** (same
  negatives, same frozen decisions);
- precision/AUPRC/AUROC with the 28-case positive class are **small-sample
  reads** — reported as asked, but the verdict reads are recall, the joint
  breakdown and the exact tests.
"""),

        md("## Cell 1 — unpack the harness (no installs needed)"),
        code(UNPACK),

        md("## Cell 2 — mount Drive and verify the frozen reference is there"),
        code(MOUNT),

        md("""
## Cell 3 — THE ANALYSIS ⚠️ the deliverable (~10 s, CPU)

Refuses to print anything unless the frozen predictions match the reference
(cross-checks listed in the notebook header). Then prints the per-head
tables, the joint breakdown, the comparisons with exact tests and CIs, and
the verdict. Everything is also written to
`cardiosentry_runs/af_lvh_joint_v1/` on Drive.
"""),
        code(ANALYSIS),

        md("## Cell 4 — the verdict, isolated"),
        code(VERDICT),

        md("## Cell 5 — morning check"),
        code(MORNING),

        md(f"""
## Where the results live

```
{RUNS}/
├── STATUS.txt               DONE
├── report.txt               the whole analysis + verdict      <- read this
├── summary.json             model, thresholds, tests, verdict
├── af_head_rows.csv         AF head per subgroup (TP/FP/FN/TN + metrics)
├── lvh_head_rows.csv        LVH head per subgroup
├── joint_breakdown.csv      4 joint categories + bootstrap CIs
├── recalls.csv              recall point estimates + CIs
└── recall_deltas.csv        AF+LVH vs AF-only / LVH-only deltas + CIs
```

All on Drive, so it survives the runtime being recycled. The frozen
`hybrid_v1/` and `af_lvh_subgroup_v1/` folders are untouched — this round
only reads them.

**Read in this order:** report.txt → cross-checks → the two per-head tables →
the joint breakdown → the comparisons → the verdict → the limitation
paragraph.
"""),
    ]
    return {"cells": cells, **_meta()}


def _meta() -> dict:
    return {"metadata": {"accelerator": "None",
                         "colab": {"provenance": [], "gpuType": "T4"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def validate(nb: dict, name: str) -> None:
    """Catch the newline bug and any broken cell BEFORE the file is written."""
    import ast
    problems = []
    for i, c in enumerate(nb["cells"]):
        src = c["source"]
        if not isinstance(src, list):
            problems.append(f"cell {i}: source is not a list")
            continue
        for j, line in enumerate(src[:-1]):
            if not line.endswith("\n"):
                problems.append(f"cell {i} line {j}: missing trailing newline "
                                f"-> Colab would concatenate it with the next line")
        if c["cell_type"] == "code":
            joined = "".join(src)
            def _neutralise(line: str) -> str:
                body = line.lstrip()
                if body.startswith(("!", "%")):
                    return line[:len(line) - len(body)] + "pass"
                return line
            py = "\n".join(_neutralise(l) for l in joined.split("\n"))
            try:
                ast.parse(py)
            except SyntaxError as e:
                problems.append(f"cell {i}: python does not parse: {e}")
    if problems:
        raise SystemExit(f"{name} VALIDATION FAILED:\n  - " + "\n  - ".join(problems))


if __name__ == "__main__":
    name = "AF_LVH_Joint_Analysis.ipynb"
    print(f"embedding {len(FILES)} files ({len(B64)/1024:.0f} KB base64)\n")
    nb = notebook()
    validate(nb, name)
    out = HERE / name
    out.write_text(json.dumps(nb, indent=1))
    print(f"  {name:<48}{out.stat().st_size/1024:>7.0f} KB  "
          f"{len(nb['cells']):>3} cells")
    print("\nupload this one file to Colab - the code travels inside it.")
    print("it reads hybrid_v1/resnet50_hires/ (read-only) and writes to")
    print("cardiosentry_runs/af_lvh_joint_v1/.")
