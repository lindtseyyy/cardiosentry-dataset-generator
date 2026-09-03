#!/usr/bin/env python3
"""
Build the SELF-CONTAINED Colab notebook for the AF x LVH subgroup analysis.

    python make_notebook.py

The harness is embedded in cell 1 as a base64 tar.gz (the project convention),
so one file travels to Colab with everything it needs. The analysis itself is
pure pandas/numpy over 938 rows and runs on any CPU runtime in seconds; the
OPTIONAL weights-verification cell reloads the saved best.pt and re-runs the
test split to prove the frozen predictions file matches the saved checkpoint.

DRIVE LAYOUT (unchanged from round 2):

    MyDrive/CardioSentry/
      cardiosentry_dataset/            prototype_1152.tar + prototype_1152_768.tar
      cardiosentry_runs/
        hybrid_v1/resnet50_hires/      result.json, weights/best.pt, predictions/
        af_lvh_subgroup_v1/            THIS round writes here

Re-run this script after editing any module to refresh the embedded copy.
"""
from __future__ import annotations
import base64, io, json, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = [HERE / n for n in ("config.py", "baseline.py", "subgroups.py",
                            "metrics.py", "bootstrap.py", "model_verify.py",
                            "analyze.py")]

DRIVE_ROOT = "/content/drive/MyDrive/CardioSentry"
RUN_NAME = "af_lvh_subgroup_v1"

DATASETS = f"{DRIVE_ROOT}/cardiosentry_dataset"
SRC_TAR = f"{DATASETS}/prototype_1152.tar"
TAR_768 = f"{DATASETS}/prototype_1152_768.tar"
RUNS = f"{DRIVE_ROOT}/cardiosentry_runs/{RUN_NAME}"
HYBRID_RUNS = f"{DRIVE_ROOT}/cardiosentry_runs/hybrid_v1"
REFERENCE = f"{HYBRID_RUNS}/resnet50_hires"
DATA = "/content/prototype_1152_768"


def payload() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in FILES:
            tar.add(f, arcname=f.name)
    return base64.b64encode(buf.getvalue()).decode()


def _src(text: str) -> list[str]:
    """nbformat wants a list of lines WITH their trailing newlines.

    Splitting on "\\n" and discarding them makes Colab concatenate every line
    into one. Kept from the hybrids generator verbatim.
    """
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
print("\\nno pip installs needed for the analysis - pandas/sklearn/numpy are")
print("preinstalled in Colab. (timm is installed only by the optional")
print("weights-verification cell, pinned to the reference version.)")
'''

MOUNT = f'''
import os
from google.colab import drive
drive.mount('/content/drive')

# ---- Drive layout. Edit DRIVE_ROOT only; everything else follows. ----------
DRIVE_ROOT = "{DRIVE_ROOT}"
RUN_NAME   = "{RUN_NAME}"

DATASETS    = f"{{DRIVE_ROOT}}/cardiosentry_dataset"
SRC_TAR     = f"{{DATASETS}}/prototype_1152.tar"        # round-1 original corpus
TAR_768     = f"{{DATASETS}}/prototype_1152_768.tar"    # the frozen 768px corpus
HYBRID_RUNS = f"{{DRIVE_ROOT}}/cardiosentry_runs/hybrid_v1"
REFERENCE   = f"{{HYBRID_RUNS}}/resnet50_hires"         # the four-head model
RUNS        = f"{{DRIVE_ROOT}}/cardiosentry_runs/{{RUN_NAME}}"   # THIS round writes here

os.makedirs(RUNS, exist_ok=True)

print("Drive layout check:")
for _p, _label in (
        (REFERENCE + "/predictions/test_predictions.csv", "frozen test predictions"),
        (REFERENCE + "/result.json",                     "reference result.json"),
        (REFERENCE + "/weights/best.pt",                 "reference weights"),
        (TAR_768,                                        "768px corpus tar"),
        (SRC_TAR,                                        "original corpus tar (index fallback)")):
    print(f"  {{'FOUND ' if os.path.exists(_p) else 'missing'}}  {{_label:<28}} {{_p}}")

_need = REFERENCE + "/predictions/test_predictions.csv"
if not os.path.exists(_need):
    raise SystemExit(
        f"\\nMISSING the frozen predictions file:\\n  {{_need}}\\n"
        f"This analysis reads the EXACT file the reference numbers were\\n"
        f"computed from - copy the hybrid_v1/resnet50_hires/ run folder back\\n"
        f"to Drive if it is missing. (The weights alone are not enough: the\\n"
        f"analysis must rest on the frozen evaluation, not a re-run.)")
'''

GET_INDEX = '''
# Get index.csv - the corpus manifest with patient ids and whole-corpus
# labels. The ANALYSIS only needs this one file (938 test rows + the corpus
# counts); the optional weights cell later wants the whole 768px corpus.
#
# Resolution order: an already-extracted corpus -> single-member extraction
# from prototype_1152_768.tar -> single-member extraction from the round-1
# prototype_1152.tar (index rows are identical; prepare_dataset.py copied it
# verbatim). Single-member extraction takes seconds - no images are pulled.
import os, tarfile, tempfile, shutil

INDEX_CSV = "/content/index.csv"
EXPECTED_ROWS = 6670


def _find_member(tar_path, suffix="index.csv"):
    with tarfile.open(tar_path) as t:
        for m in t.getmembers():
            if m.name.endswith(suffix):
                return m
    return None


def _extract_member(tar_path, member, dest):
    with tarfile.open(tar_path) as t:
        with tempfile.TemporaryDirectory() as tmp:
            t.extract(member, tmp)
            src = os.path.join(tmp, member.name)
            shutil.copy2(src, dest)
    print(f"  extracted index.csv from {tar_path}")


if os.path.exists("/content/prototype_1152_768/index.csv"):
    shutil.copy2("/content/prototype_1152_768/index.csv", INDEX_CSV)
    print("index.csv: copied from the already-extracted 768px corpus")
elif os.path.exists(TAR_768):
    _m = _find_member(TAR_768)
    if _m is None:
        raise SystemExit(f"no index.csv inside {TAR_768} - re-run the hybrids "
                         f"0_Prepare notebook")
    _extract_member(TAR_768, _m, INDEX_CSV)
elif os.path.exists(SRC_TAR):
    _m = _find_member(SRC_TAR)
    if _m is None:
        raise SystemExit(f"no index.csv inside {SRC_TAR} - re-run the round-1 "
                         f"prepare notebook")
    _extract_member(SRC_TAR, _m, INDEX_CSV)
else:
    raise SystemExit(
        f"neither {TAR_768} nor {SRC_TAR} exists on Drive.\\n"
        f"Re-upload one of them to {DATASETS}/ - only the index.csv inside is "
        f"needed for the analysis.")

import pandas as pd
_df = pd.read_csv(INDEX_CSV)
print(f"index.csv: {len(_df):,} rows")
assert len(_df) == EXPECTED_ROWS, \
    f"index.csv has {len(_df)} rows, expected {EXPECTED_ROWS}"
assert len(_df[_df.split == 'test']) == 938, "test split is not 938 rows"
need = {"record", "split", "source", "patient_id", "LVH", "AF", "STEMI"}
assert need <= set(_df.columns), f"index.csv missing columns {need - set(_df.columns)}"
print("index.csv OK - test split 938 rows, all required columns present")
'''

ANALYSIS = '''
%cd /content/code
!python analyze.py \
    --predictions "$REFERENCE/predictions/test_predictions.csv" \
    --result-json  "$REFERENCE/result.json" \
    --index        "$INDEX_CSV" \
    --out          "$RUNS"
'''

VERIFY_WEIGHTS = '''
# OPTIONAL - but recommended once. Proves the frozen predictions file the
# analysis rests on matches the saved weights, by reloading best.pt and
# re-running the 938-image test split through the same eval pipeline.
#
# Needs: the 768px corpus (this cell extracts it, ~1-2 min) and timm
# (pinned to the reference version). A GPU makes it ~10 s of inference;
# without one it falls back to CPU and takes a few minutes - same math.
!pip install -q "timm==1.0.28"

import os, subprocess, tarfile

DATA = "/content/prototype_1152_768"
EXPECTED_FILES = 6670


def _images_present(root):
    if not os.path.isdir(root):
        return 0
    return sum(len([f for f in files if f.endswith(".jpg")])
               for _, _, files in os.walk(root))


def _have_dataset(root):
    return os.path.exists(os.path.join(root, "index.csv")) and \\
           _images_present(root) >= EXPECTED_FILES


if not _have_dataset(DATA):
    if not os.path.exists(TAR_768):
        raise SystemExit(
            f"MISSING: {TAR_768}\\n"
            f"Run the hybrids round's 0_Prepare notebook first, or re-upload\\n"
            f"the tar to {DATASETS}/.")
    print(f"extracting {TAR_768} ({os.path.getsize(TAR_768)/1024**3:.2f} GB)")
    with tarfile.open(TAR_768) as _t:
        _names = _t.getnames()
    print(f"  archive intact: {len(_names)} entries")
    r = subprocess.run(["tar", "-xf", TAR_768, "-C", "/content/"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"tar failed (exit {r.returncode}):\\n{r.stderr[:2000]}")
    print("  extracted ->", DATA)

n_img = _images_present(DATA)
print(f"images on disk: {n_img:,}")
if n_img < EXPECTED_FILES:
    print(f"WARNING: expected {EXPECTED_FILES:,} images, found {n_img:,} -")
    print("the tar on Drive is probably truncated; re-run 0_Prepare.")

%cd /content/code
!python model_verify.py \
    --ckpt  "$REFERENCE/weights/best.pt" \
    --data  "$DATA" \
    --frozen "$REFERENCE/predictions/test_predictions.csv" \
    --out   "$RUNS/weights_verification.json"
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
# AF × LVH subgroup analysis — does AF break the LVH head?

**The question:** LVH is the worst of the four heads of the current
**ResNet-50@768** (`resnet50_hires`: STEMI AUPRC 0.9979, AF 0.9825, NORMAL
0.9523, **LVH 0.9248** pooled / 0.9315 PTB-XL). Does the LVH head
deteriorate specifically on ECGs that also carry **AF**, and is the AF+LVH
subgroup dragging the pooled LVH number down?

**What this notebook does — nothing is trained, re-thresholded or
re-predicted:**

1. reads the **frozen per-image test predictions** of `resnet50_hires` (the
   exact file its reported numbers were computed from) and cross-checks them
   against the frozen transcription and the live `result.json`;
2. stratifies the 938-image test split by **ground-truth** labels into four
   subgroups — **LVH only** (no AF, no STEMI), **AF + LVH**, **STEMI + LVH**,
   **AF + STEMI + LVH** — and counts each, with LVH prevalence per source;
3. scores the LVH head **inside each subgroup** at the frozen 0.47 threshold:
   sensitivity, specificity, FPR, precision, F1, AUPRC (+ AUROC and
   sensitivity at FPR ≤ 0.08 as the prevalence-independent detector reads);
4. bootstraps the AF-vs-clean delta (1,200 independent resamples per group)
   and applies the **verdict rule stated below — before any number**;
5. (optional cell) reloads the **weights from Drive** and re-runs the test
   split, to prove the frozen predictions file matches the saved checkpoint.

**Runtime:** the analysis is pure pandas/numpy over 938 rows — any Colab
runtime, ~30 s. The optional weights cell wants a GPU (any card) and the
768px corpus (~2 min to extract).

**The verdict rule (frozen, applied as stated):** the two prevalence-
independent columns — **AUROC delta** and **sensitivity @ FPR ≤ 0.08 delta**
— decide: SUPPORTED if both point estimates are negative with ≥ 1 CI entirely
below 0; PARTIALLY if exactly one CI is entirely below 0; NOT SEPARATED if
the CIs include 0; NO EVIDENCE if the point estimates are ≥ 0. AUPRC,
precision and F1 are reported with CIs but never decide: their floor is the
subgroup's LVH prevalence (44.8% clean vs 9.8% AF in test). Whatever the
verdict, it is an **association on this test split, not a causal claim** —
the confound ledger is printed beside it.

The whole harness ({names}) is embedded in cell 1 — no folder upload needed.
""")

def notebook() -> dict:
    cells = [
        header(),

        md("""
## The subgroups — defined by ground truth, never by predictions

| subgroup | definition | why the name |
| --- | --- | --- |
| **LVH only** | true_AF=0 AND true_STEMI=0 | the clean stratum (its LVH-negatives are exactly the NORMAL sheets) |
| **AF + LVH** | true_AF=1 | the stratum under investigation |
| **STEMI + LVH** | true_STEMI=1 | degenerate by construction — see below |
| **AF + STEMI + LVH** | true_AF=1 AND true_STEMI=1 | does not exist in this corpus |

Using the model's own AF predictions to define "AF ECGs" would let a false
AF call drag an LVH error into the AF stratum (or a missed AF call hide one)
— the true labels are the only boundaries that do not contaminate the LVH
measurement.

**Structural zeros (corpus facts):** STEMI is exactly `source == "STEMI"`
and that database carries zero LVH positives, so STEMI+LVH and AF+STEMI+LVH
ECGs do not exist; AF and STEMI never co-occur on any row. Those subgroups
contribute LVH false-positive counts only, and their LVH scores print as
`--`, not as a number that would read like a measurement.
"""),

        md("## Cell 1 — unpack the harness (no installs needed)"),
        code(UNPACK),

        md("## Cell 2 — mount Drive and verify the reference is where it should be"),
        code(MOUNT),

        md("""
## Cell 3 — get index.csv (~5 s, no images)

The analysis needs the manifest for patient ids and corpus-wide counts. It
takes `index.csv` from the already-extracted 768px corpus if present,
otherwise extracts that one member from the tar on Drive.
"""),
        code(GET_INDEX),

        md("""
## Cell 4 — THE ANALYSIS ⚠️ the deliverable (~30 s, CPU)

Cross-checks the frozen reference first (refuses on any disagreement), then
prints: the four subgroup counts, LVH prevalence per source, the LVH head
inside each subgroup at the frozen threshold, the bootstrap delta table with
CIs, and the verdict with its confound ledger.

Everything is also written to Drive under `cardiosentry_runs/af_lvh_subgroup_v1/`.
"""),
        code(ANALYSIS),

        md("""
## Cell 5 — OPTIONAL: prove the predictions match the saved weights

Reloads `best.pt` from Drive (the current model's weights), rebuilds the
exact architecture, re-runs the 938-image test split through the same eval
pipeline (768px stored view, no resampling, batch 8, no TTA) and compares the
recomputed probabilities against the frozen file, per label.

- max|Δ| ~ 1e-6 → the frozen file **is** this checkpoint;
- up to ~1e-3 → floating-point/library drift, reported, frozen file stays authoritative;
- larger → **stop and investigate** — the file on Drive would not match the weights.

This cell cannot change an analysis number: the analysis always reads the
frozen file. It exists so the file is proven, not assumed.
"""),
        code(VERIFY_WEIGHTS),

        md("## Cell 6 — the verdict, isolated"),
        code(VERDICT),

        md("## Cell 7 — morning check"),
        code(MORNING),

        md(f"""
## Where the results live

```
{RUNS}/
├── STATUS.txt                     DONE
├── report.txt                     the whole analysis + verdict      <- read this
├── summary.json                   model, threshold, verdict, timestamp
├── subgroup_counts_test.csv       the four requested counts, test split
├── subgroup_counts_corpus.csv     same, whole 6,670-image corpus
├── lvh_prevalence_by_source_test.csv     LVH prevalence per source per subgroup
├── lvh_prevalence_by_source_corpus.csv
├── subgroup_metrics.csv           per-subgroup LVH metrics + rollups
├── bootstrap_deltas.csv           clean-vs-AF deltas with 95% CIs
├── patient_structure.csv          images/patients per subgroup
└── weights_verification.json      from the optional cell 5
```

All on Drive, so it survives the runtime being recycled.

**Read in this order:** report.txt → the counts tables → the per-subgroup
metrics table → the bootstrap delta table → the verdict → the confound
ledger. The verdict answers "does LVH deteriorate when AF is present, and is
it supported by the test data?" — the ledger states exactly what it is not
proof of.
"""),
    ]
    return {"cells": cells, **_meta()}


def _meta() -> dict:
    return {"metadata": {"accelerator": "GPU",
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
    name = "AF_LVH_Subgroup_Analysis.ipynb"
    print(f"embedding {len(FILES)} files ({len(B64)/1024:.0f} KB base64)\n")
    nb = notebook()
    validate(nb, name)
    out = HERE / name
    out.write_text(json.dumps(nb, indent=1))
    print(f"  {name:<48}{out.stat().st_size/1024:>7.0f} KB  "
          f"{len(nb['cells']):>3} cells")
    print("\nupload this one file to Colab - the code travels inside it.")
    print("it reads hybrid_v1/resnet50_hires/ and writes to")
    print("cardiosentry_runs/af_lvh_subgroup_v1/.")
