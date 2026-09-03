#!/usr/bin/env python3
"""
Build the SELF-CONTAINED Colab notebook for the LVH binary negative-transfer test.

    python make_notebooks.py

One notebook, because the round is one run (~40 min on an L4) plus its
comparison. The harness is embedded in cell 1 as a base64 tar.gz, exactly like
the hybrids round; every later cell is a command that runs those files.

THE NOTEBOOK REUSES THE ROUND-2 DATASET

No 0_Prepare step this time. The binary model must read the SAME bytes the
four-head reference read, so the notebook extracts `prototype_1152_768.tar` —
the copy round 2's 0_Prepare built once — and refuses to fall back to anything
else. Rebuilding a fresh 768 copy here would add a second JPEG generation
between the two runs and give the delta a place to hide.

RESULTS LAND IN THEIR OWN ROUND FOLDER

    cardiosentry_runs/lvh_binary_v1/resnet50_lvh_binary/

The hybrid_v1 folder stays untouched. compare.py reads the binary run from the
new folder and the four-head reference from hybrid_v1 (with the frozen
baseline.py transcription as fallback and cross-check).

Re-run this script after editing any module to refresh the embedded copy.
"""
from __future__ import annotations
import base64, io, json, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = sorted([p for p in HERE.glob("*.py") if p.name != "make_notebooks.py"]) + \
        [HERE / "requirements.txt", HERE / "run_lvh_binary.sh"]

# ---------------------------------------------------------------- Drive layout
#
# Same root as the hybrids round - the notebook only adds one folder:
#
#   MyDrive/CardioSentry/
#     cardiosentry_dataset/            prototype_1152.tar + prototype_1152_768.tar
#     cardiosentry_runs/
#       v1/                            round 1 (five backbones)
#       hybrid_v1/                     round 2 - THE FOUR-HEAD REFERENCE lives here
#       lvh_binary_v1/                 THIS round
DRIVE_ROOT = "/content/drive/MyDrive/CardioSentry"
RUN_NAME = "lvh_binary_v1"

DRIVE_DATASETS = f"{DRIVE_ROOT}/cardiosentry_dataset"
SRC_TAR = f"{DRIVE_DATASETS}/prototype_1152.tar"
TAR_768 = f"{DRIVE_DATASETS}/prototype_1152_768.tar"
DRIVE_HF_CACHE = f"{DRIVE_ROOT}/hf_cache"
RUNS = f"{DRIVE_ROOT}/cardiosentry_runs/{RUN_NAME}"
HYBRID_RUNS = f"{DRIVE_ROOT}/cardiosentry_runs/hybrid_v1"
REFERENCE = f"{HYBRID_RUNS}/resnet50_hires"


def payload() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in FILES:
            if f.exists():
                tar.add(f, arcname=f.name)
    return base64.b64encode(buf.getvalue()).decode()


def _src(text: str) -> list[str]:
    """nbformat wants a list of lines WITH their trailing newlines.

    Splitting on "\n" and discarding them makes Colab concatenate every line
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
os.chmod("/content/code/run_lvh_binary.sh", 0o755)
print("unpacked:")
for f in sorted(os.listdir("/content/code")):
    print("   ", f)

!pip install -q timm
!nvidia-smi --query-gpu=name,memory.total --format=csv

# What this runtime actually is. vCPU count is the number that matters most:
# the 768px corpus is JPEG-decode-bound, and decode scales with cores, not RAM.
import os, shutil
_cores = os.cpu_count()
_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
try:
    _disk = shutil.disk_usage("/content").free / 1024**3
except OSError:
    _disk = float("nan")
print(f"\\nvCPU {{_cores}}   RAM {{_ram:.0f}} GB   free disk {{_disk:.0f}} GB")
if _cores <= 2:
    print("  2 vCPU -> decode-bound at ~27 img/s. On Colab Pro switch to L4:")
    print("  Runtime > Change runtime type > L4. ~40 min instead of ~2 h.")
else:
    print(f"  projected decode ceiling ~{{27 * min(_cores, 8) / 2:.0f}} img/s")
'''

MOUNT = f'''
import os
from google.colab import drive
drive.mount('/content/drive')

# Cache pretrained weights on Drive (~1.4 GB, once). Without this every runtime
# reset re-downloads them from HuggingFace.
os.environ["HF_HOME"] = "{DRIVE_HF_CACHE}"

# HuggingFace token - raises Hub rate limits. Stored in Colab Secrets, never in
# this notebook.
try:
    from google.colab import userdata
    _tok = userdata.get("HF_TOKEN")
    if _tok:
        os.environ["HF_TOKEN"] = _tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _tok
        print("HF token loaded from Colab Secrets")
    else:
        print("HF_TOKEN secret is empty - continuing unauthenticated (works fine)")
except Exception as exc:
    print(f"no HF_TOKEN in Colab Secrets ({{type(exc).__name__}}) - "
          f"continuing unauthenticated (works fine)")

# ---- Drive layout. Edit DRIVE_ROOT only; everything else follows. ----------
DRIVE_ROOT = "{DRIVE_ROOT}"
RUN_NAME   = "{RUN_NAME}"

DATASETS     = f"{{DRIVE_ROOT}}/cardiosentry_dataset"
SRC_TAR      = f"{{DATASETS}}/prototype_1152.tar"      # the original 1686x1311 corpus
TAR_768      = f"{{DATASETS}}/prototype_1152_768.tar"  # built once by the hybrids 0_Prepare
RUNS         = f"{{DRIVE_ROOT}}/cardiosentry_runs/{{RUN_NAME}}"
HYBRID_RUNS  = f"{{DRIVE_ROOT}}/cardiosentry_runs/hybrid_v1"
REFERENCE    = f"{{HYBRID_RUNS}}/resnet50_hires"       # the four-head model to beat

# ALWAYS local. Training off the mounted Drive turns every image read into a
# network round trip and starves the GPU.
DATA = "/content/prototype_1152_768"

os.makedirs(RUNS, exist_ok=True)
for _p, _label in ((SRC_TAR, "source corpus"), (TAR_768, "768px corpus"),
                   (REFERENCE + "/result.json", "four-head reference")):
    print(f"  {{'FOUND ' if os.path.exists(_p) else 'missing'}}  {{_label:<20}} {{_p}}")
print("\\nDATA =", DATA, "\\nRUNS =", RUNS)
if not os.path.exists(REFERENCE + "/result.json"):
    print("\\nNOTE: the four-head reference result.json was not found under")
    print("hybrid_v1/. compare.py will fall back to the FROZEN transcription in")
    print("baseline.py, but the paired bootstrap needs the reference's")
    print("test_predictions.csv - copy hybrid_v1/ back to Drive if it is missing.")
'''

EXTRACT_768 = '''
# Extract the 768px copy built by the hybrids round's 0_Prepare notebook.
#
# DO NOT rebuild a fresh copy here. The binary model must read the SAME bytes
# the four-head reference read - same JPEGs, same decode - or the comparison
# acquires a second JPEG generation and the delta a place to hide.
#
# Every step is checked: a failed shell command does NOT stop a Colab cell, so
# an unchecked `!tar -xf` prints a traceback into the void and the cell sails on.
import os, subprocess

EXPECTED_FILES = 6670          # images; index.csv makes 6671


def _images_present(root):
    if not os.path.isdir(root):
        return 0
    return sum(len([f for f in files if f.endswith(".jpg")])
               for _, _, files in os.walk(root))


def _have_dataset(root):
    return os.path.exists(os.path.join(root, "index.csv")) and \
           _images_present(root) >= EXPECTED_FILES


if _have_dataset(DATA):
    print("already extracted:", DATA)
elif not os.path.exists(TAR_768):
    raise SystemExit(
        f"MISSING: {TAR_768}\\n"
        f"This notebook reuses round 2's 768px corpus on purpose (identical\\n"
        f"bytes for both models). Run the hybrids round's 0_Prepare notebook\\n"
        f"first to build it, or re-upload the tar to\\n{DATASETS}/")
else:
    gb = os.path.getsize(TAR_768) / 1024**3
    print(f"extracting {TAR_768}  ({gb:.2f} GB)")

    # Verify the archive BEFORE trusting it. prepare_dataset.py copied index.csv
    # LAST, so a truncated tar loses exactly that file - check it exists.
    try:
        with __import__("tarfile").open(TAR_768) as _t:
            _names = _t.getnames()
        print(f"  archive intact: {len(_names)} entries")
        _has_index = any(n.endswith("index.csv") for n in _names)
    except Exception as exc:
        raise SystemExit(
            f"\\nTHE TAR ON DRIVE IS TRUNCATED OR CORRUPT: {type(exc).__name__}: {exc}\\n"
            f"  size on Drive: {gb:.2f} GB\\n"
            f"  Delete it and re-run the hybrids round's 0_Prepare notebook:\\n"
            f"      !rm -f '{TAR_768}'")

    r = subprocess.run(["tar", "-xf", TAR_768, "-C", "/content/"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"tar failed (exit {r.returncode}):\\n{r.stderr[:2000]}")
    print("  extracted ->", DATA)

    if not _has_index and _images_present(DATA) >= EXPECTED_FILES:
        print("  index.csv missing from the archive but the images are all here")
        print("  recovering it from the original tar (copied verbatim by")
        print("  prepare_dataset.py; rel_path is unchanged by design)")
        import shutil, tempfile, tarfile
        with tarfile.open(SRC_TAR) as _st:
            _member = next((m for m in _st.getnames()
                            if m.endswith("index.csv")), None)
            if _member is None:
                raise SystemExit(f"no index.csv inside {SRC_TAR} either - "
                                 f"re-run the hybrids 0_Prepare from the original corpus.")
            with tempfile.TemporaryDirectory() as _tmp:
                _st.extract(_member, _tmp)
                shutil.copy2(os.path.join(_tmp, _member),
                             os.path.join(DATA, "index.csv"))
        print(f"  recovered from {_member}")

n_img = _images_present(DATA)
print(f"\\nimages on disk: {n_img:,}")
if not os.path.exists(os.path.join(DATA, "index.csv")):
    raise SystemExit(
        f"no index.csv at {DATA} even after extraction.\\n"
        f"images found: {n_img:,} of {EXPECTED_FILES:,}\\n"
        f"If the image count is right, only index.csv is missing - recover it with:\\n"
        f"    !tar -xf '{SRC_TAR}' -C /content/ prototype_1152/index.csv\\n"
        f"    !cp /content/prototype_1152/index.csv '{DATA}/index.csv'\\n"
        f"Otherwise delete the tar on Drive and re-run the hybrids 0_Prepare.")
if n_img < EXPECTED_FILES:
    print(f"WARNING: expected {EXPECTED_FILES:,} images, found {n_img:,}. "
          f"The archive was probably truncated - preflight will fail the "
          f"integrity check.")

from PIL import Image
import pandas as pd
_df = pd.read_csv(os.path.join(DATA, "index.csv"))
print(f"index.csv: {len(_df):,} rows")
with Image.open(os.path.join(DATA, _df.rel_path.iloc[0])) as _im:
    print("stored image size:", _im.size)
    assert min(_im.size) >= 768, (
        f"images are {_im.size} - this is the round-1 384px copy, not the "
        f"768px one; the binary model must see the same bytes the four-head "
        f"reference saw.")
print("\\ndataset OK")
!df -h /content/drive/MyDrive | tail -1
'''

MORNING = '''
!cat "$RUNS/_SUMMARY.txt"
'''

FAILCHECK = '''
# did anything fail?
!for f in "$RUNS"/*/STATUS.txt; do echo "--- $f"; head -2 "$f"; done
!ls "$RUNS"/*/ERROR.txt 2>/dev/null && echo "^ failures above" || echo "no failures"
'''


def header() -> dict:
    names = ", ".join(f.name for f in FILES if f.exists())
    return md(f"""
# LVH binary — negative-transfer test

**The question:** is the shared multi-label objective (STEMI, AF, LVH, NORMAL)
itself holding LVH back?

**The test:** train a ResNet-50 at the same **768×768**, on **LVH only**, as a
binary task — same split, same bytes, same augmentation, same optimiser,
same protocol as the four-head reference (`resnet50_hires`, round 2), which
scored **PTB-XL LVH AUPRC 0.9315** (pooled 0.9248). If the binary model beats
it, that is evidence of **negative transfer**.

**Run the cells in order, top to bottom.**
**Before starting: Runtime → Change runtime type → L4 if you have it.**

One run, micro-batch 8 (the same value the reference used — BatchNorm
normalises over the micro-batch, so do not change it):

| runtime | vCPU | expected |
| --- | ---: | --- |
| T4 standard | 2 | ~1.5–2 h (decode-bound) |
| **L4 high-RAM** | 12 | **~40 min** |

The whole harness ({names}) is embedded in cell 1 — no folder upload needed.
""")

def notebook() -> dict:
    cells = [
        header(),

        md("""
## What is held constant, and what is the experiment

| held constant | what |
| --- | --- |
| splits, corpus, view | the frozen `index.csv`, the SAME `prototype_1152_768` folder, 768×768 |
| preprocessing | ImageNet normalisation, ±10% brightness/contrast only |
| training | AdamW 1e-4, cosine + 2 warmup epochs, effective batch 32, micro-batch **8**, 20 epochs, patience 5, seed 42 |
| loss | `BCEWithLogitsLoss` + pos_weight (LVH = 2.5524, the same number) |
| selection | monitor `macro_auprc` — over one head this IS val LVH AUPRC |
| evaluation | F1-optimal threshold on val, frozen for test; per-source breakdown |

**The one declared difference is the task:** the shared 2-layer MLP head keeps
its exact structure and only its final width changes 512×4 → 512×1.

**Primary read: PTB-XL LVH AUPRC.** The STEMI database contributes 146 images
and zero LVH positives, so pooled gains can be database recognition wearing an
LVH costume. PTB-XL carries 234 of the 255 positives — the shortcut cannot
reach it.

`compare.py` verifies the shared protocol against the reference field by field
and refuses to print deltas if anything else changed.
"""),

        md("## Cell 1 — unpack the code and install timm"),
        code(UNPACK),

        md("""
## Cell 2 — mount Drive

Add your HuggingFace token first if you have not (🔑 key icon in the left
sidebar → **+ Add new secret** → name `HF_TOKEN` → toggle notebook access on).
It only raises Hub rate limits; the run works without it.
"""),
        code(MOUNT),

        md("""
## Cell 3 — extract the 768px dataset (~1 min)

Built once by the hybrids round's **0_Prepare** notebook. This cell refuses to
fall back to the 384px copy: the binary model must read the same bytes the
four-head reference read, or the delta acquires a second JPEG generation.
"""),
        code(EXTRACT_768),

        md("""
## Cell 4 — preflight ⚠️ do not skip

GPU, dataset integrity, the 768px stored-size check, one real loader batch,
then the model built and run through a real forward+backward at micro-batch 8,
reporting peak VRAM.
"""),
        code('''
%cd /content/code
!python preflight.py --data "$DATA"
'''),

        md("""
## Cell 5 — train  ⏳ ~40 min on an L4 (~2 h on a 2-vCPU T4)

Writes to Drive as it goes, so a disconnect does not lose finished work, and
re-running resumes from the last completed epoch.

`MICRO_BATCH=8` is **not** a knob to tweak for speed: BatchNorm normalises
over the micro-batch, and the reference ran at 8.
"""),
        code('''
%cd /content/code
!MICRO_BATCH=8 ./run_lvh_binary.sh "$DATA" "$RUNS"
'''),

        md("## Cell 6 — progress check"),
        code(MORNING),
        code(FAILCHECK),

        md("""
## Cell 7 — THE COMPARISON  ⚠️ the deliverable

Binary vs four-head, all requested metrics (AUPRC, AUROC, sensitivity,
specificity, FPR, precision, F1) — pooled AND PTB-XL — with deltas, paired
bootstrap CIs and the negative-transfer verdict.

`compare.py` first proves the shared protocol matched the reference (live
result.json under `hybrid_v1/`, cross-checked against the frozen transcription
in `baseline.py`), then refuses to print deltas if anything else differs.
"""),
        code('''
%cd /content/code
!python compare.py --runs "$RUNS" --reference "$HYBRID_RUNS"
'''),

        md("""
## Cell 8 — the verdict, isolated

The one paragraph that decides the round. `comparison.txt` is rewritten on
every compare, so re-run this cell after cell 7 rather than reading a stale
copy.
"""),
        code('''
%cd /content/code
!sed -n "/VERDICT - does this support/,/CONTEXT ROWS/p" "$RUNS/comparison.txt"
'''),

        md("""
## Cell 9 — learning curves

Val LVH AUPRC per epoch, with the reference's TEST number as the bar. The
dashed line is a test metric on the same split, so do not read a val curve
crossing it as "beaten" — read it as the level a val curve has to reach and
hold for the test delta to be plausible.
"""),
        code('''
import json, glob
import matplotlib.pyplot as plt

REF_POOLED = 0.9248      # four-head resnet50_hires, pooled TEST LVH AUPRC
REF_PTBXL  = 0.9315      # four-head resnet50_hires, PTB-XL TEST LVH AUPRC

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for f in sorted(glob.glob(f"{RUNS}/*/result.json")):
    r = json.load(open(f))
    h = r["history"]
    if not h:
        continue
    ep = [x["epoch"] for x in h]
    ax[0].plot(ep, [x["train_loss"] for x in h], label=r["model"])
    ax[1].plot(ep, [x.get("val_macro_auprc") for x in h], label="val LVH AUPRC")
    ax[1].plot(ep, [x.get("val_macro_auroc") for x in h], label="val LVH AUROC",
               ls="--", alpha=.7)
ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].set_title("validation LVH AUPRC / AUROC (macro over one head)")
ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
ax[1].axhline(REF_POOLED, ls=":", c="0.5", lw=1)
ax[1].annotate(f"reference pooled test {REF_POOLED}", (0.02, REF_POOLED),
               xycoords=("axes fraction", "data"), fontsize=7, va="bottom", color="0.4")
ax[1].axhline(REF_PTBXL, ls="--", c="crimson", lw=1.2)
ax[1].annotate(f"reference PTB-XL test {REF_PTBXL} - THE BAR", (0.02, REF_PTBXL),
               xycoords=("axes fraction", "data"), fontsize=7, va="bottom",
               color="crimson")
plt.tight_layout(); plt.show()

print("val curves vs test bars: same split, different sets - the bar is where")
print("a val curve has to LIVE for the test delta to be plausible, not a")
print("line the val curve has to cross.")
'''),

        md("""
## Cell 10 — full report for the binary run
"""),
        code('''
MODEL = "resnet50_lvh_binary"
print(open(f"{RUNS}/{MODEL}/report.txt").read())
'''),

        md(f"""
## Where the results live

```
{RUNS}/
├── _SUMMARY.txt              the run at a glance, delta vs the four-head reference
├── comparison.txt            binary vs four-head, CIs, verdict  <- read this
└── resnet50_lvh_binary/
    ├── STATUS.txt            RUNNING | DONE | FAILED
    ├── report.txt            full metrics + per-source breakdown
    ├── train_log.txt         complete console output
    ├── history.csv           per epoch
    ├── metrics.csv
    ├── result.json
    ├── config_snapshot.json  protocol/task/shared fingerprints
    ├── weights/best.pt       best-on-val weights + frozen threshold
    └── predictions/          per-image probabilities, for the paired bootstrap
```

All on Drive, so it survives the runtime being recycled.

**Read in this order:** `comparison.txt` → the PTB-XL table → the bootstrap
table → the verdict → `_SUMMARY.txt` for the morning glance.
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
    name = "LVH_Binary_NegativeTransfer.ipynb"
    print(f"embedding {sum(1 for f in FILES if f.exists())} files "
          f"({len(B64)/1024:.0f} KB base64)\n")
    nb = notebook()
    validate(nb, name)
    out = HERE / name
    out.write_text(json.dumps(nb, indent=1))
    print(f"  {name:<48}{out.stat().st_size/1024:>7.0f} KB  "
          f"{len(nb['cells']):>3} cells")
    print("\nupload this one file to Colab - the code travels inside it.")
    print("it writes to cardiosentry_runs/lvh_binary_v1/ and reads the")
    print("four-head reference from cardiosentry_runs/hybrid_v1/.")
