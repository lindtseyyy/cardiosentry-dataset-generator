#!/usr/bin/env python3
"""
Build a SELF-CONTAINED Colab notebook.

    python make_notebook.py

The harness is 11 Python modules. You do not paste modules into notebook cells -
that is not what cells are for. Cell 1 of the generated notebook carries the
whole package as a base64 tar.gz and unpacks it to /content/code; every later
cell is a one-line command that runs those files.

So the notebook is the ONLY file you upload. Re-run this script after editing any
module to refresh the embedded copy.
"""
from __future__ import annotations
import base64, io, json, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = sorted([p for p in HERE.glob("*.py") if p.name != "make_notebook.py"]) + \
        [HERE / "requirements.txt", HERE / "run_all.sh"]


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
    into one - `import os` + `os.makedirs(...)` becomes `import osos.makedirs(...)`.
    """
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": _src(s.strip())}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": _src(s.strip("\n"))}


def build() -> dict:
    b64 = payload()
    names = ", ".join(f.name for f in FILES if f.exists())

    cells = [
        md(f"""
# CardioSentry — controlled architecture comparison

Compares **VGG-16, ResNet-50, EfficientNet-B3, ConvNeXt-Base, ViT-B/384** with
data, preprocessing, training protocol and evaluation held identical, so any
difference between them is attributable to the architecture.

Four independent sigmoid heads: **STEMI, AF, LVH, NORMAL**.

**Run the cells in order, top to bottom.** Cells 1–4 are setup and take a few
minutes. Cell 5 is the long one.

Before starting: **Runtime → Change runtime type → GPU** (T4 minimum).

---
*The whole harness ({names}) is embedded in cell 1 — no folder upload needed.*
"""),

        md("## Cell 1 — unpack the code and install timm"),
        code(f'''
import base64, io, tarfile, os
os.makedirs("/content/code", exist_ok=True)
_PKG = "{b64}"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(_PKG)), mode="r:gz").extractall("/content/code")
os.chmod("/content/code/run_all.sh", 0o755)
print("unpacked:")
for f in sorted(os.listdir("/content/code")):
    print("   ", f)

!pip install -q timm
!nvidia-smi --query-gpu=name,memory.total --format=csv
'''),

        md("""
## Cell 2 — mount Drive, set up caching, extract the dataset

**Add your HuggingFace token first** (one-time, ~30 s):

1. Click the 🔑 **key icon** in Colab's left sidebar
2. **+ Add new secret** → Name: `HF_TOKEN` → Value: your `hf_...` token
3. Toggle **Notebook access** on

The cell reads it from there. Do not paste the token into a cell — it would be
saved inside the notebook file. If you skip this the run still works; the token
only raises Hub rate limits.

Extract to **local disk** (`/content/`), never train off the mounted Drive —
every image read would become a network round trip and starve the GPU.

Adjust `TAR` if your file is somewhere else in Drive.
"""),
        code('''
import os
from google.colab import drive
drive.mount('/content/drive')

# Cache pretrained weights on Drive (~1.35 GB, once). Without this every runtime
# reset re-downloads them from HuggingFace - slow, and it is what triggers the
# "unauthenticated requests" rate limit over a night of reconnects.
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"

# HuggingFace token - raises Hub rate limits and speeds up weight downloads.
# Stored in Colab Secrets, never in this notebook: a pasted token gets saved
# into the .ipynb and travels with any copy you share.
try:
    from google.colab import userdata
    _tok = userdata.get("HF_TOKEN")
    if _tok:
        os.environ["HF_TOKEN"] = _tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _tok   # older hub versions
        print("HF token loaded from Colab Secrets")
    else:
        print("HF_TOKEN secret is empty - continuing unauthenticated (works fine)")
except Exception as exc:
    print(f"no HF_TOKEN in Colab Secrets ({type(exc).__name__}) - "
          f"continuing unauthenticated (works fine)")

TAR  = "/content/drive/MyDrive/cardiosentry_dataset/prototype_1152.tar"
DATA = "/content/prototype_1152"
RUNS = "/content/drive/MyDrive/cardiosentry_runs"   # results land here

!tar -xf "$TAR" -C /content/
!ls "$DATA"
!mkdir -p "$RUNS"
print("\\nDATA =", DATA, "\\nRUNS =", RUNS)

# Drive space check - the overnight batch needs roughly 7 GB free.
!df -h /content/drive/MyDrive | tail -1
'''),

        md("""
## Cell 2b — pre-resize the images (~10 min, run ONCE)

The corpus stores 1686x1311 sheets, but training uses 384x384. Resizing on the
fly decodes 2.2 megapixels to produce 0.15 — every image, every epoch, every
model — which starves the GPU on Colab's 2 CPU cores (~21 img/s instead of
70-100).

Doing it once removes ~15x of decode work and shrinks the folder ~8x.
Fair for the comparison: resolution, interpolation and normalisation are
unchanged, and all five models use the same resized copy.

**Skip this cell only if `data-wait` in the training log is under ~15%.**
"""),
        code('''
%cd /content/code
!python resize_dataset.py --src "$DATA" --size 384

# A failed shell command does NOT stop a Colab cell, so verify the output really
# exists before switching DATA - otherwise every later cell points at nothing.
import os
_resized = DATA.rstrip("/") + "_384"
if os.path.exists(os.path.join(_resized, "index.csv")):
    DATA = _resized
    print()
    print("DATA is now", DATA)
    # Switching datasets invalidates earlier runs - clear them so nothing
    # resumes across the change or lands incomparable in the table.
    !rm -rf "$RUNS"/* 2>/dev/null
    print("cleared", RUNS)
else:
    print()
    print("RESIZE DID NOT COMPLETE - DATA left as", DATA)
    print("read the error above; do not continue until this is fixed")
'''),

        md("""
## Cell 3 — preflight (~2 min)

Checks the GPU, loads and integrity-checks the dataset, resolves all five timm
pretrained tags, and runs a real forward+backward per model reporting peak VRAM.

**Do not skip this.** It turns an out-of-memory crash at epoch 7 into a message
in two minutes.
"""),
        code('''
%cd /content/code
!python preflight.py --data "$DATA"
'''),

        md("""
## Cell 4 — smoke test (~2 min)

20 train / 10 eval batches. The metrics are meaningless — this exists only to
prove the whole path runs before you commit hours to it.
"""),
        code('''
!python train.py --model resnet50 --data "$DATA" --out /content/smoke --smoke
'''),

        md("""
## Cell 5 — train all five  ⏳ THE LONG ONE

Leave this running overnight. It writes everything to Drive as it goes, so a
disconnect does not lose finished work, and re-running resumes each model from
its last completed epoch.

`run_all.sh` does **not** abort if one model fails — the rest still run.
"""),
        code('''
!./run_all.sh "$DATA" "$RUNS"
'''),

        md("""
### Prefer one model at a time?

Use these instead of cell 5. Re-running a line after a disconnect resumes it.
"""),
        code('''
# !python train.py --model resnet50        --data "$DATA" --out "$RUNS"
# !python train.py --model efficientnet_b3 --data "$DATA" --out "$RUNS"
# !python train.py --model vgg16           --data "$DATA" --out "$RUNS"
# !python train.py --model convnext_base   --data "$DATA" --out "$RUNS"
# !python train.py --model vit_b_384       --data "$DATA" --out "$RUNS"
'''),

        md("""
## Cell 6 — morning check

Works even if the batch is only half finished.
"""),
        code('''
!cat "$RUNS/_SUMMARY.txt"
'''),
        code('''
# did anything fail overnight?
!for f in "$RUNS"/*/STATUS.txt; do echo "--- $f"; head -2 "$f"; done
!ls "$RUNS"/*/ERROR.txt 2>/dev/null && echo "^ failures above" || echo "no failures"
'''),

        md("""
## Cell 7 — comparison

`compare.py` first verifies the runs are actually comparable — matching config
fingerprints, same dataset, same row counts, no smoke runs — and **refuses to
rank them if not**.
"""),
        code('''
!python compare.py --runs "$RUNS" --csv "$RUNS/comparison.csv"
'''),

        md("## Cell 8 — results table"),
        code('''
import pandas as pd
pd.set_option("display.width", 200, "display.max_columns", 40)
df = pd.read_csv(f"{RUNS}/comparison.csv")
cols = ["model", "macro_auroc", "macro_auprc", "macro_sensitivity",
        "macro_specificity", "macro_precision", "macro_f1", "macro_fpr",
        "exact_match_accuracy", "params_total_m", "size_mb", "latency_ms_b1"]
display(df[[c for c in cols if c in df.columns]]
        .sort_values("macro_auprc", ascending=False))
'''),

        md("## Cell 9 — training curves"),
        code('''
import json, glob
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(17, 4))
for f in sorted(glob.glob(f"{RUNS}/*/result.json")):
    r = json.load(open(f))
    h = r["history"]
    if not h:
        continue
    ep = [x["epoch"] for x in h]
    ax[0].plot(ep, [x["train_loss"] for x in h], label=r["model"])
    ax[1].plot(ep, [x["val_macro_auprc"] for x in h], label=r["model"])
    ax[2].plot(ep, [x["val_macro_auroc"] for x in h], label=r["model"])
for a, t in zip(ax, ["train loss", "val macro AUPRC", "val macro AUROC"]):
    a.set_title(t); a.set_xlabel("epoch"); a.legend(fontsize=8); a.grid(alpha=.3)
plt.tight_layout(); plt.show()
'''),

        md("## Cell 10 — full report for one model"),
        code('''
MODEL = "resnet50"          # change and re-run
print(open(f"{RUNS}/{MODEL}/report.txt").read())
'''),

        md("""
## Cell 11 — source confound check  ⚠️ do not skip

All 1,442 STEMI records come from a single database, and stage 5 of the image
pipeline measured residual source signal (probe accuracy 0.515 vs 0.372 chance).
A high STEMI score cannot by itself separate *recognising infarction* from
*recognising the Chongqing scanner*.

**Report these numbers in your results chapter next to the headline**, whatever
they show.
"""),
        code('''
import json, glob
for f in sorted(glob.glob(f"{RUNS}/*/result.json")):
    r = json.load(open(f))
    ps = r.get("test_per_source", {})
    if not ps:
        continue
    v = [d["macro_auroc"] for d in ps.values()]
    print(f"{r['model']:18} " +
          "  ".join(f"{s}={d['macro_auroc']:.3f}(n={d['n']})" for s, d in ps.items()) +
          f"   spread={max(v)-min(v):.3f}")
'''),

        md("""
## Where the results live

```
<RUNS>/
├── _SUMMARY.txt              all models at a glance   <- read first
├── comparison.txt / .csv
└── <model>/
    ├── STATUS.txt            RUNNING | DONE | FAILED
    ├── report.txt            full metrics, plain text
    ├── train_log.txt         complete console output
    ├── history.csv           per epoch
    ├── metrics.csv
    ├── result.json
    ├── weights/best.pt       best-on-val weights + thresholds
    └── predictions/          per-image probabilities
```

All of it is on Drive, so it survives the runtime being recycled.
"""),
    ]

    return {"cells": cells,
            "metadata": {"accelerator": "GPU",
                         "colab": {"provenance": [], "gpuType": "T4"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def validate(nb: dict) -> None:
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
            # strip IPython magics/shell lines, then the rest must be valid python
            py = "\n".join(l for l in joined.split("\n")
                            if not l.lstrip().startswith(("!", "%")))
            try:
                ast.parse(py)
            except SyntaxError as e:
                problems.append(f"cell {i}: python does not parse: {e}")
    if problems:
        raise SystemExit("NOTEBOOK VALIDATION FAILED:\n  - " + "\n  - ".join(problems))
    print(f"validated {len(nb['cells'])} cells: newlines OK, python parses")


if __name__ == "__main__":
    nb = build()
    validate(nb)
    out = HERE / "CardioSentry_Colab.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({kb:.0f} KB, {len(nb['cells'])} cells)")
    print(f"embedded: {sum(1 for f in FILES if f.exists())} files")
    print("\nupload ONLY this notebook to Colab - the code travels inside it.")
