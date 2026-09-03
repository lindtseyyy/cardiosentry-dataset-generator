#!/usr/bin/env python3
"""
Build the SELF-CONTAINED Colab notebook.

    python make_notebook.py   ->  CardioSentry_Baselines_v2.ipynb

ONE notebook, because the point is to start it and go to sleep.

The study is two experiments - pick the input geometry, then rank the nine
architectures - and the second needs the first's answer. That handoff used to be
"read a table, edit a variable", which is exactly the kind of instruction that
does not survive an overnight run. `run_study.sh` + `pick_dim.py` do it
automatically instead, on VALIDATION macro AUPRC, so the whole thing is one cell.

The harness is 16 Python modules. You do not paste modules into notebook cells -
that is not what cells are for. Cell 1 carries the whole package as a base64
tar.gz and unpacks it to /content/code; every later cell is a one-line command
that runs those files.

So the notebook is the ONLY file you upload. Re-run this script after editing any
module to refresh the embedded copy.
"""
from __future__ import annotations
import base64, io, json, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = sorted([p for p in HERE.glob("*.py") if p.name != "make_notebook.py"]) + \
        sorted(HERE.glob("*.sh")) + [HERE / "requirements.txt"]

# Drive layout, per the project's directory map. Notebook in codes/, extracted
# and intermediate artifacts in experiments/, finished runs in runs/, HF weights
# in hf_cache/. Nothing is written to the CardioSentry root.
DRIVE = "/content/drive/MyDrive/CardioSentry"
OUT_NAME = "CardioSentry_Baselines_v2.ipynb"


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
        md("""
# CardioSentry — nine-architecture baselines

Compares **VGG-16, ResNet-50, DenseNet-121, EfficientNet-B3, EfficientNetV2-S,
ConvNeXt-Base, ConvNeXt-V2-Base, ViT-B/16 and Swin Transformer V2-Base** with
data, preprocessing, training protocol and evaluation held identical, so any
difference between them is attributable to the architecture.

Four independent sigmoid heads: **STEMI, AF, LVH, NORMAL**. Same metrics as
`cardiosentry_runs_v1`, including the full per-condition breakdown and the
per-source confound check.

## The input geometry is already settled

Every run here uses **`nat768` — 768x1024, the full sheet at its own aspect,
with the printed ID redacted.** That decision was made from measurement, not by
sweeping it here, so this notebook trains nine models once and nothing else.

| vs the previous `v1_768` runs | `sq768` | **`nat768`** |
|---|---|---|
| px/mm **time** | 2.690 | **3.587** (+33%) |
| px/mm **voltage** | 3.460 | 3.460 (equal) |
| aspect distortion | 0.78 | **1.04** |
| time Nyquist | 33.6 Hz | **44.8 Hz** |

A square resize squeezed the **time** axis by 22% on a sheet that is 3x4 leads
plus a rhythm strip. `nat768` removes that for free on the voltage axis, and it
is the first geometry whose time-Nyquist clears the 40 Hz band the corpus was
rendered in — a mechanism for the 384 -> 768 gain, and a prediction that it
continues. Both dimensions are multiples of 256, so Swin-V2 needs no padding
wrapper and the nine-model cost table stays directly comparable.

## Two data corrections baked in

1. **The printed record ID was a STEMI label.** Stage 1 numbered records by
   source in order, so CHAPMAN and PTB-XL are <= `ECG007796` and every STEMI
   record is >= `ECG007797`, with no overlap — a model reading one digit of
   ~25 px type gets the head for free. A fixed rectangle now covers that block,
   verified against all 9,238 clean renders to contain printed text and nothing
   else on 9,230 of them. It is baked into the pre-resized corpus, so it costs
   nothing per epoch.
2. **Two corrupt records excluded.** `ECG009191` and `ECG009005` carry a
   saturation artifact spanning the full sheet; one of them was in **test**.

Because of (1), the v1/v1_768 STEMI scores are not directly comparable to these
— the gap is the honest measure of how much of that 0.986-0.991 AUROC was the
digit. Say so in the writeup.

## Time

About **14 hours** for the nine models, plus ~12 minutes of one-time
pre-resizing, scaled from the measured `v1_768` L4 timings. **That is two Colab
sessions, not one** — and that is fine: re-run the training cell with the same
command and it resumes, skipping finished models and continuing an interrupted
one from its last completed epoch. Everything is written to Drive as it happens.

---
Before starting: **Runtime -> Change runtime type -> GPU**. L4 or A100 preferred.
On a 16 GB T4, halve every `micro_batch` in `models.py` first.
"""),

        md(f"## Cell 1 — unpack the code and install timm\n\n"
           f"*The whole harness ({names}) is embedded in this cell — no folder "
           f"upload needed.*"),
        code(f'''
import base64, io, tarfile, os
os.makedirs("/content/code", exist_ok=True)
_PKG = "{b64}"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(_PKG)), mode="r:gz").extractall("/content/code")
for _f in os.listdir("/content/code"):
    if _f.endswith(".sh"):
        os.chmod(f"/content/code/{{_f}}", 0o755)
os.environ["CS_DIM"] = "nat768"          # the geometry, fixed for every cell below
print("unpacked:")
for f in sorted(os.listdir("/content/code")):
    print("   ", f)

!pip install -q "timm>=1.0.9"
!nvidia-smi --query-gpu=name,memory.total --format=csv
'''),

        md("""
## Cell 2 — mount Drive, set up caching, extract the dataset

**Add your HuggingFace token first** (one-time, ~30 s):

1. Click the key icon in Colab's left sidebar
2. **+ Add new secret** -> Name: `HF_TOKEN` -> Value: your `hf_...` token
3. Toggle **Notebook access** on

Do not paste the token into a cell — it would be saved inside the `.ipynb` and
travel with any copy you share. If you skip this the run still works; the token
only raises Hub rate limits.

The dataset is extracted to **local disk** (`/content/`), never trained off the
mounted Drive — every image read would become a network round trip and starve
the GPU. **Everything the run produces still goes to Drive**, written as it
happens.

| folder | what lands there |
|---|---|
| `datasets/` | `prototype_1152.tar` — read only, never unpacked in place |
| `experiments/baselines_v2/samples/` | the rendered sample sheets |
| `experiments/baselines_v2/data/` | **the pre-resized corpus, cached as a tar** |
| `runs/baselines_v2/` | **every run: logs, metrics, weights, predictions** |
| `hf_cache/` | the nine pretrained checkpoints (~2.5 GB, downloaded once) |

Budget about **8 GB of Drive**: ~2.5 GB of checkpoints, ~3 GB of run artifacts
(`best.pt` is 94 MB for ResNet-50, 512 MB for VGG-16), and ~2.2 GB for the
cached corpus.
"""),
        code('''
import os
from google.colab import drive
drive.mount('/content/drive')

DRIVE = "/content/drive/MyDrive/CardioSentry"
DIM   = "nat768"
os.environ["CS_DIM"] = DIM

# Cache pretrained weights on Drive. Without this every runtime reset
# re-downloads all nine from HuggingFace - slow, and it is what triggers the
# "unauthenticated requests" rate limit over a night of reconnects.
os.environ["HF_HOME"] = f"{DRIVE}/hf_cache"

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

TAR  = f"{DRIVE}/datasets/prototype_1152.tar"
RAW  = "/content/prototype_1152"
DATA = RAW                      # cell 4 repoints this at the pre-resized copy
EXP  = f"{DRIVE}/experiments/baselines_v2"
CACHE = f"{EXP}/data"                # the pre-resized corpus is cached here
RUNS = f"{DRIVE}/runs/baselines_v2"

if not os.path.exists(f"{RAW}/index.csv"):
    !tar -xf "$TAR" -C /content/
!ls "$RAW"
!mkdir -p "$RUNS" "$EXP" "$CACHE"
print("\\nRAW =", RAW, "\\nRUNS =", RUNS, "\\nEXP  =", EXP, "\\nCACHE =", CACHE)

print("\\nDrive space:")
!df -h /content/drive/MyDrive | tail -1
print("local disk (holds the pre-resized copy):")
!df -h /content | tail -1
'''),

        md("""
## Cell 3 — see what the model will actually receive

Renders a few sheets through **the same code path training uses**, so this is
not a mock-up. Check that the redaction box covers the printed block and
nothing else — all four trace rows, twelve lead labels, both calibration pulses
and the `25mm/s 10mm/mV` footer should be untouched.

The last sheet is the **most rotated in the corpus (-3.00 deg)**, which is the
hardest case for a fixed box.
"""),
        code('''
%cd /content/code
!python dimensions.py | head -12
'''),
        code('''
import pandas as pd
from PIL import Image
from IPython.display import display, Markdown
import sys; sys.path.insert(0, "/content/code")
# `DIM` must be a geometry NAME ("nat768") for render(), never the module:
# `import dimensions as DIM` makes render() receive the module object and
# resolve() raises KeyError. Import under another name and re-derive DIM from
# DEFAULT_DIM so the call below is correct however this cell is re-run.
import dimensions as dimlib
from aspect_study import render

DIM = dimlib.DEFAULT_DIM          # the geometry NAME, e.g. "nat768"

df = pd.read_csv(f"{RAW}/index.csv")
df = df[~df.record.isin(dimlib.EXCLUDED_RECORDS)]
picks = [df[df.cls == c].iloc[0] for c in ("STEMI", "LVH", "AF", "NORMAL")]
picks.append(df.iloc[df["rotation_deg"].abs().idxmax()])      # worst rotation

os.makedirs(f"{EXP}/samples", exist_ok=True)
for r in picks:
    out = render(Image.open(f"{RAW}/{r['rel_path']}").convert("RGB"), DIM)
    p = f"{EXP}/samples/{r['cls']}_{r['record']}.png"
    out.save(p)
    display(Markdown(f"### {r['cls']}  {r['record']}  — rotation {float(r['rotation_deg']):+.2f} deg"))
    display(out)
print("saved to", f"{EXP}/samples")
'''),

        md("""
## Cell 4 — pre-resize the corpus, cached on Drive

**First session:** builds the 768x1024 corpus (~12 min) and archives it to Drive.
**Every session after:** restores it in a couple of minutes instead of rebuilding.

Colab wipes `/content` on every runtime reset and this study needs two sessions,
so without the cache you would pay the resize twice.

Why the resize at all: the corpus stores 1686x1311 sheets (2.2 Mpx). Resizing on
the fly decodes 2.2 Mpx to produce 0.79 — every image, every epoch, **for every
one of the nine models**. Measured: **41.3 ms/image on the fly vs 6.2 ms from a
pre-resized copy**, a 6.6x decode saving. Your v1 logs showed ~21 img/s where the
GPU could do 70-100. The **ID redaction is baked in here too**, so it costs
nothing per epoch.

Why a **tar** and not a folder on Drive: Drive is mounted over the network, so
6,670 individual files means 6,670 round trips — slower than the resize it was
meant to replace. One ~2.2 GB tar is a single sequential stream each way. Not
gzipped; JPEG does not compress.

**Training still reads from `/content`.** The Drive copy exists to survive a
restart, never to train off — every image read has to come off local disk or the
GPU starves. The cached tar carries the `.cs_geometry.json` marker, and its
geometry is checked *before* extraction, so a tar built at a different geometry
is refused rather than silently used.
"""),
        code('''
%cd /content/code
!./prepare_corpus.sh "$RAW" "/content/prototype_{DIM}" "$CACHE"

# A failed shell command does NOT stop a Colab cell, so verify before switching.
_pre = f"/content/prototype_{DIM}"
if os.path.exists(os.path.join(_pre, "index.csv")):
    DATA = _pre
    print("\\nDATA is now", DATA)
    !du -sh "$CACHE"/*.tar 2>/dev/null || true
else:
    DATA = RAW
    print("\\nPREPARE DID NOT COMPLETE - DATA left as", DATA)
    print("training still works, just slower; read the error above")
'''),

        md("""
## Cell 5 — selftest + preflight (~5 min)

**`selftest.py` is not a smoke test.** Every check in it is a bug whose symptom
is a model that trains normally, converges normally, and reports a number meaning
something other than what you think: `(H,W)` transposed, the redaction applied
twice, a micro-batch that silently changes the effective batch, the fingerprints
failing to separate protocol from geometry. It also verifies on real sheets that
**no ink survives inside the redaction box**.

**`preflight.py`** checks the GPU, integrity-checks the dataset, resolves all
nine timm tags, and runs a real forward+backward per model reporting peak VRAM —
turning an out-of-memory crash at epoch 7 into a message in four minutes. It
prints the largest micro-batch that would still fit, which is a free speedup.
"""),
        code('''
%cd /content/code
!python selftest.py --data "$DATA"
'''),
        code('''
!python preflight.py --data "$DATA"
'''),

        md("""
## Cell 6 — smoke test (~3 min)

20 train / 10 eval batches on the two models most likely to break at a
rectangular input — ResNet-50 (the reference) and Swin-V2 (the window-attention
one). The metrics are meaningless; this only proves the path runs before you
commit a night to it.
"""),
        code('''
!python train.py --model resnet50    --data "$DATA" --out /content/smoke --smoke
!python train.py --model swinv2_base --data "$DATA" --out /content/smoke --smoke
'''),

        md("""
## Cell 7 — train all nine  ⏳ **THE OVERNIGHT CELL**

**Expect this to need two sessions.** When Colab disconnects, re-run this same
cell — finished models are skipped and an interrupted one resumes from its last
completed epoch.

It does **not** abort if one model fails; the rest still run. Models are ordered
cheapest-and-most-informative first, so a batch cut short leaves ResNet-50 and
the new CNNs finished rather than three transformers half-done.
"""),
        code('''
%cd /content/code
!./run_all.sh "$DATA" "$RUNS"
'''),

        md("""
### Prefer one model at a time?

Use these instead of cell 7. Re-running a line after a disconnect resumes it.
"""),
        code('''
# !python train.py --model resnet50         --data "$DATA" --out "$RUNS"
# !python train.py --model densenet121      --data "$DATA" --out "$RUNS"
# !python train.py --model efficientnet_b3  --data "$DATA" --out "$RUNS"
# !python train.py --model efficientnetv2_s --data "$DATA" --out "$RUNS"
# !python train.py --model convnext_base    --data "$DATA" --out "$RUNS"
# !python train.py --model convnextv2_base  --data "$DATA" --out "$RUNS"
# !python train.py --model vgg16            --data "$DATA" --out "$RUNS"
# !python train.py --model swinv2_base      --data "$DATA" --out "$RUNS"
# !python train.py --model vit_b16          --data "$DATA" --out "$RUNS"
'''),

        md("## Cell 8 — morning check\n\nWorks even if the batch is only half finished."),
        code('''
!cat "$RUNS/_SUMMARY.txt"
'''),
        code('''
# did anything fail overnight?
!for f in "$RUNS"/*/STATUS.txt; do echo "--- $f"; head -3 "$f"; done
!ls "$RUNS"/*/ERROR.txt 2>/dev/null && echo "^ failures above" || echo "no failures"
'''),

        md("""
## Cell 9 — the comparison

`compare.py` first verifies the runs are actually comparable — matching config
fingerprints, same dataset, same row counts, no smoke runs — and **refuses to
rank them if not**.
"""),
        code('''
%cd /content/code
!python compare.py --runs "$RUNS" --csv "$RUNS/comparison.csv"
'''),
        code('''
import pandas as pd
pd.set_option("display.width", 220, "display.max_columns", 60)
df = pd.read_csv(f"{RUNS}/comparison.csv")
cols = ["model", "macro_auroc", "macro_auprc", "macro_sensitivity",
        "macro_specificity", "macro_precision", "macro_f1", "macro_fpr",
        "exact_match_accuracy", "params_total_m", "size_mb", "latency_ms_b1"]
display(df[[c for c in cols if c in df.columns]]
        .sort_values("macro_auprc", ascending=False))
'''),

        md("""
## Cell 10 — per-condition breakdown

The table for the results chapter: every metric, per head, per architecture.
STEMI is the minority head and the clinically critical one — read its column
first, next to cell 13.
"""),
        code('''
import pandas as pd
df = pd.read_csv(f"{RUNS}/comparison.csv")
for metric in ["auroc", "auprc", "sensitivity", "specificity", "precision",
               "f1", "fpr", "accuracy"]:
    cols = [f"{l}_{metric}" for l in ["STEMI", "AF", "LVH", "NORMAL"]]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        continue
    t = df.set_index("model")[cols + [f"macro_{metric}"]]
    t.columns = ["STEMI", "AF", "LVH", "NORMAL", "MACRO"][:len(t.columns)]
    print(f"\\n{metric.upper()} (test)")
    print(t.sort_values("MACRO", ascending=False).round(4).to_string())
'''),

        md("## Cell 11 — training curves"),
        code('''
import json, glob
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
for f in sorted(glob.glob(f"{RUNS}/*/result.json")):
    r = json.load(open(f))
    h = r["history"]
    if not h or r.get("smoke"):
        continue
    ep = [x["epoch"] for x in h]
    ax[0].plot(ep, [x["train_loss"] for x in h], label=r["model"])
    ax[1].plot(ep, [x["val_macro_auprc"] for x in h], label=r["model"])
    ax[2].plot(ep, [x["val_macro_auroc"] for x in h], label=r["model"])
for a, t in zip(ax, ["train loss", "val macro AUPRC", "val macro AUROC"]):
    a.set_title(t); a.set_xlabel("epoch"); a.legend(fontsize=8); a.grid(alpha=.3)
plt.tight_layout(); plt.show()
'''),

        md("""
## Cell 12 — accuracy vs cost

A model that wins by 0.005 AUPRC at 5x the size is not obviously the winner.
At `nat768` every model is fed exactly 768x1024 — including Swin-V2, since both
dimensions are multiples of 256 — so these cost numbers are directly comparable
with no per-model adjustment.
"""),
        code('''
import json, glob
import matplotlib.pyplot as plt

runs = [json.load(open(f)) for f in sorted(glob.glob(f"{RUNS}/*/result.json"))]
runs = [r for r in runs if not r.get("smoke")]

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for r in runs:
    fam = r["model_info"].get("family", "cnn")
    c = "C0" if fam == "cnn" else "C3"
    for a, x in ((ax[0], r["complexity"]["params_total_m"]),
                 (ax[1], r["inference"]["latency_ms_b1_mean"])):
        a.scatter(x, r["test"]["macro_auprc"], s=90, color=c, zorder=3)
        a.annotate(r["model"], (x, r["test"]["macro_auprc"]),
                   textcoords="offset points", xytext=(7, 4), fontsize=9)
ax[0].set_xlabel("parameters (M)"); ax[1].set_xlabel("latency, batch=1 (ms)")
for a in ax:
    a.set_ylabel("test macro AUPRC"); a.grid(alpha=.3)
ax[0].set_title("blue = CNN, red = transformer")
plt.tight_layout(); plt.show()
'''),

        md("""
## Cell 13 — source confound check  ⚠️ do not skip

All 1,442 STEMI records come from a single database, and stage 5 of the image
pipeline measured residual source signal (probe accuracy 0.515 vs 0.372 chance).
A high STEMI score cannot by itself separate *recognising infarction* from
*recognising the Chongqing scanner*. The printed ID is gone, which closes one
route to that shortcut — this checks the other.

**Report these next to the headline**, whatever they show.
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

        md("## Cell 14 — full report for one model"),
        code('''
import glob
MODEL = "resnet50"          # change and re-run
hits = glob.glob(f"{RUNS}/{MODEL}@*/report.txt")
print(open(hits[0]).read() if hits else f"no finished run for {MODEL}")
'''),

        md(f"""
## Where everything lives

```
{DRIVE}/runs/baselines_v2/
├── _SUMMARY.txt              all nine at a glance   <- read first
├── _batch_log.txt
├── comparison.txt / .csv     the ranking
└── <model>@nat768/
    ├── STATUS.txt            RUNNING | DONE | FAILED, rewritten each epoch
    ├── report.txt            full metrics, per condition, plain text
    ├── train_log.txt         complete console output, flushed per line
    ├── history.csv           one row appended per epoch
    ├── metrics.csv           flat per-label + macro, for plotting
    ├── result.json           everything machine-readable
    ├── config_snapshot.json
    ├── weights/best.pt       best-on-val weights + frozen thresholds
    └── predictions/          per-image probabilities, val and test
```

All on Drive, written as it happens, so it survives the runtime being recycled.
`last.pt` is deleted once a model finishes — `best.pt` has the weights that
matter.

## Three things that bound what these numbers mean

Put all three next to the headline in the results chapter:

1. **Prototype subset.** Train is 1 augmentation variant per record instead of 4.
   AUROC/AUPRC and the *ranking* transfer to the full corpus; thresholds,
   calibration and absolute scores do not.
2. **Single-source STEMI.** All 1,442 STEMI records come from one database —
   cell 13. The cross-source test is still outstanding.
3. **The v1/v1_768 STEMI scores are not comparable to these.** Those runs had the
   printed record ID in the image and it is a perfect STEMI label; these do not.

## The geometry ablation, if you ever want it

The package still ships `run_dims.sh`, `pick_dim.py` and the `sq768` / `nat672` /
`wide576` / `crop640` geometries. This notebook does not use them — `nat768` was
chosen from measurement (`crop_audit.py`, `prototype-results/baselines_v2/`)
rather than by sweeping. To run the sweep later:

```bash
./run_dims.sh <DATA> <OUT> resnet50 "sq768 nat672 nat768 wide576 crop640"
python compare.py --runs <OUT> --across-dims
```
"""),
    ]

    return {"cells": cells,
            "metadata": {"accelerator": "GPU",
                         "colab": {"provenance": [], "gpuType": "L4"},
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
            # Replace IPython magic/shell lines with `pass` at the SAME
            # indentation rather than dropping them. Dropping them silently
            # empties any block whose only body is a `!` line - which IPython
            # accepts and runs correctly - and the validator would then reject a
            # perfectly good cell.
            out = []
            for l in joined.split("\n"):
                stripped = l.lstrip()
                if stripped.startswith(("!", "%")):
                    out.append(" " * (len(l) - len(stripped)) + "pass")
                else:
                    out.append(l)
            try:
                ast.parse("\n".join(out))
            except SyntaxError as e:
                problems.append(f"cell {i}: python does not parse: {e}")
    if problems:
        raise SystemExit("NOTEBOOK VALIDATION FAILED:\n  - " + "\n  - ".join(problems))
    print(f"validated {len(nb['cells'])} cells: newlines OK, python parses")


if __name__ == "__main__":
    nb = build()
    validate(nb)
    out = HERE / OUT_NAME
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB, {len(nb['cells'])} cells)")
    print(f"embedded: {sum(1 for f in FILES if f.exists())} files")
    print(f"\nupload ONLY this notebook to Colab - the code travels inside it.")
    print(f"Drive home: {DRIVE}/codes/baselines_v2/")
