#!/usr/bin/env python3
"""
Build the THREE self-contained Colab notebooks - one per backbone family.

    python make_notebook.py
      ->  CardioSentry_Hybrids_v2_A_ResNet50.ipynb
          CardioSentry_Hybrids_v2_B_EfficientNetV2S.ipynb
          CardioSentry_Hybrids_v2_C_VGG16.ipynb

WHY THREE, AND WHY SPLIT ON THE BACKBONE

Twenty-one runs is ~31 hours in one stream. The three backbone families are
completely independent - nothing in family B reads anything family A produces -
so run them in three Colab sessions at once and the wall clock becomes the
LONGEST family (~13 h for VGG-16) rather than their sum.

Splitting on the backbone rather than, say, seven-runs-each is what makes each
notebook self-contained: a family notebook trains one backbone's whole ladder and
imports that one backbone's baseline row, so it can produce its own complete
delta table without waiting for the other two. Any other split would leave every
notebook holding a fragment of three ladders and nothing reportable until all
three finished.

WHAT THE SPLIT HAD TO FIX

Three Colab sessions on one Drive is three writers. Each notebook therefore
writes to `runs/hybrids_v2/<family>/`, its OWN directory, so `_SUMMARY.txt`,
`_batch_log.txt` and `comparison.csv` have exactly one writer each. Two shared
things remain, and both were made safe rather than assumed safe:

  * the pre-resized corpus tar, which all three read. `prepare_corpus.sh` now
    stages to a PID-suffixed .part, so three simultaneous cold builds cannot
    interleave into one file and then publish it as the cache.
  * `compare.py --runs` takes several folders, so the cross-backbone table is
    built by pointing it at all three when they are done.

The harness is 17 Python modules. You do not paste modules into notebook cells;
that is not what cells are for. Cell 1 of each notebook carries the whole package
as a base64 tar.gz and unpacks it to /content/code. So each notebook is the ONLY
file you upload.

Re-run this script after editing any module to refresh all three embedded copies.
"""
from __future__ import annotations
import base64, io, json, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = sorted([p for p in HERE.glob("*.py") if p.name != "make_notebook.py"]) + \
        sorted(HERE.glob("*.sh")) + [HERE / "requirements.txt"]

# Drive layout, per the project's directory map: notebooks in codes/, extracted
# and intermediate artifacts in experiments/, finished runs in runs/, HF weights
# in hf_cache/. Nothing is written to the CardioSentry root.
DRIVE = "/content/drive/MyDrive/CardioSentry"

# (letter, backbone key, display name, notebook file stem, hours, per-run notes)
FAMILIES = [
    ("A", "resnet50", "ResNet-50", "A_ResNet50", 9,
     "the project's reference backbone - won v1 at 384, tied v1_768. Cheapest "
     "per epoch of the three, so this is the family to watch first if you are "
     "only going to read one."),
    ("B", "efficientnetv2_s", "EfficientNetV2-S", "B_EfficientNetV2S", 9,
     "fused-MBConv early stages. Comparable cost to ResNet-50 per epoch and a "
     "quite different inductive bias, which is what makes agreement between "
     "these two ladders meaningful."),
    ("C", "vgg16", "VGG-16", "C_VGG16", 13,
     "a 512-channel feature map behind timm's 4096-d ConvMlp pre-logits. The "
     "slowest family by a wide margin (micro-batch 4 against 8), which is why "
     "it gets its own session rather than being appended to another."),
]

LADDER_ROWS = [
    ("cbam", "CBAM", "channel + spatial reweighting (Woo et al. 2018)"),
    ("eca", "ECA", "channel reweighting with NO bottleneck - one 1-D conv, "
                   "7 parameters at C=2048 (Wang et al. 2020)"),
    ("msa", "MSA", "multi-scale separable strip attention at k = 3, 7, 11 "
                   "(the MSCA construction, Guo et al. 2022)"),
    ("lt", "LT", "2 transformer blocks, width 256, over the 24x32 = 768-token "
                 "grid - REPLACING the backbone's pooling"),
    ("lt_concat", "LT + concat", "both streams, per-stream LayerNorm, "
                                 "concatenated"),
    ("lt_gated", "LT + gated", "both streams, learned per-channel mixture"),
    ("msa_lt_gated", "MSA + LT + gated", "the full stack"),
]


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


def build(letter: str, bb: str, nice: str, hours: int, bb_note: str,
          b64: str) -> dict:
    names = ", ".join(f.name for f in FILES if f.exists())
    others = [(l, n) for l, k, n, _, _, _ in FAMILIES if k != bb
              for _ in [0]]
    others = [(l, n) for l, k, n, *_ in FAMILIES if k != bb]
    other_txt = " and ".join(f"**{l}** ({n})" for l, n in others)

    ladder_md = "\n".join(
        f"| `{bb}_{sfx}` | {label} | {desc} |" for sfx, label, desc in LADDER_ROWS)

    cells = [
        md(f"""
# CardioSentry — hybrids v2 · **notebook {letter} of 3: {nice}**

This notebook trains **the seven {nice} hybrids** and nothing else. Notebooks
{other_txt} do the other two backbones and are **meant to run at the same time**.

> **Colab limits concurrent GPU sessions per account** — typically one on the
> free tier and two to three on Pro/Pro+. Running all three at once needs enough
> concurrent-session headroom (or separate accounts). If you can only run two,
> start **C (VGG-16)** first: it is the long pole at ~13 h against ~9 h each for
> A and B.

| | wall clock |
|---|---|
| all 21 runs in one session | ~31 h |
| three sessions in parallel | **~13 h** (the longest family) |

## The seven runs in this notebook

| run | component | what it is |
|---|---|---|
{ladder_md}

*{nice}: {bb_note}*

## Why the split is on the backbone

Each family is completely independent — nothing here reads anything the other
notebooks produce. Splitting this way also makes each notebook **self-contained**:
it trains one backbone's whole ladder and imports that one backbone's baseline
row, so it produces its own complete delta table without waiting for the others.
Any other split would leave all three holding fragments of three ladders and
nothing reportable until every one had finished.

**This notebook writes only to `runs/hybrids_v2/{bb}/`.** Three concurrent
sessions on one Drive is three writers, so each gets its own directory and every
shared file — `_SUMMARY.txt`, `_batch_log.txt`, `comparison.csv` — has exactly
one writer. Cell 18 builds the cross-backbone table at the end by pointing
`compare.py` at all three folders.

## The number that matters is a difference, not a score

`config.py` in this package is **byte-identical** to `baselines_v2`'s, so
`fingerprint()` matches and `compare.py` ranks a baseline run and a hybrid run in
one table without being forced to. Cell 7 imports the `{bb}` baseline row and
every result is then reported as a **delta against the bare backbone**.

That subtraction is only honest if the baseline is genuinely nested inside the
hybrid, and it is: with no components, the wrapper reduces to
`forward_features → forward_head(pre_logits=True) → Linear`, which is exactly
what `timm.create_model(tag, num_classes=4)` builds. `selftest.py` asserts the
parameter counts match rather than asking you to believe this paragraph.

## Every component starts as the identity

Each attention module is wrapped as `x + γ·(refine(x) − x)` with **γ initialised
to exactly 0**. Dropping a fresh sigmoid gate into a pretrained backbone
otherwise halves activations the ImageNet weights were tuned for, and the first
epochs go on undoing that rather than learning — which shows up as "attention
hurt" when what happened was "attention was initialised badly".

**The learned γ is a result in itself.** A γ that never leaves zero says the
backbone did not want the reweighting, and that is a cleaner finding than a noisy
±0.004 AUPRC. Same for the fusion gate: its mean is measured on **test** and
printed, so "the model learned to ignore the transformer" is something you can
write down. Cell 12 tabulates both.

## Geometry and the two data corrections — unchanged from baselines_v2

`nat768` (768×1024, the full sheet at its own aspect), printed record ID
redacted, two corrupt records excluded. Nothing here re-opens those decisions;
changing any of them would change the fingerprint and break the comparison this
notebook exists for.

---
Before starting: **Runtime → Change runtime type → GPU**. L4 or A100 preferred.
On a 16 GB T4, halve every `micro_batch` in `models.py` first.
"""),

        md(f"## Cell 1 — unpack the code and install timm\n\n"
           f"*The whole harness ({names}) is embedded in this cell — no folder "
           f"upload needed. All three notebooks carry the same package.*"),
        code(f'''
import base64, io, tarfile, os
os.makedirs("/content/code", exist_ok=True)
_PKG = "{b64}"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(_PKG)), mode="r:gz").extractall("/content/code")
for _f in os.listdir("/content/code"):
    if _f.endswith(".sh"):
        os.chmod(f"/content/code/{{_f}}", 0o755)
os.environ["CS_DIM"] = "nat768"          # the geometry, fixed for every cell below
FAMILY = "{bb}"                          # this notebook trains only this backbone
print("unpacked:")
for f in sorted(os.listdir("/content/code")):
    print("   ", f)
print("\\nfamily:", FAMILY)

!pip install -q "timm>=1.0.9"
!nvidia-smi --query-gpu=name,memory.total --format=csv
'''),

        md(f"""
## Cell 2 — mount Drive, set up caching, extract the dataset

**Add your HuggingFace token first** (one-time, ~30 s): key icon in the left
sidebar → **+ Add new secret** → Name `HF_TOKEN`, Value your `hf_...` token →
toggle **Notebook access** on. Do not paste it into a cell; it would be saved
inside the `.ipynb` and travel with any copy you share. This matters a little
more with three sessions running — unauthenticated Hub rate limits are per-IP.

The dataset is extracted to **local disk** (`/content/`), never trained off the
mounted Drive — every image read would become a network round trip and starve the
GPU. **Everything the run produces still goes to Drive**, written as it happens.

| folder | what lands there | writers |
|---|---|---|
| `datasets/` | `prototype_1152.tar` — read only | none |
| `experiments/baselines_v2/data/` | **the pre-resized corpus** — shared, read-mostly | see below |
| `experiments/hybrids_v2/{bb}/samples/` | this family's sample sheets | this notebook |
| `runs/hybrids_v2/{bb}/` | **this family's runs**: logs, metrics, weights, predictions | this notebook |
| `runs/baselines_v2/` | read only — where the `{bb}` baseline row comes from | none |
| `hf_cache/` | the pretrained backbone, downloaded once | all three (safe: per-file locks, and different files) |

**Everything this notebook writes is under a `{bb}` path.** That is what makes
three concurrent sessions safe: no two of them touch the same file.

⚠️ **Run this cell in ONE notebook first, wait for it, then start the others.**
It takes seconds, and it is the one step that must not be done in parallel.
Google Drive is not a POSIX filesystem — it permits two folders with the SAME
NAME in the same parent — and `mkdir -p` on a mounted Drive goes through the
Drive API. Three sessions creating `runs/hybrids_v2/` at the same instant can
therefore end up with two or three *different* folders all called
`hybrids_v2`, and your results would be silently split across them with cell 18
seeing only one. Creating the shared parents once, from one session, removes
that entirely. Nothing after this cell writes to a shared path.

**The pre-resized corpus cache is deliberately pointed at `baselines_v2`.** It is
the same corpus at the same geometry, keyed by `<corpus>_<dim>.tar` with its
geometry marker checked before extraction — so this restores the tar
`baselines_v2` already built instead of building an identical second copy. If it
is *not* there, all three sessions will build it; `prepare_corpus.sh` stages to a
PID-suffixed `.part` so they cannot interleave into one corrupt file. It is still
three times the work, so **if the cache is cold, run cell 4 in this notebook
alone first** and start the other two once it has finished.

Budget about **{2 + hours // 3} GB of Drive for this family** (`best.pt` is
{'~512 MB for a VGG-16 hybrid' if bb == 'vgg16' else '~90-110 MB per run'}), plus
the shared 2.2 GB corpus tar if it is not already there.
"""),
        code(f'''
import os
from google.colab import drive
drive.mount('/content/drive')

DRIVE  = "/content/drive/MyDrive/CardioSentry"
DIM    = "nat768"
FAMILY = "{bb}"
os.environ["CS_DIM"] = DIM

# Cache pretrained weights on Drive. Without this every runtime reset
# re-downloads the backbone from HuggingFace.
os.environ["HF_HOME"] = f"{{DRIVE}}/hf_cache"

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
    print(f"no HF_TOKEN in Colab Secrets ({{type(exc).__name__}}) - "
          f"continuing unauthenticated (works fine)")

TAR   = f"{{DRIVE}}/datasets/prototype_1152.tar"
RAW   = "/content/prototype_1152"
DATA  = RAW                      # cell 4 repoints this at the pre-resized copy
EXP   = f"{{DRIVE}}/experiments/hybrids_v2/{{FAMILY}}"
CACHE = f"{{DRIVE}}/experiments/baselines_v2/data"    # SHARED, read-mostly
HYB   = f"{{DRIVE}}/runs/hybrids_v2"                  # the parent of all three
RUNS  = f"{{HYB}}/{{FAMILY}}"                         # THIS notebook's only output
BASE_RUNS = f"{{DRIVE}}/runs/baselines_v2"            # read only: the reference row

if not os.path.exists(f"{{RAW}}/index.csv"):
    !tar -xf "$TAR" -C /content/
!ls "$RAW"

# The SHARED parents first and on their own. See the warning above: creating
# these concurrently from three sessions can produce duplicate same-name folders
# on Drive, which would split the results without any error.
!mkdir -p "$HYB" "$DRIVE/experiments/hybrids_v2" "$CACHE"
# ...then this family's own directories, which no other session touches.
!mkdir -p "$RUNS" "$EXP"

_siblings = [d for d in os.listdir(f"{{DRIVE}}/runs") if d.startswith("hybrids_v2")]
print("runs/ entries matching hybrids_v2*:", _siblings)
if len(_siblings) > 1:
    print("  *** MORE THAN ONE. Drive has duplicated the parent folder - merge them")
    print("      in the Drive web UI before training, or results will be split.")

print("\\nRAW  =", RAW, "\\nRUNS =", RUNS, "\\nEXP  =", EXP,
      "\\nCACHE =", CACHE, "\\nBASE_RUNS =", BASE_RUNS)

print("\\nDrive space:")
!df -h /content/drive/MyDrive | tail -1
print("local disk (holds the pre-resized copy):")
!df -h /content | tail -1
'''),

        md(f"""
## Cell 3 — see what the models will actually receive

Renders a few sheets through **the same code path training uses**. Check that the
redaction box covers the printed block and nothing else — all four trace rows,
twelve lead labels, both calibration pulses and the `25mm/s 10mm/mV` footer
should be untouched. The last sheet is the **most rotated in the corpus (−3.00°)**,
the hardest case for a fixed box.

The first cell also lists this family's seven registry entries and the token grid
the lightweight transformer builds from this geometry: at 768×1024 and stride 32
that is **24×32 = 768 tokens**, one per ~8.9 mm of time and ~9.3 mm of voltage.

*(Redundant across the three notebooks by design — each writes to its own
`samples/` folder, so running it in all three costs a minute and collides with
nothing.)*
"""),
        code(f'''
%cd /content/code
!python dimensions.py | head -12
!python models.py | grep -E "^(model|-|{bb}_)"
'''),
        code('''
import pandas as pd
from PIL import Image
from IPython.display import display, Markdown
import sys; sys.path.insert(0, "/content/code")
# `DIM` must be a geometry NAME ("nat768") for render(), never the module:
# `import dimensions as DIM` makes render() receive the module object and
# resolve() raises KeyError.
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
## Cell 4 — pre-resize the corpus (usually just a restore)

**If you ran `baselines_v2`, this is a two-minute extract**, not a rebuild — the
cache points at that experiment's `data/` folder and the tar is keyed by corpus
and geometry, so the identical `prototype_1152_nat768.tar` is reused.

⚠️ **If the cache is cold, run this cell in ONE notebook first and wait.** All
three sessions would otherwise each spend ~12 minutes building the same 2.2 GB
corpus. It is safe — staging is PID-suffixed, so they cannot interleave into one
corrupt file — but it is three times the work for one result.

Why the resize at all: the corpus stores 1686×1311 sheets (2.2 Mpx). Resizing on
the fly decodes 2.2 Mpx to produce 0.79 — every image, every epoch, for every
run. Measured: **41.3 ms/image on the fly vs 6.2 ms from a pre-resized copy**.
The **ID redaction is baked in here too**, so it costs nothing per epoch.

**Training always reads from `/content`.** The Drive copy exists to survive a
restart, never to train off — three sessions reading images over one network
mount would starve all three GPUs at once.
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

        md(f"""
## Cell 5 — selftest + preflight (~3 min)

**`selftest.py` is not a smoke test.** Every check in it is a bug whose symptom is
a model that trains normally, converges normally, and reports a number meaning
something other than what you think. The hybrid-specific ones:

* **every attention module is EXACTLY the identity at step 0** — asserted with
  `torch.equal`, not approximately
* **the componentless hybrid has the same parameter count as the baselines_v2
  model** — the claim every delta in this study rests on
* **`forward_features` is stride 32 on all three backbones** — the transformer's
  position embedding is built for that grid
* **per-stream LayerNorm actually isolates the streams**, or `ablate.py` reports
  a distribution shift rather than a contribution
* **the 3 × 7 grid is complete** — a missing cell would cost a whole
  cross-backbone claim, so it fails the selftest rather than passing quietly

`preflight.py --family {bb}` constructs **this family's seven** hybrids and runs a
real forward+backward on each, reporting peak VRAM. Checking the other fourteen
would cost three minutes for information this session cannot act on.
"""),
        code('''
%cd /content/code
!python selftest.py --data "$DATA"
'''),
        code('''
!python preflight.py --data "$DATA" --family "$FAMILY"
'''),

        md(f"""
## Cell 6 — smoke test (~3 min)

20 train / 10 eval batches on the two {nice} variants most likely to break: the
gated two-stream path and the multi-scale attention path. The metrics are
meaningless; this only proves the path runs before you commit {hours} hours.
"""),
        code(f'''
!python train.py --model {bb}_msa       --data "$DATA" --out /content/smoke --smoke
!python train.py --model {bb}_lt_gated  --data "$DATA" --out /content/smoke --smoke
'''),

        md(f"""
## Cell 7 — import the `{bb}` baseline row

Copies `{bb}@nat768` from the `baselines_v2` runs folder into this family's runs
folder — **`result.json` and the text reports only, never the checkpoints**, which
are already on Drive and would duplicate hundreds of MB to build a table that
never opens them.

It **refuses** a run whose `fingerprint` differs from the one these hybrids are
trained under. If that happens, nothing is broken: it means the baseline predates
a config change, and the honest fix is to re-train `{bb}` here rather than to
subtract across two protocols.

Without this cell the comparison still runs — it just has no reference row, and
the delta table is the whole point.
"""),
        code('''
%cd /content/code
!python import_baselines.py --from "$BASE_RUNS" --to "$RUNS" --backbone "$FAMILY"
'''),

        md(f"""
## Cell 8 — train the seven  ⏳ **THE OVERNIGHT CELL**  (~{hours} h)

**Start the other two notebooks now**, if you have the session headroom — they do
not touch anything this one writes.

When Colab disconnects, re-run this same cell: finished runs are skipped and an
interrupted one resumes from its last completed epoch. It does **not** abort if
one run fails; the rest still run.
"""),
        code('''
%cd /content/code
!./run_all.sh "$DATA" "$RUNS" --family "$FAMILY"
'''),

        md("""
### If VRAM is tighter than you like

`--micro-batch` is passed through to `train.py`. It is a VRAM accommodation, not
an experimental variable: accumulation restores the effective batch of 32 and
`steps_per_epoch` is **151** at 4, 8 or 16, so the LR schedule does not move.

It is **only** free on a backbone with no BatchNorm. BN normalises over the
*micro*-batch and accumulation sums gradients — it does not merge batch
statistics — so on ResNet-50 (53 BN layers) or EfficientNetV2-S (110) changing it
changes what the network computes, while the loss curve looks perfectly normal
and the delta against the baseline quietly stops meaning what it says.
`check_micro_batch.py` therefore **refuses** it there rather than warning.

VGG-16 has none, so it is the one family where you can trade throughput for
headroom freely — including part-way through, without invalidating the runs
already finished at another value.
"""),
        code('''
# !./run_all.sh "$DATA" "$RUNS" --family "$FAMILY" --micro-batch 8
'''),

        md("### One run at a time\n\nIf you would rather step through them."),
        code("\n".join(
            [f'# !python train.py --model {bb}_{sfx:<14} --data "$DATA" --out "$RUNS"'
             for sfx, _, _ in LADDER_ROWS])),

        md("## Cell 9 — morning check\n\nWorks even if the family is only part finished."),
        code('''
!cat "$RUNS/_SUMMARY.txt"
'''),
        code('''
# did anything fail overnight?
!for f in "$RUNS"/*/STATUS.txt; do echo "--- $f"; head -3 "$f"; done
!ls "$RUNS"/*/ERROR.txt 2>/dev/null && echo "^ failures above" || echo "no failures"
'''),

        md(f"""
## Cell 10 — this family's comparison

`compare.py` first verifies the runs are actually comparable — matching config
fingerprints, same dataset, same row counts, no smoke runs — and **refuses to
rank them if not**. Read the **DELTA AGAINST THE BACKBONE BASELINE** section at
the bottom; everything above it is context.

This is {nice} only. Cell 18 does all three together.
"""),
        code('''
%cd /content/code
!python compare.py --runs "$RUNS" --csv "$RUNS/comparison.csv"
'''),

        md(f"""
## Cell 11 — the delta table

**On reading a delta:** test is ~660 records at one canonical image each, one
seed. A macro AUPRC difference under roughly **0.01** is inside single-seed noise
on a set that size — treat it as a tie and read the parameter cost and the
learned γ/gate instead.

**Nothing here is a cross-backbone claim yet.** A component that helps {nice} is
one draw; a component that helps all three backbones is a pattern. That is what
cell 18 is for, and it is the only place a ranking should be read from.
"""),
        code('''
import pandas as pd
pd.set_option("display.width", 220, "display.max_columns", 60)
df = pd.read_csv(f"{RUNS}/comparison.csv")

base = df[df.is_baseline]
if base.empty:
    print("no baseline row imported - run cell 7")
else:
    b = base.iloc[0]
    t = df[~df.is_baseline][["model", "macro_auprc", "macro_auroc", "macro_f1",
                             "STEMI_auprc", "added_params"]].copy()
    t["model"] = t["model"].str.replace(f"{FAMILY}_", "", regex=False)
    for c in ("macro_auprc", "macro_auroc", "macro_f1", "STEMI_auprc"):
        t["d_" + c] = t[c] - b[c]
    t["added_M"] = t.pop("added_params") / 1e6
    print(f"=== {FAMILY}   baseline mAUPRC {b['macro_auprc']:.4f} ===")
    print(t.sort_values("d_macro_auprc", ascending=False).round(4).to_string(index=False))
'''),

        md("""
## Cell 12 — what the components learned  ⚠️ read before writing anything up

γ is the residual weight on the attention branch, **initialised to exactly 0**. A
γ that never left zero means the backbone did not want the reweighting, and any
score difference on that row is noise rather than the module. The gate mean is
`g` in `g·cnn + (1−g)·transformer`, measured on **test**.

A row with a positive delta and γ ≈ 0 is not "attention helped". It is
"a differently-initialised head helped", and it should be reported that way.
"""),
        code('''
import pandas as pd
df = pd.read_csv(f"{RUNS}/comparison.csv")
cols = ["model", "attention", "attention_gamma", "fusion",
        "gate_mean", "lt_tokens", "added_params", "macro_auprc"]
t = df[~df.is_baseline][[c for c in cols if c in df.columns]].copy()
t["added_M"] = t.pop("added_params") / 1e6
print(t.sort_values("model").round(4).to_string(index=False))
print("\\ngamma near 0  -> the attention module effectively turned itself OFF")
print("gate near 1   -> the model learned to IGNORE the transformer stream")
print("gate near 0   -> the model learned to ignore the CNN stream")
'''),

        md("""
## Cell 13 — stream ablation (no retraining)

Runs the test set three times on each two-stream model's trained weights — whole,
CNN zeroed, transformer zeroed — with the **frozen val-tuned thresholds**. It
answers the question the headline cannot: did the transformer contribute, or did
the model improve for some other reason and carry a passenger?

An ablated model is off-distribution — it was never trained with a stream zeroed
— so these bound what a stream *contributes here*, not how a single-stream model
would score if trained alone. For that, read the `lt`-only rung, which **is**
trained alone.
"""),
        code('''
%cd /content/code
!python ablate.py --runs "$RUNS" --data "$DATA"
'''),

        md("""
## Cell 14 — per-condition breakdown

Every metric, per head, per run. STEMI is the minority head and the clinically
critical one — read its column first, next to cell 16.
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

        md("## Cell 15 — training curves and accuracy vs cost"),
        code('''
import json, glob
import matplotlib.pyplot as plt

runs = [json.load(open(f)) for f in sorted(glob.glob(f"{RUNS}/*/result.json"))]
runs = [r for r in runs if not r.get("smoke")]

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for r in runs:
    h = r["history"]
    if not h:
        continue                                  # the imported baseline row
    ep = [x["epoch"] for x in h]
    lw = 2.5 if r["model"] == FAMILY else 1.3     # the baseline drawn heavier
    lab = r["model"].replace(FAMILY + "_", "") if r["model"] != FAMILY else "BASELINE"
    ax[0].plot(ep, [x["train_loss"] for x in h], label=lab, lw=lw)
    ax[1].plot(ep, [x["val_macro_auprc"] for x in h], label=lab, lw=lw)
for a, t in zip(ax, [f"{FAMILY}: train loss", f"{FAMILY}: val macro AUPRC"]):
    a.set_title(t); a.set_xlabel("epoch"); a.legend(fontsize=8); a.grid(alpha=.3)
plt.tight_layout(); plt.show()
'''),
        code('''
import json, glob
import matplotlib.pyplot as plt

runs = [json.load(open(f)) for f in sorted(glob.glob(f"{RUNS}/*/result.json"))]
runs = [r for r in runs if not r.get("smoke")]

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
for r in runs:
    cp = (r.get("hybrid") or {}).get("component_params", {})
    added = cp.get("added_over_backbone", 0) / 1e6
    y = r["test"]["macro_auprc"]
    is_base = r["model"] == FAMILY
    for a, x in ((ax[0], r["complexity"]["params_total_m"]), (ax[1], added)):
        a.scatter(x, y, s=170 if is_base else 80, color="C3" if is_base else "C0",
                  marker="*" if is_base else "o", zorder=3)
        a.annotate(r["model"].replace(FAMILY + "_", ""), (x, y),
                   textcoords="offset points", xytext=(7, 4), fontsize=9)
ax[0].set_xlabel("total parameters (M)")
ax[1].set_xlabel("parameters ADDED over the bare backbone (M)")
for a in ax:
    a.set_ylabel("test macro AUPRC"); a.grid(alpha=.3)
ax[0].set_title("red star = the bare backbone baseline")
plt.tight_layout(); plt.show()
'''),

        md("""
## Cell 16 — source confound check  ⚠️ do not skip

All 1,442 STEMI records come from a single database, and stage 5 of the image
pipeline measured residual source signal (probe accuracy 0.515 vs 0.372 chance).
A high STEMI score cannot by itself separate *recognising infarction* from
*recognising the Chongqing scanner*. The printed ID is redacted, which closes one
route to that shortcut — this checks the other.

Watch for a hybrid that **widens** the spread relative to its baseline: a spatial
attention gate that learned to look at an acquisition artifact would do exactly
that while the headline AUPRC went up.
"""),
        code('''
import json, glob
rows = []
for f in sorted(glob.glob(f"{RUNS}/*/result.json")):
    r = json.load(open(f))
    ps = r.get("test_per_source", {})
    if not ps:
        continue
    v = [d["macro_auroc"] for d in ps.values()]
    rows.append((r["model"], ps, max(v) - min(v)))
base = next((sp for m, _, sp in rows if m == FAMILY), None)
for m, ps, spread in sorted(rows):
    delta = f"   d vs baseline {spread - base:+.3f}" if base is not None and m != FAMILY else ""
    print(f"{m:32} " +
          "  ".join(f"{s}={d['macro_auroc']:.3f}(n={d['n']})" for s, d in ps.items()) +
          f"   spread={spread:.3f}{delta}")
'''),

        md("## Cell 17 — full report for one run"),
        code(f'''
import glob
MODEL = "{bb}_lt_gated"          # change and re-run
hits = glob.glob(f"{{RUNS}}/{{MODEL}}@*/report.txt")
print(open(hits[0]).read() if hits else f"no finished run for {{MODEL}}")
'''),

        md(f"""
## Cell 18 — the cross-backbone table  🔗 **run this once all three notebooks are done**

`compare.py --runs` takes several folders, so this builds the full 21-run table
by pointing it at all three family directories at once. It works from **any** of
the three notebooks — they all see the same Drive — and it is safe to re-run.

Until all three families have finished it will simply tabulate whatever exists,
and say how many runs it found. **The ranking is only meaningful once it reports
21 runs plus 3 baselines.**

This is the only cell in this notebook whose output is a cross-backbone claim.
Everything above it describes {nice} alone, and a component that helps one
backbone is one draw.
"""),
        code('''
%cd /content/code
import os
COMBINED = f"{HYB}/_combined"
os.makedirs(COMBINED, exist_ok=True)
FAMS = " ".join(f'"{HYB}/{f}"' for f in ["resnet50", "efficientnetv2_s", "vgg16"])

!python compare.py --runs {FAMS} --out "{COMBINED}/comparison.txt" --csv "{COMBINED}/comparison.csv"
'''),
        code('''
import pandas as pd
pd.set_option("display.width", 240, "display.max_columns", 80)
df = pd.read_csv(f"{HYB}/_combined/comparison.csv")

done = df[~df.is_baseline].groupby("backbone").size()
print("runs finished per backbone:\\n" + done.to_string())
if len(done) < 3 or done.min() < 7:
    print("\\nNOT ALL 21 RUNS ARE IN YET - the table below is partial.")

# Does a component help on ALL THREE backbones, or only on one? That question is
# the only thing three ladders buy you, so ask it directly.
base = df[df.is_baseline].set_index("backbone")["macro_auprc"]
h = df[~df.is_baseline].copy()
h["variant"] = [m.replace(b + "_", "") for m, b in zip(h.model, h.backbone)]
h["delta"] = h.macro_auprc - h.backbone.map(base)
piv = h.pivot_table(index="variant", columns="backbone", values="delta")
piv["mean"] = piv.mean(axis=1)
piv["helps_all"] = (piv.drop(columns="mean") > 0).all(axis=1)
print("\\ndelta in test macro AUPRC vs each backbone's own baseline:")
print(piv.sort_values("mean", ascending=False).round(4).to_string())
print("\\nhelps_all = positive on every backbone that has finished.")
print("A delta under ~0.01 is a tie; rank on agreement across the row, not on "
      "the size of any one cell.")
'''),

        md(f"""
## Where everything lives

```
{DRIVE}/runs/hybrids_v2/
├── resnet50/           <- notebook A writes only here
├── efficientnetv2_s/   <- notebook B writes only here
├── vgg16/              <- notebook C writes only here
│   ├── _SUMMARY.txt          every run in the family at a glance
│   ├── _batch_log.txt
│   ├── comparison.txt/.csv   this family's ranking + delta table
│   ├── ablation.txt          stream ablation, two-stream runs
│   ├── <backbone>@nat768/    imported from baselines_v2 (result.json only)
│   └── <backbone>_<variant>@nat768/
│       ├── STATUS.txt        RUNNING | DONE | FAILED, rewritten each epoch
│       ├── report.txt        full metrics + the HYBRID COMPONENTS section
│       ├── train_log.txt     complete console output, flushed per line
│       ├── history.csv       one row appended per epoch
│       ├── result.json       everything machine-readable, incl. learned gamma/gate
│       ├── weights/best.pt   best-on-val weights + frozen thresholds
│       └── predictions/      per-image probabilities, val and test
└── _combined/                written by cell 18, from whichever notebook runs it
    └── comparison.txt/.csv   THE 21-run cross-backbone table
```

`last.pt` is deleted once a run finishes — `best.pt` has the weights that matter.

## Four things that bound what these numbers mean

Put all four next to the headline in the results chapter:

1. **A delta under ~0.01 macro AUPRC is a tie.** ~660 test records, one seed.
2. **One seed per cell.** Twenty-one runs at three seeds would be four days.
   Nothing here estimates run-to-run variance, so no confidence interval can be
   attached to any single delta — the complete 3 × 7 grid is what stands in for
   it. A component that helps on one backbone is one draw; one that helps on all
   three is a pattern. **This notebook alone cannot make that call** — cell 18
   can.
3. **Prototype subset.** Train is 1 augmentation variant per record instead of 4.
   AUROC/AUPRC and the *ranking* transfer to the full corpus; thresholds,
   calibration and absolute scores do not.
4. **Single-source STEMI.** All 1,442 STEMI records come from one database —
   cell 16. The cross-source test is still outstanding.
"""),
    ]

    return {"cells": cells,
            "metadata": {"accelerator": "GPU",
                         "colab": {"provenance": [], "gpuType": "L4"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def validate(nb: dict, label: str) -> None:
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
            out, cont = [], False
            for l in joined.split("\n"):
                stripped = l.lstrip()
                if cont:
                    # A continuation of a `!` line is part of that shell command,
                    # not python. Blanking it (rather than writing an indented
                    # `pass`) is what keeps the stand-in parseable.
                    cont = l.rstrip().endswith("\\")
                    out.append("")
                    continue
                if stripped.startswith(("!", "%")):
                    cont = l.rstrip().endswith("\\")
                    out.append(" " * (len(l) - len(stripped)) + "pass")
                else:
                    out.append(l)
            try:
                ast.parse("\n".join(out))
            except SyntaxError as e:
                problems.append(f"cell {i}: python does not parse: {e}")
    if problems:
        raise SystemExit(f"NOTEBOOK VALIDATION FAILED ({label}):\n  - "
                         + "\n  - ".join(problems))
    print(f"  {label}: validated {len(nb['cells'])} cells")


if __name__ == "__main__":
    b64 = payload()
    print(f"embedded package: {sum(1 for f in FILES if f.exists())} files, "
          f"{len(b64)/1024:.0f} KB base64\n")
    for letter, bb, nice, stem, hours, note in FAMILIES:
        nb = build(letter, bb, nice, hours, note, b64)
        name = f"CardioSentry_Hybrids_v2_{stem}.ipynb"
        validate(nb, name)
        out = HERE / name
        out.write_text(json.dumps(nb, indent=1))
        print(f"  wrote {name}  ({out.stat().st_size/1024:.0f} KB)  "
              f"{nice}: 7 runs, ~{hours} h")
    print("\nupload the THREE notebooks to Colab and run them in parallel -")
    print("the code travels inside each one, and each writes to its own")
    print("runs/hybrids_v2/<family>/ folder, so nothing collides.")
    print(f"Drive home: {DRIVE}/codes/hybrids_v2/")
