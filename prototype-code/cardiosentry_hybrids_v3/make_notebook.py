#!/usr/bin/env python3
"""
Build the TWO self-contained Colab notebooks - one per backbone group.

    python make_notebook.py
      ->  CardioSentry_Hybrids_v3_A_CBAM_CNNs.ipynb
          CardioSentry_Hybrids_v3_B_CBAM_Transformers.ipynb

WHY TWO, AND WHY SPLIT ON THE FEATURE FORMAT

Five runs is ~18 hours in one stream. The two groups are completely independent -
nothing in B reads anything A produces - so run them in two Colab sessions at
once and the wall clock becomes the LONGER group (~10 h) rather than the sum.

The split is on what `forward_features` hands back, which is also the cost
boundary and the interesting scientific boundary:

    A  cnn           densenet121, efficientnet_b3, convnext_base   NCHW,   3 runs
    B  transformer   swinv2_base, vit_b16                          NHWC / tokens

That is what makes each notebook a self-contained claim - "CBAM across three
convolutional backbones" and "CBAM across two attention backbones" - rather than
a fragment. A split by run count (three and two, ordered by cost) would leave
each session holding pieces of both claims and nothing reportable until the other
had finished too.

WHAT THE SPLIT HAD TO FIX

Two Colab sessions on one Drive is two writers. Each notebook writes to
`runs/hybrids_v3/<group>/`, its OWN directory, so `_SUMMARY.txt`, `_batch_log.txt`
and `comparison.csv` have exactly one writer each. Two things are genuinely
shared, and both were made safe rather than assumed safe:

  * the pre-resized corpus tar, which both read. `prepare_corpus.sh` stages to a
    PID-suffixed .part, so two simultaneous cold builds cannot interleave into
    one file and then publish it as the cache. It is the SAME tar baselines_v2
    and hybrids_v2 built - same corpus, same geometry - so it is usually a
    two-minute restore.
  * the cross-group table. `compare.py --runs` takes several folders, so the
    five-run table is built at the end by pointing it at both.

The harness is 17 Python modules. You do not paste modules into notebook cells;
that is not what cells are for. Cell 1 of each notebook carries the whole package
as a base64 tar.gz and unpacks it to /content/code. So each notebook is the ONLY
file you upload.

Re-run this script after editing any module to refresh both embedded copies.
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

GROUP_KEYS = ["cnn", "transformer"]

# (letter, group key, display name, notebook file stem, hours, the group's point)
GROUPS = [
    ("A", "cnn", "the three CNNs", "A_CBAM_CNNs", 8,
     "DenseNet-121, EfficientNet-B3 and ConvNeXt-Base. All three hand back a "
     "plain NCHW feature map at stride 32, so CBAM sits on them exactly as it "
     "did in hybrids_v2 - which is what makes this group the control for the "
     "other one."),
    ("B", "transformer", "the two transformers", "B_CBAM_Transformers", 10,
     "SwinV2-Base and ViT-B/16. Neither returns an NCHW map: Swin is "
     "channels-LAST and ViT is a token sequence with a class token at stride "
     "16. This is the group that tests whether CBAM's effect survives leaving "
     "the convolutional family at all."),
]

# per-backbone notes for the run table in each notebook
BACKBONE_NOTES = {
    "densenet121": ("DenseNet-121", "nchw", 4,
                    "8 M parameters and the largest activation footprint here - "
                    "dense connectivity keeps every feature map alive"),
    "efficientnet_b3": ("EfficientNet-B3", "nchw", 8,
                        "recovered strongly from 384 to 768 in v1 "
                        "(mAUPRC 0.857 -> 0.931), so the most resolution-"
                        "sensitive backbone in the study"),
    "convnext_base": ("ConvNeXt-Base", "nchw", 4,
                      "7x7 depthwise convolutions already give it a wide "
                      "receptive field - the sharpest test of whether CBAM adds "
                      "anything a modern CNN lacks"),
    "swinv2_base": ("SwinV2-Base", "nhwc", 2,
                    "hierarchical windowed attention; 768x1024 is already a "
                    "multiple of 256 so nothing is padded"),
    "vit_b16": ("ViT-B/16", "tokens", 2,
                "stride 16, so a 48x64 = 3072-cell grid - CBAM's spatial gate "
                "reads ~4.5 mm cells here against ~9 mm everywhere else"),
}


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


def build(letter: str, group: str, nice: str, hours: int, group_note: str,
          b64: str) -> dict:
    import models as M

    names = ", ".join(f.name for f in FILES if f.exists())
    backbones = M.GROUPS[group]
    runs = M.models_in_group(group)
    other = next((l, n) for l, g, n, *_ in GROUPS if g != group)
    other_txt = f"**{other[0]}** ({other[1]})"
    bb_csv = ",".join(backbones)

    run_rows = "\n".join(
        f"| `{b}_cbam` | {BACKBONE_NOTES[b][0]} | `{BACKBONE_NOTES[b][1]}` | "
        f"{BACKBONE_NOTES[b][2]} | {BACKBONE_NOTES[b][3]} |"
        for b in backbones)
    first_run = runs[0]

    cells = [
        md(f"""
# CardioSentry — hybrids v3 · **notebook {letter} of 2: {nice}**

**One component, five backbones.** Every run here is a bare backbone from
`cardiosentry_baselines_v2` with **CBAM inserted** and nothing else changed, so
every result is reported as a **delta against that backbone's own baseline row**.

This notebook trains **{len(runs)} of the 5** and nothing else. Notebook
{other_txt} trains the rest and is **meant to run at the same time**.

> **Colab limits concurrent GPU sessions per account** — typically one on the
> free tier and two to three on Pro/Pro+. Two at once needs that headroom, or a
> second account. If you can only run one, start **B (the transformers)** first:
> it is the long pole.

| | wall clock (estimated) |
|---|---|
| all 5 runs in one session | ~18 h |
| two sessions in parallel | **~10 h** (the longer group) |

*Those are estimates scaled from the hybrids_v2 runs at this geometry, not
measurements — three of these five backbones have never been trained through this
harness. Cell 5 measures the real per-step cost before you commit the night.*

## The {len(runs)} runs in this notebook

| run | backbone | feature layout | micro-batch | why it is here |
|---|---|---|---|---|
{run_rows}

*{group_note}*

## What this package asks that hybrids_v2 could not

`hybrids_v2` ran seven components over **three convolutional backbones**. It can
say "CBAM helped or hurt ResNet-50, EfficientNetV2-S and VGG-16" — three
backbones with the same stride, the same NCHW feature layout and broadly the same
inductive bias. This package puts **the same CBAM** on a densely-connected CNN, a
compound-scaled CNN, a modernised CNN, a windowed transformer and a plain ViT.

Together the two packages give **eight backbones spanning three feature
layouts**, which is what turns "CBAM helps" from an observation about one model
family into a claim that either replicates across families or does not.

## The number that matters is a difference, not a score

`config.py` here is **byte-identical** to `baselines_v2`'s, so `fingerprint()`
matches and `compare.py` ranks a baseline run and a hybrid run in one table
without being forced to. Cell 7 imports this group's baseline rows.

That subtraction is only honest if the baseline is genuinely **nested** inside
the hybrid, and it is: with `attn=None` the wrapper reduces to
`forward_features → forward_head(pre_logits=True) → Linear`, which is exactly
what `timm.create_model(tag, num_classes=4)` builds. `selftest.py` asserts the
parameter counts match on all five real backbones rather than asking you to
believe this paragraph.

## The part that is new, and the part that could have gone silently wrong

Only three of the five backbones hand back an NCHW feature map. SwinV2 returns
**channels-last**; ViT returns a **token sequence with a class token**. CBAM is
written for NCHW, so `hybrids.py` converts the features rather than forking the
module — one CBAM on all five, or "CBAM on ViT" and "CBAM on DenseNet" would be
two different modules and this comparison would not exist.

Two properties of that conversion are load-bearing, and `selftest.py` checks both
on the **real** backbones, not on stand-in tensors:

* **the round trip is exact** — `from_map(to_map(f)) == f` bit for bit, so
  "CBAM is the identity at step 0" stays true on all five rather than on the
  three where it is easy;
* **ViT's class token is held out of the grid entirely** — it has no position on
  a 48×64 sheet, and timm's ViT head *pools that token*, so folding it in would
  quietly damage the exact vector the classifier reads.

## Every component starts as the identity

CBAM is wrapped as `x + γ·(refine(x) − x)` with **γ initialised to exactly 0**.
Dropping a fresh sigmoid gate into a pretrained backbone otherwise halves
activations the ImageNet weights were tuned for, and the first epochs go on
undoing that rather than learning — which shows up as "attention hurt" when what
happened was "attention was initialised badly".

**The learned γ is a result in itself.** A γ that never leaves zero says the
backbone did not want the reweighting, and that is a cleaner finding than a noisy
±0.004 AUPRC. Cell 12 tabulates it.

## Geometry and the two data corrections — unchanged from baselines_v2

`nat768` (768×1024, the full sheet at its own aspect), printed record ID
redacted, two corrupt records excluded. Nothing here re-opens those decisions;
changing any of them would change the fingerprint and break the comparison this
notebook exists for.

---
Before starting: **Runtime → Change runtime type → GPU**. L4 or A100 preferred.
On a 16 GB T4, halve every `micro_batch` in `models.py` — but read cell 8's note
first, because that is only free on three of the five.
"""),

        md(f"## Cell 1 — unpack the code and install timm\n\n"
           f"*The whole harness ({names}) is embedded in this cell — no folder "
           f"upload needed. Both notebooks carry the same package.*"),
        code(f'''
import base64, io, tarfile, os
os.makedirs("/content/code", exist_ok=True)
_PKG = "{b64}"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(_PKG)), mode="r:gz").extractall("/content/code")
for _f in os.listdir("/content/code"):
    if _f.endswith(".sh"):
        os.chmod(f"/content/code/{{_f}}", 0o755)
os.environ["CS_DIM"] = "nat768"          # the geometry, fixed for every cell below
GROUP = "{group}"                        # this notebook trains only this group
print("unpacked:")
for f in sorted(os.listdir("/content/code")):
    print("   ", f)
print("\\ngroup:", GROUP)

!pip install -q "timm>=1.0.9"
!nvidia-smi --query-gpu=name,memory.total --format=csv
'''),

        md(f"""
## Cell 2 — mount Drive, set up caching, extract the dataset

**Add your HuggingFace token first** (one-time, ~30 s): key icon in the left
sidebar → **+ Add new secret** → Name `HF_TOKEN`, Value your `hf_...` token →
toggle **Notebook access** on. Do not paste it into a cell; it would be saved
inside the `.ipynb` and travel with any copy you share.

The dataset is extracted to **local disk** (`/content/`), never trained off the
mounted Drive — every image read would become a network round trip and starve the
GPU. **Everything the run produces still goes to Drive**, written as it happens.

| folder | what lands there | writers |
|---|---|---|
| `datasets/` | `prototype_1152.tar` — read only | none |
| `experiments/baselines_v2/data/` | **the pre-resized corpus** — shared, read-mostly | see below |
| `experiments/hybrids_v3/{group}/samples/` | this group's sample sheets | this notebook |
| `runs/hybrids_v3/{group}/` | **this group's runs**: logs, metrics, weights, predictions | this notebook |
| `runs/baselines_v2/` | read only — where this group's baseline rows come from | none |
| `hf_cache/` | the pretrained backbones, downloaded once | both (safe: per-file locks, and different files) |

**Everything this notebook writes is under a `{group}` path.** That is what makes
two concurrent sessions safe: no two of them touch the same file.

⚠️ **Run this cell in ONE notebook first, wait for it, then start the other.**
It takes seconds, and it is the one step that must not be done in parallel.
Google Drive is not a POSIX filesystem — it permits two folders with the SAME
NAME in the same parent — and `mkdir -p` on a mounted Drive goes through the
Drive API. Two sessions creating `runs/hybrids_v3/` at the same instant can end
up with two *different* folders both called `hybrids_v3`, and your results would
be silently split across them with the final cell seeing only one.

**The pre-resized corpus cache is deliberately pointed at `baselines_v2`.** Same
corpus, same geometry, keyed by `<corpus>_<dim>.tar` with its geometry marker
checked before extraction — so this restores the tar `baselines_v2` (or
`hybrids_v2`) already built instead of building a third identical copy. If it is
**not** there, run cell 4 in this notebook alone first and start the other once it
finishes.

Budget about **{2 + hours // 3} GB of Drive for this group** (`best.pt` is
{'~350 MB for the two Base transformers' if group == 'transformer' else '~30-350 MB per run'}),
plus the shared 2.2 GB corpus tar if it is not already there.
"""),
        code(f'''
import os
from google.colab import drive
drive.mount('/content/drive')

DRIVE = "/content/drive/MyDrive/CardioSentry"
DIM   = "nat768"
GROUP = "{group}"
os.environ["CS_DIM"] = DIM

# Cache pretrained weights on Drive. Without this every runtime reset
# re-downloads the backbones from HuggingFace - and this group's are
# {'86-88 M parameters apiece' if group == 'transformer' else '7-88 M parameters apiece'}.
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
EXP   = f"{{DRIVE}}/experiments/hybrids_v3/{{GROUP}}"
CACHE = f"{{DRIVE}}/experiments/baselines_v2/data"    # SHARED, read-mostly
HYB   = f"{{DRIVE}}/runs/hybrids_v3"                  # the parent of both groups
RUNS  = f"{{HYB}}/{{GROUP}}"                          # THIS notebook's only output
BASE_RUNS = f"{{DRIVE}}/runs/baselines_v2"            # read only: the reference rows
BACKBONES = "{bb_csv}"                                # this group's baseline rows

if not os.path.exists(f"{{RAW}}/index.csv"):
    !tar -xf "$TAR" -C /content/
!ls "$RAW"

# The SHARED parents first and on their own. See the warning above: creating
# these concurrently from two sessions can produce duplicate same-name folders
# on Drive, which would split the results without any error.
!mkdir -p "$HYB" "$DRIVE/experiments/hybrids_v3" "$CACHE"
# ...then this group's own directories, which the other session never touches.
!mkdir -p "$RUNS" "$EXP"

_siblings = [d for d in os.listdir(f"{{DRIVE}}/runs") if d.startswith("hybrids_v3")]
print("runs/ entries matching hybrids_v3*:", _siblings)
if len(_siblings) > 1:
    print("  *** MORE THAN ONE. Drive has duplicated the parent folder - merge them")
    print("      in the Drive web UI before training, or results will be split.")

print("\\nRAW  =", RAW, "\\nRUNS =", RUNS, "\\nEXP  =", EXP,
      "\\nCACHE =", CACHE, "\\nBASE_RUNS =", BASE_RUNS,
      "\\nBACKBONES =", BACKBONES)

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

The first cell also prints this package's registry: the stride, the attention
grid and the feature layout for every backbone. **Read the `grid` column** — at
768×1024 the stride-32 backbones give CBAM a 24×32 grid (one cell ≈ 8.9 mm of
time, 9.3 mm of voltage) and ViT gives it 48×64. Same module, different
millimetres.

*(Redundant across the two notebooks by design — each writes to its own
`samples/` folder, so running it in both costs a minute and collides with
nothing.)*
"""),
        code(f'''
%cd /content/code
!python dimensions.py | head -12
!python models.py
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

**If you ran `baselines_v2` or `hybrids_v2`, this is a two-minute extract**, not a
rebuild — the cache points at that experiment's `data/` folder and the tar is
keyed by corpus and geometry, so the identical `prototype_1152_nat768.tar` is
reused.

⚠️ **If the cache is cold, run this cell in ONE notebook first and wait.** Both
sessions would otherwise each spend ~12 minutes building the same 2.2 GB corpus.
It is safe — staging is PID-suffixed, so they cannot interleave into one corrupt
file — but it is twice the work for one result.

Why the resize at all: the corpus stores 1686×1311 sheets (2.2 Mpx). Resizing on
the fly decodes 2.2 Mpx to produce 0.79 — every image, every epoch, for every
run. Measured: **41.3 ms/image on the fly vs 6.2 ms from a pre-resized copy**.
The **ID redaction is baked in here too**, so it costs nothing per epoch.

**Training always reads from `/content`.** The Drive copy exists to survive a
restart, never to train off.
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
## Cell 5 — selftest + preflight (~5 min)

**`selftest.py` is not a smoke test.** Every check in it is a bug whose symptom is
a model that trains normally, converges normally, and reports a number meaning
something other than what you think. The ones this package added:

* **the feature-format round trip is EXACT on every real backbone** — checked
  with `torch.equal` on the actual `forward_features` output, so "CBAM is the
  identity at step 0" is verified on ViT and Swin, not assumed from the CNNs
* **ViT's class token is held out of the grid**, and the patch tokens map
  **row-major** onto it — the transposed alternative runs fine and reads a 7×7
  window across TIME believing it is reading across VOLTAGE
* **the recorded stride and feature layout match the real network** — a stride
  recorded as 32 on stride-16 ViT builds the gate over a quarter of the
  millimetres the registry claims
* **the componentless hybrid has the same parameter count as the baselines_v2
  model**, on all five — the claim every delta in this study rests on
* **the two groups partition the run list** — a backbone in neither group is a
  run that never happens, and the table would still look complete

`preflight.py --group {group}` constructs **this group's {len(runs)}** hybrids and
runs a real forward+backward on each, reporting peak VRAM. Checking the other
group's would cost minutes and a Hub download for information this session cannot
act on.

**Read the peak-VRAM column before cell 8.** Three of these backbones have never
been trained through this harness at 768×1024; this is where an OOM is cheap.
"""),
        code('''
%cd /content/code
!python selftest.py --data "$DATA"
'''),
        code('''
!python preflight.py --data "$DATA" --group "$GROUP"
'''),

        md(f"""
## Cell 6 — smoke test (~3 min)

20 train / 10 eval batches on the most expensive run in this group. The metrics
are meaningless; this only proves the path runs end to end — data, model, AMP,
accumulation, checkpointing — before you commit {hours} hours.
"""),
        code(f'''
!python train.py --model {runs[-1]} --data "$DATA" --out /content/smoke --smoke
'''),

        md(f"""
## Cell 7 — import this group's baseline rows

Copies `{'`, `'.join(b + '@nat768' for b in backbones)}` from the `baselines_v2`
runs folder into this group's runs folder — **`result.json` and the text reports
only, never the checkpoints**, which are already on Drive and would duplicate
hundreds of MB to build a table that never opens them.

It **refuses** a run whose `fingerprint` differs from the one these hybrids are
trained under. If that happens, nothing is broken: it means the baseline predates
a config change, and the honest fix is to re-train that backbone rather than to
subtract across two protocols.

Without this cell the runs still train — they just have no reference row, and the
delta is the whole point.
"""),
        code('''
%cd /content/code
!python import_baselines.py --from "$BASE_RUNS" --to "$RUNS" --backbone "$BACKBONES"
'''),

        md(f"""
## Cell 8 — train the {len(runs)}  ⏳ **THE OVERNIGHT CELL**  (~{hours} h estimated)

**Start the other notebook now**, if you have the session headroom — it does not
touch anything this one writes.

When Colab disconnects, re-run this same cell: finished runs are skipped and an
interrupted one resumes from its last completed epoch. It does **not** abort if
one run fails; the rest still run.
"""),
        code('''
%cd /content/code
!./run_all.sh "$DATA" "$RUNS" --group "$GROUP"
'''),

        md(f"""
### If VRAM is tighter than you like

`--micro-batch` is passed through to `train.py`. It is a VRAM accommodation, not
an experimental variable: accumulation restores the effective batch of 32 and
`steps_per_epoch` is **151** at 2, 4, 8 or 16, so the LR schedule does not move.

It is **only** free on a backbone with no BatchNorm. BN normalises over the
*micro*-batch and accumulation sums gradients — it does not merge batch
statistics — so changing it changes what the network computes, while the loss
curve looks perfectly normal and the delta against the baseline quietly stops
meaning what it says. `check_micro_batch.py` therefore **refuses** it there
rather than warning:

| backbone | BatchNorm layers | micro-batch |
|---|:--:|---|
| DenseNet-121 | 121 | **LOCKED** at 4 |
| EfficientNet-B3 | 78 | **LOCKED** at 8 |
| ConvNeXt-Base | 0 | free |
| SwinV2-Base | 0 | free |
| ViT-B/16 | 0 | free |

The three free ones are LayerNorm networks, so raising or halving is a pure
throughput/VRAM trade with an identical accumulated gradient.
"""),
        code(f'''
# !./run_all.sh "$DATA" "$RUNS" --group "$GROUP" --micro-batch 4
'''),

        md("### One run at a time\n\nIf you would rather step through them."),
        code("\n".join(
            [f'# !python train.py --model {r:<22} --data "$DATA" --out "$RUNS"'
             for r in runs])),

        md("## Cell 9 — morning check\n\nWorks even if the group is only part finished."),
        code('''
!cat "$RUNS/_SUMMARY.txt"
'''),
        code('''
# did anything fail overnight?
!for f in "$RUNS"/*/STATUS.txt; do echo "--- $f"; head -3 "$f"; done
!ls "$RUNS"/*/ERROR.txt 2>/dev/null && echo "^ failures above" || echo "no failures"
'''),

        md(f"""
## Cell 10 — this group's comparison

`compare.py` first verifies the runs are actually comparable — matching config
fingerprints, same dataset, same row counts, no smoke runs — and **refuses to
rank them if not**. Read the **DELTA AGAINST THE BACKBONE BASELINE** section at
the bottom; everything above it is context.

This is {nice} only. The last cell does both groups together.
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
learned γ instead.

CBAM adds between **0.08 M and 0.30 M** parameters depending on the backbone's
feature width, which is +0.09% on ViT-B/16 and +2.8% on EfficientNet-B3. Those
are different bargains for the same component, and the `added_M` column is what
makes them comparable.
"""),
        code('''
import pandas as pd
pd.set_option("display.width", 220, "display.max_columns", 60)
df = pd.read_csv(f"{RUNS}/comparison.csv")

base = df[df.is_baseline].set_index("backbone")
if base.empty:
    print("no baseline rows imported - run cell 7")
else:
    h = df[~df.is_baseline].copy()
    rows = []
    for _, r in h.iterrows():
        b = base.loc[r.backbone] if r.backbone in base.index else None
        rows.append({
            "model": r.model,
            "macro_auprc": r.macro_auprc,
            "d_macro_auprc": (r.macro_auprc - b.macro_auprc) if b is not None else float("nan"),
            "d_macro_auroc": (r.macro_auroc - b.macro_auroc) if b is not None else float("nan"),
            "d_macro_f1":    (r.macro_f1    - b.macro_f1)    if b is not None else float("nan"),
            "d_STEMI_auprc": (r.STEMI_auprc - b.STEMI_auprc) if b is not None else float("nan"),
            "gamma": r.get("attention_gamma", float("nan")),
            "added_M": r.added_params / 1e6 if "added_params" in r else float("nan"),
        })
    t = pd.DataFrame(rows).sort_values("d_macro_auprc", ascending=False)
    print(t.round(4).to_string(index=False))
    print("\\nd_* are against each backbone's OWN baselines_v2 row.")
    print("Under ~0.01 macro AUPRC is a tie on ~660 test records at one seed.")
'''),

        md("""
## Cell 12 — what CBAM learned  ⚠️ read before writing anything up

γ is the residual weight on the attention branch, **initialised to exactly 0**:

    output = features + γ · (CBAM(features) − features)

A γ that never left zero means the backbone did not want the reweighting, and any
score difference on that row is noise rather than the module. **A row with a
positive delta and γ ≈ 0 is not "attention helped"** — it is "a
differently-initialised head helped", and it should be reported that way.

γ is also the one number that is directly comparable across all five backbones,
because the wrapper is identical on every one of them.
"""),
        code('''
import pandas as pd
df = pd.read_csv(f"{RUNS}/comparison.csv")
cols = ["model", "backbone", "attention", "attention_gamma",
        "added_params", "macro_auprc"]
t = df[~df.is_baseline][[c for c in cols if c in df.columns]].copy()
if "added_params" in t:
    t["added_M"] = t.pop("added_params") / 1e6
print(t.sort_values("model").round(4).to_string(index=False))
print("\\ngamma near 0 -> CBAM effectively turned itself OFF on that backbone.")
print("Compare gamma ACROSS backbones: same wrapper, same init, so the numbers")
print("are on one scale even though the feature layouts are not.")
'''),

        md("""
## Cell 13 — per-condition breakdown

Every metric, per head, per run. STEMI is the minority head and the clinically
critical one — read its column first, next to cell 15.
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

        md("## Cell 14 — training curves and accuracy vs cost"),
        code('''
import json, glob
import matplotlib.pyplot as plt

runs = [json.load(open(f)) for f in sorted(glob.glob(f"{RUNS}/*/result.json"))]
runs = [r for r in runs if not r.get("smoke")]

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for r in runs:
    h = r["history"]
    if not h:
        continue                                  # an imported baseline row
    ep = [x["epoch"] for x in h]
    ax[0].plot(ep, [x["train_loss"] for x in h], label=r["model"], lw=1.6)
    ax[1].plot(ep, [x["val_macro_auprc"] for x in h], label=r["model"], lw=1.6)
for a, t in zip(ax, [f"{GROUP}: train loss", f"{GROUP}: val macro AUPRC"]):
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
    is_base = not cp                              # the imported bare backbones
    for a, x in ((ax[0], r["complexity"]["params_total_m"]), (ax[1], added)):
        a.scatter(x, y, s=170 if is_base else 80, color="C3" if is_base else "C0",
                  marker="*" if is_base else "o", zorder=3)
        a.annotate(r["model"], (x, y), textcoords="offset points",
                   xytext=(7, 4), fontsize=8)
ax[0].set_xlabel("total parameters (M)")
ax[1].set_xlabel("parameters ADDED over the bare backbone (M)")
for a in ax:
    a.set_ylabel("test macro AUPRC"); a.grid(alpha=.3)
ax[0].set_title("red stars = the bare backbone baselines")
plt.tight_layout(); plt.show()
'''),

        md("""
## Cell 15 — source confound check  ⚠️ do not skip

All 1,442 STEMI records come from a single database, and stage 5 of the image
pipeline measured residual source signal (probe accuracy 0.515 vs 0.372 chance).
A high STEMI score cannot by itself separate *recognising infarction* from
*recognising the Chongqing scanner*. The printed ID is redacted, which closes one
route to that shortcut — this checks the other.

**This is the check CBAM specifically demands.** Its spatial gate learns *which
region of the sheet matters*, and an acquisition artifact is a region. Watch for a
hybrid that **widens** the per-source spread relative to its baseline: that is
what a gate which learned to look at the scanner rather than the ST segment would
do, while the headline AUPRC went up.
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
    is_base = not (r.get("hybrid") or {}).get("component_params")
    rows.append((r["model"], ps, max(v) - min(v), is_base))
base_spread = {m.split("_cbam")[0]: sp for m, _, sp, b in rows if b}
for m, ps, spread, is_base in sorted(rows):
    key = m.split("_cbam")[0]
    delta = ""
    if not is_base and key in base_spread:
        delta = f"   d vs baseline {spread - base_spread[key]:+.3f}"
    print(f"{m:26} " +
          "  ".join(f"{s}={d['macro_auroc']:.3f}(n={d['n']})" for s, d in ps.items()) +
          f"   spread={spread:.3f}{delta}")
print("\\nA POSITIVE d here means CBAM made the per-source gap WIDER.")
'''),

        md("## Cell 16 — full report for one run"),
        code(f'''
import glob
MODEL = "{first_run}"          # change and re-run
hits = glob.glob(f"{{RUNS}}/{{MODEL}}@*/report.txt")
print(open(hits[0]).read() if hits else f"no finished run for {{MODEL}}")
'''),

        md(f"""
## Cell 17 — the five-run table  🔗 **run this once both notebooks are done**

`compare.py --runs` takes several folders, so this builds the full table by
pointing it at both group directories at once. It works from **either** notebook —
they see the same Drive — and it is safe to re-run.

Until both groups have finished it will simply tabulate whatever exists, and say
how many runs it found. **The pattern is only readable once it reports 5 runs plus
5 baselines.**

This is the only cell whose output is a cross-backbone claim. Everything above it
describes {nice} alone.
"""),
        code('''
%cd /content/code
import os
COMBINED = f"{HYB}/_combined"
os.makedirs(COMBINED, exist_ok=True)
FAMS = " ".join(f'"{HYB}/{g}"' for g in ["cnn", "transformer"])

!python compare.py --runs {FAMS} --out "{COMBINED}/comparison.txt" --csv "{COMBINED}/comparison.csv"
'''),
        code('''
import pandas as pd
pd.set_option("display.width", 240, "display.max_columns", 80)
df = pd.read_csv(f"{HYB}/_combined/comparison.csv")

n_h = (~df.is_baseline).sum()
n_b = df.is_baseline.sum()
print(f"hybrid runs finished: {n_h}/5    baseline rows imported: {n_b}/5")
if n_h < 5 or n_b < 5:
    print("NOT EVERYTHING IS IN YET - the table below is partial.\\n")

base = df[df.is_baseline].set_index("backbone")
h = df[~df.is_baseline].copy()
h["baseline_auprc"] = h.backbone.map(base["macro_auprc"])
h["delta"] = h.macro_auprc - h.baseline_auprc
h["added_M"] = h.added_params / 1e6 if "added_params" in h else float("nan")
cols = ["backbone", "baseline_auprc", "macro_auprc", "delta",
        "attention_gamma", "added_M"]
print("CBAM vs the bare backbone, one row per backbone:")
print(h[[c for c in cols if c in h.columns]]
      .sort_values("delta", ascending=False).round(4).to_string(index=False))

pos = (h.delta > 0).sum()
print(f"\\nCBAM improved macro AUPRC on {pos} of {len(h)} backbones.")
print("A delta under ~0.01 is a tie. What is worth writing up is the SHAPE:")
print("  helps almost everywhere            -> a real, backbone-independent effect")
print("  helps the CNNs, not the attention  -> CBAM is supplying something")
print("                                        self-attention already has")
print("  scattered, all inside +/-0.01      -> one seed of noise; say so")
print("\\nRead this next to the gamma column. A positive delta with gamma near 0")
print("is a differently-initialised head, not attention.")
'''),

        md(f"""
## Where everything lives

```
{DRIVE}/runs/hybrids_v3/
├── cnn/                <- notebook A writes only here
├── transformer/        <- notebook B writes only here
│   ├── _SUMMARY.txt          every run in the group at a glance
│   ├── _batch_log.txt
│   ├── comparison.txt/.csv   this group's ranking + delta table
│   ├── <backbone>@nat768/    imported from baselines_v2 (result.json only)
│   └── <backbone>_cbam@nat768/
│       ├── STATUS.txt        RUNNING | DONE | FAILED, rewritten each epoch
│       ├── report.txt        full metrics + the HYBRID COMPONENTS section
│       ├── train_log.txt     complete console output, flushed per line
│       ├── history.csv       one row appended per epoch
│       ├── result.json       everything machine-readable, incl. the learned gamma
│       ├── weights/best.pt   best-on-val weights + frozen thresholds
│       └── predictions/      per-image probabilities, val and test
└── _combined/                written by cell 17, from whichever notebook runs it
    └── comparison.txt/.csv   THE 5-run table
```

`last.pt` is deleted once a run finishes — `best.pt` has the weights that matter.

## Five things that bound what these numbers mean

Put all of them next to the headline in the results chapter:

1. **A delta under ~0.01 macro AUPRC is a tie.** ~660 test records, one seed.
2. **One seed per cell.** Nothing here estimates run-to-run variance, so no
   confidence interval can be attached to any single delta. What stands in for it
   is the shape across five very different backbones — and **this notebook alone
   cannot see that shape**; cell 17 can.
3. **Pretraining is not uniform.** ViT-B/16's checkpoint is in21k→in1k; the other
   four are supervised in1k. This does *not* contaminate any delta (each hybrid is
   measured against a baseline with the same weights) but it does contaminate any
   statement about absolute scores across backbones.
4. **Prototype subset.** Train is 1 augmentation variant per record instead of 4.
   AUROC/AUPRC and the *ranking* transfer to the full corpus; thresholds,
   calibration and absolute scores do not.
5. **Single-source STEMI.** All 1,442 STEMI records come from one database —
   cell 15, which matters more for a spatial attention module than for a plain
   backbone. The cross-source test is still outstanding.
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
    import models as M

    # The notebooks select their work with --group, so a group the registry does
    # not know about would produce a notebook that trains nothing and reports an
    # empty table. Fail here instead.
    unknown = [g for _, g, *_ in GROUPS if g not in M.GROUPS]
    missing = [g for g in M.GROUPS if g not in [x[1] for x in GROUPS]]
    if unknown or missing:
        raise SystemExit(f"make_notebook.py and models.GROUPS disagree: "
                         f"unknown here {unknown}, missing here {missing}")

    b64 = payload()
    print(f"embedded package: {sum(1 for f in FILES if f.exists())} files, "
          f"{len(b64)/1024:.0f} KB base64\n")
    for letter, group, nice, stem, hours, note in GROUPS:
        nb = build(letter, group, nice, hours, note, b64)
        name = f"CardioSentry_Hybrids_v3_{stem}.ipynb"
        validate(nb, name)
        out = HERE / name
        out.write_text(json.dumps(nb, indent=1))
        print(f"  wrote {name}  ({out.stat().st_size/1024:.0f} KB)  "
              f"{nice}: {len(M.models_in_group(group))} runs, ~{hours} h")
    print("\nupload BOTH notebooks to Colab and run them in parallel -")
    print("the code travels inside each one, and each writes to its own")
    print("runs/hybrids_v3/<group>/ folder, so nothing collides.")
    print(f"Drive home: {DRIVE}/codes/hybrids_v3/")
