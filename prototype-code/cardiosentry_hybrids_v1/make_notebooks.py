#!/usr/bin/env python3
"""
Build the SELF-CONTAINED Colab notebooks for round 2.

    python make_notebooks.py

The harness is a dozen Python modules. You do not paste modules into notebook
cells - that is not what cells are for. Cell 1 of every notebook carries the
whole package as a base64 tar.gz and unpacks it to /content/code; every later
cell is a one-line command that runs those files.

WHY FOUR NOTEBOOKS AND NOT ONE

Seven runs is roughly 8 GPU-hours on a T4, which no single Colab session will
survive. Splitting them means a disconnect costs you one notebook, not the
night. They are ordered so that each one is worth reading on its own:

  0_Prepare   build the 768px dataset once, tar it to Drive        ~15 min
  A_Controls  resnet50_base, resnet50_hires, cbam, fpn             ~5 h
  B_Hybrids   dual, cbam@768, fpn@768, effb3_dual                  ~4.5 h
  C_ViT       resnet50_vit (768+768), then the full comparison    ~2-3 h (L4)

ONE RUN FOLDER, NOT ONE PER NOTEBOOK

Every notebook writes into the SAME run folder, one subdirectory per model:

    cardiosentry_runs/hybrid_v1/resnet50_cbam_hires/
                               /resnet50_dual/
                               /resnet50_effb3_dual/   ...

That is what keeps them comparable. compare.py reads ONE directory and ranks
whatever it finds there, so giving each notebook its own Drive folder would mean
no table could ever show an A model next to a B model - and the whole question
("does any architecture beat plain ResNet-50 at 768?") spans both. Organisation
comes from the per-model subdirectory names, which are already unique and
already sort sensibly.

Measured on a 2-CPU T4 runtime, not estimated: the 768px corpus is decode-bound
at about 27 img/s, which is slower than the GPU for every 384-only model. A is
the long one and may outlast a free-tier session - run it two models at a time
using its per-model cells; every run resumes from its last completed epoch.

Run 0 first. After that every notebook is independent - run them in any order,
in parallel accounts, or across nights. Each one's comparison cells tabulate
whatever has finished so far, so a half-done round is still readable.

Re-run this script after editing any module to refresh the embedded copy.
"""
from __future__ import annotations
import base64, io, json, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = sorted([p for p in HERE.glob("*.py") if p.name != "make_notebooks.py"]) + \
        [HERE / "requirements.txt", HERE / "run_hybrids.sh"]

# ---------------------------------------------------------------- Drive layout
#
# ONE root, everything derived from it. Change DRIVE_ROOT and every notebook
# follows; nothing else in this file hard-codes a Drive path.
#
#   MyDrive/CardioSentry/
#     cardiosentry_code/          the harness, for reference - notebooks carry
#                                 their own copy embedded in cell 1
#     cardiosentry_dataset/       every dataset tar
#     cardiosentry_runs/          one subfolder per experiment round
#       v1/                       round 1 (five backbones)
#       hybrid_v1/                THIS round
#     hf_cache/                   pretrained weights, cached once
#
# Run folders are named by round only - the parent is already called
# cardiosentry_runs, so a "cardiosentry_" prefix on the child says nothing.
DRIVE_ROOT = "/content/drive/MyDrive/CardioSentry"
RUN_NAME = "hybrid_v1"

DRIVE_DATASETS = f"{DRIVE_ROOT}/cardiosentry_dataset"
DRIVE_SRC_TAR = f"{DRIVE_DATASETS}/prototype_1152.tar"
DRIVE_768_TAR = f"{DRIVE_DATASETS}/prototype_1152_768.tar"
DRIVE_HF_CACHE = f"{DRIVE_ROOT}/hf_cache"
RUNS = f"{DRIVE_ROOT}/cardiosentry_runs/{RUN_NAME}"


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

B64 = payload()

UNPACK = f'''
import base64, io, tarfile, os
os.makedirs("/content/code", exist_ok=True)
_PKG = "{B64}"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(_PKG)), mode="r:gz").extractall("/content/code")
os.chmod("/content/code/run_hybrids.sh", 0o755)
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
print(f"loader will use {{min(_cores, 8)}} worker(s)")
if _cores <= 2:
    print("  2 vCPU -> decode-bound at ~27 img/s. On Colab Pro switch to L4:")
    print("  Runtime > Change runtime type > L4. Same compute units for this")
    print("  round, under half the wall clock.")
else:
    print(f"  projected decode ceiling ~{{27 * min(_cores, 8) / 2:.0f}} img/s "
          f"(the 2-vCPU T4 baseline was 27)")
print("  RAM is NOT the bottleneck - peak usage here is about 5 GB. Turn on")
print("  High-RAM only if it raises the vCPU count on your runtime; if it")
print("  grants memory alone it costs compute units and buys nothing.")
'''

MOUNT = f'''
import os
from google.colab import drive
drive.mount('/content/drive')

# Cache pretrained weights on Drive (~1.4 GB, once). Without this every runtime
# reset re-downloads them from HuggingFace - slow, and it is what triggers the
# "unauthenticated requests" rate limit over a night of reconnects.
os.environ["HF_HOME"] = "{DRIVE_HF_CACHE}"

# HuggingFace token - raises Hub rate limits. Stored in Colab Secrets, never in
# this notebook: a pasted token gets saved into the .ipynb and travels with any
# copy you share.
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

DATASETS = f"{{DRIVE_ROOT}}/cardiosentry_dataset"
SRC_TAR  = f"{{DATASETS}}/prototype_1152.tar"      # the original 1686x1311 corpus
TAR_768  = f"{{DATASETS}}/prototype_1152_768.tar"  # built once by notebook 0
RUNS     = f"{{DRIVE_ROOT}}/cardiosentry_runs/{{RUN_NAME}}"

# ALWAYS local. Training off the mounted Drive turns every image read into a
# network round trip and starves the GPU - /content is wiped on each runtime
# reset, which is why the notebooks re-extract (about a minute) every session.
DATA = "/content/prototype_1152_768"

os.makedirs(RUNS, exist_ok=True)
for _p, _label in ((SRC_TAR, "source corpus"), (TAR_768, "768px corpus")):
    print(f"  {{'FOUND ' if os.path.exists(_p) else 'missing'}}  {{_label:<14}} {{_p}}")
print("\\nDATA =", DATA, "\\nRUNS =", RUNS)
'''

EXTRACT_768 = '''
# Extract the 768px copy built by notebook 0.
#
# Every step is checked. A failed shell command does NOT stop a Colab cell, so
# an unchecked `!tar -xf` prints a traceback into the void and the cell sails on
# to report success - which is exactly how you end up staring at a missing
# index.csv with no idea which step actually failed.
import os, subprocess, tarfile

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
        f"Run notebook 0_Prepare first - it builds this once and tars it to Drive.")
else:
    gb = os.path.getsize(TAR_768) / 1024**3
    print(f"extracting {TAR_768}  ({gb:.2f} GB)")

    # Verify the archive BEFORE trusting it. Writing ~2 GB straight to the Drive
    # FUSE mount truncates more often than anyone expects, especially if the
    # notebook-0 session ended before Drive finished flushing. prepare_dataset.py
    # copies index.csv LAST, so it sits near the end of the archive - a truncated
    # tar loses precisely that file and nothing else, which looks like a mystery
    # rather than a broken download.
    try:
        with tarfile.open(TAR_768) as _t:
            _names = _t.getnames()
        print(f"  archive intact: {len(_names)} entries")
        _has_index = any(n.endswith("index.csv") for n in _names)
    except Exception as exc:
        raise SystemExit(
            f"\\nTHE TAR ON DRIVE IS TRUNCATED OR CORRUPT: {type(exc).__name__}: {exc}\\n"
            f"  size on Drive: {gb:.2f} GB\\n"
            f"  It never finished writing. Delete it and re-run notebook 0:\\n"
            f"      !rm -f '{TAR_768}'\\n"
            f"  Notebook 0 now builds the tar on local disk and verifies it\\n"
            f"  before copying it to Drive, so this will not recur.")

    r = subprocess.run(["tar", "-xf", TAR_768, "-C", "/content/"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"tar failed (exit {r.returncode}):\\n{r.stderr[:2000]}")
    print("  extracted ->", DATA)

    # index.csv is copied verbatim by prepare_dataset.py and rel_path is unchanged
    # by design, so the copy in the ORIGINAL tar is byte-identical. If only that
    # file is missing there is no need to rebuild 6,670 images to recover it.
    if not _has_index and _images_present(DATA) >= EXPECTED_FILES:
        print("  index.csv missing from the archive but the images are all here")
        print("  recovering it from the original tar (it is copied verbatim by")
        print("  prepare_dataset.py, and rel_path is unchanged by design)")
        import shutil, tempfile
        with tarfile.open(SRC_TAR) as _st:
            _member = next((m for m in _st.getnames()
                            if m.endswith("index.csv")), None)
            if _member is None:
                raise SystemExit(f"no index.csv inside {SRC_TAR} either - "
                                 f"re-run notebook 0 from the original corpus.")
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
        f"Otherwise delete the tar on Drive and re-run notebook 0.")
if n_img < EXPECTED_FILES:
    print(f"WARNING: expected {EXPECTED_FILES:,} images, found {n_img:,}. "
          f"The archive was probably truncated - preflight will fail the "
          f"integrity check. Re-run notebook 0.")

from PIL import Image
import pandas as pd
_df = pd.read_csv(os.path.join(DATA, "index.csv"))
print(f"index.csv: {len(_df):,} rows")
with Image.open(os.path.join(DATA, _df.rel_path.iloc[0])) as _im:
    print("stored image size:", _im.size)
    assert min(_im.size) >= 768, (
        "images are %s - this is the round-1 384px copy, not the 768px one. "
        "The resolution experiment would measure nothing." % (_im.size,))
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


def header(title: str, subtitle: str) -> dict:
    names = ", ".join(f.name for f in FILES if f.exists())
    return md(f"""
# {title}

{subtitle}

Four independent sigmoid heads: **STEMI, AF, LVH, NORMAL**.
This round exists to improve **LVH**, which in round 1 accumulated 2.20 errors
per positive case against 1.11 for AF — the clearest finding in that run, and
unanimous across all five backbones.

**Run the cells in order, top to bottom.**
**Before starting: Runtime → Change runtime type.**

**Measured**, identical code and micro-batch, on `resnet50_base` epoch 2:

| runtime | vCPU | steady rate | whole round | units/hr | **total units** |
|---|---:|---:|---:|---:|---:|
| T4 standard | 2 | 25 img/s | ~11 h | 1.0 | ~11 |
| T4 high-RAM | 8 | 84.5 img/s | ~8.0 h | 1.24 | ~10.0 |
| **L4** | 12 | **149.7 img/s** | **~4.5 h** | 1.54 | **~7.0** |

**L4 is the cheaper option, not the dearer one.** It is 1.77x faster for 1.24x
the hourly rate, so it uses ~30% *fewer* compute units and half the wall clock.
Hourly rate is the wrong number to optimise — you rent by the hour and need
fewer of them.

The 22 GB matters too: it protects the uniform micro-batch of 8 (cell 5), which
is what keeps BatchNorm comparable across runs. On a 14.6 GB T4 the heaviest
models are tight at micro 8, and an OOM there forces *every* run down to 4.

---
*The whole harness ({names}) is embedded in cell 1 — no folder upload needed.*
""")


# --------------------------------------------------------------- notebook 0
def nb_prepare() -> dict:
    return {"cells": [
        header("CardioSentry hybrids — 0. Prepare the dataset",
               """Builds the **768×768** copy of `prototype_1152` that every round-2 run reads,
and tars it to Drive so notebooks A, B and C can each extract it in a minute
instead of rebuilding it in fifteen.

**Run this once.** Roughly 15 minutes, and it needs ~2 GB of Drive space."""),

        md("""
## Why 768, and why you cannot reuse round 1's folder

Round 1 trained off `prototype_1152_384`. Resizing that to 768 would **upsample**
a 384px JPEG and invent detail that is not in it — the resolution comparison
would then measure nothing at all.

So this copy is built from the **original** `prototype_1152.tar` (1686×1311
sheets). Every view any model wants is derived from it:

| view | px/mm vertical | px per mV | px/mm horizontal |
|---:|---:|---:|---:|
| 384 (round 1) | 1.73 | 17.3 | 1.35 |
| **768 (round 2)** | **3.46** | **34.6** | **2.71** |

Sokolow-Lyon (S<sub>V1</sub>+R<sub>V5</sub> > 35 mm) and Cornell turn on
differences of a few millimetres. Whether that doubling matters is the question
the round asks.
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
## Cell 2b — reset (only if a previous attempt left a mess)

**Skip this on a first run.** Use it when notebook 0 died partway and you want
to start clean.

The build is **resumable**, so the gentle option is usually right: delete only
the bad tar on Drive and let cell 3 continue from whatever images it already
wrote. Images written before a crash are complete, valid files — the crash
happens between files, never inside one.

`prototype_1152.tar` is **never** deleted. That is the original corpus, it is
the only thing here you cannot rebuild, and `index.csv` is recovered from it if
the 768 archive ever loses it.
"""),
        code('''
import os, shutil

RESET = "none"        # "none" | "tar" | "all"

# "tar"  - drop the Drive archive only. Keeps any local images already built,
#          so cell 3 resumes instead of restarting. Use this after a failed
#          or interrupted notebook-0 run.
# "all"  - drop the archive AND the local 768px folder. Full rebuild, ~15 min.
#          Use this if you suspect the images themselves are wrong, e.g. they
#          were built at the wrong size.

targets = {
    "none": [],
    "tar":  [TAR_768],
    "all":  [TAR_768, DATA, "/content/prototype_1152"],
}[RESET]

if not targets:
    print("RESET = 'none' - nothing deleted.")
    print("Set RESET to 'tar' or 'all' above and re-run this cell if you need to.")
else:
    for t in targets:
        if os.path.isdir(t):
            n = sum(len(f) for _, _, f in os.walk(t))
            shutil.rmtree(t); print(f"removed dir  {t}  ({n:,} files)")
        elif os.path.isfile(t):
            gb = os.path.getsize(t) / 1024**3
            os.remove(t); print(f"removed file {t}  ({gb:.2f} GB)")
        else:
            print(f"not present  {t}")

# The source corpus is never a target. Confirm it survived.
print(f"\\nsource corpus intact: {os.path.exists(SRC_TAR)}  {SRC_TAR}")
assert os.path.exists(SRC_TAR), (
    "prototype_1152.tar is missing from Drive - that is the one file that "
    "cannot be rebuilt. Re-upload it before continuing.")
'''),

        md("""
## Cell 3 — extract the original corpus and build the 768px copy  ⏳ ~15 min

Extract to **local disk** (`/content/`), never work off the mounted Drive —
every image read would become a network round trip.

`prepare_dataset.py` refuses to upsample: if you point it at an already-resized
folder it stops with an explanation rather than producing a corpus that looks
fine and is not.
"""),
        code('''
%cd /content/code
import os, subprocess, sys

EXPECTED = 6670

def n_images(root):
    if not os.path.isdir(root):
        return 0
    return sum(len([f for f in fs if f.endswith(".jpg")])
               for _, _, fs in os.walk(root))

if os.path.exists(os.path.join(DATA, "index.csv")):
    print(f"768px copy already present at {DATA} ({n_images(DATA):,} images)")
elif os.path.exists(TAR_768):
    # The expensive build already happened in an earlier session and the result
    # is on Drive. /content was wiped by the runtime reset, not the dataset.
    # Extracting takes about a minute; rebuilding would take fifteen.
    import tarfile
    try:
        with tarfile.open(TAR_768) as _t:
            _ok = any(n.endswith("index.csv") for n in _t.getnames())
    except Exception:
        _ok = False
    if _ok:
        print(f"768px corpus already built and on Drive - extracting it")
        r = subprocess.run(["tar", "-xf", TAR_768, "-C", "/content/"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"tar failed (exit {r.returncode}):\\n{r.stderr[:2000]}")
        print(f"  {n_images(DATA):,} images at {DATA}")
    else:
        raise SystemExit(
            f"the tar on Drive is incomplete or corrupt:\\n  {TAR_768}\\n"
            f"Set RESET = 'tar' in cell 2b, run it, then run this cell again to "
            f"rebuild.")
else:
    # Extract the source corpus. Checked, because a `!tar` that fails prints a
    # traceback and the cell carries on regardless.
    if not os.path.isdir("/content/prototype_1152"):
        r = subprocess.run(["tar", "-xf", SRC_TAR, "-C", "/content/"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"extracting {SRC_TAR} failed "
                             f"(exit {r.returncode}):\\n{r.stderr[:2000]}")
    n_src = n_images("/content/prototype_1152")
    print(f"source corpus: {n_src:,} images")
    if n_src < EXPECTED:
        raise SystemExit(f"the SOURCE corpus is incomplete: {n_src:,} of "
                         f"{EXPECTED:,}. Re-upload prototype_1152.tar.")

    # Build the 768px copy. Streamed so you can watch it, and its exit status is
    # CHECKED - leaving this unchecked is what quietly produced a 10%-complete
    # corpus that then got tarred to Drive and failed three notebooks later.
    print()
    # sys.executable, not "python": guarantees the same interpreter that is
    # running this notebook, whatever the PATH looks like.
    proc = subprocess.Popen([sys.executable, "prepare_dataset.py",
                             "--src", "/content/prototype_1152"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
    rc = proc.wait()

    n_out = n_images(DATA)
    if rc != 0 or n_out < EXPECTED:
        raise SystemExit(
            f"\\nprepare_dataset.py exited {rc} with {n_out:,} of {EXPECTED:,} images.\\n"
            f"  It is RESUMABLE - run this cell again and it continues from where\\n"
            f"  it stopped. If it dies at the same point every time, the usual\\n"
            f"  causes are a full disk or the OOM killer; retry with\\n"
            f"      !python prepare_dataset.py --src /content/prototype_1152 --workers 1\\n"
            f"  index.csv has deliberately not been written, so no later cell can\\n"
            f"  mistake this for a finished corpus.")

    print(f"\\nbuilt {n_out:,} images at {DATA}")

# Reclaim the ~5 GB the 1686x1311 source occupies; the 768px copy is what every
# run reads from here on.
if os.path.exists(os.path.join(DATA, "index.csv")):
    !rm -rf /content/prototype_1152
!df -h /content | tail -1
'''),

        md("""
## Cell 4 — tar it to Drive so the other notebooks can skip cell 3

~2 GB. Worth it: every later notebook then starts in about a minute.

**Built on local disk first, verified, and only then copied to Drive.** Writing
a multi-gigabyte tar straight to the Drive FUSE mount truncates more often than
anyone expects — and because `prepare_dataset.py` copies `index.csv` *last*, a
truncated archive loses exactly that one file and nothing else. The symptom is a
folder full of images and a missing `index.csv`, which looks like a mystery
rather than a failed download. This cell makes that impossible to miss.
"""),
        code('''
import os, shutil, subprocess, tarfile

EXPECTED_FILES = 6670
LOCAL_TAR = "/content/prototype_1152_768.tar"
os.makedirs(os.path.dirname(TAR_768), exist_ok=True)


def verify(path):
    """Walk the whole archive. Size alone proves nothing about a truncated tar."""
    try:
        with tarfile.open(path) as t:
            names = t.getnames()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not any(n.endswith("index.csv") for n in names):
        return None, "index.csv is not in the archive"
    n_jpg = sum(1 for n in names if n.endswith(".jpg"))
    if n_jpg < EXPECTED_FILES:
        return None, f"only {n_jpg} images in the archive, expected {EXPECTED_FILES}"
    return len(names), None


if os.path.exists(TAR_768):
    n, err = verify(TAR_768)
    if err:
        print(f"the tar already on Drive is BAD ({err}) - rebuilding")
        os.remove(TAR_768)
    else:
        print(f"already on Drive and verified: {TAR_768} ({n} entries)")

if not os.path.exists(TAR_768):
    print("building the archive on local disk ...")
    r = subprocess.run(["tar", "-cf", LOCAL_TAR, "-C", "/content",
                        "prototype_1152_768"], capture_output=True, text=True)
    assert r.returncode == 0, f"tar failed:\\n{r.stderr[:2000]}"

    n, err = verify(LOCAL_TAR)
    assert not err, f"the archive we just built is bad: {err}"
    print(f"  verified locally: {n} entries, "
          f"{os.path.getsize(LOCAL_TAR)/1024**3:.2f} GB")

    print("copying to Drive (this is the slow part) ...")
    shutil.copyfile(LOCAL_TAR, TAR_768)

    # Re-verify AFTER the copy. This is the step that catches a truncated
    # Drive write, and it is the whole reason for the local-first detour.
    n2, err2 = verify(TAR_768)
    if err2:
        os.remove(TAR_768)
        raise SystemExit(
            f"the copy to Drive did not complete ({err2}).\\n"
            f"The bad file has been deleted. Check free space below and re-run "
            f"this cell.")
    assert n2 == n, f"entry count changed in transit: {n} -> {n2}"
    print(f"  verified on Drive: {n2} entries")
    os.remove(LOCAL_TAR)

!ls -lh "$TAR_768"
!df -h /content/drive/MyDrive | tail -1
'''),

        md("""
## Cell 5 — preflight ⚠️ do not skip

Checks the GPU, integrity-checks the dataset, verifies the stored images really
are 768px, loads one real dual-view batch, then builds every model and runs a
real forward+backward reporting **peak VRAM**.

That last part is what turns "OOM at epoch 7 of the dual-resolution run" into a
message in three minutes. Two of the seven models hold two backbones and a 768px
activation stack on the same card.
"""),
        code('''
%cd /content/code
!python preflight.py --data "$DATA"
'''),

        md("""
## Cell 6 — smoke test (~3 min)

20 train / 10 eval batches on the heaviest model. The metrics are meaningless —
this exists only to prove the whole path runs, including the stream ablation,
before you commit hours to it.
"""),
        code('''
!python train.py --model resnet50_dual --data "$DATA" --out /content/smoke --smoke
'''),

        md(f"""
## Done

The dataset is at `{DRIVE_768_TAR}`.\n\nRuns land in `{RUNS}`.

Now open **A_Controls**, then **B_Hybrids**, then **C_ViT** — or run them in
any order, in any number of sessions.

**They all write into the same `{RUNS}` folder**, one subdirectory per model.
That is deliberate: `compare.py` reads a single directory, so a shared folder is
what lets a table show a B1 model next to a B2 one. Each run resumes
independently after a disconnect, and nothing is overwritten — a model already
finished is skipped.
"""),
    ], **_meta()}


# --------------------------------------------------------------- training nbs
def nb_train(title: str, subtitle: str, models: list[str], body_md: str,
             hours: str, parallel: int = 1) -> dict:
    per_model = "\n".join(
        f'# !python train.py --model {m:<20} --data "$DATA" --out "$RUNS"'
        for m in models)
    model_args = " ".join(models)
    return {"cells": [
        header(title, subtitle),
        md(body_md),

        md("## Cell 1 — unpack the code and install timm"),
        code(UNPACK),

        md("## Cell 2 — mount Drive"),
        code(MOUNT),

        md("""
## Cell 3 — extract the 768px dataset (~1 min)

Built once by notebook **0_Prepare**. This cell refuses to fall back to round 1's
384px folder, because feeding upsampled data to the high-resolution models would
invalidate the comparison silently.
"""),
        code(EXTRACT_768),

        md("""
## Cell 4 — preflight for these models only (~2 min)
"""),
        code(f'''
%cd /content/code
!python preflight.py --data "$DATA" --model {" ".join(models)} \
    --micro-batch 8 --parallel {parallel}
'''),

        md(f"""
## Cell 5 — train  ⏳ THE LONG ONE (~{hours})

Writes to Drive as it goes, so a disconnect does not lose finished work, and
re-running resumes each model from its last completed epoch.

`run_hybrids.sh` does **not** abort if one model fails — the rest still run.
"""),
        code(f'''
%cd /content/code

# MICRO_BATCH applies the SAME value to every run, and that matters more than it
# sounds. BatchNorm normalises over the MICRO-batch, and gradient accumulation
# does not touch it - so runs at different micro-batches differ in BN statistics
# on top of whatever they were meant to differ in. On the T4 defaults
# resnet50_base runs at 16 and resnet50_hires at 4, which means the control pair
# is NOT "only the resolution differs".
#
#   24 GB card (L4 / A100) -> "8", and the comparison is clean
#   16 GB card (T4)        -> "4" fits everything but is slower; or leave it
#                             unset for per-model defaults and DECLARE the confound
import os
os.environ["MICRO_BATCH"] = "8"

# PARALLEL runs N models AT ONCE on the one GPU. LEAVE IT AT 1 ON AN L4.
#
# It was tried at 2 and MEASURED SLOWER, on this exact card and corpus:
#
#     model                PARALLEL=2      alone
#     resnet50_dual        19.5 img/s      ~34 img/s
#     resnet50_cbam_hires  11.7 img/s      ~30 img/s
#     aggregate            31.2 img/s
#
# Two models together move fewer images per second than resnet50_dual moves
# on its own. For that group it was ~137 min against ~95 min sequential.
#
# WHY, so nobody re-derives the wrong answer: an L4 at ~6 TFLOP/s against a
# ~121 TFLOP/s fp16 peak LOOKS like it has idle SMs to fill. It does not.
# 768px convolutions move huge activation tensors and the card has only
# ~300 GB/s of memory bandwidth, so the workload is BANDWIDTH-bound, not
# occupancy-bound, and a second process buys nothing while costing cache
# locality. The giveaway is in round A's own numbers: 159.4 -> 43.7 img/s
# from 384 to 768 is 3.65x for exactly 4x the pixels. Near-linear scaling
# with pixel count means the device is already saturated; idle SMs would
# have shown SUBLINEAR scaling.
#
# It also corrupts the COST table. train_seconds and the end-of-run
# inference benchmark both run under contention - and the benchmark runs
# after training, while the other model in the group is still going. That
# is half the "was the second branch worth it" decision, measured wrong.
#
# Accuracy is unaffected either way: same seeds, same micro-batch, same
# batch order, same arithmetic. Only the clock changes.
#
# Worth retrying only on a bandwidth-rich card (A100 ~2 TB/s, H100 ~3 TB/s).
# If a group OOMs there, lower PARALLEL - NEVER MICRO_BATCH, which would
# change BatchNorm statistics and break comparability with every finished run.
os.environ["PARALLEL"] = "{parallel}"

!./run_hybrids.sh "$DATA" "$RUNS" {model_args}
'''),

        md("""
### Prefer one model at a time?

Use these instead of cell 5. Re-running a line after a disconnect resumes it.
If you hit an OOM, add `--micro-batch 2` — gradient accumulation keeps the
effective batch at 32, so the run stays comparable.
"""),
        code(per_model),

        md("## Cell 6 — progress check"),
        code(MORNING),
        code(FAILCHECK),

        md(f"""
## Cell 7 — what these runs said about LVH

Sorted by LVH AUPRC, with deltas against round 1's ResNet-50. Works on a
half-finished batch.
"""),
        code('''
%cd /content/code
!python compare.py --runs "$RUNS" --sort lvh_auprc --csv "$RUNS/comparison.csv" | head -60
'''),

        md("""
## Cell 7b — did anything else break?  ⚠️ read this before the headline

This round targets LVH. **Macro averages dilute a single-head regression by 4×**,
so a model that gains 0.04 on LVH and loses 0.04 on STEMI shows an unchanged
macro AUPRC and reads as a clean win in every headline table.

It is not one. An over-called LVH earns a follow-up echo; a missed STEMI does
not. Round 1's ResNet-50 missed **16 of 146 STEMIs** — that is the number a
"better" model has to not make worse.
"""),
        code('''
%cd /content/code
!python compare.py --runs "$RUNS" 2>/dev/null | sed -n "/PER-CONDITION REGRESSION/,/AF x LVH SUBGROUP/p" | head -50
'''),

        md("""
## Cell 7c — LVH by source  🔴 THE PRIMARY READ, before any headline number

Notebook A is the reason this cell exists. Its pooled result looked clean —
768px lifted LVH AUPRC from 0.9036 to 0.9248, beating both attention hybrids.
Stratifying the test split by originating database took most of it away:

| | pooled | PTB-XL only |
|---|---:|---:|
| `resnet50_base` (384) | 0.9036 | 0.9270 |
| `resnet50_hires` (768) | 0.9248 | **0.9315** |
| delta | **+0.0212** | **+0.0045**  (95% CI −0.0124 … +0.0208) |

**PTB-XL carries 234 of the 255 LVH positives. Chapman carries 21. The STEMI
database carries none** — all 146 of its images are LVH-negative, so a model
that gets better at *recognising that database* stops calling LVH on them and
its pooled AUPRC rises without it having read a single QRS complex. That is
exactly what happened: LVH false positives on STEMI-database images went 11 → 1
from 384 to 768, while inside PTB-XL they went 30 → 28.

Worse, on this split **the STEMI label is *exactly* `source == "STEMI"`**, 146 of
146. Any STEMI metric here is a database classifier in disguise — which is why
`resnet50_hires` posted 0.9996 STEMI AUROC. Do not report that number.

**So: read the PTB-XL column. It is the one the shortcut cannot reach.**
`sens@.08` fixes specificity instead of trusting a globally-tuned threshold on a
subgroup, so it compares detectors rather than operating points.
"""),
        code('''
%cd /content/code
!python compare.py --runs "$RUNS" 2>/dev/null | sed -n "/BY SOURCE - is the gain/,/PER-CONDITION REGRESSION/p" | head -60
'''),

        md("""
## Cell 7d — the resolution ladder, and what each branch was worth

Two questions, one after the other, and they have to be answered in this order.

**1. The ladder: 384 → 768 → both.** Each rung is scored against *the rung below
it*, on PTB-XL LVH AUPRC. Scoring `resnet50_dual` against round 1 measures three
changes at once and credits all of them to the second branch. The bar is
`resnet50_hires` at **0.9315** — 24.6 M params, one forward pass. A dual model
that gains less than about 0.01 over that has not paid for 2× the parameters and
a second pass per image, and the round's finding is *"higher-resolution
preprocessing was sufficient; the dual-branch complexity was unnecessary."*

**2. The ablation, which can overturn (1).** `without_hi` zeroes the 768 feature
group after its LayerNorm and re-scores TEST with the same frozen thresholds.
**If zeroing `hi` barely moves the score, the model never used the extra pixels**
— it ensembled two ImageNet backbones, and any gain is an ensembling gain that a
second 384 backbone would also have produced, more cheaply.

Notebook A's FPN is the cautionary case: zeroing *any* pyramid level moved macro
AUPRC by ≤0.0016. It had multi-scale structure and used none of it.
"""),
        code('''
%cd /content/code
!python compare.py --runs "$RUNS" 2>/dev/null | sed -n "/RESOLUTION LADDER/,/TEST SET/p"
'''),

        md("""
## Cell 7e — AF × LVH subgroup: does LVH survive when AF is present?  🔴 THE QUESTION C ASKS

The AF × LVH subgroup analyses scored the frozen 768 reference and found it
detects AF correctly but **misses LVH when the two coexist**:

| head | recall on AF+LVH (n=28) | comparand | delta | Fisher p |
|---|---:|---:|---:|---:|
| LVH | **0.6429** (18/28) | 0.8722 on LVH-only (n=227) | −0.2294 | **0.0038** |
| AF | 0.9286 (26/28) | 0.9416 on AF-only (n=257) | −0.0131 | 0.5142 |

Verdict: **SUPPORTED** — 10 of the 28 AF+LVH cases were "AF correct + LVH
missed" against 2 the other way (McNemar p=0.0386).

Every run in this round is scored the same way, from its **own frozen
decisions**, so this table answers the question notebook C exists for: does
`resnet50_vit` (768+768) raise LVH recall on AF+LVH from 0.6429 without
dropping AF recall below 0.9286? AUPRC columns use the reference-pool design
(subgroup positives vs every true_<head> == 0 ECG) — small-sample reads, not
verdict reads. The subgroup has 28 ECGs, so the Fisher p and bootstrap CI say
how much of any delta is real.
"""),
        code('''
%cd /content/code
!python compare.py --runs "$RUNS" 2>/dev/null | sed -n "/AF x LVH SUBGROUP/,/THE CONTROL/p"
'''),

        md("## Cell 8 — full report for one model"),
        code(f'''
MODEL = "{models[0]}"          # change and re-run
print(open(f"{{RUNS}}/{{MODEL}}/report.txt").read())
'''),

        md("""
## Cell 9 — LVH learning curves

The per-epoch LVH AUPRC is tracked separately from the macro monitor on purpose.
A model that improves macro AUPRC while flattening or degrading LVH has traded
the head you care about for the three you already had.
"""),
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
    lab = f"{r['model']} ({'+'.join(map(str, r['model_info']['views']))})"
    ax[0].plot(ep, [x["train_loss"] for x in h], label=lab)
    ax[1].plot(ep, [x["val_macro_auprc"] for x in h], label=lab)
    ax[2].plot(ep, [x.get("val_LVH_auprc") for x in h], label=lab)
for a, t in zip(ax, ["train loss", "val macro AUPRC", "val LVH AUPRC"]):
    a.set_title(t); a.set_xlabel("epoch"); a.legend(fontsize=7); a.grid(alpha=.3)
# TWO reference lines, because they say different things. Round 1 is where this
# work started; resnet50_hires is what a new model actually has to beat. Drawing
# only the first one makes every model in this notebook look like a triumph.
ax[2].axhline(0.8805, ls=":", c="0.5", lw=1)
ax[2].annotate("round 1 resnet50 (test)", (0.02, 0.8805), xycoords=("axes fraction", "data"),
               fontsize=7, va="bottom", color="0.4")
ax[2].axhline(0.9248, ls="--", c="crimson", lw=1.2)
ax[2].annotate("notebook A resnet50_hires 768 (test) - THE BAR", (0.02, 0.9248),
               xycoords=("axes fraction", "data"), fontsize=7, va="bottom", color="crimson")
plt.tight_layout(); plt.show()

# ...and every head separately. LVH is the target, but STEMI is the one whose
# regression matters clinically, so it gets its own panel rather than being
# averaged into the macro line above.
R1 = {"STEMI": 0.9560, "AF": 0.9777, "LVH": 0.8805, "NORMAL": 0.9534}
# Notebook A's single-stream 768 run, on the same frozen test split. This is the
# row a dual-resolution model has to beat, head by head - not round 1.
HIRES = {"STEMI": 0.9979, "AF": 0.9825, "LVH": 0.9248, "NORMAL": 0.9523}
fig, ax = plt.subplots(1, 4, figsize=(20, 3.6), sharey=True)
for a, lab in zip(ax, ["STEMI", "AF", "LVH", "NORMAL"]):
    for f in sorted(glob.glob(f"{RUNS}/*/result.json")):
        r = json.load(open(f))
        h = r["history"]
        if not h:
            continue
        a.plot([x["epoch"] for x in h],
               [x.get(f"val_{lab}_auprc") for x in h], label=r["model"])
    a.axhline(R1[lab], ls=":", c="0.5", lw=1)
    a.axhline(HIRES[lab], ls="--", c="crimson", lw=1.2)
    a.set_title(f"val {lab} AUPRC\\n(dotted = round-1 resnet50, dashed = A's hires 768)",
                fontsize=8)
    a.set_xlabel("epoch"); a.grid(alpha=.3)
ax[0].legend(fontsize=7)
plt.tight_layout(); plt.show()

# STEMI's panel is the one to distrust, not to celebrate. Its label is exactly
# "source == STEMI" on this split, so the crimson line at 0.9979 is what perfect
# database recognition scores - not what reading infarcts scores. A model that
# reaches it has learned the corpus. See cell 7c.
print("STEMI panel: label == source on this split. 0.9979 is the database-"
      "recognition ceiling,\\n             not a clinical result. Do not report it.")
'''),
    ], **_meta()}


# --------------------------------------------------------------- notebook C extras
def nb_vit() -> dict:
    nb = nb_train(
        "CardioSentry hybrids — C. ResNet-50 + ViT, and the final comparison",
        """Trains `resnet50_vit`, then tabulates **everything** that has finished
across every notebook in the round.""",
        ["resnet50_vit"],
        """
## Why this hybrid, and what it is for

Both branches see the same **768×768** view — the exact view the round's best
LVH model (`resnet50_hires`: 0.9248 pooled / 0.9315 PTB-XL LVH AUPRC) saw, so
any gain over that bar is attributable to the second branch's **attention**,
not to extra pixels.

The question it answers comes from the AF × LVH subgroup analyses of the
frozen 768 reference, on the 938-image test split with the model's own frozen
thresholds:

| head | recall on AF+LVH (n=28) | comparand | delta | Fisher p |
|---|---:|---:|---:|---:|
| LVH | **0.6429** (18/28) | 0.8722 on LVH-only (n=227) | −0.2294 | **0.0038** |
| AF | 0.9286 (26/28) | 0.9416 on AF-only (n=257) | −0.0131 | 0.5142 |

**Verdict (frozen): SUPPORTED — the 768 model correctly detects AF and
disproportionately misses LVH when the two coexist** (10 of 28 cases "AF
correct + LVH missed" vs 2 the other way, McNemar p=0.0386).

That is the failure this run targets: **does a 768 representation enriched
with global self-attention preserve LVH information when AF is present?**
This run is scored the same way, automatically, at the end of training —
`report.txt`, `compare.py`'s *AF × LVH SUBGROUP* section and `_SUMMARY.txt`'s
`A+Lrec` column all report LVH recall inside the 28 AF+LVH cases, the
LVH-only comparand, the Fisher test and the reference-pool AUPRCs for both
heads. It must raise LVH recall on AF+LVH from 0.6429 without dropping AF
recall below 0.9286.

**The mechanism.** Sokolow-Lyon compares S in V1 against R in V5 — leads in
different columns, hundreds of pixels apart. A ResNet only bridges that
distance at stage 5; every ViT block attends across it directly. At 768 the
ViT-B/16 grid is 48×48 patches of ~3 mm each, so the attention operates on
detail round 1's 384 view had already thrown away. Its pos-embed is
interpolated from the 24×24 (384px) pretrained grid at first forward — the
checkpoint itself never trained at 768.

**Carry the caveats into the write-up.**
1. ViT-B/16-384 has no in1k-only checkpoint in timm, so this run enters with
   ImageNet-**21k** pretraining while the other models have in1k. If it wins,
   the claim is *"ResNet-50 + in21k ViT at 768"*, not *"the hybrid is better"*.
2. The AF+LVH subgroup has 28 ECGs, so the Fisher p and bootstrap CI decide
   how much of any recall delta is real. No causal claim ("AF causes the
   miss") is available from one split.
3. This is the heaviest run of the round: both branches at 768. Epoch 1's
   img/s is the authoritative rate — expect roughly 15–30 img/s on an L4
   (~2–3 h for 20 epochs) and several times slower on a T4.
""",
        "2-3 h (L4)")
    nb["cells"] += [
        md("""
---

# The final comparison

Everything below reads whatever has finished in `RUNS`, across all three
notebooks. Run it once B is done for the complete picture.
"""),
        md("""
## Cell 10 — full comparison

`compare.py` first verifies every run shared the **training protocol** — seed,
optimiser, schedule, epochs, augmentation, monitor, threshold rule — and refuses
to rank them if not. Input resolution is *expected* to differ and is printed as a
column, because it is the experimental variable this round.
"""),
        code('''
%cd /content/code
!python compare.py --runs "$RUNS" --sort lvh_auprc --csv "$RUNS/comparison.csv"
'''),

        md("""
## Cell 10b — AF × LVH subgroup, standalone  🔴 the question this notebook asks

The same table as the *AF × LVH SUBGROUP* section of the full comparison,
written to `af_lvh.csv` for spreadsheets. Every run is scored from its **own
frozen decisions** on the ground-truth subgroups; the frozen bar is
`resnet50_hires`: LVH recall 0.6429 on AF+LVH vs 0.8722 on LVH-only
(Fisher p=0.0038), AF recall 0.9286 (not degraded).

**A pass is: LVHrec@A+L rises materially from 0.6429 AND AFrec@A+L holds at
~0.9286.** With 28 cases, the Fisher p and CI say how much of any delta is
real — a +0.1 delta whose CI spans zero is a hint, not a finding.
"""),
        code('''
%cd /content/code
!python af_lvh.py --runs "$RUNS" --csv "$RUNS/af_lvh.csv"
'''),

        md("""
## Cell 11 — the control, isolated

`resnet50_base` vs `resnet50_hires` are the same model, same head, same protocol,
same source files. **Only the input resolution differs.**

If most of the round's LVH gain lives in this one row, the finding is
*"preprocess at higher resolution"* — cheaper to deploy and easier to defend
than any hybrid.
"""),
        code('''
import json, os
b = f"{RUNS}/resnet50_base/result.json"
h = f"{RUNS}/resnet50_hires/result.json"
if os.path.exists(b) and os.path.exists(h):
    B, H = json.load(open(b))["test"], json.load(open(h))["test"]
    print(f"{'metric':<26}{'384':>10}{'768':>10}{'delta':>10}")
    print("-" * 56)
    rows = [("LVH AUPRC",       lambda t: t["per_label"]["LVH"]["auprc"]),
            ("LVH sensitivity", lambda t: t["per_label"]["LVH"]["sensitivity"]),
            ("LVH FPR",         lambda t: t["per_label"]["LVH"]["fpr"]),
            ("LVH F1",          lambda t: t["per_label"]["LVH"]["f1"]),
            ("LVH false pos",   lambda t: t["per_label"]["LVH"]["fp"]),
            ("LVH false neg",   lambda t: t["per_label"]["LVH"]["fn"]),
            ("macro AUPRC",     lambda t: t["macro_auprc"]),
            ("macro AUROC",     lambda t: t["macro_auroc"])]
    for name, f in rows:
        print(f"{name:<26}{f(B):10.4f}{f(H):10.4f}{f(H)-f(B):+10.4f}")
else:
    print("run notebook A first - this needs resnet50_base and resnet50_hires")
'''),

        md("""
## Cell 12 — stream ablation: did the extra pixels actually do anything?

Each row re-scores the test set with one feature group zeroed, using the same
val-frozen thresholds.

**A dual-resolution model whose score barely moves when the `hi` group is zeroed
did not use the extra resolution — it ensembled two backbones.** That is a
different, and much less interesting, claim than the one this round set out to
test.

For `resnet50_fpn` the groups are pyramid levels, so the table says which
*scale* the diagnosis was read at. P2 is stride 4 (≈2.9 mm per cell at 384);
P5 is stride 32 (≈23 mm), which is wider than the entire QRS complex.
"""),
        code('''
import json, glob
for f in sorted(glob.glob(f"{RUNS}/*/result.json")):
    r = json.load(open(f))
    ab = r.get("test_stream_ablation") or {}
    if not ab:
        continue
    t = r["test"]
    fa, ff = t["macro_auprc"], t["per_label"]["LVH"]["auprc"]
    print(f"\\n{r['model']}   views {r['model_info']['views']}")
    print(f"  {'zeroed':<16}{'mAUPRC':>10}{'d':>9}{'LVH AUPRC':>12}{'d':>9}")
    print(f"  {'(none)':<16}{fa:10.4f}{'':>9}{ff:12.4f}{'':>9}")
    for k, v in ab.items():
        m, l = v["macro_auprc"], v["per_label"]["LVH"]["auprc"]
        print(f"  {k:<16}{m:10.4f}{m-fa:+9.4f}{l:12.4f}{l-ff:+9.4f}")
'''),

        md("""
## Cell 12b — the regression verdict, all four conditions

The single table to check before calling any of this a win.

`compare.py` flags a head as regressed when it loses more than **0.02 AUPRC**
against round 1's ResNet-50 (roughly the run-to-run noise floor on a 938-image
test split), and flags STEMI separately when **sensitivity** drops by more than
**0.01** — a tighter bar, because that failure is the one that harms a patient.
"""),
        code('''
%cd /content/code
!python compare.py --runs "$RUNS" 2>/dev/null | sed -n "/PER-CONDITION REGRESSION/,/AF x LVH SUBGROUP/p"
'''),

        md("""
### Per-head trade-off view

Every run, every condition, against round 1 — so a gain on LVH bought with a
loss elsewhere is impossible to miss.
"""),
        code('''
import json, glob
import pandas as pd
R1 = {"STEMI": 0.9560, "AF": 0.9777, "LVH": 0.8805, "NORMAL": 0.9534}
rows = []
for f in sorted(glob.glob(f"{RUNS}/*/result.json")):
    r = json.load(open(f)); t = r["test"]
    row = {"model": r["model"],
           "views": "+".join(map(str, r["model_info"]["views"]))}
    for l in R1:
        row[f"{l}"] = round(t["per_label"][l]["auprc"], 4)
        row[f"d{l}"] = round(t["per_label"][l]["auprc"] - R1[l], 4)
    row["macro"] = round(t["macro_auprc"], 4)
    row["worst_head"] = min(R1, key=lambda l: t["per_label"][l]["auprc"] - R1[l])
    row["worst_delta"] = round(min(t["per_label"][l]["auprc"] - R1[l] for l in R1), 4)
    rows.append(row)
df = pd.DataFrame(rows).sort_values("dLVH", ascending=False)
pd.set_option("display.width", 220, "display.max_columns", 40)
display(df)

print()
print("AUPRC deltas vs round-1 resnet50. `worst_head` is the condition each")
print("model gave up the most on. A model with a big +dLVH and a big negative")
print("worst_delta traded one head for another - that is not an improvement,")
print("it is a different operating point.")
'''),

        md("""
## Cell 13 — source confound ⚠️ do not skip

All 1,442 STEMI records come from a single database, and stage 5 of the image
pipeline measured residual source signal (probe 0.515 vs 0.372 chance).

**This round raises the stakes on it.** The 768px and FPN models can see fine
texture the 384px models could not — grid colour, paper wear, JPEG signature —
and that texture identifies the source database. Round 1's spreads were
resnet50 0.0225, convnext 0.0287, vgg16 0.0188, vit 0.0096, effnet 0.0347.

**A resolution gain that arrives together with a widening source spread is a
leakage gain, not a diagnostic one.** Report these numbers next to the headline
whatever they show.
"""),
        code('''
import json, glob
R1 = {"resnet50": 0.0225, "convnext_base": 0.0287, "vgg16": 0.0188,
      "vit_b_384": 0.0096, "efficientnet_b3": 0.0347}
print("round 1 spreads: " + "  ".join(f"{k}={v:.4f}" for k, v in R1.items()))
print()
for f in sorted(glob.glob(f"{RUNS}/*/result.json")):
    r = json.load(open(f))
    ps = r.get("test_per_source", {})
    if not ps:
        continue
    v = [d["macro_auroc"] for d in ps.values()]
    spread = max(v) - min(v)
    flag = "  <-- WIDER than any round-1 model" if spread > max(R1.values()) else ""
    print(f"{r['model']:<22} " +
          "  ".join(f"{s}={d['macro_auroc']:.4f}(n={d['n']})" for s, d in ps.items()) +
          f"   spread={spread:.4f}{flag}")
'''),

        md("""
## Cell 14 — cost, because accuracy is half the comparison

A dual-stream model that wins by 0.005 AUPRC at 2× the parameters and 4× the
latency is not obviously the winner, and the deployment target here is a clinic.
Round 1's ResNet-50 reference: 23.5 M params, 89.9 MB, 12.21 ms/image at batch 1.
"""),
        code('''
import pandas as pd
pd.set_option("display.width", 220, "display.max_columns", 40)
df = pd.read_csv(f"{RUNS}/comparison.csv")
cols = ["model", "views", "LVH_auprc", "LVH_sensitivity", "LVH_fpr",
        "PTBXL_LVH_auprc", "macro_auprc", "macro_auroc", "params_total_m",
        "size_mb", "latency_ms_b1", "train_min", "source_spread"]
display(df[[c for c in cols if c in df.columns]]
        .sort_values("PTBXL_LVH_auprc", ascending=False))
'''),

        md(f"""
## Where the results live

```
{RUNS}/
├── _SUMMARY.txt              every run at a glance, LVH columns first
├── comparison.txt / .csv
└── <model>/
    ├── STATUS.txt            RUNNING | DONE | FAILED
    ├── report.txt            full metrics + stream ablation, plain text
    ├── train_log.txt         complete console output
    ├── history.csv           per epoch, including val LVH AUPRC
    ├── metrics.csv
    ├── result.json
    ├── weights/best.pt       best-on-val weights + frozen thresholds
    └── predictions/          per-image probabilities, for later error review
```

All on Drive, so it survives the runtime being recycled.

**Read in this order:** the control pair first (cell 11), then the ablation
(cell 12), then the headline. If resolution alone explains the gain, the hybrids
are an interesting negative result rather than the deliverable — and that is a
perfectly good chapter.
"""),
    ]
    return nb


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
            # IPython magics and shell lines are not python, but they DO occupy
            # a statement slot - a `!cmd` alone inside an if-block is legal in a
            # notebook. Replacing them with `pass` AT THE SAME INDENT keeps the
            # rest parseable instead of leaving an empty block behind.
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


NOTEBOOKS = {
    "CardioSentry_Hybrids_0_Prepare.ipynb": nb_prepare,

    "CardioSentry_Hybrids_A_Controls.ipynb": lambda: nb_train(
        "CardioSentry hybrids — A. Controls and the single-stream variants",
        """Four runs: the in-family **baseline**, the **resolution control**, **CBAM** and
**FPN**. Run this notebook first — it can answer the question on its own.""",
        ["resnet50_base", "resnet50_hires", "resnet50_cbam", "resnet50_fpn"],
        """
## What these four are for

| run | views | what it isolates |
|---|---|---|
| `resnet50_base` | 384 | the baseline. Same head and pipeline as everything else, so it — not round 1 — is the control |
| `resnet50_hires` | **768** | **resolution alone.** No architectural change whatsoever |
| `resnet50_cbam` | 384 | channel + spatial attention, no extra pixels |
| `resnet50_fpn` | 384 | multi-scale fusion, no extra pixels |

### ⏱ How long this takes, and what actually limits it

**On a 2-vCPU Colab VM these runs are decode-bound.** The corpus is stored at
768px, so every image costs 4× round 1's decode, and the loader delivers ~27
img/s while the GPU could take 61–72. That is where the ~5 h estimate comes
from, and it is a known, accepted cost of the design — see below.

**On a 12-vCPU high-RAM VM they are not.** Round A measured, on this exact
harness:

| model | img/s | data-wait | peak VRAM |
|---|---:|---:|---:|
| `resnet50_base` 384 | 159.4 | 2.1% | 1.2 G |
| `resnet50_cbam` 384 | 109.9 | 1.4% | 1.4 G |
| `resnet50_fpn` 384 | 121.8 | 1.6% | 1.5 G |
| `resnet50_hires` 768 | 43.7 | 0.7% | 4.1 G |

Data-wait under 2% means the loader is keeping up and **the GPU is the
bottleneck** — but that is ~6 TFLOP/s against an L4's ~121 TFLOP/s fp16 peak,
and 4.1 G of 22 G. The card is bottlenecked on being *fed*: micro-batch 8 makes
GEMMs too small to fill the SMs, and more loader workers cannot fix that.

Data-wait under 2% means the loader is keeping up and **the GPU is the
bottleneck** — and it is a genuine one. The tell is in the table above:
159.4 → 43.7 img/s from 384 to 768 is **3.65× for exactly 4× the pixels**.
Near-linear scaling with pixel count means the card is already saturated. It
looks underused on paper (~6 TFLOP/s against a ~121 TFLOP/s fp16 peak, 4.1 G of
22 G) but an L4 has only ~300 GB/s of memory bandwidth, and 768px convolutions
are bandwidth-bound, not occupancy-bound.

**So there is no free speed here, and two things that look like free speed are
not:**

- **Raising `MICRO_BATCH` to fill the VRAM.** BatchNorm normalises over the
  micro-batch, so this changes BN statistics and breaks comparability with every
  finished run. Never do it to buy speed.
- **`PARALLEL=2`, running two models at once.** Tried and *measured slower*:
  19.5 + 11.7 img/s together, against ~34 img/s for one model alone. It also
  corrupts `train_seconds` and the inference benchmark, which is half the cost
  comparison. Cell 5 leaves it at 1.

**Run two models at a time** using the per-model cells further down if a session
may not last. Every run resumes from its last completed epoch, so a disconnect
costs at most one epoch.

**Run `resnet50_hires` before committing to the dual-stream notebooks.** If
doubling the input resolution fixes LVH by itself, the answer is "change the
preprocessing", and notebooks B and C become confirmation rather than the
deliverable. You will know that two runs in instead of seven.

### A note on the CBAM implementation

Dropping a stock CBAM into a pretrained ResNet multiplies every feature map by a
freshly-initialised sigmoid, whose expected value at init is 0.5 — halving the
activations of a network whose downstream weights were tuned for the unhalved
ones. On 4,810 training images that cost is not recoverable.

This version wraps it as `x + gamma * (cbam(x) - x)` with `gamma` initialised to
**zero**, so the module is *exactly* the identity at step 0 and the ImageNet
features arrive intact. The learned gammas are printed in `report.txt`. **Gammas
that stay near zero are a real finding**, not a bug: it means attention was
offered and declined, and that reweighting existing features was not the
bottleneck.

### And one on FPN

FPN gives the classifier access to stride-4 features. On a rendered ECG sheet,
stride-4 features include the **grid colour and JPEG texture that identify the
source database**. Check the source-confound spread in `report.txt` for this run
specifically — a gain that arrives with a widening spread is leakage.
""",
        "~5 h; about 3.5 h on a 12-vCPU high-RAM VM",
        parallel=1),

    "CardioSentry_Hybrids_B_Hybrids.ipynb": lambda: nb_train(
        "CardioSentry hybrids \u2014 B. Everything round A could not settle",
        """Four runs in one session: **CBAM at 768**, **FPN at 768**, the
**dual-resolution** hybrid, and the **heterogeneous** dual.""",
        ["resnet50_dual", "resnet50_cbam_hires", "resnet50_fpn_hires",
         "resnet50_effb3_dual"],
        """
## What notebook A settled, and the one flaw in how it settled it

A ran the control: `resnet50_base` and `resnet50_hires` are the same model, same
head, same protocol, same files, same micro-batch \u2014 **only the input resolution
differs.** 768 won, and beat both attention hybrids.

| model | views | pooled LVH AUPRC | **PTB-XL LVH AUPRC** |
|---|---:|---:|---:|
| **`resnet50_hires`** | **768** | **0.9248** | **0.9315** |
| `resnet50_base` | 384 | 0.9036 | 0.9270 |
| `resnet50_cbam` | 384 | 0.8901 | 0.9097 |
| `resnet50_fpn` | 384 | 0.8767 | 0.8976 |

**Now read the `views` column.** The only model that beat the 384 baseline is the
only model that was given more pixels. CBAM and FPN were never run at 768, so
that table compares an *architecture* change against a *resolution* change and
concludes the architectures were worthless. **That conclusion is not available
from that experiment**, and half this notebook exists to fix it.

Two findings from A do carry in unchanged:

**Most of the pooled 768 gain was not about LVH.** Stratified by database,
+0.0212 pooled becomes **+0.0045 inside PTB-XL** (95% CI \u22120.0124 \u2026 +0.0208),
where 234 of the 255 LVH positives live. The rest came from the 768 model
learning to stop calling LVH on STEMI-database images \u2014 11 false positives \u2192 1
\u2014 which is database recognition, not electrocardiography. There *is* a genuine
residue: inside PTB-XL, 768 missed 15 fewer LVH at essentially unchanged FPR
(sensitivity 0.786 \u2192 0.850). Real, but about a third the headline size.
**Cell 7c is the primary read of this notebook.**

**768 did not cost the other heads.** No AUPRC or sensitivity regression on
STEMI, AF or NORMAL. NORMAL is the only head trending down (\u22120.017 inside
PTB-XL, not significant) and it is the rule-out head, so watch it.

## The four runs, in the order they run

**1 \u00b7 `resnet50_dual` (384+768) \u2014 the headline hybrid.** One ResNet-50 on the
whole sheet at 384, a second on the same sheet at 768, fused by concatenation.
*Was higher-resolution preprocessing sufficient, making the dual-branch
complexity unnecessary?* It runs first because it is the round's central
question and because run 4 is uninterpretable without it.

**2 \u00b7 `resnet50_cbam_hires` (768) \u2014 does attention become useful once
fine-grained information is preserved?** At 384 the sheet is **1.73 px/mm**
vertically, so a 0.5 mV LVH voltage difference \u2014 what Sokolow-Lyon measures \u2014
spans **under 9 pixels**. Attention cannot reweight detail the resampling already
destroyed. At 768 it is 3.46 px/mm.

**3 \u00b7 `resnet50_fpn_hires` (768) \u2014 does multi-scale extraction benefit from
higher-resolution input?** An FPN's stride-4 level is only finer than its
stride-8 level if the pixels underneath differ. In round A, zeroing *any* pyramid
level moved macro AUPRC by **\u22640.0016** \u2014 it had multi-scale structure and used
none of it. If that is because every level read the same blurred waveform, 768
should change it. If it ablates just as flat here, the FPN is inert on this task
at any resolution, and that is worth knowing once rather than suspecting forever.

**4 \u00b7 `resnet50_effb3_dual` (384+768) \u2014 does a different CNN provide
complementary fine-detail features?** `resnet50_dual` fuses two identical
ResNet-50s, so whatever the 768 branch adds, it adds through the same inductive
biases. This swaps the detail branch for EfficientNet-B3.

Runs 2 and 3 each pair exactly with their 384 twin from notebook A \u2014 same
everything, **only the resample differs** \u2014 so cell 7d can print them as pairs and
separate "this architecture is worthless" from "this architecture was starved."

## \u26a0\ufe0f Before interpreting run 4

In round 1 EfficientNet-B3 finished last on all nine headline metrics \u2014 but its
**epoch-1 validation AUPRC was 0.4467 against 0.77\u20130.83 for every other
backbone**. That is the signature of a learning rate mismatched to the
architecture, not of a weak architecture, and no LR sweep was ever run
(`sweep = false` in every round-1 `result.json`). Fusing a possibly-mis-optimised
branch under the same shared 1e-4 risks re-measuring that confound rather than
testing heterogeneous fusion.

**A poor result on run 4 is UNINTERPRETABLE** until an equal-budget sweep has run
(`--sweep`, the same three LRs for every model). A good result is still
informative \u2014 the confound can only hurt this model, not flatter it. It runs last
so a dead session costs you the least important run.

## The bar, and the column that decides

`resnet50_hires` reached **0.9315 PTB-XL LVH AUPRC** with 24.6 M parameters, one
forward pass and 37 minutes. That is the bar \u2014 **not** round 1's 0.8805, and not
`resnet50_base`. Cell 7d scores each rung against the rung below it.

**A PTB-XL gain under about 0.01 does not pay for 2\u00d7 the parameters and a second
forward pass per image**, and the round's finding becomes *"higher-resolution
preprocessing was sufficient; the dual-branch complexity was unnecessary."* That
is a perfectly good result, cheaper to deploy and easier to defend than any
hybrid. Cell 7d states the threshold before the numbers arrive, so the call is
not made post hoc.

**Rank on the PTB-XL column, not the pooled one.** A model that merely learns to
recognise the STEMI database stops calling LVH on its 146 images and its pooled
score rises without it reading a single QRS complex.

**Cell 7d also runs the stream ablation.** `without_hi` zeroes the 768 feature
group and re-scores TEST with the same frozen thresholds. If zeroing `hi` barely
moves the score, the dual never used the extra pixels \u2014 it ensembled two ImageNet
backbones, and a second 384 backbone would have produced the same gain more
cheaply. Read the ablation *and* the ladder; when they disagree, the ablation
wins.

## \u23f1\ufe0f This is a ~4.5 h session \u2014 how to survive a disconnect

`run_hybrids.sh` writes to Drive as it goes and **does not abort if one model
fails** \u2014 the rest still run. Every run resumes from its last completed epoch, and
a model that already finished is skipped, so re-running cell 5 after a
disconnect costs a minute rather than a night.

If your session will not last 4.5 h, **use the per-model cells below cell 5**
instead. They run one model at a time in the order above, and each line is
independently resumable.

Results land in `cardiosentry_runs/hybrid_v1/<model>/` \u2014 the same folder notebook
A wrote to. That is deliberate: `compare.py` reads a single directory and ranks
whatever it finds, so the shared folder is what lets one table put a 768 CBAM run
next to a dual-resolution run. Organisation comes from the per-model
subdirectory names.
""",
        "~3.3 h on a 12-vCPU L4 high-RAM",
        parallel=1),

    "CardioSentry_Hybrids_C_ViT.ipynb": nb_vit,
}


if __name__ == "__main__":
    print(f"embedding {sum(1 for f in FILES if f.exists())} files "
          f"({len(B64)/1024:.0f} KB base64)\n")
    for name, fn in NOTEBOOKS.items():
        nb = fn()
        validate(nb, name)
        out = HERE / name
        out.write_text(json.dumps(nb, indent=1))
        print(f"  {name:<44}{out.stat().st_size/1024:>7.0f} KB  "
              f"{len(nb['cells']):>3} cells")
    print("\nupload these to Colab - the code travels inside each one.")
    print("run 0_Prepare first; A, B and C are independent after that.")
    print("all of them write into one run folder - that is what keeps the")
    print("cross-notebook comparison possible. Do not split it per notebook.")
