#!/usr/bin/env python3
"""
Build a SELF-CONTAINED Colab notebook for the external validation run.

    python make_notebook.py

Same trick as the training harness's make_notebook.py: the eval modules are
carried inside cell 1 as a base64 tar.gz and unpacked to /content/eval, so the
notebook is the only file that needs uploading. Re-run this after editing
evaluate.py or report.py.

The notebook does NOT retrain anything and does NOT touch the checkpoints - it
opens each best.pt read-only, reads the weights and the frozen thresholds, and
writes only into a results folder.
"""
from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = [HERE / "evaluate.py", HERE / "report.py"]


def payload() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in FILES:
            if f.exists():
                tar.add(f, arcname=f.name)
    return base64.b64encode(buf.getvalue()).decode()


def _src(text: str) -> list[str]:
    """nbformat wants lines WITH their trailing newlines, or Colab runs them
    together - `import os` + `os.makedirs(...)` becomes `import osos.makedirs(...)`."""
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(s.strip())}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(s.strip("\n"))}


def build() -> dict:
    cells = [
        md("""
# CardioSentry — external STEMI validation

**Nothing is trained here.** Five existing checkpoints are scored, read-only, on
a test corpus they have never seen, to answer one question the training corpus
could not:

> Did the STEMI head learn ST elevation, or did it learn the Chongqing scanner?

In the training corpus every STEMI positive came from a single database and no
other source contributed one, so both hypotheses predicted identical outputs.
This corpus contains STEMI positives from sources that never supplied a STEMI
label, which makes them predict **opposite** things.

### How to read the result

| STEMI | AF / LVH / NORMAL | Conclusion |
|---|---|---|
| collapses | hold | **confound confirmed** — the failure is specific to the confounded head |
| collapses | collapse too | **inconclusive** — generic domain shift, says nothing either way |
| holds | hold | **transfers** — the confound was in the data, the model did not use it |

AF, LVH and NORMAL are the control: they had multi-source training, so they
carry no confound and ride on the identical sheets. The STEMI row is not
interpretable without them.

### What you need

1. `external_testset.tar` in Drive (from `build/04_package.py`)
2. the training run folder in Drive, holding `<model>/weights/best.pt`
"""),
        md("## 1 · Runtime\n\nGPU is optional — 376 sheets score on CPU in a "
           "few minutes. `Runtime ▸ Change runtime type ▸ T4` if one is free."),
        code("""
import torch, platform
print("python  ", platform.python_version())
print("torch   ", torch.__version__)
print("cuda    ", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(cpu)")
"""),
        md("## 2 · Unpack the eval code\n\nCarried inline — no upload, no clone."),
        code(f"""
import base64, io, tarfile, os, pathlib

PAYLOAD = "{payload()}"

dest = pathlib.Path("/content/eval")
dest.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as t:
    t.extractall(dest)
print("unpacked:", sorted(p.name for p in dest.iterdir()))
"""),
        md("""
## 3 · The training harness

`evaluate.py` imports `config.py`, `models.py`, `metrics.py` and `data.py` from
the training code rather than reimplementing them. That is deliberate: if the
transforms or the metric definitions differed even slightly, these numbers could
not be compared against the internal ones they are meant to be contrasted with.

Point `CARDIOSENTRY_CODE` at that folder. If it is not in Drive, upload the
`cardiosentry_runs_v1` code folder or re-run its `make_notebook.py` once to drop
the modules at `/content/code`.
"""),
        code("""
from google.colab import drive
drive.mount('/content/drive')

import os
# EDIT THESE THREE if your Drive layout differs.
CODE    = "/content/drive/MyDrive/cardiosentry_code/cardiosentry_runs_v1"
TAR     = "/content/drive/MyDrive/cardiosentry_dataset/external_testset.tar"
WEIGHTS = "/content/drive/MyDrive/cardiosentry_runs_v1"

for label, path in [("code", CODE), ("tar", TAR), ("weights", WEIGHTS)]:
    print(f"{label:8} {'OK  ' if os.path.exists(path) else 'MISSING'} {path}")
"""),
        code("""
!pip -q install timm
import os
os.environ["CARDIOSENTRY_CODE"] = CODE
!ls "$WEIGHTS"/*/weights/best.pt 2>/dev/null || echo "no best.pt found under $WEIGHTS"
"""),
        md("## 4 · Extract the test corpus\n\nTo local disk, not Drive — reading "
           "376 JPEGs over the Drive FUSE mount is far slower than the copy."),
        code("""
import os, tarfile, pandas as pd

os.makedirs("/content/data", exist_ok=True)
if not os.path.exists("/content/data/external_testset/index.csv"):
    with tarfile.open(TAR) as t:
        t.extractall("/content/data")

TESTSET = "/content/data/external_testset"
df = pd.read_csv(f"{TESTSET}/index.csv")
print(f"{len(df)} sheets")
print(df.groupby("source")[["STEMI", "AF", "LVH", "NORMAL"]].sum())
print()
print("gold STEMI (LUDB, asserted) :", int(df[df.source == "LUDB"].STEMI.sum()))
print("proxy STEMI (PTB-XL acute)  :", int(df[df.source == "PTBXL_MI"].STEMI.sum()))
print("hard negatives              :", int((df.n_positive_labels == 0).sum()))
print("patients seen in training   :", int(df.seen_in_training.sum()), "(must be 0)")
"""),
        md("""
## 5 · Score every checkpoint

Thresholds come from each `best.pt`, frozen on the **original validation split**,
and are applied unchanged. They are never re-tuned here — re-tuning on this
corpus would be tuning on test, and it would hand a model that ranks external
positives barely above chance an operating point that manufactures respectable
sensitivity. AUROC and AUPRC are reported alongside for the same reason: no
threshold can rescue them.
"""),
        code("""
!cd /content/eval && python evaluate.py \\
    --weights "$WEIGHTS" \\
    --testset "/content/data/external_testset" \\
    --out /content/results
"""),
        md("## 6 · Report"),
        code("""
!cd /content/eval && python report.py \\
    --metrics /content/results/external_metrics.json \\
    -o /content/results/EXTERNAL_VALIDATION_REPORT.md
"""),
        code("""
from IPython.display import Markdown, display
display(Markdown(open("/content/results/EXTERNAL_VALIDATION_REPORT.md").read()))
"""),
        md("## 7 · Save results back to Drive\n\nPredictions are saved per model "
           "so the report can be regenerated, or new cohorts sliced, without "
           "re-running inference."),
        code("""
import shutil, os
OUT = "/content/drive/MyDrive/cardiosentry_external_validation"
os.makedirs(OUT, exist_ok=True)
shutil.copytree("/content/results", OUT, dirs_exist_ok=True)
print("saved ->", OUT)
for root, _, files in os.walk(OUT):
    for f in sorted(files):
        print("   ", os.path.relpath(os.path.join(root, f), OUT))
"""),
        md("""
---

### Before quoting anything

- **Eight gold positives.** Every LUDB STEMI interval is wide. Enough to detect
  a collapse; nowhere near enough to certify a head works.
- **`mean p(pos)` when sensitivity is zero.** A model can rank external
  positives correctly and still fire on none of them, because its frozen
  threshold sits above every probability it assigned. That is a calibration
  failure, not a blindness — and it has a different fix.
- **High specificity on the mimics is not automatically good.** A model that
  learnt "Chongqing paper ⇒ STEMI" gets a perfect score on every mimic in the
  corpus, because none of them are on Chongqing paper. Only meaningful from a
  model that also fired on the true positives.
- **Do not merge the LUDB and PTB-XL STEMI figures.** One is asserted, one is a
  proxy on a source the models already know.
"""),
    ]
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


if __name__ == "__main__":
    out = HERE / "CardioSentry_External_Validation.ipynb"
    out.write_text(json.dumps(build(), indent=1))
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({kb:.0f} KB)")
    print("upload this one file to Colab; it carries evaluate.py and report.py inside.")
