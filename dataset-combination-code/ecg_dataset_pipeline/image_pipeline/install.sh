#!/usr/bin/env bash
# Create the virtualenv that ecg-image-kit and the corpus pipeline need.
#
#   bash install.sh                 create, or top up an existing venv
#   RECREATE=1 bash install.sh      delete and rebuild from scratch
#   VENV=/some/path bash install.sh install somewhere else
#
# You do NOT need to activate anything afterwards, and this venv is separate
# from the dataset pipeline's own .venv on purpose - the kit needs Python 3.11
# and NumPy 1.26, which the rest of the project does not. image_pipeline/run_all.sh
# calls this interpreter by absolute path; if you installed elsewhere, point it
# there with ECGKIT_PY=/some/path/bin/python.
set -euo pipefail

VENV="${VENV:-$HOME/.cache/ecgkit-venv}"
PY_BIN="${PY_BIN:-python3.11}"
RECREATE="${RECREATE:-0}"

if ! command -v "$PY_BIN" >/dev/null 2>&1; then
    echo "error: $PY_BIN not found." >&2
    echo "The kit needs Python 3.11 - imgaug breaks on NumPy 2.x, which newer" >&2
    echo "interpreters ship by default. Install it, or set PY_BIN=python3.10" >&2
    exit 1
fi

if [[ -d "$VENV" ]]; then
    if [[ "$RECREATE" == "1" ]]; then
        echo "==> removing existing venv at $VENV"
        rm -rf "$VENV"
    else
        echo "==> reusing existing venv at $VENV"
        echo "    package versions will be brought up to the pins below, but"
        echo "    anything already installed is left in place."
        echo "    Use RECREATE=1 for a clean rebuild."
    fi
fi

echo "==> creating venv at $VENV using $PY_BIN"
"$PY_BIN" -m venv "$VENV"
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

"$PIP" install --quiet --upgrade pip

echo "==> installing packages (this pulls TensorFlow, ~2 GB, give it a few minutes)"
# Both opencv distributions are held below 5.0: the 5.x wheels are built
# against the NumPy 2 ABI and this environment is pinned to NumPy 1.26 for
# imgaug's sake. Both are named because imgaug depends on opencv-python while
# we ask for opencv-python-headless, and whichever lands last owns "cv2".
"$PIP" install --quiet \
    wfdb matplotlib pandas scipy pyyaml qrcode \
    "opencv-python<5" "opencv-python-headless<5" \
    scikit-image scikit-learn Pillow \
    imgaug seaborn imutils validators beautifulsoup4 html5lib \
    tensorflow-cpu spacy

# Must come last. TensorFlow pulls NumPy 2.x, but imgaug calls np.sctypes,
# which was removed in NumPy 2.0, so the pin has to be re-applied afterwards.
echo "==> pinning numpy 1.26.4 (must be last)"
"$PIP" install --quiet "numpy==1.26.4"

echo "==> verifying"
"$PY" - <<'PYCHECK'
import importlib, sys
mods = ["numpy", "wfdb", "scipy", "pandas", "matplotlib", "PIL",
        "cv2", "skimage", "sklearn", "imgaug", "seaborn", "yaml",
        "qrcode", "tensorflow", "spacy"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        bad.append(f"{m}: {type(exc).__name__}: {exc}")
import numpy, cv2
print(f"    python {sys.version.split()[0]}, numpy {numpy.__version__}, "
      f"cv2 {cv2.__version__}")
if cv2.__version__.split(".")[0] != "4":
    bad.append(f"cv2 {cv2.__version__} is not 4.x - NumPy ABI mismatch")
if bad:
    print("    FAILED:")
    for b in bad:
        print(f"      {b}")
    raise SystemExit(1)
print(f"    all {len(mods)} imports OK")
PYCHECK

cat <<EOF

Done. Interpreter: $PY

pip may have warned that ml-dtypes wants numpy >= 2.0 - ignore it, every
import resolves and the renders are correct.

Next:
    git clone https://github.com/alphanumericslab/ecg-image-kit.git ../ecg-image-kit
    ./run_all.sh
EOF
