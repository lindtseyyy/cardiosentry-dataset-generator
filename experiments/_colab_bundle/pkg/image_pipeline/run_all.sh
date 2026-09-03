#!/usr/bin/env bash
# Build the synthetic paper-ECG image corpus from the dataset pipeline's CSV.
#
#   ./run_all.sh              full build
#   ./run_all.sh 3            resume from stage 3
#   CSV=../data/harmonized/harmonized_ecg_metadata.csv ./run_all.sh
#   WORKERS=4 ./run_all.sh
#
# Every stage is idempotent: stages 3 and 4 skip work that already exists, so an
# interrupted build can be restarted with the same command.
#
# Before resuming an interrupted build, check for orphaned kit workers:
#   pgrep -af "stage3_render|gen_ecg_images|stage4_augment"
set -euo pipefail

cd "$(dirname "$0")"
PY="${ECGKIT_PY:-$HOME/.cache/ecgkit-venv/bin/python}"
FROM="${1:-1}"
WORKERS="${WORKERS:-$(nproc 2>/dev/null || echo 4)}"
CSV="${CSV:-}"

if [[ ! -x "$PY" ]]; then
    echo "interpreter not found: $PY" >&2
    echo "create it with: bash install.sh   (or set ECGKIT_PY)" >&2
    exit 1
fi

run() {
    local n=$1; shift
    if (( n < FROM )); then
        echo "--- stage $n: skipped"
        return
    fi
    echo
    echo "=== stage $n: $* ==="
    local start=$SECONDS
    "$PY" "$@"
    echo "--- stage $n done in $(( (SECONDS - start) / 60 )) min"
}

echo "=== preflight: patching ecg-image-kit ==="
"$PY" patch_kit.py

if [[ -n "$CSV" ]]; then
    run 1 stage1_manifest_from_csv.py --input-csv "$CSV"
else
    run 1 stage1_manifest_from_csv.py
fi
run 2 stage2_transcode.py
run 3 stage3_render.py -j "$WORKERS"
run 4 stage4_augment.py -j "$WORKERS"
run 5 stage5_verify.py

BUILD="${ECG_IMAGE_BUILD:-$(cd .. && pwd)/build}"
echo
echo "corpus ready: $BUILD/images"
echo "index:        $BUILD/index.csv"
