#!/usr/bin/env bash
#
# Render the external test corpus.
#
# THIS SCRIPT DELIBERATELY WRITES NO RENDERING CODE OF ITS OWN.
#
# It shells out to the same image_pipeline/stage{1,2,3,4}.py that produced the
# training corpus, with the same config.py, the same ecg-image-kit checkout and
# the same interpreter. That is the whole point. If these sheets were rendered
# by a second implementation - even a faithful-looking one - then a model
# scoring badly on them would be evidence of a rendering mismatch and not of a
# source confound, and the experiment would answer nothing.
#
# The only two things that change are the input CSV and the output directory,
# both injected the way the pipeline already supports:
#
#   CSV=...                 stage 1's --input-csv
#   ECG_IMAGE_BUILD=...     config.BUILD, and every path derived from it
#
# config.VARIANTS_PER_SPLIT pins val and test to exactly one variant per record,
# and 01_build_metadata.py stamps split=test on every row, so each ECG here
# contributes exactly one sheet. Nothing needs to be passed to enforce that.
#
#   ./02_render_images.sh          full build
#   ./02_render_images.sh 3        resume from stage 3
#   WORKERS=4 ./02_render_images.sh
#
# Stages 3 and 4 skip work that already exists, so an interrupted run restarts
# with the same command. Before resuming, check for orphaned kit workers:
#   pgrep -af "stage3_render|gen_ecg_images|stage4_augment"

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PIPELINE="$ROOT/dataset-combination-code/ecg_dataset_pipeline/image_pipeline"

DATA="$HERE/../data"
PY="${ECGKIT_PY:-$HOME/.cache/ecgkit-venv/bin/python}"
FROM="${1:-1}"
WORKERS="${WORKERS:-$(nproc 2>/dev/null || echo 4)}"

if [[ ! -x "$PY" ]]; then
    echo "interpreter not found: $PY" >&2
    echo "create it with: bash $PIPELINE/install.sh   (or set ECGKIT_PY)" >&2
    exit 1
fi
if [[ ! -f "$DATA/external_test_metadata.csv" ]]; then
    echo "input CSV not found: $DATA/external_test_metadata.csv" >&2
    echo "run:  python $HERE/01_build_metadata.py" >&2
    exit 1
fi

cd "$PIPELINE"
echo "=== preflight: patching ecg-image-kit ==="
"$PY" patch_kit.py

# Two passes, into two build directories. They cannot share one, because stage 1
# writes a single image_manifest.csv at config.BUILD and the second pass would
# overwrite the first. See 01_build_metadata.py for why the unlabelled pass
# exists at all; 03_merge.py stitches the two index.csv files back together and
# strips the sentinel label.
render_pass() {
    local name=$1 csv=$2
    export ECG_IMAGE_BUILD="$DATA/image_build/$name"

    echo
    echo "############################################################"
    echo "# pass: $name"
    echo "#   input : $csv"
    echo "#   build : $ECG_IMAGE_BUILD"
    echo "############################################################"

    run() {
        local n=$1; shift
        if (( n < FROM )); then echo "--- stage $n: skipped"; return; fi
        echo; echo "=== [$name] stage $n: $* ==="
        local start=$SECONDS
        "$PY" "$@"
        echo "--- stage $n done in $(( (SECONDS - start) / 60 )) min"
    }

    run 1 stage1_manifest_from_csv.py --input-csv "$csv"
    run 2 stage2_transcode.py
    run 3 stage3_render.py -j "$WORKERS"
    run 4 stage4_augment.py -j "$WORKERS"

    # Stage 5 is advisory here, not a gate. Several of its checks assume a
    # corpus with all three splits and will report a zero train count on a
    # test-only build, which is expected rather than a failure - hence `|| true`.
    #
    # Its source-leakage probe, though, is worth more on this corpus than on
    # the training one. It fits a small classifier to predict SOURCE from a
    # 64x64 greyscale thumbnail. If LUDB sheets are separable from PTB-XL
    # sheets by paper appearance alone, that is a number the final report has
    # to carry, because it bounds how much of any cross-source result is
    # really about paper rather than about the trace.
    echo; echo "=== [$name] stage 5: verify (advisory) ==="
    "$PY" stage5_verify.py 2>&1 | tee "$ECG_IMAGE_BUILD/stage5_verify.txt" || true
}

render_pass labelled   "$DATA/external_test_metadata.csv"
render_pass unlabelled "$DATA/external_test_metadata_unlabelled.csv"

echo
echo "both passes rendered. now run:"
echo "    python $HERE/03_merge.py"
