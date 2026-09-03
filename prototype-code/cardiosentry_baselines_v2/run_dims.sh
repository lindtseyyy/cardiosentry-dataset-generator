#!/usr/bin/env bash
# Screen the INPUT GEOMETRIES with ONE architecture, then compare across them.
#
#   ./run_dims.sh /content/prototype_1152 /content/drive/MyDrive/cardiosentry_dims
#   ./run_dims.sh "$DATA" "$OUT" resnet50 "sq768 nat672 crop608"
#
# RUN THIS BEFORE run_all.sh.
#
# Nine architectures x seven geometries is 63 runs and roughly a fortnight of L4
# time. It is also the wrong experiment: the geometry question is "how should the
# pixel budget be spent", and one architecture answers it. So:
#
#   1. this script: 1 architecture x N geometries   -> pick the geometry
#   2. run_all.sh:  9 architectures x 1 geometry    -> rank the architectures
#
# resnet50 is the default screen model because it won v1 at 384 and tied at 768,
# it is the cheapest of the strong ones, and it is fully convolutional - so it
# adapts to every geometry with NO per-model accommodation at all. Screening on
# ViT or Swin would confound the geometry with a position-embedding resample or a
# padding overhead, which is exactly the thing you do not want in the run that
# picks the geometry for everything else.
#
# Same overnight rules as run_all.sh: no `set -e` on the loop, resume on re-run.
set -uo pipefail

DATA="${1:?usage: ./run_dims.sh /path/to/prototype_1152 [OUTDIR] [MODEL] [\"dim1 dim2 ...\"]}"
OUT="${2:-runs_dims}"
MODEL="${3:-resnet50}"
DIMS="${4:-sq768 nat672 nat768 wide576 crop640}"
mkdir -p "$OUT"

BATCH_LOG="$OUT/_batch_log.txt"
{
  echo "=============================================================="
  echo "geometry screen started $(date '+%Y-%m-%d %H:%M:%S')"
  echo "data:  $DATA"
  echo "out:   $OUT"
  echo "model: $MODEL"
  echo "dims:  $DIMS"
  echo "=============================================================="
} | tee -a "$BATCH_LOG"

declare -a OK=() FAILED=()
for d in $DIMS; do
  echo | tee -a "$BATCH_LOG"
  echo "##############################################################" | tee -a "$BATCH_LOG"
  echo "# $MODEL @ $d   started $(date '+%H:%M:%S')" | tee -a "$BATCH_LOG"
  echo "##############################################################" | tee -a "$BATCH_LOG"

  # selftest per geometry: the crop window and the (H,W) plumbing are what this
  # loop varies, so they are what has to be re-checked each time round.
  if ! CS_DIM="$d" python selftest.py --data "$DATA" 2>&1 | tee -a "$BATCH_LOG"; then
    FAILED+=("$d")
    echo "# $d SELFTEST FAILED - skipping" | tee -a "$BATCH_LOG"
    continue
  fi

  if CS_DIM="$d" python train.py --model "$MODEL" --data "$DATA" --out "$OUT"; then
    OK+=("$d")
    echo "# $d DONE $(date '+%H:%M:%S')" | tee -a "$BATCH_LOG"
  else
    FAILED+=("$d")
    echo "# $d FAILED $(date '+%H:%M:%S') - see $OUT/$MODEL@$d/ERROR.txt" | tee -a "$BATCH_LOG"
  fi
done

{
  echo
  echo "=============================================================="
  echo "geometry screen finished $(date '+%Y-%m-%d %H:%M:%S')"
  echo "succeeded (${#OK[@]}): ${OK[*]:-none}"
  echo "failed    (${#FAILED[@]}): ${FAILED[*]:-none}"
  echo "=============================================================="
} | tee -a "$BATCH_LOG"

python compare.py --runs "$OUT" --across-dims --csv "$OUT/comparison_dims.csv" \
  2>&1 | tee -a "$BATCH_LOG" || true

echo
echo "read:"
echo "   $OUT/comparison_dims.txt   geometries ranked, plus the header-crop ablation"
echo
echo "then pick the winner and run the architectures at it:"
echo "   CS_DIM=<winner> ./run_all.sh \"$DATA\" <RUNS>"
