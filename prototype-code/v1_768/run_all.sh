#!/usr/bin/env bash
# Train all five architectures under identical conditions, then compare.
# v1_768: 768x768 variant of cardiosentry_runs_v1 (only IMG_SIZE and the
# VRAM-accommodating micro-batches differ).
#
#   ./run_all.sh /content/prototype_1152 /content/drive/MyDrive/CardioSentry/runs/v1_768
#
# Built to be left running overnight:
#   * NO `set -e` on the training loop. If one model OOMs at 3am the rest still
#     run - losing four models because the fifth failed is the worst outcome.
#   * every model writes its own STATUS.txt, so the morning check is one glance
#   * _SUMMARY.txt is rewritten after each model, so partial progress is readable
#   * re-running resumes each model from its last completed epoch
set -uo pipefail

DATA="${1:?usage: ./run_all.sh /path/to/prototype_1152 [OUTDIR]}"
OUT="${2:-runs}"
mkdir -p "$OUT"

MODELS=(resnet50 efficientnet_b3 vgg16 convnext_base vit_b_768)

BATCH_LOG="$OUT/_batch_log.txt"
{
  echo "=============================================================="
  echo "batch started $(date '+%Y-%m-%d %H:%M:%S')"
  echo "data: $DATA"
  echo "out:  $OUT"
  echo "=============================================================="
} | tee -a "$BATCH_LOG"

echo "preflight ..." | tee -a "$BATCH_LOG"
if ! python preflight.py --data "$DATA" --quick 2>&1 | tee -a "$BATCH_LOG"; then
  echo "PREFLIGHT FAILED - stopping before wasting GPU hours" | tee -a "$BATCH_LOG"
  exit 1
fi

declare -a OK=() FAILED=()
for m in "${MODELS[@]}"; do
  echo | tee -a "$BATCH_LOG"
  echo "##############################################################" | tee -a "$BATCH_LOG"
  echo "# $m   started $(date '+%H:%M:%S')" | tee -a "$BATCH_LOG"
  echo "##############################################################" | tee -a "$BATCH_LOG"

  if python train.py --model "$m" --data "$DATA" --out "$OUT"; then
    OK+=("$m")
    echo "# $m DONE $(date '+%H:%M:%S')" | tee -a "$BATCH_LOG"
  else
    FAILED+=("$m")
    echo "# $m FAILED $(date '+%H:%M:%S') - see $OUT/$m/ERROR.txt" | tee -a "$BATCH_LOG"
    echo "# continuing with the remaining models" | tee -a "$BATCH_LOG"
  fi
done

{
  echo
  echo "=============================================================="
  echo "batch finished $(date '+%Y-%m-%d %H:%M:%S')"
  echo "succeeded (${#OK[@]}): ${OK[*]:-none}"
  echo "failed    (${#FAILED[@]}): ${FAILED[*]:-none}"
  echo "=============================================================="
} | tee -a "$BATCH_LOG"

python compare.py --runs "$OUT" --csv "$OUT/comparison.csv" 2>&1 | tee -a "$BATCH_LOG" || true

echo
echo "read these first:"
echo "   $OUT/_SUMMARY.txt        all models at a glance"
echo "   $OUT/comparison.txt      full comparison"
echo "   $OUT/<model>/report.txt  per-model detail"
