#!/usr/bin/env bash
# Train all NINE architectures at ONE input geometry, then compare.
#
#   ./run_all.sh /content/prototype_1152 /content/drive/MyDrive/cardiosentry_runs_v2
#   CS_DIM=sq768 ./run_all.sh /content/prototype_1152 "$RUNS"
#
# The geometry comes from CS_DIM (default: dimensions.DEFAULT_DIM). Runs land in
# <OUT>/<model>@<dim>/, so several geometries can share one OUT without
# colliding - which is what run_dims.sh relies on.
#
# Built to be left running overnight:
#   * NO `set -e` on the training loop. If one model OOMs at 3am the rest still
#     run - losing eight models because the ninth failed is the worst outcome.
#   * every model writes its own STATUS.txt, so the morning check is one glance
#   * _SUMMARY.txt is rewritten after each model, so partial progress is readable
#   * re-running resumes each model from its last completed epoch
set -uo pipefail

DATA="${1:?usage: ./run_all.sh /path/to/prototype_1152 [OUTDIR]}"
OUT="${2:-runs}"
DIM="${CS_DIM:-nat768}"
export CS_DIM="$DIM"
mkdir -p "$OUT"

# Cheapest and most informative first. If the batch is going to be cut short by a
# disconnect, you want resnet50 (the v1 winner, the reference point) and the two
# new CNNs finished rather than three transformers half-done.
MODELS=(resnet50 densenet121 efficientnet_b3 efficientnetv2_s convnext_base convnextv2_base vgg16 swinv2_base vit_b16)

BATCH_LOG="$OUT/_batch_log.txt"
{
  echo "=============================================================="
  echo "batch started $(date '+%Y-%m-%d %H:%M:%S')"
  echo "data:     $DATA"
  echo "out:      $OUT"
  echo "geometry: CS_DIM=$DIM"
  echo "models:   ${MODELS[*]}"
  echo "=============================================================="
} | tee -a "$BATCH_LOG"

echo "selftest ..." | tee -a "$BATCH_LOG"
if ! python selftest.py --data "$DATA" 2>&1 | tee -a "$BATCH_LOG"; then
  echo "SELFTEST FAILED - stopping. These are silent-wrongness bugs, not crashes." \
    | tee -a "$BATCH_LOG"
  exit 1
fi

echo "preflight ..." | tee -a "$BATCH_LOG"
if ! python preflight.py --data "$DATA" --quick 2>&1 | tee -a "$BATCH_LOG"; then
  echo "PREFLIGHT FAILED - stopping before wasting GPU hours" | tee -a "$BATCH_LOG"
  exit 1
fi

declare -a OK=() FAILED=()
for m in "${MODELS[@]}"; do
  echo | tee -a "$BATCH_LOG"
  echo "##############################################################" | tee -a "$BATCH_LOG"
  echo "# $m @ $DIM   started $(date '+%H:%M:%S')" | tee -a "$BATCH_LOG"
  echo "##############################################################" | tee -a "$BATCH_LOG"

  if python train.py --model "$m" --data "$DATA" --out "$OUT"; then
    OK+=("$m")
    echo "# $m DONE $(date '+%H:%M:%S')" | tee -a "$BATCH_LOG"
  else
    FAILED+=("$m")
    echo "# $m FAILED $(date '+%H:%M:%S') - see $OUT/$m@$DIM/ERROR.txt" | tee -a "$BATCH_LOG"
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
echo "   $OUT/_SUMMARY.txt              all runs at a glance"
echo "   $OUT/comparison.txt            full comparison"
echo "   $OUT/<model>@$DIM/report.txt   per-model detail"
