#!/usr/bin/env bash
# Train the LVH-only binary ResNet-50@768, then refresh the summary.
#
#   ./run_lvh_binary.sh /content/prototype_1152_768 /content/drive/MyDrive/CardioSentry/cardiosentry_runs/lvh_binary_v1
#   ./run_lvh_binary.sh <DATA> <OUT>
#
# MICRO_BATCH must stay 8 to match the four-head reference run: BatchNorm
# normalises over the micro-batch, so a different value here changes BN
# statistics and confounds the comparison. The env var exists only so the
# notebook states the value out loud.
set -uo pipefail

MICRO_BATCH="${MICRO_BATCH:-8}"
MB_ARG=(--micro-batch "$MICRO_BATCH")

DATA="${1:?usage: run_lvh_binary.sh <DATA> <OUT>}"
OUT="${2:?usage: run_lvh_binary.sh <DATA> <OUT>}"

python train.py --model resnet50_lvh_binary --data "$DATA" --out "$OUT" "${MB_ARG[@]}"
STATUS=$?

echo "================================================================"
if [ $STATUS -eq 0 ]; then
  echo "training finished - summary:"
else
  echo "training exited $STATUS - partial summary:"
fi
cat "$OUT/_SUMMARY.txt"
exit $STATUS
