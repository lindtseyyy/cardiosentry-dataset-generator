#!/usr/bin/env bash
# Reproduce this experiment end to end. Idempotent: re-running skips finished stages.
set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE="$EXP/../../dataset-combination-code/ecg_dataset_pipeline/image_pipeline"
PY="${ECGKIT_PY:-$HOME/.cache/ecgkit-venv/bin/python}"   # the only interpreter with wfdb + the kit

export ECG_RELEASE=v3
export ECG_LOWPASS_HZ=150.0   # this experiment IS the 150 Hz variant; not the default
export ECG_IMAGE_BUILD="$EXP/build"

cd "$PIPE"
"$PY" stage1_manifest_from_csv.py --input-csv "$EXP/input/record_subset.csv"
"$PY" stage2_transcode.py
"$PY" stage3_render.py  -j 1
"$PY" stage4_augment.py -j 1
"$PY" stage5_verify.py
