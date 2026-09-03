#!/usr/bin/env bash
# Regenerate the 5 validation photographs. Idempotent; stages 1-3 skip if done.
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE="$EXP/../../dataset-combination-code/ecg_dataset_pipeline/image_pipeline"
PY="${ECGKIT_PY:-$HOME/.cache/ecgkit-venv/bin/python}"

export ECG_RELEASE=v3
# ECG_LOWPASS_HZ deliberately unset: 0.05-40 Hz is the config default
export ECG_IMAGE_BUILD="$EXP/build"
export ECG_DPI=300                 # print resolution; the camera downsamples
export ECG_LINE_WIDTH=0.40         # trace stroke, matplotlib points
export ECG_GRID_LINE_WIDTH=0.40
# export ECG_PHOTO_LONG_EDGE=3400   # uncomment for 10.2 px/mm instead of 7.6

cd "$PIPE"
"$PY" patch_kit.py
"$PY" stage1_manifest_from_csv.py --input-csv "$EXP/input/validation_records.csv"
"$PY" stage2_transcode.py
"$PY" stage3_render.py -j 5
"$PY" stage4_photo.py  -j 3        # NOT stage4_augment.py
cp "$ECG_IMAGE_BUILD/images/test/"*.jpg "$ECG_IMAGE_BUILD/index.csv" "$EXP/validation/"
