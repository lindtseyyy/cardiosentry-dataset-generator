#!/usr/bin/env bash
# v2 renderer (150 dpi, stock ink, stage4_augment) with the layout fix.
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE="$EXP/../../dataset-combination-code/ecg_dataset_pipeline/image_pipeline"
PY="${ECGKIT_PY:-$HOME/.cache/ecgkit-venv/bin/python}"

export ECG_RELEASE=v3
# ECG_LOWPASS_HZ deliberately unset: 0.05-40 Hz is the config default
export ECG_IMAGE_BUILD="$EXP/build"
# Everything else deliberately UNSET: DPI 150, stock stroke widths, 4 columns
# -- i.e. exactly the v2 render settings. Only the sheet layout differs.
unset ECG_DPI ECG_LINE_WIDTH ECG_GRID_LINE_WIDTH ECG_PHOTO_LONG_EDGE

cd "$PIPE"
"$PY" patch_kit.py
"$PY" stage1_manifest_from_csv.py --input-csv "$EXP/input/validation_records.csv"
"$PY" stage2_transcode.py
"$PY" stage3_render.py   -j 5
"$PY" stage4_augment.py  -j 5      # the v2 augmenter, NOT stage4_photo.py
cp "$ECG_IMAGE_BUILD/images/test/"*.jpg "$ECG_IMAGE_BUILD/index.csv" "$EXP/validation/"
