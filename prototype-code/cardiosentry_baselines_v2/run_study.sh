#!/usr/bin/env bash
# THE WHOLE STUDY, unattended, in one command.
#
#   ./run_study.sh /content/prototype_1152 /content/drive/MyDrive/CardioSentry/runs/baselines_v2
#
# Runs, in order:
#   A. geometry screen   ResNet-50 once per candidate geometry        (~6 runs)
#   .. picks the winner on VALIDATION macro AUPRC (pick_dim.py)
#   B. architecture run  all nine backbones at the winning geometry   (9 runs)
#   .. both comparison tables
#
# The handoff between A and B is automatic. That is the whole point: you should
# be able to start this and go to sleep, rather than waking up to a table that
# needs reading before anything else can run.
#
# TO SKIP THE SCREEN, pass the geometry you already want as $3:
#   ./run_study.sh "$DATA" "$OUT" crop608
#
# RESUMING. Everything here is idempotent. Re-run the exact same command after a
# Colab disconnect and it picks up where it stopped: finished runs are skipped,
# an interrupted run resumes from its last completed epoch, and an already
# pre-resized corpus is reused. Expect to need this - the full study is longer
# than one Colab session.
#
# NO `set -e` on the training loops, deliberately. If one model OOMs at 3am the
# rest still run; losing eight models because the ninth failed is the worst
# outcome available.
set -uo pipefail

DATA="${1:?usage: ./run_study.sh /path/to/prototype_1152 OUTDIR [DIM|auto] [\"dim1 dim2 ...\"]}"
OUT="${2:?give an output directory - put it on Drive}"
CHOICE="${3:-auto}"
SCREEN_DIMS="${4:-sq768 nat672 nat768 wide576 crop640}"

SCREEN_MODEL="${SCREEN_MODEL:-resnet50}"
DIMS_OUT="$OUT/dims"
ARCH_OUT="$OUT/baselines"
mkdir -p "$DIMS_OUT" "$ARCH_OUT"

LOG="$OUT/_study_log.txt"
say() { echo "$@" | tee -a "$LOG"; }
rule() { say "=============================================================="; }

rule
say "CardioSentry baselines v2 - full study"
say "started  $(date '+%Y-%m-%d %H:%M:%S')"
say "data     $DATA"
say "out      $OUT"
say "choice   $CHOICE"
rule

# ---------------------------------------------------------------- pre-resize
# The corpus stores 1686x1311 (2.2 Mpx). Training resizes every image on the CPU
# every epoch, and on Colab's 2 cores that starves the GPU - the v1 logs show
# ~21 img/s where the GPU could do 70-100. Doing it once per geometry roughly
# halves experiment A's wall clock.
#
# Each copy is ~1.5-2 GB on local disk (NOT Drive). Six of them is ~10 GB, which
# /content has room for. They are left in place so a re-run after a disconnect
# does not pay for them again; `rm -rf /content/prototype_*` reclaims it.
prepare() {
  local dim="$1" src="$2" dst
  dst="/content/prototype_${dim}"
  if [ -f "$dst/index.csv" ]; then
    echo "$dst"; return 0
  fi
  if ! CS_DIM="$dim" python resize_dataset.py --src "$src" --dst "$dst" \
        >>"$LOG" 2>&1; then
    # Falling back to the original corpus is CORRECT, not a silent failure: the
    # transforms produce the identical tensor either way, it is just slower.
    say "  pre-resize failed for $dim - training from the original corpus instead"
    echo "$src"; return 0
  fi
  echo "$dst"
}

# ------------------------------------------------------- A. geometry screen
DIM=""
if [ "$CHOICE" = "auto" ]; then
  rule
  say "EXPERIMENT A - geometry screen ($SCREEN_MODEL, dims: $SCREEN_DIMS)"
  rule

  for d in $SCREEN_DIMS; do
    say ""
    say "### $SCREEN_MODEL @ $d   $(date '+%H:%M:%S')"

    if ! CS_DIM="$d" python selftest.py --data "$DATA" >>"$LOG" 2>&1; then
      say "  SELFTEST FAILED for $d - skipping. See $LOG."
      continue
    fi

    d_data="$(prepare "$d" "$DATA")"
    say "  data: $d_data"

    if CS_DIM="$d" python train.py --model "$SCREEN_MODEL" --data "$d_data" \
         --out "$DIMS_OUT"; then
      say "  $d DONE $(date '+%H:%M:%S')"
    else
      say "  $d FAILED - see $DIMS_OUT/$SCREEN_MODEL@$d/ERROR.txt"
    fi
  done

  python compare.py --runs "$DIMS_OUT" --across-dims \
    --csv "$DIMS_OUT/comparison_dims.csv" >>"$LOG" 2>&1 || true

  # pick_dim prints the TABLE on stderr and the WINNER alone on stdout, so the
  # two can be separated without parsing. Report first, then capture.
  rule
  python pick_dim.py --runs "$DIMS_OUT" --report 2>&1 >/dev/null | tee -a "$LOG"
  DIM="$(python pick_dim.py --runs "$DIMS_OUT" 2>/dev/null)"
  rule
  say "GEOMETRY SELECTED (on validation): $DIM"
  rule
else
  DIM="$CHOICE"
  say "skipping experiment A - geometry fixed to $DIM"
fi

if [ -z "$DIM" ]; then
  say "no geometry could be selected - stopping before experiment B"
  exit 1
fi

# --------------------------------------------------- B. architecture ranking
rule
say "EXPERIMENT B - nine architectures @ $DIM"
rule

B_DATA="$(prepare "$DIM" "$DATA")"
say "data: $B_DATA"

if ! CS_DIM="$DIM" python selftest.py --data "$B_DATA" 2>&1 | tee -a "$LOG" | tail -3; then
  say "SELFTEST FAILED at $DIM - stopping. These are silent-wrongness bugs."
  exit 1
fi
CS_DIM="$DIM" python preflight.py --data "$B_DATA" --quick 2>&1 | tee -a "$LOG" | tail -14

# Cheapest and most informative first: if the batch is cut short you want
# resnet50 (the v1 reference point) and the new CNNs finished, rather than three
# transformers half-done.
MODELS=(resnet50 densenet121 efficientnet_b3 efficientnetv2_s convnext_base convnextv2_base vgg16 swinv2_base vit_b16)

declare -a OK=() FAILED=()
for m in "${MODELS[@]}"; do
  say ""
  say "### $m @ $DIM   $(date '+%H:%M:%S')"
  if CS_DIM="$DIM" python train.py --model "$m" --data "$B_DATA" --out "$ARCH_OUT"; then
    OK+=("$m"); say "  $m DONE $(date '+%H:%M:%S')"
  else
    FAILED+=("$m"); say "  $m FAILED - see $ARCH_OUT/$m@$DIM/ERROR.txt"
  fi
done

# ------------------------------------------------------------------ results
rule
say "finished  $(date '+%Y-%m-%d %H:%M:%S')"
say "geometry  $DIM"
say "succeeded (${#OK[@]}): ${OK[*]:-none}"
say "failed    (${#FAILED[@]}): ${FAILED[*]:-none}"
rule

CS_DIM="$DIM" python compare.py --runs "$ARCH_OUT" \
  --csv "$ARCH_OUT/comparison.csv" 2>&1 | tee -a "$LOG" || true

echo
echo "read these:"
echo "   $ARCH_OUT/comparison.txt        the nine architectures ranked"
echo "   $ARCH_OUT/_SUMMARY.txt          all runs at a glance"
echo "   $DIMS_OUT/comparison_dims.txt   the geometry screen + header-crop ablation"
echo "   $OUT/_study_log.txt             everything this driver printed"
