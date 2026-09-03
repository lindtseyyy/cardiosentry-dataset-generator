#!/usr/bin/env bash
# Train the five CBAM hybrids at one input geometry, then compare them against
# the baselines_v2 backbones.
#
#   ./run_all.sh /content/prototype_nat768 "$RUNS"
#   ./run_all.sh /content/prototype_nat768 "$RUNS" --group cnn
#   ./run_all.sh /content/prototype_nat768 "$RUNS" --group transformer
#   ./run_all.sh /content/prototype_nat768 "$RUNS" densenet121_cbam
#   ./run_all.sh /content/prototype_nat768 "$RUNS" --family convnext_base
#   ./run_all.sh /content/prototype_nat768 "$RUNS" --group cnn --micro-batch 8
#
# --group is what the two notebooks pass, and it is the unit of parallelism:
#
#   cnn           densenet121, efficientnet_b3, convnext_base    ~8 h
#   transformer   swinv2_base, vit_b16                           ~10 h
#
# With no selection it runs models.RUN_ORDER - all five, CNNs first and cheapest
# first within each group, so a session cut short by a disconnect leaves finished
# runs rather than an expensive one half-done.
#
# Re-run the identical command to continue: finished runs are skipped, an
# interrupted run resumes from its last completed epoch, and _SUMMARY.txt is
# rewritten after every model so a half-done batch is readable in the morning.
#
# Built to be left running:
#   * NO `set -e` on the training loop. If one model OOMs at 3am the rest still
#     run - losing four models because the fifth failed is the worst possible
#     outcome.
#   * every run writes its own STATUS.txt, so the morning check is one glance
#   * re-running resumes each model from its last completed epoch
set -uo pipefail

DATA="${1:?usage: ./run_all.sh <DATA> [OUTDIR] [model ...]}"
OUT="${2:-runs}"
shift 2 2>/dev/null || shift $#
DIM="${CS_DIM:-nat768}"
export CS_DIM="$DIM"
mkdir -p "$OUT"

# ---------------------------------------------------------------- micro-batch
# A VRAM accommodation, passed straight through to train.py. Accumulation
# restores the effective batch of 32 and steps_per_epoch is 151 at any of 2, 4, 8 or 16,
# so the LR schedule does not move. It is NOT free on a BatchNorm backbone -
# check_micro_batch.py is the gate, and it refuses rather than warns.
MB=""
_args=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --micro-batch) MB="${2:?--micro-batch needs a number}"; shift 2 ;;
    *) _args+=("$1"); shift ;;
  esac
done
set -- ${_args+"${_args[@]}"}

# ---------------------------------------------------------------- model list
if [ "${1:-}" = "--group" ]; then
  GROUP="${2:?--group needs a group name}"
  mapfile -t MODELS < <(python - "$GROUP" <<'PY'
import sys, models
g = sys.argv[1]
try:
    sel = models.models_in_group(g)
except KeyError as exc:
    sys.exit(str(exc))
if not sel:
    sys.exit(f"group '{g}' selects no models")
print("\n".join(sel))
PY
) || { echo "could not resolve --group $GROUP"; exit 1; }
elif [ "${1:-}" = "--family" ]; then
  FAMILY="${2:?--family needs a backbone name}"
  mapfile -t MODELS < <(python - "$FAMILY" <<'PY'
import sys, models
fam = sys.argv[1]
sel = [n for n in models.RUN_ORDER if models.REGISTRY[n]["backbone"] == fam]
if not sel:
    sys.exit(f"no models for backbone '{fam}'. known: "
             + ", ".join(sorted(models.BACKBONES)))
print("\n".join(sel))
PY
) || { echo "could not resolve --family $FAMILY"; exit 1; }
elif [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  mapfile -t MODELS < <(python -c "import models; print('\n'.join(models.RUN_ORDER))")
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "no models selected"; exit 1
fi

TRAIN_EXTRA=()
if [ -n "$MB" ]; then
  if ! python check_micro_batch.py "$MB" "${MODELS[@]}"; then
    exit 1
  fi
  TRAIN_EXTRA+=(--micro-batch "$MB")
fi

BATCH_LOG="$OUT/_batch_log.txt"
{
  echo "=============================================================="
  echo "batch started $(date '+%Y-%m-%d %H:%M:%S')"
  echo "data:     $DATA"
  echo "out:      $OUT"
  echo "geometry: CS_DIM=$DIM"
  echo "models:   ${#MODELS[@]}  -  ${MODELS[*]}"
  echo "micro:    ${MB:-per-model registry value}"
  echo "=============================================================="
} | tee -a "$BATCH_LOG"

echo "selftest ..." | tee -a "$BATCH_LOG"
if ! python selftest.py --data "$DATA" 2>&1 | tee -a "$BATCH_LOG"; then
  echo "SELFTEST FAILED - stopping. These are silent-wrongness bugs, not crashes." \
    | tee -a "$BATCH_LOG"
  exit 1
fi

echo "preflight ..." | tee -a "$BATCH_LOG"
# Scoped to the group when there is one. --quick still CONSTRUCTS every model it
# checks, which means downloading every backbone it checks - and the transformer
# session has no reason to pull three CNNs it will never train.
PF_SCOPE=()
[ -n "${GROUP:-}" ] && PF_SCOPE=(--group "$GROUP")
if ! python preflight.py --data "$DATA" --quick ${PF_SCOPE+"${PF_SCOPE[@]}"} 2>&1 | tee -a "$BATCH_LOG"; then
  echo "PREFLIGHT FAILED - stopping before wasting GPU hours" | tee -a "$BATCH_LOG"
  exit 1
fi

declare -a OK=() FAILED=() SKIPPED=()
for m in "${MODELS[@]}"; do
  # A FINISHED run is skipped, not retrained. This matters because train.py
  # DELETES last.pt on success (it is dead weight once best.pt exists), so a
  # finished run has nothing to resume from and would silently start again at
  # epoch 0 - burning its full wall clock and overwriting a good result.json.
  # STATUS.txt says DONE and result.json exists only after finalize(ok=True).
  RD="$OUT/$m@$DIM"
  if [ -f "$RD/result.json" ] && grep -q '^status:   DONE' "$RD/STATUS.txt" 2>/dev/null; then
    SKIPPED+=("$m")
    echo "# $m already DONE - skipping (rm -rf '$RD' to force a retrain)" \
      | tee -a "$BATCH_LOG"
    continue
  fi

  echo | tee -a "$BATCH_LOG"
  echo "##############################################################" | tee -a "$BATCH_LOG"
  echo "# $m @ $DIM   started $(date '+%H:%M:%S')" | tee -a "$BATCH_LOG"
  echo "##############################################################" | tee -a "$BATCH_LOG"

  if python train.py --model "$m" --data "$DATA" --out "$OUT" \
       ${TRAIN_EXTRA+"${TRAIN_EXTRA[@]}"}; then
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
  echo "skipped   (${#SKIPPED[@]}): ${SKIPPED[*]:-none}"
  echo "failed    (${#FAILED[@]}): ${FAILED[*]:-none}"
  echo "=============================================================="
} | tee -a "$BATCH_LOG"

python compare.py --runs "$OUT" --csv "$OUT/comparison.csv" 2>&1 | tee -a "$BATCH_LOG" || true

echo
echo "read these first:"
echo "   $OUT/_SUMMARY.txt              all runs at a glance"
echo "   $OUT/comparison.txt            the delta-against-baseline table"
echo "   $OUT/<model>@$DIM/report.txt   per-run detail"
echo
echo "if the delta table says NO BASELINE RUNS PRESENT, import them:"
echo "   python import_baselines.py --from <baselines_v2 runs> --to $OUT"
