#!/usr/bin/env bash
# Train the round-2 architectures under one protocol, then compare.
#
#   ./run_hybrids.sh /content/prototype_1152_768 /content/drive/MyDrive/cardiosentry_hybrids
#   ./run_hybrids.sh <DATA> <OUT> resnet50_base resnet50_hires     # a subset
#
# Built to be left running:
#   * NO `set -e` on the training loop. If the dual-stream run OOMs at 3am the
#     rest still run - losing six models because the seventh failed is the worst
#     outcome for an overnight batch.
#   * every model writes its own STATUS.txt, so the morning check is one glance
#   * _SUMMARY.txt is rewritten after each model, so partial progress is readable
#   * re-running resumes each model from its last completed epoch
#
# ORDER IS DELIBERATE. resnet50_base and resnet50_hires run FIRST because they
# are the control pair: if 768 alone fixes LVH, you may not need the hybrids at
# all, and you will know that two runs in rather than seven.
set -uo pipefail

# MICRO_BATCH applies to EVERY model, which is the point: BatchNorm normalises
# over the micro-batch, so runs that disagree on it are not cleanly comparable.
# Unset means each model uses its own T4-sized default (see hybrids.REGISTRY),
# which is faster but leaves the control pair confounded.
#
#     MICRO_BATCH=8 ./run_hybrids.sh <data> <out>        # 24 GB card
MICRO_BATCH="${MICRO_BATCH:-}"
MB_ARG=()
if [ -n "$MICRO_BATCH" ]; then
  MB_ARG=(--micro-batch "$MICRO_BATCH")
fi

# ---------------------------------------------------------------- PARALLEL
#
#     PARALLEL=2 MICRO_BATCH=8 ./run_hybrids.sh <data> <out>
#
# Runs N models at once ON ONE GPU.
#
# DEFAULT 1, AND ON AN L4 LEAVE IT THERE. It was tried at 2 on this exact card
# and corpus and measured SLOWER:
#
#     model                PARALLEL=2      alone
#     resnet50_dual        19.5 img/s      ~34 img/s
#     resnet50_cbam_hires  11.7 img/s      ~30 img/s
#     aggregate            31.2 img/s
#
# Two models together move fewer images per second than resnet50_dual moves on
# its own: ~137 min for that group against ~95 min sequential.
#
# WHY, so nobody re-derives the wrong answer from the same numbers. Round A
# measured on this L4:
#
#     model            img/s   data-wait   peak VRAM   approx TFLOP/s
#     resnet50_base    159.4      2.1%        1.2 G         5.8
#     resnet50_hires    43.7      0.7%        4.1 G         6.3
#
# ~6 TFLOP/s against an L4's ~121 TFLOP/s fp16 peak, using 4.1 G of 22 G, LOOKS
# like a card with idle SMs waiting to be filled. It is not. 768px convolutions
# move huge activation tensors and an L4 has only ~300 GB/s of memory
# bandwidth, so this is BANDWIDTH-bound, not occupancy-bound - a second process
# buys nothing and costs cache locality. The giveaway is in the table itself:
# 159.4 -> 43.7 img/s from 384 to 768 is 3.65x for exactly 4x the pixels.
# Near-linear scaling with pixel count means the device is already saturated;
# idle SMs would have shown SUBLINEAR scaling.
#
# IT ALSO CORRUPTS THE COST TABLE. train_seconds and the end-of-run inference
# benchmark both run under contention, and the benchmark runs AFTER training
# while the other model in the group is still going. That is half the
# "was the second branch worth it" decision, measured wrong.
#
# Accuracy is unaffected either way - same seeds, same micro-batch, same batch
# order, same arithmetic. Only the clock changes. So a run already part-done
# under PARALLEL=2 is scientifically fine; only its timings are junk.
#
# WORTH RETRYING on a bandwidth-rich card (A100 ~2 TB/s, H100 ~3 TB/s), where
# the premise actually holds. Preflight checks the concurrent VRAM sum first.
# If a group OOMs there, lower PARALLEL - NEVER MICRO_BATCH, which would change
# BatchNorm statistics and invalidate every run already finished.
PARALLEL="${PARALLEL:-1}"
if ! [ "$PARALLEL" -ge 1 ] 2>/dev/null; then
  echo "PARALLEL must be a positive integer, got '$PARALLEL'" >&2
  exit 1
fi

# Workers are per PROCESS, so N processes on 12 vCPUs must not each spawn 10.
# Two cores stay with the main processes, which are not idle - they drive the
# GPU and copy into pinned memory.
# nproc first: it is in coreutils and cannot fail the way a `python` that is
# only installed as `python3` can. A wrong core count here silently starves
# every loader, so it falls back rather than trusting an empty string.
CORES="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
case "$CORES" in ''|*[!0-9]*) CORES=2 ;; esac

if [ "$PARALLEL" -gt 1 ]; then
  W=$(( (CORES - 2) / PARALLEL ))
  [ "$W" -lt 1 ] && W=1
  WORKER_ARG=(--workers "$W")
else
  WORKER_ARG=()          # let data.py auto-detect, as before
fi

DATA="${1:?usage: ./run_hybrids.sh /path/to/prototype_1152_768 [OUTDIR] [models...]}"
OUT="${2:-runs}"
shift 2 2>/dev/null || shift 1
mkdir -p "$OUT"

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  # In registry order, which is also 384-then-768 within each architecture so a
  # full sequential batch produces every resolution pair back to back.
  MODELS=(resnet50_base resnet50_hires \
          resnet50_cbam resnet50_cbam_hires \
          resnet50_fpn resnet50_fpn_hires \
          resnet50_dual resnet50_vit resnet50_effb3_dual)
fi

BATCH_LOG="$OUT/_batch_log.txt"
{
  echo "=============================================================="
  echo "batch started $(date '+%Y-%m-%d %H:%M:%S')"
  echo "data:   $DATA"
  echo "out:    $OUT"
  echo "models: ${MODELS[*]}"
  echo "micro-batch: ${MICRO_BATCH:-per-model defaults (NOT uniform - see hybrids.py)}"
  if [ "$PARALLEL" -gt 1 ]; then
    echo "parallel:    $PARALLEL models at once, ${WORKER_ARG[1]} loader workers each"
    echo "             (micro-batch is UNCHANGED, so BN statistics are unchanged"
    echo "              and these runs stay comparable with every earlier one)"
    echo "             WARNING: measured SLOWER than sequential on an L4, and it"
    echo "             corrupts train_seconds and the inference benchmark."
    echo "             Only worth it on an A100/H100. See run_hybrids.sh."
  else
    echo "parallel:    1 (sequential) - the right setting on an L4; see run_hybrids.sh"
  fi
  echo "=============================================================="
} | tee -a "$BATCH_LOG"

# Models are checked too, at the micro-batch this batch will actually use.
# Skipping that is how an OOM reaches you six hours in instead of here: the heavy
# models hold two backbones and a 768px activation stack on one card, and the
# fix for an OOM is to lower MICRO_BATCH for EVERY run, which means restarting
# the ones already finished.
PREFLIGHT_ARGS=(--data "$DATA" --model "${MODELS[@]}")
if [ -n "$MICRO_BATCH" ]; then
  PREFLIGHT_ARGS+=(--micro-batch "$MICRO_BATCH")
fi
if [ "${SKIP_MODEL_PREFLIGHT:-0}" = "1" ]; then
  PREFLIGHT_ARGS+=(--skip-models)
fi
if [ "$PARALLEL" -gt 1 ]; then
  PREFLIGHT_ARGS+=(--parallel "$PARALLEL")
fi

echo "preflight ..." | tee -a "$BATCH_LOG"
if ! python preflight.py "${PREFLIGHT_ARGS[@]}" 2>&1 | tee -a "$BATCH_LOG"; then
  echo "PREFLIGHT FAILED - stopping before wasting GPU hours" | tee -a "$BATCH_LOG"
  echo "(set SKIP_MODEL_PREFLIGHT=1 to bypass the per-model VRAM check)" \
    | tee -a "$BATCH_LOG"
  exit 1
fi

declare -a OK=() FAILED=()

run_one() {                       # $1 = model. Sequential path, output inline.
  local m="$1"
  echo | tee -a "$BATCH_LOG"
  echo "##############################################################" | tee -a "$BATCH_LOG"
  echo "# $m   started $(date '+%H:%M:%S')" | tee -a "$BATCH_LOG"
  echo "##############################################################" | tee -a "$BATCH_LOG"

  if python train.py --model "$m" --data "$DATA" --out "$OUT" \
       "${MB_ARG[@]}" "${WORKER_ARG[@]}"; then
    OK+=("$m")
    echo "# $m DONE $(date '+%H:%M:%S')" | tee -a "$BATCH_LOG"
  else
    FAILED+=("$m")
    echo "# $m FAILED $(date '+%H:%M:%S') - see $OUT/$m/ERROR.txt" | tee -a "$BATCH_LOG"
    echo "# continuing with the remaining models" | tee -a "$BATCH_LOG"
  fi
}

if [ "$PARALLEL" -le 1 ]; then
  for m in "${MODELS[@]}"; do run_one "$m"; done
else
  # Concurrent groups of $PARALLEL. Each process still gets its own
  # train_log.txt and STATUS.txt under $OUT/<model>/, so interleaved stdout is
  # not a problem - the per-run logs stay clean and are what you actually read.
  # A group waits for all its members before the next starts: simpler than a
  # rolling pool, and it keeps peak VRAM to the group preflight actually checked.
  i=0
  while [ "$i" -lt "${#MODELS[@]}" ]; do
    group=("${MODELS[@]:$i:$PARALLEL}")
    echo | tee -a "$BATCH_LOG"
    echo "##############################################################" | tee -a "$BATCH_LOG"
    echo "# GROUP: ${group[*]}   started $(date '+%H:%M:%S')" | tee -a "$BATCH_LOG"
    echo "#   stdout below is INTERLEAVED. Read $OUT/<model>/train_log.txt" | tee -a "$BATCH_LOG"
    echo "##############################################################" | tee -a "$BATCH_LOG"

    declare -a pids=() names=()
    for m in "${group[@]}"; do
      # Backgrounding a PIPELINE makes $! the pid of its last stage (sed), so
      # whether `wait` reports python's failure or sed's success depends on the
      # `set -o pipefail` sixty lines above. It is set, so this works either
      # way - the explicit PIPESTATUS just stops that from being load-bearing
      # at a distance. Getting it wrong records an OOMed run as DONE.
      (
        python train.py --model "$m" --data "$DATA" --out "$OUT" \
          "${MB_ARG[@]}" "${WORKER_ARG[@]}" 2>&1 | sed "s/^/[$m] /"
        exit "${PIPESTATUS[0]}"
      ) &
      pids+=($!); names+=("$m")
      sleep 20   # stagger: two processes allocating their peak activation
                 # stack at the same instant is how a run that fits alone
                 # OOMs in a pair. 20 s puts them out of phase for good.
    done

    for j in "${!pids[@]}"; do
      if wait "${pids[$j]}"; then
        OK+=("${names[$j]}")
        echo "# ${names[$j]} DONE $(date '+%H:%M:%S')" | tee -a "$BATCH_LOG"
      else
        FAILED+=("${names[$j]}")
        echo "# ${names[$j]} FAILED $(date '+%H:%M:%S') - see $OUT/${names[$j]}/ERROR.txt" \
          | tee -a "$BATCH_LOG"
      fi
    done
    unset pids names
    i=$(( i + PARALLEL ))
  done
fi

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
echo "   $OUT/comparison.txt      full comparison, LVH first"
echo "   $OUT/<model>/report.txt  per-model detail"
