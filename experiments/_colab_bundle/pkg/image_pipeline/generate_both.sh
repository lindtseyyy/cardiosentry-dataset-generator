#!/usr/bin/env bash
# =============================================================================
# generate_both.sh - sequentially regenerate the balanced synthetic ECG corpus
# at two low-pass cutoffs: v2_100hz (0.05-100 Hz) then v2_150hz (0.05-150 Hz).
#
# Reuses the existing image_pipeline stages 1-5 verbatim. The ONLY intended
# experimental change vs the baseline release v2 (0.05-40 Hz) is the stage-2
# band-pass low-pass cutoff, passed via ECG_LOWPASS_HZ (see config.py).
# balanced/ and harmonized/ are filter-independent: they are hardlink-copied
# from v2 (zero extra disk, bit-identical, relative paths stay valid).
#
# The second release starts only after the first passes validate_release.py.
# Everything is tee'd to logs/generate_both_<ts>.log.
#
# THE ONE COMMAND (run it from this directory):
#
#     ./generate_both.sh                                  # foreground
#     nohup ./generate_both.sh >/dev/null 2>&1 &          # detach (overnight)
#     tail -f ../logs/generate_both_*.log                 # follow progress
#
# Stages are idempotent, so an interrupted run resumes where it stopped.
# A release with .release_complete.json is refused (delete the marker to
# rebuild it on purpose).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd .. && pwd)"                    # ecg_dataset_pipeline/
RELEASES="$ROOT/data/releases"
V2="$RELEASES/v2"
KIT_PY="${ECGKIT_PY:-$HOME/.cache/ecgkit-venv/bin/python}"
DATASET_PY="${DATASET_PY:-$ROOT/../../.venv/bin/python}"
[[ -x "$DATASET_PY" ]] || DATASET_PY="$(command -v python3 || true)"
WORKERS="${WORKERS:-$(nproc 2>/dev/null || echo 4)}"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/generate_both_$TS.log"
exec > >(tee -a "$LOG") 2>&1

echo "======================================================================"
echo "generate_both.sh - started $(date -u '+%F %T UTC')"
echo "log: $LOG"
echo "workers: $WORKERS | kit interpreter: $KIT_PY"
echo "======================================================================"

# ---------------------------------------------------------------- preflight
fail() { echo "FATAL: $*" >&2; exit 1; }

[[ -x "$KIT_PY" ]] || fail "kit interpreter not found: $KIT_PY (run install.sh, or set ECGKIT_PY)"
[[ -f "$V2/VERSION.json" ]] || fail "reference release v2 missing or incomplete: $V2"
for rel in v2_100hz v2_150hz; do
    if [[ -f "$RELEASES/$rel/.release_complete.json" ]]; then
        fail "$rel already contains a COMPLETED release. Inspect it, and delete \
$RELEASES/$rel/.release_complete.json only if you intend to rebuild."
    fi
done

echo
echo "--- preflight: verifying reference release v2 ---"
if [[ -n "$DATASET_PY" ]] && (cd "$ROOT/scripts" && "$DATASET_PY" export_dataset.py --check v2); then
    echo "v2 checksums OK"
else
    fail "export_dataset.py --check v2 failed - do not build variants from a broken baseline"
fi

# Snapshot v2 (path/size/mtime) so the very end can prove it was not touched.
SNAP="$LOG_DIR/v2_snapshot_$TS.txt"
( cd "$V2" && find . -type f -printf '%p\t%s\t%T@\n' | sort ) > "$SNAP"
echo "v2 snapshot: $(wc -l < "$SNAP") files -> $SNAP"

AVAIL_KB=$(df -P -B1 "$RELEASES" | awk 'NR==2 {print $4}')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024 / 1024))
echo "free disk on releases volume: ${AVAIL_GB} GB (two builds need ~105 GB)"
if (( AVAIL_GB < 120 )); then
    echo "WARN: this may not be enough for both staged+rendered+images builds"
fi

# ---------------------------------------------------------------- build_one
build_one() {
    local rel="$1" lowpass="$2"
    local rel_dir="$RELEASES/$rel"

    echo
    echo "######################################################################"
    echo "# BUILD $rel  (band-pass 0.05-${lowpass} Hz)                          #"
    echo "######################################################################"

    mkdir -p "$rel_dir"
    # Filter-independent cohorts: hardlink copies of v2 (bit-identical, free).
    for cohort in balanced harmonized; do
        if [[ ! -d "$rel_dir/$cohort" ]]; then
            cp -al "$V2/$cohort" "$rel_dir/$cohort"
            echo "$cohort/ hardlinked from v2"
        else
            echo "$cohort/ already present - reusing"
        fi
    done

    echo
    echo "--- $rel: running image_pipeline stages 1-5 ---"
    ECG_RELEASE="$rel" ECG_LOWPASS_HZ="$lowpass" WORKERS="$WORKERS" ./run_all.sh

    echo
    echo "--- $rel: finalizing release (rel_path, READMEs, VERSION.json, prototypes) ---"
    ECG_RELEASE="$rel" ECG_LOWPASS_HZ="$lowpass" \
        "$KIT_PY" finalize_release.py --release "$rel" --lowpass "$lowpass"

    echo
    echo "--- $rel: validation gate ---"
    ECG_RELEASE="$rel" ECG_LOWPASS_HZ="$lowpass" \
        "$KIT_PY" validate_release.py --release "$rel" --lowpass "$lowpass"

    printf '{\n  "release": "%s",\n  "lowpass_hz": %s,\n  "created_utc": "%s",\n  "note": "completed and validated by generate_both.sh"\n}\n' \
        "$rel" "$lowpass" "$(date -u '+%FT%TZ')" > "$rel_dir/.release_complete.json"
    echo "$rel COMPLETE and VALIDATED -> $rel_dir"
}

build_one v2_100hz 100
build_one v2_150hz 150

# ---------------------------------------------------------------- v2 intact?
echo
echo "--- verifying releases/v2 was not modified ---"
SNAP_END="$LOG_DIR/v2_snapshot_end_$TS.txt"
( cd "$V2" && find . -type f -printf '%p\t%s\t%T@\n' | sort ) > "$SNAP_END"
if cmp -s "$SNAP" "$SNAP_END"; then
    echo "OK: releases/v2 unchanged (path/size/mtime identical across the run)"
else
    echo "FATAL: releases/v2 differs from the pre-build snapshot:"
    diff "$SNAP" "$SNAP_END" | head -40 || true
    exit 1
fi

echo
echo "======================================================================"
echo "ALL DONE $(date -u '+%F %T UTC')"
echo "  v2_100hz -> $RELEASES/v2_100hz  (0.05-100 Hz)"
echo "  v2_150hz -> $RELEASES/v2_150hz  (0.05-150 Hz)"
echo "  log: $LOG"
echo "======================================================================"
