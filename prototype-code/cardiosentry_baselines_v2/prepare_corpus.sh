#!/usr/bin/env bash
# Get a pre-resized corpus onto LOCAL disk, building it only if we have to.
#
#   ./prepare_corpus.sh <RAW_DIR> <LOCAL_DEST> <DRIVE_CACHE_DIR>
#
#   ./prepare_corpus.sh /content/prototype_1152 \
#                       /content/prototype_nat768 \
#                       "$DRIVE/experiments/baselines_v2/data"
#
# WHY THIS EXISTS
# ---------------
# Colab wipes /content on every runtime reset, and the full study needs two
# sessions. Without a cache you pay the ~12 minute resize again every time you
# reconnect. With one, you pay a ~2-4 minute sequential read instead.
#
# WHY A TAR AND NOT A FOLDER ON DRIVE
# -----------------------------------
# Drive is mounted over the network. 6,670 individual files means 6,670 round
# trips - copying the folder is far slower than the resize it was meant to
# replace, and listing it is slow too. One ~2.15 GB tar is a single sequential
# stream in each direction. Not gzipped: the contents are already JPEG, so
# compression would burn CPU to save nothing.
#
# WHY TRAINING STILL READS FROM /content
# --------------------------------------
# The cache is for SURVIVING A RESTART, not for training off. Every image read
# during training must come off local disk or the GPU starves waiting on network
# round trips - which is the entire reason the corpus is extracted to /content in
# the first place. This script always leaves the usable copy on LOCAL_DEST.
#
# SAFETY
# ------
# A cached tar carries the .cs_geometry.json marker that resize_dataset.py wrote.
# Before extracting, the marker is read OUT OF THE TAR and checked against the
# current CS_DIM. A tar built at a different geometry is refused rather than
# silently extracted, because a 768x768 corpus fed to a nat768 run would be
# upscaled from the wrong aspect and nothing downstream would notice.
set -uo pipefail

RAW="${1:?usage: ./prepare_corpus.sh <RAW_DIR> <LOCAL_DEST> <DRIVE_CACHE_DIR>}"
DEST="${2:?give the local destination, e.g. /content/prototype_nat768}"
CACHE_DIR="${3:-}"
DIM="${CS_DIM:-nat768}"
export CS_DIM="$DIM"

MARKER=".cs_geometry.json"
NAME="$(basename "$DEST")"
# The cache is named from the SOURCE CORPUS + GEOMETRY, never from the
# destination directory. Naming it after $DEST means a run with a mismatched
# --dst overwrites a perfectly good cache for another geometry with its own
# content: the marker check would still refuse the bad tar later, but the good
# one is already gone.
TAR=""
[ -n "$CACHE_DIR" ] && TAR="$CACHE_DIR/$(basename "$RAW")_${DIM}.tar"

say() { echo "[prepare_corpus] $*"; }

# ---------------------------------------------------------------- already here?
if [ -f "$DEST/index.csv" ] && [ -f "$DEST/$MARKER" ]; then
  have="$(python - "$DEST/$MARKER" <<'PY'
import json,sys; print(json.load(open(sys.argv[1])).get("dim_name",""))
PY
)"
  if [ "$have" = "$DIM" ]; then
    say "already present at $DEST (geometry $have) - nothing to do"
    echo "$DEST"; exit 0
  fi
  say "WARNING: $DEST holds geometry '$have' but CS_DIM=$DIM - rebuilding"
  rm -rf "$DEST"
fi

# ---------------------------------------------------------------- restore?
if [ -n "$TAR" ] && [ -f "$TAR" ]; then
  say "found cache $TAR"
  cached="$(tar -xOf "$TAR" "$NAME/$MARKER" 2>/dev/null \
            | python -c 'import json,sys; print(json.load(sys.stdin).get("dim_name",""))' 2>/dev/null)"
  if [ "$cached" = "$DIM" ]; then
    say "geometry matches ($cached) - extracting, no resize needed"
    t0=$SECONDS
    if tar -xf "$TAR" -C "$(dirname "$DEST")"; then
      say "restored in $(( SECONDS - t0 ))s"
      echo "$DEST"; exit 0
    fi
    say "extract FAILED - falling through to a rebuild"
    rm -rf "$DEST"
  else
    say "cache is geometry '${cached:-unreadable}', need '$DIM' - ignoring it"
    say "  (delete $TAR if it is stale)"
  fi
fi

# ---------------------------------------------------------------- build
say "building $DEST at geometry $DIM (this is the slow path, ~12 min)"
if ! python resize_dataset.py --src "$RAW" --dst "$DEST"; then
  say "RESIZE FAILED - training can still run off $RAW, just slower"
  echo "$RAW"; exit 1
fi

# ---------------------------------------------------------------- save cache
if [ -n "$CACHE_DIR" ]; then
  mkdir -p "$CACHE_DIR"
  say "archiving to $TAR (~2.2 GB, one sequential write)"
  t0=$SECONDS
  # Write to a .part first: a half-written tar left by a disconnect would
  # otherwise be picked up as a valid cache on the next run.
  if tar -cf "$TAR.part" -C "$(dirname "$DEST")" "$NAME" && mv "$TAR.part" "$TAR"; then
    say "cached in $(( SECONDS - t0 ))s - next session skips the resize"
    du -h "$TAR" 2>/dev/null | awk '{print "[prepare_corpus] cache size: "$1}'
  else
    rm -f "$TAR.part"
    say "archiving FAILED (out of Drive space?) - continuing without a cache"
  fi
fi

echo "$DEST"
