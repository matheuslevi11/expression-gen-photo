#!/bin/bash
# Run ON THE WORKSTATION (in the background: nohup bash pack_mead_front.sh > pack.log 2>&1 &).
# Packs MEAD's front-view clips into one .tar per emotion, in the layout preprocess_mead.py reads
# (MEAD_raw_subset/<actor>/video/front/<emotion>/level_<n>/<clip>.mp4), with a .sha256 per tar.
# Rerunnable: emotions whose .sha256 exists are skipped. Checks free space first.
#
#   GP=<dir containing MEAD/<actor>/video_extracted*/...> OUT=<staging dir> \
#   EXCLUDE_ACTORS="M003 M005 ..." bash pack_mead_front.sh
#
# EXCLUDE_ACTORS: actors already on the cluster (2026-11-15: the 10 actors of the HF mirror,
# see docs/experiments/2026-11-15-multi-au-e0-design.md). Neutral is already there for all 47.
set -euo pipefail
: "${GP:?set GP}"; : "${OUT:?set OUT}"
EMOS=${EMOS:-"surprised angry sad happy fear disgusted contempt"}
mkdir -p "$OUT"; cd "$GP"
find MEAD -path "*/front/*" -name "*.mp4" -print0 > "$OUT/front_list.bin"
sel() {  # NUL-separated front clips of emotion $1, minus EXCLUDE_ACTORS (empty output is fine)
  { grep -z "/front/$1/" "$OUT/front_list.bin" || true; } \
    | if [ -n "${EXCLUDE_ACTORS:-}" ]; then { grep -zv -E "^MEAD/($(echo $EXCLUDE_ACTORS | tr ' ' '|'))/" || true; }; else cat; fi
}
need=0
for emo in $EMOS; do
  n=$(sel "$emo" | tr -cd '\0' | wc -c)
  b=$(sel "$emo" | du -cb --files0-from=- | tail -1 | cut -f1)
  echo "$emo $n clips $(awk -v b="$b" 'BEGIN{printf "%.2f", b/1e9}') GB"
  [ -s "$OUT/mead_$emo.tar.sha256" ] || need=$((need + b))
done
avail=$(df -B1 --output=avail "$OUT" | tail -1)
echo "to pack $(awk -v b=$need 'BEGIN{printf "%.1f", b/1e9}') GB; free $(awk -v b=$avail 'BEGIN{printf "%.1f", b/1e9}') GB"
[ "$avail" -gt $((need + 2000000000)) ] || { echo "ERROR: not enough space in $OUT"; exit 1; }
for emo in $EMOS; do
  [ -s "$OUT/mead_$emo.tar.sha256" ] && continue
  sel "$emo" | sort -z \
    | tar cf "$OUT/mead_$emo.tar.part" --null -T - \
        --transform "s|^MEAD/\([^/]*\)/.*/front/|MEAD_raw_subset/\1/video/front/|"
  mv "$OUT/mead_$emo.tar.part" "$OUT/mead_$emo.tar"
  (cd "$OUT" && sha256sum "mead_$emo.tar" > "mead_$emo.tar.sha256")
  echo "$(date +%T) $emo ready: $(stat -c %s "$OUT/mead_$emo.tar") bytes"
done
echo DONE
