#!/bin/bash
# Run ON A MACHINE THAT REACHES BOTH the workstation and the cluster login (inside tmux).
# Copies each packed .tar (pack_mead_front.sh) to the cluster through this machine, waiting for
# each archive to be packed. Resumable: if a connection drops it retries from the byte reached,
# and rerunning the script continues where it stopped. Measured rate on 2026-10-29: ~1.7 MB/s.
#
#   WS=<user>@<workstation> CL=<user>@<cluster-login> SRC=<staging dir on WS> DST=<dir on cluster> \
#   bash copy_resumable.sh
#
# Then on the cluster: (cd $DST && sha256sum -c mead_<emotion>.tar.sha256) and
# tar xf $DST/mead_<emotion>.tar -C <parent of MEAD_raw_subset>.
: "${WS:?set WS}"; : "${CL:?set CL}"; : "${SRC:?set SRC}"; : "${DST:?set DST}"
EMOS=${EMOS:-"surprised angry sad happy fear disgusted contempt"}
O="-o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%C -o ControlPersist=2h"
r() { local out; until out=$(ssh -n $O "$@"); do echo "  (ssh failed, retrying in 30 s)" >&2; sleep 30; done; echo "$out"; }
r "$CL" "mkdir -p $DST" >/dev/null
for emo in $EMOS; do
  until [ "$(r "$WS" "test -s $SRC/mead_$emo.tar.sha256 && echo ok || echo no")" = ok ]; do
    echo "$(date +%T) $emo: waiting for packing on the workstation"; sleep 120
  done
  total=$(r "$WS" "stat -c %s $SRC/mead_$emo.tar")
  while :; do
    have=$(r "$CL" "stat -c %s $DST/mead_$emo.tar 2>/dev/null || echo 0")
    [ "$have" -ge "$total" ] && break
    echo "$(date +%T) $emo: $((have / 1048576)) of $((total / 1048576)) MB"
    ssh -n $O "$WS" "tail -c +$((have + 1)) $SRC/mead_$emo.tar" | ssh $O "$CL" "cat >> $DST/mead_$emo.tar" || sleep 30
  done
  r "$WS" "cat $SRC/mead_$emo.tar.sha256" | ssh $O "$CL" "cat > $DST/mead_$emo.tar.sha256"
  echo "$(date +%T) $emo: complete ($((total / 1048576)) MB)"
done
echo DONE
