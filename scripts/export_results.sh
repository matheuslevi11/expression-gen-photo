#!/usr/bin/env bash
# Stage lightweight experiment results into a dated tarball under exports/,
# ready to pull to a personal machine over SSH.
#
# Default bundle (~10 MB): all inference GIFs/figures + per-run config.yaml,
# log.txt and sanity-check GIFs (training provenance). Checkpoints and raw
# data are never included.
#
# Usage:
#   scripts/export_results.sh                 # light bundle
#   scripts/export_results.sh --with-samples  # also include validation GIFs (~77 MB)
#
# Then, from the personal computer:
#   scp <user>@<this-host>:<repo>/exports/results_<stamp>.tar.gz .

set -euo pipefail
cd "$(dirname "$0")/.."

WITH_SAMPLES=0
[[ "${1:-}" == "--with-samples" ]] && WITH_SAMPLES=1

STAMP=$(date +%Y-%m-%d_%H%M%S)
STAGE="exports/results_${STAMP}"
mkdir -p "${STAGE}"

# 1. All inference outputs (GIFs, ablation figures, eval sets) — small by design.
if [[ -d inference_output ]]; then
    cp -r inference_output "${STAGE}/inference_output"
fi

# 2. Training provenance per run: config, full log, sanity-check GIFs.
for run in output/expression*/expression*/; do
    [[ -d "${run}" ]] || continue
    dest="${STAGE}/runs/$(basename "${run}")"
    mkdir -p "${dest}"
    for f in config.yaml log.txt; do
        [[ -f "${run}${f}" ]] && cp "${run}${f}" "${dest}/"
    done
    [[ -d "${run}sanity_check" ]] && cp -r "${run}sanity_check" "${dest}/sanity_check"
    if [[ ${WITH_SAMPLES} -eq 1 && -d "${run}samples" ]]; then
        cp -r "${run}samples" "${dest}/samples"
    fi
done

# 3. Manifest: what this bundle is and the repo state it came from.
{
    echo "exported : $(date --iso-8601=seconds)"
    echo "host     : $(hostname)"
    echo "repo     : $(pwd)"
    echo "commit   : $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a') ($(git log -1 --format=%s 2>/dev/null || true))"
    echo "dirty    : $(git status --porcelain 2>/dev/null | grep -c . || true) uncommitted change(s)"
    echo "samples  : $([[ ${WITH_SAMPLES} -eq 1 ]] && echo included || echo excluded)"
} > "${STAGE}/MANIFEST.txt"

TARBALL="exports/results_${STAMP}.tar.gz"
tar czf "${TARBALL}" -C exports "results_${STAMP}"
rm -rf "${STAGE}"

echo "Bundle: $(pwd)/${TARBALL} ($(du -h "${TARBALL}" | cut -f1))"
echo
echo "Pull it from your personal computer with:"
echo "  scp $(whoami)@$(hostname):$(pwd)/${TARBALL} ."
echo "or keep a synced local mirror of all bundles:"
echo "  rsync -avz $(whoami)@$(hostname):$(pwd)/exports/ ./genphoto-results/"
