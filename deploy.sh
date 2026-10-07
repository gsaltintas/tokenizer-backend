#!/bin/bash
# Deploy a ref (default: main) to the live server. Run from the serving worktree, which has
# the `serve` branch checked out (see README, "Working vs. serving branch"):
#   bash -ic '/h/371/gsaltintas/tokenizer-exploration/tokenizer-backend-serve/deploy.sh [ref]'
# (bash -ic, so the submitted job inherits HF_TOKEN from ~/.bashrc.)
# - fast-forwards `serve` to <ref> and syncs the venv
# - drops any queued successor (it may have been submitted from another directory)
# - submits a fresh serve.sbatch from here, on a node other than the running ones; once it
#   is healthy it cancels the old job, as on a normal handover
set -euo pipefail
cd "$(dirname "$0")"
NAME=tokenizer-serve
REF=${1:-main}

log() { echo "$(date '+%F %T') deploy: $*"; }

[ "$(git branch --show-current)" = serve ] || { log "not on the serve branch; refusing"; exit 1; }
git diff --quiet HEAD || { log "uncommitted changes in $(pwd); refusing"; exit 1; }
[ -z "${HF_TOKEN:-}" ] && log "warning: HF_TOKEN is not set (run via bash -ic)"

git merge --ff-only "$REF"
uv sync --frozen
log "serving $(git log --oneline -1)"

rm -f STOP_SERVING
mkdir -p logs
for job in $(squeue -h -u "$USER" -n "$NAME" -t PENDING -o %i); do
    log "cancelling queued job $job"
    scancel "$job"
done
nodes=$(squeue -h -u "$USER" -n "$NAME" -t RUNNING,CONFIGURING -o %N | paste -sd, -)
sbatch ${nodes:+--exclude="$nodes"} serve.sbatch
