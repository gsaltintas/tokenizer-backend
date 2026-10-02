#!/bin/bash
# Make sure a tokenizer-serve job is running; idempotent, so safe to run from cron:
#   */15 * * * * /h/371/gsaltintas/tokenizer-exploration/tokenizer-backend/ensure_serving.sh >> /h/371/gsaltintas/tokenizer-exploration/tokenizer-backend/logs/ensure.log 2>&1
# - nothing queued         -> submit serve.sbatch
# - only a held successor  -> release it now (the running job died unexpectedly)
# Does nothing while STOP_SERVING exists.
set -euo pipefail
cd "$(dirname "$0")"
NAME=tokenizer-serve

log() { echo "$(date '+%F %T') ensure_serving: $*"; }
[ -e STOP_SERVING ] && exit 0

running=$(squeue -h -u "$USER" -n "$NAME" -t RUNNING,CONFIGURING -o %i)
pending=$(squeue -h -u "$USER" -n "$NAME" -t PENDING -o %i)

if [ -n "$running" ]; then
    exit 0
elif [ -n "$pending" ]; then
    for job in $pending; do
        # Only the --begin hold; jobs waiting on resources are left alone.
        if [ "$(squeue -h -j "$job" -o %r)" = BeginTime ]; then
            log "no running job; releasing held successor $job"
            scontrol update JobId="$job" StartTime=now
        fi
    done
else
    mkdir -p logs
    log "no job queued; submitting"
    sbatch serve.sbatch
fi
