#!/bin/bash
# Long-running, self-renewing server job. Start (or check on) the chain with:
#   ./ensure_serving.sh
# As soon as this job starts it queues its successor, held until LEAD before this job's
# time limit, on a different node. When the successor's server is healthy and its
# cloudflared connector is up, it cancels the older job, so traffic fails over with
# no gap. Crashes and hangs are handled inside serve.sh; node failures by --requeue.
# To stop the chain: touch STOP_SERVING && scancel -n tokenizer-serve
#SBATCH --job-name=tokenizer-serve
#SBATCH --partition=cpunodes
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.log
#SBATCH --open-mode=append
#SBATCH --requeue
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

LEAD=$((12 * 3600))   # how early the successor may start; covers queue wait
HEALTH_URL="http://127.0.0.1:8000/api/health"

log() { echo "$(date '+%F %T') serve.sbatch[$SLURM_JOB_ID]: $*"; }
serve_jobs() { squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t "$1" -o %i; }

# Queue the successor, unless one is already waiting (e.g. this job was requeued).
if [ -e STOP_SERVING ]; then
    log "STOP_SERVING found, not queueing a successor"
elif [ -n "$(serve_jobs PENDING)" ]; then
    log "successor already queued: $(serve_jobs PENDING | tr '\n' ' ')"
else
    END=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)
    BEGIN=$(date -d "@$((END - LEAD))" +%Y-%m-%dT%H:%M:%S)
    # Exclude this node: the successor binds the same port on 127.0.0.1.
    log "queueing successor to begin at $BEGIN"
    sbatch --begin="$BEGIN" --exclude="$(hostname)" serve.sbatch
fi

./serve.sh &
SERVE_PID=$!
trap 'kill -TERM "$SERVE_PID" 2>/dev/null' TERM INT

# Once this server is up and its tunnel connector has registered, retire the jobs it replaces.
until curl -fsS --max-time 10 "$HEALTH_URL" >/dev/null 2>&1; do
    kill -0 "$SERVE_PID" 2>/dev/null || break
    sleep 10
done
sleep 60
for job in $(serve_jobs RUNNING); do
    if [ "$job" -lt "$SLURM_JOB_ID" ]; then
        log "taking over from job $job"
        scancel "$job"
    fi
done

# `wait` returns early when a trapped signal arrives; keep waiting on the server.
while kill -0 "$SERVE_PID" 2>/dev/null; do
    wait "$SERVE_PID"
done
