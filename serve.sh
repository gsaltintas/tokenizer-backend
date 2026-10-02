#!/bin/bash
# Start the backend (and the Cloudflare tunnel, if a token file exists) on the current node,
# and keep them up: the backend is restarted if it exits or stops answering /api/health,
# and the tunnel is restarted if it exits. The tunnel connector only starts once the
# backend is healthy, so a new node doesn't take traffic before it can serve it.
# Run inside a Slurm job, e.g.: srun --jobid=<id> --overlap ./serve.sh
set -uo pipefail
cd "$(dirname "$0")"

# Hugging Face cache on node-local disk; /localscratch if the node has it, else /tmp.
if [ -d /localscratch ]; then SCRATCH=/localscratch; else SCRATCH=/tmp; fi
export HF_HOME="$SCRATCH/$USER/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
mkdir -p "$HF_HUB_CACHE" || exit 1
chmod 700 "$SCRATCH/$USER"
echo "$(hostname): HF cache at $HF_HUB_CACHE"

PORT=8000
HEALTH_URL="http://127.0.0.1:$PORT/api/health"
TOKEN_FILE="$HOME/.cloudflared-token"
STARTUP_TIMEOUT=600   # seconds a fresh backend gets to become healthy
MAX_FAILS=4           # consecutive failed checks (15 s apart) before a restart

APP_PID=
TUNNEL_PID=

log() { echo "$(date '+%F %T') serve.sh: $*"; }

healthy() { curl -fsS --max-time 10 "$HEALTH_URL" >/dev/null 2>&1; }

# Each child runs in its own process group (setsid), so stopping it also stops
# whatever it spawned (uv -> uvicorn) instead of leaving an orphan on the port.
stop_group() {
    local pid=$1
    [ -n "$pid" ] || return 0
    kill -TERM -- "-$pid" 2>/dev/null
    for _ in $(seq 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
}

start_app() {
    setsid uv run uvicorn app.main:app --host 127.0.0.1 --port "$PORT" &
    APP_PID=$!
    APP_STARTED=$(date +%s)
    APP_WAS_HEALTHY=false
    FAILS=0
    log "started backend (pid $APP_PID)"
}

start_tunnel() {
    [ -f "$TOKEN_FILE" ] || return 0
    # Token via env var, not argv, so it doesn't show up in `ps`.
    TUNNEL_TOKEN="$(cat "$TOKEN_FILE")" setsid "$HOME/bin/cloudflared" tunnel run &
    TUNNEL_PID=$!
    log "started tunnel connector (pid $TUNNEL_PID)"
}

shutdown() {
    log "shutting down"
    stop_group "$TUNNEL_PID"
    stop_group "$APP_PID"
}
trap shutdown EXIT
trap 'exit 0' TERM INT

BACKOFF=5
start_app
while true; do
    # Background sleep + wait, so a SIGTERM is handled immediately.
    sleep 15 &
    wait $!

    if ! kill -0 "$APP_PID" 2>/dev/null; then
        wait "$APP_PID"
        log "backend exited with status $?; restarting in ${BACKOFF}s"
        sleep "$BACKOFF"
        BACKOFF=$(( BACKOFF * 2 > 300 ? 300 : BACKOFF * 2 ))
        start_app
        continue
    fi

    if healthy; then
        APP_WAS_HEALTHY=true
        FAILS=0
        BACKOFF=5
        if [ -f "$TOKEN_FILE" ] && ! { [ -n "$TUNNEL_PID" ] && kill -0 "$TUNNEL_PID" 2>/dev/null; }; then
            [ -n "$TUNNEL_PID" ] && log "tunnel connector exited; restarting"
            start_tunnel
        fi
    elif $APP_WAS_HEALTHY; then
        FAILS=$((FAILS + 1))
        if [ "$FAILS" -ge "$MAX_FAILS" ]; then
            log "backend failed $FAILS health checks; restarting"
            stop_group "$APP_PID"
            start_app
        fi
    elif [ $(( $(date +%s) - APP_STARTED )) -gt "$STARTUP_TIMEOUT" ]; then
        log "backend not healthy after ${STARTUP_TIMEOUT}s; restarting"
        stop_group "$APP_PID"
        start_app
    fi
done
