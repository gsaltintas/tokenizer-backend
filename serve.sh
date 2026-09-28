#!/bin/bash
# Start the backend (and the Cloudflare tunnel, if a token file exists) on the current node.
# Run inside a Slurm job, e.g.: srun --jobid=<id> --overlap ./serve.sh
set -euo pipefail
cd "$(dirname "$0")"

# Hugging Face cache on node-local disk; /localscratch if the node has it, else /tmp.
if [ -d /localscratch ]; then SCRATCH=/localscratch; else SCRATCH=/tmp; fi
export HF_HOME="$SCRATCH/$USER/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
mkdir -p "$HF_HUB_CACHE"
chmod 700 "$SCRATCH/$USER"
echo "$(hostname): HF cache at $HF_HUB_CACHE"

TOKEN_FILE="$HOME/.cloudflared-token"
if [ -f "$TOKEN_FILE" ]; then
    # Token via env var, not argv, so it doesn't show up in `ps`.
    TUNNEL_TOKEN="$(cat "$TOKEN_FILE")" "$HOME/bin/cloudflared" tunnel run &
    TUNNEL_PID=$!
    trap 'kill $TUNNEL_PID 2>/dev/null' EXIT
fi

uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
