#!/bin/bash
#SBATCH --job-name=rag-api-tunnel
#SBATCH --partition=Main,lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/api_tunnel_%j.out
#SBATCH --error=slurm/logs/api_tunnel_%j.err
#
# Serves the FastAPI app on a GPU compute node, reaching the external gemma-4
# LLM server (172.30.42.129:8080, lab network) through an SSH port-forward via
# the demetra login node. Compute nodes cannot route to the lab network directly;
# only the login node can. Requires passwordless SSH compute->login (key in
# ~/.ssh, shared home). The GPU runs the local BGE embedding + reranker.
#
#   Reach from the login node once RUNNING:
#     curl http://<node>:8000/health
#     RAG_API_URL=http://<node>:8000/answer uv run python scripts/run_eval.py

for _cand in "${SLURM_SUBMIT_DIR:-}/slurm" "$(dirname "$0")" "/u/$USER/phd_projects/RAG_EU_AI/slurm"; do
  if [ -f "$_cand/00_common.sh" ]; then source "$_cand/00_common.sh"; _common_ok=1; break; fi
done
[ -n "${_common_ok:-}" ] || { echo "FATAL: cannot locate slurm/00_common.sh" >&2; exit 1; }

# --- Self-healing SSH tunnel: localhost:8080 (this node) -> gemma via demetra --
# A single `ssh -f` drops if the login node reaps the idle/long connection (seen
# to die mid model-load). Supervise it in a reconnect loop with aggressive
# keepalives so a blip self-heals instead of 500ing every request.
GEMMA_HOSTPORT="${GEMMA_HOSTPORT:-172.30.42.129:8080}"
LLM_LOCAL_PORT="${LLM_LOCAL_PORT:-8080}"
echo "[api] starting self-healing tunnel localhost:${LLM_LOCAL_PORT} -> ${GEMMA_HOSTPORT} via demetra"
(
  while true; do
    ssh -N \
        -o BatchMode=yes -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=no \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes \
        -L "${LLM_LOCAL_PORT}:${GEMMA_HOSTPORT}" demetra
    echo "[tunnel] disconnected (rc=$?), reconnecting in 3s..." >&2
    sleep 3
  done
) &
TUNNEL_LOOP_PID=$!
trap 'kill "$TUNNEL_LOOP_PID" 2>/dev/null; pkill -f "${LLM_LOCAL_PORT}:${GEMMA_HOSTPORT}" 2>/dev/null' EXIT

# Wait for the forward to come up (first connect can take a moment).
_up=""
for _i in $(seq 1 20); do
  if curl -sf -m 5 "http://localhost:${LLM_LOCAL_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[api] tunnel verified: gemma reachable via localhost:${LLM_LOCAL_PORT} (attempt ${_i})"; _up=1; break
  fi
  sleep 2
done
[ -n "$_up" ] || { echo "FATAL: gemma not reachable through tunnel" >&2; exit 1; }

# Point the app at the tunnel (env overrides .env in pydantic-settings).
export SGLANG_BASE_URL="http://localhost:${LLM_LOCAL_PORT}/v1"
echo "[api] SGLANG_BASE_URL=$SGLANG_BASE_URL"

uv sync --extra ml

echo "[api] serving on $(hostname -f):8000"
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
