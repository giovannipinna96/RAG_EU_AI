#!/bin/bash
# start_challenge.sh — bring up the EU AI Act RAG for the competition.
#
# One command starts the FastAPI app on a GPU node via slurm/api_tunnel.slurm,
# which opens the SSH tunnel to the external gemma LLM (through demetra), warms
# the models at startup, and serves POST /answer on <node>:8000. Config comes
# entirely from .env (LightRAG mix, rerank/voting off, thinking off, embedded
# Qdrant, 24h cache TTL).
#
# Usage:
#   scripts/start_challenge.sh [--warmup] [--public] [--time HH:MM:SS]
#   scripts/start_challenge.sh --stop
#
#   --warmup   after the API is up, pre-compute the ~152 predictable answers
#              (served at ~5ms afterwards).
#   --public   expose a temporary public HTTPS URL via cloudflared (trycloudflare).
#   --time     override the slurm walltime (default 24h from the job script).
#   --stop     tear down: scancel the API job and kill cloudflared.
#
# Prereqs (one-time, already done in this repo): passwordless SSH compute->login
# (~/.ssh key), ingested data (bm25_index/, lightrag_data/, data/qdrant), and a
# reachable gemma server. Run from the login node (demetra).
set -euo pipefail

PROJECT_ROOT="/u/$USER/phd_projects/RAG_EU_AI"
cd "$PROJECT_ROOT"
STATE="slurm/logs/.challenge_state"
GEMMA="http://172.30.42.129:8080"
CF_BIN="$HOME/bin/cloudflared"

DO_WARMUP=0; DO_PUBLIC=0; DO_STOP=0; TIME_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --warmup) DO_WARMUP=1 ;;
    --public) DO_PUBLIC=1 ;;
    --stop)   DO_STOP=1 ;;
    --time)   shift; TIME_OVERRIDE="$1" ;;
    -h|--help) tail -n +2 "$0" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

# ----------------------------------------------------------------- teardown ---
if [ "$DO_STOP" = 1 ]; then
  [ -f "$STATE" ] && . "$STATE" || true
  [ -n "${JID:-}" ] && { scancel "$JID" 2>/dev/null && echo "[stop] scancel $JID" || true; }
  [ -n "${CF_PID:-}" ] && { kill "$CF_PID" 2>/dev/null && echo "[stop] killed cloudflared $CF_PID" || true; }
  pkill -f "cloudflared tunnel" 2>/dev/null || true
  rm -f "$STATE"
  echo "[stop] done"; exit 0
fi

mkdir -p slurm/logs

# --------------------------------------------------- gemma precondition check --
echo "[check] gemma inference reachable?"
if curl -sf -m 20 "$GEMMA/v1/chat/completions" -H "Content-Type: application/json" \
     -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":3,"chat_template_kwargs":{"enable_thinking":false}}' \
     >/dev/null 2>&1; then
  echo "[check] gemma OK"
else
  echo "[check] WARNING: gemma did not answer inference in 20s — the API will start"
  echo "        but fresh /answer queries will hang until gemma is back up."
fi

# ----------------------------------------------------------- submit API job ---
SBATCH_ARGS=(--parsable)
[ -n "$TIME_OVERRIDE" ] && SBATCH_ARGS+=(--time="$TIME_OVERRIDE")
JID=$(sbatch "${SBATCH_ARGS[@]}" slurm/api_tunnel.slurm)
echo "JID=$JID" > "$STATE"
echo "[api] submitted job $JID"

echo "[api] waiting for it to run + finish startup (models load ~40s)…"
for _ in $(seq 1 120); do
  st=$(squeue -j "$JID" -h -o "%t" 2>/dev/null || echo "")
  [ "$st" = "R" ] && grep -q "Application startup complete" "slurm/logs/api_tunnel_${JID}.err" 2>/dev/null && break
  if [ -z "$st" ]; then echo "[api] ERROR: job left the queue early — see slurm/logs/api_tunnel_${JID}.err" >&2; exit 1; fi
  sleep 5
done

NODE=$(squeue -j "$JID" -h -o "%N" 2>/dev/null)
API="http://$NODE:8000"
echo "NODE=$NODE" >> "$STATE"
echo "API=$API"   >> "$STATE"

if curl -sf -m 10 "$API/health" >/dev/null; then
  echo "[api] UP  →  $API   (health OK)"
else
  echo "[api] WARNING: health check failed at $API" >&2
fi

# ------------------------------------------------------------------- warmup ---
if [ "$DO_WARMUP" = 1 ]; then
  echo "[warmup] pre-computing predictable answers (~152, several minutes)…"
  RAG_API_URL="$API/answer" uv run python scripts/warmup_cache.py || echo "[warmup] non-fatal error"
fi

# ------------------------------------------------------------------- public ---
if [ "$DO_PUBLIC" = 1 ]; then
  if [ ! -x "$CF_BIN" ]; then
    echo "[public] downloading cloudflared…"
    mkdir -p "$HOME/bin"
    curl -fsSL -m 120 -o "$CF_BIN" \
      https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x "$CF_BIN"
  fi
  CF_LOG="slurm/logs/cloudflared_${JID}.log"
  nohup "$CF_BIN" tunnel --no-autoupdate --url "$API" > "$CF_LOG" 2>&1 &
  CF_PID=$!
  echo "CF_PID=$CF_PID" >> "$STATE"
  for _ in $(seq 1 20); do
    URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$CF_LOG" 2>/dev/null | head -1)
    [ -n "$URL" ] && break
    sleep 3
  done
  echo "PUBLIC_URL=${URL:-<not-ready>}" >> "$STATE"
  echo "[public] URL: ${URL:-<check $CF_LOG>}"
fi

# ------------------------------------------------------------------ summary ---
echo
echo "================ challenge up ================"
echo "  job     : $JID  (node $NODE, walltime ${TIME_OVERRIDE:-24:00:00})"
echo "  API     : $API/answer"
echo "  browser : $API/  (test page)   |   $API/docs"
[ "$DO_PUBLIC" = 1 ] && echo "  public  : ${URL:-<not-ready>}"
echo "  logs    : slurm/logs/api_tunnel_${JID}.{out,err}"
echo "  stop    : scripts/start_challenge.sh --stop"
echo "=============================================="
