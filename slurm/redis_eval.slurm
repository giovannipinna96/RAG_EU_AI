#!/bin/bash
#SBATCH --job-name=redis-eval
#SBATCH --partition=Main,lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/redis_eval_%j.out
#SBATCH --error=slurm/logs/redis_eval_%j.err
#
# CPU-ONLY validation of the SemanticCache against a REAL Redis (the same
# redis:7-alpine image infra.slurm uses). No GPU: the embedding model is stubbed
# inside scripts/eval_redis.py. Catches Redis-specific bugs the in-memory mock
# can't (bytes round-trip, scan_iter keys, invalidate across namespaces,
# cross-client persistence). Schedules fast on Main; does not touch the A100.

set -euo pipefail
for _cand in "${SLURM_SUBMIT_DIR:-}/slurm" "$(dirname "$0")" "/u/$USER/phd_projects/RAG_EU_AI/slurm"; do
  if [ -f "$_cand/00_common.sh" ]; then source "$_cand/00_common.sh"; _common_ok=1; break; fi
done
[ -n "${_common_ok:-}" ] || { echo "FATAL: cannot locate slurm/00_common.sh" >&2; exit 1; }

REDIS_IMG="${REDIS_IMG:-docker://redis:7-alpine}"
REDIS_DATA="$DATA_DIR/redis_eval"
rm -rf "$REDIS_DATA"; mkdir -p "$REDIS_DATA"
export REDIS_URL="redis://localhost:6379/0"

# Core+dev deps only — redis-py is in core; no ml/torch needed.
uv sync

echo "[redis-eval] starting Redis (appendonly) on :6379 ..."
apptainer exec --writable-tmpfs --bind "$REDIS_DATA:/data" \
  "$REDIS_IMG" redis-server --appendonly yes --dir /data \
  --bind 0.0.0.0 --protected-mode no &
REDIS_PID=$!
trap 'echo "[redis-eval] teardown"; kill $REDIS_PID 2>/dev/null || true' EXIT

echo "[redis-eval] waiting for PONG ..."
ready=0
for i in $(seq 1 30); do
  if apptainer exec "$REDIS_IMG" redis-cli -h localhost ping 2>/dev/null | grep -q PONG; then
    echo "[redis-eval] Redis ready after ~${i}s"; ready=1; break
  fi
  kill -0 "$REDIS_PID" 2>/dev/null || { echo "[redis-eval] FATAL: Redis exited early"; exit 1; }
  sleep 1
done
[ "$ready" = 1 ] || { echo "[redis-eval] FATAL: Redis not ready after 30s"; exit 1; }

echo "[redis-eval] === running SemanticCache infra eval ==="
uv run python scripts/eval_redis.py
rc=$?

echo "[redis-eval] checking append-only persistence on disk ..."
ls -la "$REDIS_DATA" || true

echo "[redis-eval] eval exit code = $rc"
exit $rc
