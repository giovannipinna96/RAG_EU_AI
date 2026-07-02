#!/bin/bash
#SBATCH --job-name=rag-api
#SBATCH --partition=lovelace,Main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/api_%j.out
#SBATCH --error=slurm/logs/api_%j.err
#
# Serves the FastAPI app with uvicorn. A GPU is requested so the local embedding
# and reranker models run fast. Reaches SGLang / Qdrant / Redis at the URLs in
# .env. After it is up, warm the cache and evaluate from the login node or here:
#   RAG_API_URL=http://<this-node>:8000/answer uv run python scripts/warmup_cache.py
#   RAG_API_URL=http://<this-node>:8000/answer uv run python scripts/run_eval.py

# Locate 00_common.sh under both `sbatch` (script is relocated to a spool
# dir, so $0 is useless) and interactive `bash slurm/<x>.slurm`.
for _cand in "${SLURM_SUBMIT_DIR:-}/slurm" "$(dirname "$0")" "/u/$USER/phd_projects/RAG_EU_AI/slurm"; do
  if [ -f "$_cand/00_common.sh" ]; then source "$_cand/00_common.sh"; _common_ok=1; break; fi
done
[ -n "${_common_ok:-}" ] || { echo "FATAL: cannot locate slurm/00_common.sh" >&2; exit 1; }

uv sync --extra ml

echo "[api] serving on $(hostname -f):8000"
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
