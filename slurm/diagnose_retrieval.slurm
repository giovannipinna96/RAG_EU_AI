#!/bin/bash
#SBATCH --job-name=diag-retrieval
#SBATCH --partition=lovelace,Main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/diag_retrieval_%j.out
#SBATCH --error=slurm/logs/diag_retrieval_%j.err
#
# Phase-1 retrieval diagnostic (NO SGLang). Loads only the embedding model +
# Qdrant (embedded) + BM25 + BGE reranker, so a small GPU slice is plenty and it
# schedules fast without competing for the full A100. Reports where the gold
# article drops out of the pipeline for comp_2/comp_11 vs passing controls.

set -euo pipefail
for _cand in "${SLURM_SUBMIT_DIR:-}/slurm" "$(dirname "$0")" "/u/$USER/phd_projects/RAG_EU_AI/slurm"; do
  if [ -f "$_cand/00_common.sh" ]; then source "$_cand/00_common.sh"; _common_ok=1; break; fi
done
[ -n "${_common_ok:-}" ] || { echo "FATAL: cannot locate slurm/00_common.sh" >&2; exit 1; }

export QDRANT_LOCAL_PATH="${QDRANT_LOCAL_PATH:-$PROJECT_ROOT/data/qdrant}"
unset QDRANT_URL || true

uv sync --extra ml --extra data
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "[diag] === retrieval stage diagnostic ==="
uv run python scripts/diagnose_retrieval.py
