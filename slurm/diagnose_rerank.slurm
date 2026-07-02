#!/bin/bash
#SBATCH --job-name=diag-rerank
#SBATCH --partition=lovelace,Main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/diag_rerank_%j.out
#SBATCH --error=slurm/logs/diag_rerank_%j.err
#
# Phase-2 LLM-rerank diagnostic. Requires:
#   - the same Qdrant + bm25_index used by the eval (embedded mode, on disk)
#   - GPU slice for the BGE reranker (FlagEmbedding) and embedding model
#
# If RAG_LLM_URL is set in the env (e.g. http://lovelace-02:8000/v1) the script
# will also call the LLM with three prompt variants. Without it, only the
# prompt is printed.

set -euo pipefail
for _cand in "${SLURM_SUBMIT_DIR:-}/slurm" "$(dirname "$0")" "/u/$USER/phd_projects/RAG_EU_AI/slurm"; do
  if [ -f "$_cand/00_common.sh" ]; then source "$_cand/00_common.sh"; _common_ok=1; break; fi
done
[ -n "${_common_ok:-}" ] || { echo "FATAL: cannot locate slurm/00_common.sh" >&2; exit 1; }

export QDRANT_LOCAL_PATH="${QDRANT_LOCAL_PATH:-$PROJECT_ROOT/data/qdrant}"
unset QDRANT_URL || true

# By default skip LLM calls -- only print the prompts (fast, no SGLang needed).
# Set RAG_LLM_URL via `--export=ALL,RAG_LLM_URL=http://host:port/v1` to enable.
export RAG_LLM_URL="${RAG_LLM_URL:-}"

uv sync --extra ml --extra data

echo "[diag-rerank] RAG_LLM_URL=${RAG_LLM_URL}"
uv run python scripts/diagnose_rerank.py
