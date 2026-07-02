#!/bin/bash
#SBATCH --job-name=fix-bm25
#SBATCH --partition=lovelace,Main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/fix_bm25_%j.out
#SBATCH --error=slurm/logs/fix_bm25_%j.err
#
# Apply + verify the BM25 article_id fix (NO SGLang, small GPU slice):
#   1. rebuild the bm25s index from Qdrant payloads (adds article_id/granularity)
#   2. re-run the retrieval-stage diagnostic — bm25 hits must now carry real
#      article_ids (were all None), and we check whether the gold article's rank
#      improves for comp_2 / comp_11 vs the passing controls.

set -euo pipefail
for _cand in "${SLURM_SUBMIT_DIR:-}/slurm" "$(dirname "$0")" "/u/$USER/phd_projects/RAG_EU_AI/slurm"; do
  if [ -f "$_cand/00_common.sh" ]; then source "$_cand/00_common.sh"; _common_ok=1; break; fi
done
[ -n "${_common_ok:-}" ] || { echo "FATAL: cannot locate slurm/00_common.sh" >&2; exit 1; }

export QDRANT_LOCAL_PATH="${QDRANT_LOCAL_PATH:-$PROJECT_ROOT/data/qdrant}"
unset QDRANT_URL || true

uv sync --extra ml --extra data

echo "[fix-bm25] === rebuilding BM25 index from Qdrant (article_id metadata) ==="
uv run python scripts/rebuild_bm25.py

echo "[fix-bm25] === re-running retrieval diagnostic ==="
uv run python scripts/diagnose_retrieval.py
