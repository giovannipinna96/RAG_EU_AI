#!/bin/bash
#SBATCH --job-name=rag-ingest
#SBATCH --partition=lovelace,Main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/ingest_%j.out
#SBATCH --error=slurm/logs/ingest_%j.err
#
# One-shot offline ingestion: download -> parse -> chunk -> embed/index ->
# BM25 -> LightRAG graph. Requires the SGLang server (sglang_server.slurm) and,
# unless QDRANT_LOCAL_PATH is set, the infra job (infra.slurm) to be running.
# Set SGLANG_BASE_URL / QDRANT_URL in .env to the nodes hosting those services.

# Locate 00_common.sh under both `sbatch` (script is relocated to a spool
# dir, so $0 is useless) and interactive `bash slurm/<x>.slurm`.
for _cand in "${SLURM_SUBMIT_DIR:-}/slurm" "$(dirname "$0")" "/u/$USER/phd_projects/RAG_EU_AI/slurm"; do
  if [ -f "$_cand/00_common.sh" ]; then source "$_cand/00_common.sh"; _common_ok=1; break; fi
done
[ -n "${_common_ok:-}" ] || { echo "FATAL: cannot locate slurm/00_common.sh" >&2; exit 1; }

# Install the GPU-capable ML extra into the project venv (idempotent).
uv sync --extra ml

uv run python scripts/download.py
uv run python scripts/ingest.py

echo "[ingest] complete"
