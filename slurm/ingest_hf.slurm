#!/bin/bash
#SBATCH --job-name=rag-ingest-hf
#SBATCH --partition=lovelace,Main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/ingest_hf_%j.out
#SBATCH --error=slurm/logs/ingest_hf_%j.err
#
# Ingestion using the Hugging Face mirror as the corpus (small, reliable — no
# EUR-Lex 202 issue). Requires the SGLang server and, unless QDRANT_LOCAL_PATH is
# set, the infra job. See slurm/README.md.

# Locate 00_common.sh under both `sbatch` (script is relocated to a spool
# dir, so $0 is useless) and interactive `bash slurm/<x>.slurm`.
for _cand in "${SLURM_SUBMIT_DIR:-}/slurm" "$(dirname "$0")" "/u/$USER/phd_projects/RAG_EU_AI/slurm"; do
  if [ -f "$_cand/00_common.sh" ]; then source "$_cand/00_common.sh"; _common_ok=1; break; fi
done
[ -n "${_common_ok:-}" ] || { echo "FATAL: cannot locate slurm/00_common.sh" >&2; exit 1; }

# ML extras for embeddings/reranker/graph + data extra for the parquet loader.
uv sync --extra ml --extra data

uv run python scripts/download_hf.py
uv run python scripts/ingest_hf.py

echo "[ingest-hf] complete"
