#!/bin/bash
#SBATCH --job-name=chk-snip
#SBATCH --partition=lovelace,Main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/chk_snip_%j.out
#SBATCH --error=slurm/logs/chk_snip_%j.err

set -euo pipefail
for _cand in "${SLURM_SUBMIT_DIR:-}/slurm" "$(dirname "$0")" "/u/$USER/phd_projects/RAG_EU_AI/slurm"; do
  if [ -f "$_cand/00_common.sh" ]; then source "$_cand/00_common.sh"; _common_ok=1; break; fi
done
[ -n "${_common_ok:-}" ] || { echo "FATAL: cannot locate slurm/00_common.sh" >&2; exit 1; }

export QDRANT_LOCAL_PATH="${QDRANT_LOCAL_PATH:-$PROJECT_ROOT/data/qdrant}"
unset QDRANT_URL || true

uv sync --extra ml --extra data
uv run python scripts/check_snippet_len.py
