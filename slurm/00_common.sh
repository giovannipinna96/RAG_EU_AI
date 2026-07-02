#!/bin/bash
# Shared environment for all SLURM jobs. Source this from each job script:
#   source "$(dirname "$0")/00_common.sh"
#
# It loads Apptainer, points caches at fast local scratch (home is slow NFS),
# and exports the project root + venv paths.

set -euo pipefail

# --- Project layout -------------------------------------------------------
export PROJECT_ROOT="${PROJECT_ROOT:-/u/$USER/phd_projects/RAG_EU_AI}"

# Heavy artifacts (multi-GB container images, model weights, CUDA wheels) go to
# group shared storage — node-local /tmp (~56 GB) is too small for the SGLang
# CUDA image, and home has a tight quota. Override BIG_STORE to relocate.
export BIG_STORE="${BIG_STORE:-/share/malelab/RAG_EU_AI}"
if [ ! -d "$BIG_STORE" ] || ! touch "$BIG_STORE/.wtest_$$" 2>/dev/null; then
  echo "[common] WARN: BIG_STORE=$BIG_STORE not writable; falling back to /tmp/$USER" >&2
  export BIG_STORE="/tmp/$USER/rag_store"
else
  rm -f "$BIG_STORE/.wtest_$$"
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$BIG_STORE/uv-cache}"
# FORCE (not `:-`) these onto BIG_STORE. The user's ~/.bashrc points HF_HOME and
# APPTAINER_TMPDIR at /share/malelab/gpinna/sf-data, which is quota-full (~5 MB
# free) even though the /share filesystem has TBs free. Overriding here (scoped
# to these jobs) sends model weights + build scratch to the roomy RAG_EU_AI store.
export HF_HOME="$BIG_STORE/huggingface"
export HF_HUB_CACHE="$BIG_STORE/huggingface/hub"
# Put the heavy GPU venv (torch + CUDA, several GB) on the shared store too, so
# it is separate from the lightweight login-node .venv and off the home quota.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$BIG_STORE/.venv-gpu}"

# --- Apptainer ------------------------------------------------------------
# Module name as seen via `module avail` on Demetra (2026-05).
module load apptainer/1.1.9-gcc-13.2.0-i4ns3xh 2>/dev/null || module load apptainer

# Image cache + build tmp on the shared store (forced, same quota reason as HF).
export APPTAINER_CACHEDIR="$BIG_STORE/apptainer/cache"
export APPTAINER_TMPDIR="$BIG_STORE/apptainer/tmp"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# --- Shared LLM-server constants (overridable per job) -------------------
# The node GPU driver (550.54.15) supports CUDA up to 12.4. We use the vLLM
# cu124 *image* because it matches that driver. Recent SGLang *container images*
# are built on CUDA 12.5-12.8 (> 12.4) so they fail on this driver — but SGLang
# itself supports CUDA 12.4 when installed from cu124 wheels (see sglang_probe.slurm).
# vLLM is OpenAI-compatible, so the RAG code is unchanged (it just calls /v1).
# Kept here so the image/port can't drift between single_node and llm_server.
export LLM_IMG="${LLM_IMG:-docker://vllm/vllm-openai:v0.6.6}"
export LLM_PORT="${LLM_PORT:-8899}"

# --- Data directories for stateful services ------------------------------
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
mkdir -p "$DATA_DIR/qdrant" "$DATA_DIR/redis"

cd "$PROJECT_ROOT"
echo "[common] host=$(hostname) project=$PROJECT_ROOT"
echo "[common] big_store=$BIG_STORE apptainer=$(command -v apptainer)"
