#!/bin/bash
#SBATCH --job-name=rag-llm
#SBATCH --partition=lovelace,Main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/llm_%j.out
#SBATCH --error=slurm/logs/llm_%j.err
#
# Standalone LLM server for the multi-node path (OpenAI-compatible /v1 on
# port $LLM_PORT). Other jobs reach it at http://<this-node>:$LLM_PORT/v1 — set
# SGLANG_BASE_URL in .env accordingly.
#
# Backend is vLLM (not SGLang): the cluster GPU driver caps at CUDA 12.4 and
# current SGLang images need 12.8+. vLLM v0.6.6 is the cu124-compatible build.
# MODEL CHOICE: vLLM v0.6.6 supports Qwen2.5, Llama-3.x, Gemma-2, Mistral, etc.
# It does NOT support Gemma-3 (needs a newer vLLM => newer CUDA driver). Pick a
# v0.6.6-supported model, or upgrade the node driver for newer models.

# Locate 00_common.sh under both `sbatch` (script is relocated to a spool
# dir, so $0 is useless) and interactive `bash slurm/<x>.slurm`.
for _cand in "${SLURM_SUBMIT_DIR:-}/slurm" "$(dirname "$0")" "/u/$USER/phd_projects/RAG_EU_AI/slurm"; do
  if [ -f "$_cand/00_common.sh" ]; then source "$_cand/00_common.sh"; _common_ok=1; break; fi
done
[ -n "${_common_ok:-}" ] || { echo "FATAL: cannot locate slurm/00_common.sh" >&2; exit 1; }

echo "[llm] node=$(hostname -f) ip=$(hostname -I | awk '{print $1}') -> port $LLM_PORT"

# Default to a capable, ungated, vLLM-0.6.6-supported model. Override as needed:
#   MODEL=meta-llama/Llama-3.1-8B-Instruct QUANT= sbatch slurm/sglang_server.slurm
MODEL="${MODEL:-${LLM_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}}"
QUANT="${QUANT:-}"                # e.g. "awq" for an AWQ-quantized checkpoint
MEM_FRACTION="${MEM_FRACTION:-0.85}"

quant_arg=()
[ -n "$QUANT" ] && quant_arg=(--quantization "$QUANT")

# Gated models need a token; pass it into the container if set.
token_env=()
[ -n "${HF_TOKEN:-}" ] && token_env=(--env "HF_TOKEN=$HF_TOKEN")
[ -n "${HF_TOKEN:-}" ] || echo "[llm] note: HF_TOKEN unset — gated models will 401 at download"

# Pre-build a reusable SIF once (exec-ing docker:// re-unpacks the image to NFS
# every run — slow). Subsequent runs mount the squashfs SIF read-only.
LLM_SIF="$BIG_STORE/apptainer/vllm.sif"
if [ ! -f "$LLM_SIF" ]; then
  echo "[llm] building vLLM SIF once -> $LLM_SIF ..."
  apptainer build "$LLM_SIF" "$LLM_IMG" || { echo "[llm] FATAL: SIF build failed"; exit 1; }
fi

echo "[llm] model=$MODEL quant=${QUANT:-none} mem_fraction=$MEM_FRACTION"
apptainer exec --nv --writable-tmpfs \
  --bind "$HF_HOME" --env "HF_HOME=$HF_HOME" \
  "${token_env[@]}" \
  "$LLM_SIF" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name default \
    --host 0.0.0.0 \
    --port "$LLM_PORT" \
    --dtype float16 \
    --gpu-memory-utilization "$MEM_FRACTION" \
    "${quant_arg[@]}"
