#!/bin/bash
#SBATCH --job-name=rag-single
#SBATCH --partition=lovelace,Main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/single_%j.out
#SBATCH --error=slurm/logs/single_%j.err
#
# End-to-end run on ONE GPU node, no infrastructure containers:
#   SGLang (Apptainer) + embedded Qdrant + in-process cache + FastAPI, then
#   download -> ingest -> warmup -> eval, all on localhost.
#
# Run it two ways:
#   sbatch slurm/single_node.slurm
#   # or interactively:
#   srun --partition=lovelace --gres=gpu:1 --cpus-per-task=8 --mem=64G \
#        --time=04:00:00 --pty bash
#   bash slurm/single_node.slurm
#
# Tunables (env): MODEL, MEM_FRACTION, SKIP_LIGHTRAG, QUANT, KEEP_API, SKIP_INGEST.

set -euo pipefail
# Locate 00_common.sh under both `sbatch` (script is relocated to a spool
# dir, so $0 is useless) and interactive `bash slurm/<x>.slurm`.
for _cand in "${SLURM_SUBMIT_DIR:-}/slurm" "$(dirname "$0")" "/u/$USER/phd_projects/RAG_EU_AI/slurm"; do
  if [ -f "$_cand/00_common.sh" ]; then source "$_cand/00_common.sh"; _common_ok=1; break; fi
done
[ -n "${_common_ok:-}" ] || { echo "FATAL: cannot locate slurm/00_common.sh" >&2; exit 1; }

# --- Single-node configuration (mode-specific; LLM_IMG/PORT come from common) ---
# Embedded Qdrant (file-locked, single process) + in-process cache fallback —
# this is what makes "single node, no infra containers" work.
export QDRANT_LOCAL_PATH="${QDRANT_LOCAL_PATH:-$PROJECT_ROOT/data/qdrant}"
unset QDRANT_URL || true
export SGLANG_BASE_URL="http://localhost:$LLM_PORT/v1"   # OpenAI /v1 (vLLM here)
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"   # unreachable -> memory cache

# Small, ungated model so the test runs without HF gating/licenses. Swap to
# google/gemma-3-27b-it (QUANT=awq) for a competition-grade run.
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MEM_FRACTION="${MEM_FRACTION:-0.55}"   # leave GPU room for embeddings + reranker
SKIP_LIGHTRAG="${SKIP_LIGHTRAG:-1}"    # graph build is slow; off by default here
KEEP_API="${KEEP_API:-1}"              # keep API running after eval for manual curl
QUANT="${QUANT:-}"                     # e.g. "awq" for quantized large models
SKIP_INGEST="${SKIP_INGEST:-0}"        # reuse on-disk Qdrant/BM25/LightRAG indices
# --- Inference speed knobs (vLLM) ---
# Speculative decoding: a small same-family draft predicts tokens the AWQ target
# verifies in one pass (the runnable vLLM equivalent of the guide's SGLang+EAGLE
# path, which SGLang's CUDA-12.8 image can't run here). Set SPEC_MODEL="" to disable,
# or SPEC_MODEL="[ngram]" for draft-free n-gram speculation (good for grounded RAG).
SPEC_MODEL="${SPEC_MODEL:-}"           # e.g. Qwen/Qwen2.5-0.5B-Instruct
SPEC_TOKENS="${SPEC_TOKENS:-5}"        # lookahead tokens per step
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}" # cap context; frees KV cache (we need <8k)
PREFIX_CACHE="${PREFIX_CACHE:-0}"      # shared system-prompt prefix reuse (off w/ spec)
export SKIP_LIGHTRAG

echo "[single] model=$MODEL mem_fraction=$MEM_FRACTION skip_lightrag=$SKIP_LIGHTRAG"
echo "[single] quant='${QUANT}' spec_model='${SPEC_MODEL}' spec_tokens=$SPEC_TOKENS max_len=$MAX_MODEL_LEN"
echo "[single] qdrant_local=$QDRANT_LOCAL_PATH"

# --- Dependencies ---------------------------------------------------------
uv sync --extra ml --extra data

# --- Start the vLLM LLM server (background) -------------------------------
quant_arg=()
[ -n "$QUANT" ] && quant_arg=(--quantization "$QUANT")

# Speculative decoding (optional): draft model or "[ngram]".
# NOTE: a draft model must share the target's vocab size. Qwen2.5-0.5B (vocab
# 151936) is NOT valid for Qwen2.5-7B (vocab 152064) — vLLM aborts in _vocab_size.
# "[ngram]" needs no draft and suits grounded RAG (answers quote the provisions).
spec_arg=()
if [ "$SPEC_MODEL" = "[ngram]" ]; then
  spec_arg=(
    --speculative-model "[ngram]"
    --num-speculative-tokens "$SPEC_TOKENS"
    --ngram-prompt-lookup-max "${NGRAM_MAX:-4}"
    --ngram-prompt-lookup-min "${NGRAM_MIN:-1}"
  )
  echo "[single] speculative decoding ON: n-gram (k=$SPEC_TOKENS, lookup=${NGRAM_MIN:-1}..${NGRAM_MAX:-4})"
elif [ -n "$SPEC_MODEL" ]; then
  spec_arg=(--speculative-model "$SPEC_MODEL" --num-speculative-tokens "$SPEC_TOKENS")
  echo "[single] speculative decoding ON: draft=$SPEC_MODEL k=$SPEC_TOKENS"
fi

# Throughput/KV knobs.
perf_arg=(--max-model-len "$MAX_MODEL_LEN")
[ "$PREFIX_CACHE" = "1" ] && perf_arg+=(--enable-prefix-caching)

# Pre-build a reusable SIF. `apptainer exec docker://...` re-unpacks the multi-GB
# image into an NFS sandbox on EVERY run (slow, >20 min, floods xattr warnings).
# Building a squashfs .sif once and exec-ing that mounts read-only — fast startup.
LLM_SIF="$BIG_STORE/apptainer/vllm.sif"
if [ ! -f "$LLM_SIF" ]; then
  echo "[single] building vLLM SIF once -> $LLM_SIF (one-time, slow)..."
  apptainer build "$LLM_SIF" "$LLM_IMG" || { echo "[single] FATAL: SIF build failed"; exit 1; }
fi

echo "[single] launching vLLM ($LLM_SIF, port $LLM_PORT)..."
apptainer exec --nv --writable-tmpfs \
  --bind "$HF_HOME" --env "HF_HOME=$HF_HOME" \
  "$LLM_SIF" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name default \
    --host 0.0.0.0 --port "$LLM_PORT" \
    --dtype float16 \
    --gpu-memory-utilization "$MEM_FRACTION" \
    "${quant_arg[@]}" "${spec_arg[@]}" "${perf_arg[@]}" &
LLM_PID=$!
trap 'echo "[single] cleanup"; kill $LLM_PID ${API_PID:-} 2>/dev/null || true' EXIT

echo "[single] waiting for vLLM on :$LLM_PORT (model load) ..."
llm_ready=0
for i in $(seq 1 120); do
  if curl -sf "http://localhost:$LLM_PORT/health" >/dev/null 2>&1; then
    echo "[single] vLLM ready after ~${i}0s"; llm_ready=1; break
  fi
  kill -0 "$LLM_PID" 2>/dev/null || { echo "[single] FATAL: vLLM process exited early — see err log"; exit 1; }
  sleep 10
done
# C-1: fail loudly instead of ingesting against a dead LLM.
[ "$llm_ready" = 1 ] || { echo "[single] FATAL: vLLM not ready after 1200s; aborting"; exit 1; }

# --- Ingest (must finish + release the embedded Qdrant lock before the API) -
# SKIP_INGEST=1 reuses the persisted indices (data/qdrant, bm25_index/,
# lightrag_data/) from a previous run — e.g. to re-test a generation/API change
# without paying for re-embedding + the LightRAG graph rebuild.
if [ "$SKIP_INGEST" = "1" ]; then
  echo "[single] SKIP_INGEST=1 — reusing existing on-disk indices, no re-ingest"
  [ -d "$QDRANT_LOCAL_PATH" ] || { echo "[single] FATAL: SKIP_INGEST set but $QDRANT_LOCAL_PATH missing"; exit 1; }
else
  echo "[single] downloading + ingesting HF corpus..."
  uv run python scripts/download_hf.py
  uv run python scripts/ingest_hf.py
  echo "[single] ingest done (Qdrant lock released)"
fi

# --- Start API (background) ----------------------------------------------
echo "[single] launching API on :8000 ..."
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!
api_ready=0
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "[single] API ready"; api_ready=1; break
  fi
  kill -0 "$API_PID" 2>/dev/null || { echo "[single] FATAL: API process exited early"; exit 1; }
  sleep 2
done
[ "$api_ready" = 1 ] || { echo "[single] FATAL: API not ready after 120s; aborting"; exit 1; }

# --- Smoke query + evaluation --------------------------------------------
echo "[single] smoke query:"
curl -s -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '[{"role":"user","content":"Are AI systems for emotion recognition always prohibited?"}]'
echo

echo "[single] running evaluation..."
RAG_API_URL=http://localhost:8000/answer uv run python scripts/run_eval.py || true

if [ "$KEEP_API" = "1" ]; then
  echo "[single] API stays up on $(hostname -f):8000 (Ctrl-C / scancel to stop). Try:"
  echo "  curl -X POST http://localhost:8000/answer -H 'Content-Type: application/json' \\"
  echo "    -d '[{\"role\":\"user\",\"content\":\"What does Annex III list?\"}]'"
  wait "$API_PID"
else
  echo "[single] done; tearing down."
fi
