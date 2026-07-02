#!/bin/bash
#SBATCH --job-name=sglang-eagle
#SBATCH --partition=lovelace
# NOTE: lovelace ONLY — babbage's A100 is just 40GB (and via Main, --gres=gpu:a100:1
# would land there). The full 80GB A100 lives on lovelace-01/-02. MIG slices
# (1g.20gb/1g.10gb) are avoided by requesting the typed 'a100' gres below.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/sglangeagle_%j.out
#SBATCH --error=slurm/logs/sglangeagle_%j.err
#
# SGLang + EAGLE3 speculative decoding (the guide's intended technique), via the
# official SGLang container. Key facts that make this work on this node:
#   * driver is CUDA 12.4 (r550); the cu130 images (CUDA 13) won't run, but a
#     cu129 image (CUDA 12.9, same major 12) runs via CUDA minor-version compat
#     (already proven: torch cu126 works here).
#   * SGLang's container ships flashinfer pre-compiled — no pip/nvcc needed.
#   * v0.5.6.post2 is the min version the Qwen2.5-14B EAGLE3 draft requires.
# The RAG code is unchanged: SGLang serves the OpenAI /v1 API on :8899.
#
# Tunables: TARGET, DRAFT, SGLANG_IMG, MEM_FRACTION, SPEC_STEPS/TOPK/DRAFT_TOKENS,
#           SKIP_INGEST, KEEP_API.

set -euo pipefail
for _cand in "${SLURM_SUBMIT_DIR:-}/slurm" "$(dirname "$0")" "/u/$USER/phd_projects/RAG_EU_AI/slurm"; do
  if [ -f "$_cand/00_common.sh" ]; then source "$_cand/00_common.sh"; _common_ok=1; break; fi
done
[ -n "${_common_ok:-}" ] || { echo "FATAL: cannot locate slurm/00_common.sh" >&2; exit 1; }

# --- Config ---------------------------------------------------------------
TARGET="${TARGET:-Qwen/Qwen2.5-14B-Instruct}"
DRAFT="${DRAFT:-ruipeterpan/Qwen2.5-14B-Instruct_EAGLE3_UltraChat}"
SGLANG_IMG="${SGLANG_IMG:-docker://lmsysorg/sglang:v0.5.6.post2-cu129-amd64}"
SGLANG_SIF="$BIG_STORE/apptainer/sglang_cu129.sif"
MEM_FRACTION="${MEM_FRACTION:-0.75}"      # 0.75*80GB=60GB for sglang; ~20GB left for embed+reranker
SPEC_STEPS="${SPEC_STEPS:-8}"
SPEC_TOPK="${SPEC_TOPK:-10}"
SPEC_DRAFT_TOKENS="${SPEC_DRAFT_TOKENS:-60}"
CTX_LEN="${CTX_LEN:-}"                     # cap context (e.g. 32768) — models with huge
                                          # native context (Qwen3 262k) can otherwise OOM
SKIP_INGEST="${SKIP_INGEST:-1}"           # reuse on-disk Qdrant/BM25/LightRAG indices
KEEP_API="${KEEP_API:-1}"

# Embedded Qdrant + in-process cache (same single-node, no-infra setup).
export QDRANT_LOCAL_PATH="${QDRANT_LOCAL_PATH:-$PROJECT_ROOT/data/qdrant}"
unset QDRANT_URL || true
export SGLANG_BASE_URL="http://localhost:$LLM_PORT/v1"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export SKIP_LIGHTRAG="${SKIP_LIGHTRAG:-0}"

echo "[eagle] target=$TARGET"
echo "[eagle] draft=$DRAFT"
echo "[eagle] img=$SGLANG_IMG  spec(steps=$SPEC_STEPS topk=$SPEC_TOPK draft_tokens=$SPEC_DRAFT_TOKENS)"
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader || true

# --- API-side deps (embeddings/reranker/lightrag run natively, not in the container)
uv sync --extra ml --extra data

# --- Build the SGLang SIF once -------------------------------------------
if [ ! -f "$SGLANG_SIF" ]; then
  echo "[eagle] building SGLang SIF once -> $SGLANG_SIF (one-time, slow)..."
  apptainer build "$SGLANG_SIF" "$SGLANG_IMG" || { echo "[eagle] FATAL: SIF build failed"; exit 1; }
fi

# --- Compat probe: does this cu129 image's torch see the GPU on a 12.4 driver?
# Catch a minor-version-compat failure HERE, before the long 14B + draft load.
echo "[eagle] === CUDA minor-compat probe ==="
apptainer exec --nv "$SGLANG_SIF" python3 -c \
  "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); assert torch.cuda.is_available(), 'GPU not visible — cu129 runtime rejected by the 12.4 driver'" \
  || { echo "[eagle] FATAL: CUDA not available in cu129 image (minor-compat failed). Try a cu126 image + older sglang, or vLLM AWQ."; exit 1; }

# --- Launch SGLang + EAGLE3 (background) ----------------------------------
ctx_arg=()
[ -n "$CTX_LEN" ] && ctx_arg=(--context-length "$CTX_LEN")
# Some EAGLE3 drafts (e.g. the Qwen3-30B SpecForge head) declare a tiny derived
# context (2048); SGLang aborts when target ctx > draft ctx unless this override
# is set. The draft only proposes a few tokens at the decode head, so the target
# still serves the full --context-length window. (Set as an --env above.)
echo "[eagle] === launching SGLang + EAGLE3 on :$LLM_PORT ==="
apptainer exec --nv --writable-tmpfs \
  --bind "$HF_HOME" --env "HF_HOME=$HF_HOME" \
  --env "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" \
  "$SGLANG_SIF" \
  python3 -m sglang.launch_server \
    --model-path "$TARGET" \
    --served-model-name default \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps "$SPEC_STEPS" \
    --speculative-eagle-topk "$SPEC_TOPK" \
    --speculative-num-draft-tokens "$SPEC_DRAFT_TOKENS" \
    --mem-fraction-static "$MEM_FRACTION" \
    --tp 1 "${ctx_arg[@]}" \
    --host 0.0.0.0 --port "$LLM_PORT" &
SGL_PID=$!
trap 'echo "[eagle] cleanup"; kill $SGL_PID ${API_PID:-} 2>/dev/null || true' EXIT

echo "[eagle] waiting for SGLang on :$LLM_PORT (14B + draft load + cuda graphs) ..."
sgl_ready=0
for i in $(seq 1 150); do
  if curl -sf "http://localhost:$LLM_PORT/health" >/dev/null 2>&1; then
    echo "[eagle] SGLang ready after ~$((i*10))s"; sgl_ready=1; break
  fi
  kill -0 "$SGL_PID" 2>/dev/null || { echo "[eagle] FATAL: SGLang exited early — see err log"; exit 1; }
  sleep 10
done
[ "$sgl_ready" = 1 ] || { echo "[eagle] FATAL: SGLang not ready after 1500s"; exit 1; }

echo "[eagle] === test /v1/chat/completions (EAGLE3 path) ==="
curl -s -X POST "http://localhost:$LLM_PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Reply with exactly: EAGLE_OK"}],"max_tokens":16,"temperature":0}'
echo

# --- Ingest reuse (indices already built by a prior run) ------------------
if [ "$SKIP_INGEST" = "1" ]; then
  echo "[eagle] SKIP_INGEST=1 — reusing existing on-disk indices"
  [ -d "$QDRANT_LOCAL_PATH" ] || { echo "[eagle] FATAL: SKIP_INGEST set but $QDRANT_LOCAL_PATH missing"; exit 1; }
else
  echo "[eagle] downloading + ingesting HF corpus..."
  uv run python scripts/download_hf.py
  uv run python scripts/ingest_hf.py
fi

# --- Start the RAG API (background) --------------------------------------
echo "[eagle] launching API on :8000 ..."
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!
api_ready=0
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then echo "[eagle] API ready"; api_ready=1; break; fi
  kill -0 "$API_PID" 2>/dev/null || { echo "[eagle] FATAL: API exited early"; exit 1; }
  sleep 2
done
[ "$api_ready" = 1 ] || { echo "[eagle] FATAL: API not ready after 120s"; exit 1; }

# --- Smoke + evaluation (latency with EAGLE vs the AWQ baseline) ----------
echo "[eagle] smoke query:"
curl -s -X POST http://localhost:8000/answer -H "Content-Type: application/json" \
  -d '[{"role":"user","content":"Are AI systems for emotion recognition always prohibited?"}]'
echo
# Flush the cache so the smoke/warmup answer (a near-duplicate of comp_2) cannot
# be served as a false-positive cache hit during the eval — every eval question
# must be generated fresh through retrieval+rerank.
echo "[eagle] flushing semantic cache before eval..."
curl -s -X POST http://localhost:8000/cache/invalidate; echo
echo "[eagle] running evaluation..."
RAG_API_URL=http://localhost:8000/answer uv run python scripts/run_eval.py || true

if [ "$KEEP_API" = "1" ]; then
  echo "[eagle] API stays up on $(hostname -f):8000 (scancel to stop)."
  wait "$API_PID"
else
  echo "[eagle] done; tearing down."
fi
