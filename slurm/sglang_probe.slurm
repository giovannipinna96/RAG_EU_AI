#!/bin/bash
#SBATCH --job-name=sglang-probe
#SBATCH --partition=lovelace,Main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/sglangprobe_%j.out
#SBATCH --error=slurm/logs/sglangprobe_%j.err
#
# FEASIBILITY PROBE — can we run SGLang natively (via uv, no container) on this
# CUDA 12.4 / driver 550 node, and does its EAGLE speculative path work?
#
# Strategy: the prebuilt SGLang *container* ships CUDA 12.5-12.8 (> our 12.4
# driver), but SGLang itself supports CUDA 12.4 when installed from cu124 wheels.
# We install into the existing GPU venv (already has the cu124 torch stack from
# the vLLM runs) and serve the AWQ model. flashinfer (the [all] extra) is NOT a
# normal PyPI wheel, so we install core `sglang` and use the Triton attention
# backend instead — enough to prove serving + (optionally) EAGLE.
#
# Tunables: MODEL, MEM_FRACTION, TRY_EAGLE, EAGLE_DRAFT, ATTN_BACKEND.

set -euo pipefail
for _cand in "${SLURM_SUBMIT_DIR:-}/slurm" "$(dirname "$0")" "/u/$USER/phd_projects/RAG_EU_AI/slurm"; do
  if [ -f "$_cand/00_common.sh" ]; then source "$_cand/00_common.sh"; _common_ok=1; break; fi
done
[ -n "${_common_ok:-}" ] || { echo "FATAL: cannot locate slurm/00_common.sh" >&2; exit 1; }

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
MEM_FRACTION="${MEM_FRACTION:-0.70}"
ATTN_BACKEND="${ATTN_BACKEND:-triton}"     # triton avoids the flashinfer wheel
TRY_EAGLE="${TRY_EAGLE:-0}"                # 1 = add --speculative-algorithm EAGLE
EAGLE_DRAFT="${EAGLE_DRAFT:-}"             # EAGLE draft head path (model-specific)

echo "[probe] node=$(hostname) model=$MODEL attn=$ATTN_BACKEND try_eagle=$TRY_EAGLE"
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader || true

# --- Step 1: install SGLang into the GPU venv via uv sync (declared extra, so
# `uv run` won't wipe it the way it does a bare `uv pip install`) --------------
echo "[probe] === uv sync --extra ml --extra gpu-native (core sglang, no flashinfer) ==="
if ! uv sync --extra ml --extra gpu-native; then
  echo "[probe] FATAL: uv sync failed to resolve sglang — see resolver error above."
  echo "[probe] Likely a torch-version clash with the ml stack, or sgl-kernel"
  echo "[probe] has no cu124 wheel. Next step: pin a specific sglang/torch pair."
  exit 1
fi

echo "[probe] === import check ==="
uv run python - <<'PY' || { echo "[probe] FATAL: sglang import failed"; exit 1; }
import torch, sglang
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("sglang", getattr(sglang, "__version__", "?"))
assert torch.cuda.is_available(), "torch cannot see the GPU"
PY
echo "[probe] install + import OK"

# --- Step 2: launch the SGLang OpenAI-compatible server ----------------------
spec_arg=()
if [ "$TRY_EAGLE" = "1" ]; then
  spec_arg=(--speculative-algorithm EAGLE)
  [ -n "$EAGLE_DRAFT" ] && spec_arg+=(--speculative-draft-model-path "$EAGLE_DRAFT")
  echo "[probe] EAGLE speculative ON (draft='${EAGLE_DRAFT:-builtin}')"
fi

echo "[probe] === launching sglang.launch_server on :$LLM_PORT ==="
uv run python -m sglang.launch_server \
  --model-path "$MODEL" \
  --quantization awq \
  --host 0.0.0.0 --port "$LLM_PORT" \
  --mem-fraction-static "$MEM_FRACTION" \
  --attention-backend "$ATTN_BACKEND" \
  "${spec_arg[@]}" &
SGL_PID=$!
trap 'echo "[probe] cleanup"; kill $SGL_PID 2>/dev/null || true' EXIT

echo "[probe] waiting for SGLang health on :$LLM_PORT ..."
ready=0
for i in $(seq 1 90); do
  if curl -sf "http://localhost:$LLM_PORT/health" >/dev/null 2>&1; then
    echo "[probe] SGLang ready after ~$((i*10))s"; ready=1; break
  fi
  kill -0 "$SGL_PID" 2>/dev/null || { echo "[probe] FATAL: SGLang process exited early — see err log"; exit 1; }
  sleep 10
done
[ "$ready" = 1 ] || { echo "[probe] FATAL: SGLang not ready after 900s"; exit 1; }

# --- Step 3: one real completion through the OpenAI /v1 API -------------------
echo "[probe] === test /v1/chat/completions ==="
curl -s -X POST "http://localhost:$LLM_PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Reply with exactly: SGLANG_OK"}],"max_tokens":16,"temperature":0}'
echo
echo "[probe] DONE — if you see a completion above, native SGLang serves on CUDA 12.4."
