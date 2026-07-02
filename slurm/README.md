# Running the EU AI Act RAG on Demetra (SLURM + Apptainer)

Docker is unavailable on Demetra, so infrastructure runs as **Apptainer**
containers and Python runs via **uv**. All compute goes through SLURM — never
run these on the `demetra` login node.

## Job inventory

| Script | Resources | Purpose |
|---|---|---|
| `infra.slurm` | CPU, `Main`/`lovelace` | Qdrant + Redis + Prometheus + Grafana (Apptainer) |
| `sglang_server.slurm` | 1× A100, `lovelace` | SGLang LLM server on port 8899 |
| `ingest.slurm` | 1× A100 | One-shot download → parse → chunk → index → graph (EUR-Lex HTML) |
| `ingest_hf.slurm` | 1× A100 | Same, but corpus from the Hugging Face mirror (recommended — EUR-Lex blocks scripted downloads) |
| `api_server.slurm` | 1× A100 | FastAPI app on port 8000 |
| `single_node.slurm` | 1× A100 | **Everything on one node, no infra containers** (embedded Qdrant + memory cache) — easiest full test |

`00_common.sh` is sourced by each job (loads Apptainer, sets caches, cd's to the
project). Create the log dir once: `mkdir -p slurm/logs`.

## Orchestration order

```bash
mkdir -p slurm/logs

# 1. Long-lived services (note the node each lands on — see the *.out logs)
sbatch slurm/sglang_server.slurm      # -> SGLANG node, port 8899
sbatch slurm/infra.slurm              # -> INFRA node, ports 6333/6379/9090/3000

# 2. Point .env at those nodes, e.g.:
#      SGLANG_BASE_URL=http://<sglang-node>:8899/v1
#      QDRANT_URL=http://<infra-node>:6333
#      REDIS_URL=redis://<infra-node>:6379/0
#    (squeue -u $USER + the job .out files show the hostnames)

# 3. Ingest once the SGLang + Qdrant services report ready
sbatch slurm/ingest.slurm

# 4. Serve the API
sbatch slurm/api_server.slurm         # -> API node, port 8000

# 5. Warm cache + evaluate against the running API
RAG_API_URL=http://<api-node>:8000/answer uv run python scripts/warmup_cache.py
RAG_API_URL=http://<api-node>:8000/answer uv run python scripts/run_eval.py
```

Watch jobs with `squeue -u $USER` and tail logs with `tail -f slurm/logs/<job>_<id>.out`.

## Single-node, no-infra (one script)

`single_node.slurm` does the whole thing on one A100 with **embedded Qdrant +
in-process cache** (no Qdrant/Redis containers): SGLang → ingest → API → smoke
query → eval, all on `localhost`.

```bash
mkdir -p slurm/logs
sbatch slurm/single_node.slurm
# or interactively:
srun --partition=lovelace --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 --pty bash
bash slurm/single_node.slurm
```

Tunables (env): `MODEL` (default `Qwen/Qwen2.5-1.5B-Instruct` — small & ungated for
testing; use `MODEL=google/gemma-3-27b-it QUANT=awq` for a real run), `MEM_FRACTION`
(default 0.55, leaves GPU for embeddings+reranker), `SKIP_LIGHTRAG` (default 1 —
skips the slow graph build), `KEEP_API` (default 1 — leaves the API up for curl).

Sequencing matters: embedded Qdrant is single-writer, so the script runs ingest to
completion (releasing the file lock) **before** starting the API. To rebuild the
index, stop the API first.

## Embedded / no-infra mode

To skip the `infra.slurm` containers entirely, set in `.env`:

```bash
QDRANT_LOCAL_PATH=./data/qdrant   # embedded on-disk Qdrant, no server
# leave REDIS_URL unreachable      # cache transparently uses in-process fallback
```

This is the lowest-friction way to get a working pipeline on a single node; the
trade-off is no shared Qdrant dashboard and a per-process (non-shared) cache.
