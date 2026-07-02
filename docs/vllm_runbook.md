# vLLM Runbook — PandasGI su Demetra

Guida operativa per far girare correttamente il server **vLLM** che alimenta le
componenti LLM di PandasGI (Cell Analyzer priors, estensione del catalogo,
rule-mining post-hoc). Qui sono documentati **i problemi incontrati, come sono
stati risolti, e la procedura completa per lanciarlo**.

> **Policy:** solo modelli open-source ≤ 14B parametri. Default
> `Qwen/Qwen2.5-Coder-14B-Instruct`. L'allow-list è hard-coded in
> `src/pandasgi/llm/client.py` (`MODELS`).

---

## 1. Architettura in due pezzi

vLLM gira **separato** dal lavoro di ricerca:

```
┌──────────────────────────────┐        HTTP (OpenAI API)        ┌────────────────────────────┐
│ vllm_server.slurm            │  ◀───────────────────────────▶  │ llm_extension.slurm        │
│ partizione lovelace (A100)   │   http://<NODE_IP>:8000/v1      │ partizioni CPU (turing-*)  │
│ serve il modello ≤14B        │                                 │ proponi rewrite + gate     │
└──────────────────────────────┘                                 └────────────────────────────┘
```

- Il **server** occupa una GPU A100 sulla partizione `lovelace`.
- I **consumer** (es. `llm_extension.slurm`) girano su partizioni **CPU**: non
  toccano la GPU, parlano solo via HTTP. Il modello *propone* i rewrite; la
  validazione (fidelity gate) è LLM-free.
- Il collante è la variabile **`PANDASGI_LLM_BASE_URL`** (oppure
  `LLMConfig.base_url`, default `http://127.0.0.1:8000/v1`).

File chiave:
- `scripts/slurm/vllm_server.slurm` — avvia il server
- `scripts/slurm/llm_extension.slurm` — consumer di esempio (job array)
- `src/pandasgi/llm/client.py` — client OpenAI-compatibile + allow-list

---

## 2. I due problemi risolti (e perché)

### Problema A — la slice MIG (job 58545, FAILED in 46s)

La partizione `lovelace` non espone solo A100 intere: espone anche **slice MIG**.
Verifica reale del cluster:

```
$ sinfo -o "%P %G %N"
lovelace  gpu:a100:1(S:0),gpu:1g.20gb:2(S:1),gpu:1g.10gb:3(S:1)  lovelace-[01-02]
```

Quindi `lovelace-01/02` offrono **1× A100 da 80GB** + **slice MIG da 10/20GB**.

Con il vecchio `#SBATCH --gres=gpu:1` SLURM poteva assegnare una **slice MIG
`1g.10gb`/`1g.20gb`**. Due conseguenze, entrambe fatali:

1. **Troppo piccola.** Qwen2.5-Coder-14B in fp16 pesa ~28GB → non entra in
   10/20GB.
2. **Bug di vLLM 0.8.5.** Quando SLURM passa una slice MIG, esporta
   `CUDA_VISIBLE_DEVICES=MIG-<UUID>`. vLLM 0.8.5 prova a fare `int(...)` su quella
   stringa → `ValueError: invalid literal for int() with base 10: 'MIG-...'` e il
   processo muore subito (job **58545** morto in 46s).

**Fix** (commit `96a4915`):

```diff
-#SBATCH --gres=gpu:1
+#SBATCH --gres=gpu:a100:1   # richiede la A100 INTERA (80GB), non una slice MIG
```

Più una **guardia difensiva** nello script: se per qualunque motivo SLURM
consegna ancora un device MIG-UUID, lo si re-indicizza a `0` prima di lanciare
vLLM:

```bash
case "${CUDA_VISIBLE_DEVICES:-}" in
    MIG-*) echo "[vllm] MIG UUID detected -> remapping CUDA_VISIBLE_DEVICES=0";
           export CUDA_VISIBLE_DEVICES=0 ;;
esac
```

### Problema B — mismatch torch/CUDA driver (job 61131, FAILED)

Lanciare `uv sync --extra llm` sul nodo GPU tira la torch più recente, **buildata
per una CUDA più nuova del driver di lovelace**. Il driver lovelace è **CUDA
12.4**; `uv sync` portava `torch 2.11+cu130` → incompatibile col driver → crash
(job **61131**). In più churn-a la `.venv` condivisa del progetto.

**Fix:** un **venv GPU pinnato e separato**,
`/share/malelab/PandasGI/venv-gpu`, con lo stack allineato al driver cu124.
Stato verificato del venv:

```
$ /share/malelab/PandasGI/venv-gpu/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
vllm 0.8.5
torch 2.6.0+cu124
```

Lo script usa quel venv **di default** (variabile `GPU_VENV`) e ricade su
`uv sync --extra llm` **solo** se il venv pinnato manca davvero (con warning
esplicito che quel fallback può tirare una torch incompatibile):

```bash
GPU_VENV="${GPU_VENV:-/share/malelab/PandasGI/venv-gpu}"
if [ -x "$GPU_VENV/bin/python" ]; then
    PY=("$GPU_VENV/bin/python")          # path felice
else
    uv sync --extra llm                  # fallback rischioso (cu130 vs driver cu124)
    PY=(uv run python)
fi
```

> **Regola operativa:** per la GPU **non** usare `uv sync --extra llm`. Usa
> sempre il venv pinnato `venv-gpu`.

---

## 3. Procedura completa di run

### 3.1 Prerequisiti (una tantum)

1. Repo clonato sotto `~/phd_projects/PandasGI/PandasGI` (override con
   `REPO_ROOT`).
2. Venv GPU pinnato presente in `/share/malelab/PandasGI/venv-gpu`
   (torch 2.6.0+cu124 + vllm 0.8.5). Verifica:
   ```bash
   /share/malelab/PandasGI/venv-gpu/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
   ```
3. Setup ambiente CPU (per i consumer):
   ```bash
   sbatch scripts/slurm/setup_env.slurm
   ```

### 3.2 Avvio del server vLLM

Dal **login node**:

```bash
sbatch scripts/slurm/vllm_server.slurm        # default: Qwen2.5-Coder-14B su A100
squeue -u $USER
```

Override comuni (variabili d'ambiente passate a `sbatch`):

| Variabile  | Default                              | Note |
|------------|--------------------------------------|------|
| `MODEL`    | `Qwen/Qwen2.5-Coder-14B-Instruct`    | deve stare nell'allow-list `MODELS` |
| `PORT`     | `8000`                               | porta del server |
| `GPU_VENV` | `/share/malelab/PandasGI/venv-gpu`   | interprete GPU pinnato |
| `REPO_ROOT`| `$HOME/phd_projects/PandasGI/PandasGI` | root del repo |

Esempio con un modello più piccolo:

```bash
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct sbatch scripts/slurm/vllm_server.slurm
```

### 3.3 Trovare l'indirizzo del server

Lo script stampa l'header in cima al log `slurm-pandasgi-vllm-<jobid>.out`:

```
[vllm] node     : lovelace-01 (10.x.x.x)
[vllm] model    : Qwen/Qwen2.5-Coder-14B-Instruct
[vllm] port     : 8000
[vllm] base_url : http://10.x.x.x:8000/v1
[vllm] gpus     : GPU 0: NVIDIA A100-SXM4-80GB ...
[vllm] CUDA_VISIBLE_DEVICES=0
```

Copia il `base_url`.

### 3.4 Smoke test (il server è pronto?)

vLLM impiega ~1-3 min a caricare il modello. Verifica che risponda:

```bash
# da un nodo che vede la rete del cluster
curl http://<NODE_IP>:8000/v1/models
curl http://<NODE_IP>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-Coder-14B-Instruct",
       "messages":[{"role":"user","content":"ping"}],
       "max_tokens":5}'
```

### 3.5 Lanciare i consumer

```bash
PANDASGI_LLM_BASE_URL="http://<NODE_IP>:8000/v1" \
  sbatch scripts/slurm/llm_extension.slurm        # median band, array 0-59
# corpus intero:
PANDASGI_LLM_BASE_URL="http://<NODE_IP>:8000/v1" \
  MANIFEST=/share/malelab/PandasGI/corpus/manifest.txt \
  sbatch --array=0-91%10 scripts/slurm/llm_extension.slurm
```

Il consumer fallisce subito se `PANDASGI_LLM_BASE_URL` non è settata
(`: "${PANDASGI_LLM_BASE_URL:?...}"`).

### 3.6 Da codice Python

```python
from pandasgi.llm.client import LLMConfig, LLMClient

cfg = LLMConfig(base_url="http://<NODE_IP>:8000/v1",
                model="Qwen/Qwen2.5-Coder-14B-Instruct")
client = LLMClient(cfg)
print(client.chat("You are a pandas expert.", "Rewrite df.apply(...) vectorized."))
```

`api_key` è `"EMPTY"` (vLLM accetta qualunque chiave non vuota). Il client usa
l'SDK `openai` se disponibile, altrimenti fa fallback su `urllib`.

---

## 4. Parametri vLLM usati e perché

Comando di avvio effettivo:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --dtype auto \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192
```

| Flag | Valore | Motivo |
|------|--------|--------|
| `--host 0.0.0.0` | — | il consumer gira su un altro nodo; serve raggiungibilità sulla rete del cluster |
| `--dtype auto`   | fp16/bf16 | il 14B sta in fp16 (~28GB) dentro la A100 80GB |
| `--gpu-memory-utilization 0.85` | 85% | lascia headroom alla KV-cache senza OOM |
| `--max-model-len 8192` | 8192 token | sufficiente per i prompt di rewrite; tiene la KV-cache contenuta |

---

## 5. Troubleshooting

| Sintomo | Causa | Rimedio |
|---------|-------|---------|
| Job muore in ~45s con `int('MIG-...')` ValueError | assegnata slice MIG | usa `--gres=gpu:a100:1` (già nello script); la guardia rimappa a 0 |
| OOM al caricamento del modello | slice troppo piccola o `--gpu-memory-utilization` troppo alto | assicurati A100 intera; abbassa util o usa modello 7B |
| Crash torch/CUDA all'import (`libcuda` / cu130 vs driver) | usato `uv sync --extra llm` invece del venv pinnato | usa `GPU_VENV=/share/malelab/PandasGI/venv-gpu` |
| Consumer esce subito | `PANDASGI_LLM_BASE_URL` non settata | esporta il base_url letto dal log del server |
| `curl /v1/models` rifiuta connessione | server ancora in caricamento, o porta/IP sbagliati | aspetta 1-3 min; ricontrolla l'header `[vllm]` nel log |
| `ValueError: model ... not in allow-list` | modello fuori dalla allow-list ≤14B | usa uno dei `MODELS` in `client.py` |

---

## 6. Riferimenti

- Server: `scripts/slurm/vllm_server.slurm`
- Consumer: `scripts/slurm/llm_extension.slurm`, `scripts/llm/run_catalog_extension.py`
- Client: `src/pandasgi/llm/client.py`
- Indice job SLURM: `scripts/slurm/README.md`
- Commit del fix MIG: `96a4915` *fix(vllm): request full A100 (gpu:a100:1) not a MIG slice*
- Driver lovelace: CUDA 12.4 → stack pinnato torch 2.6.0+cu124 / vllm 0.8.5
