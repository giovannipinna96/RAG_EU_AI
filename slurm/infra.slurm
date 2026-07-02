#!/bin/bash
#SBATCH --job-name=rag-infra
#SBATCH --partition=Main,lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/infra_%j.out
#SBATCH --error=slurm/logs/infra_%j.err
#
# Starts Qdrant, Redis, Prometheus and Grafana as Apptainer containers on one
# compute node. Apptainer shares the host network namespace by default, so the
# services listen on the node's own ports (6333 / 6379 / 9090 / 3000). Note the
# node's hostname from the log — other jobs (API, ingest) point at it.
#
# Lighter alternative: skip this entirely and set QDRANT_LOCAL_PATH in .env
# (embedded Qdrant) — the cache then falls back to in-process. See CLAUDE.md.

# Locate 00_common.sh under both `sbatch` (script is relocated to a spool
# dir, so $0 is useless) and interactive `bash slurm/<x>.slurm`.
for _cand in "${SLURM_SUBMIT_DIR:-}/slurm" "$(dirname "$0")" "/u/$USER/phd_projects/RAG_EU_AI/slurm"; do
  if [ -f "$_cand/00_common.sh" ]; then source "$_cand/00_common.sh"; _common_ok=1; break; fi
done
[ -n "${_common_ok:-}" ] || { echo "FATAL: cannot locate slurm/00_common.sh" >&2; exit 1; }

echo "[infra] services will bind on host: $(hostname -f) / $(hostname -I | awk '{print $1}')"

QDRANT_IMG="docker://qdrant/qdrant:v1.14.0"
REDIS_IMG="docker://redis:7-alpine"
PROM_IMG="docker://prom/prometheus:latest"
GRAFANA_IMG="docker://grafana/grafana:latest"

# Apptainer mounts the container root READ-ONLY. Persistent data goes on host
# binds; everything else a service writes needs --writable-tmpfs (ephemeral
# overlay) or it crashes ("read-only file system").
mkdir -p "$DATA_DIR/qdrant" "$DATA_DIR/redis" "$DATA_DIR/prometheus" "$DATA_DIR/grafana"

# Qdrant — persistent storage bind + tmpfs for snapshots/runtime writes.
apptainer exec --writable-tmpfs --pwd /qdrant \
  --bind "$DATA_DIR/qdrant:/qdrant/storage" \
  "$QDRANT_IMG" ./qdrant &
QDRANT_PID=$!

# Redis — append-only persistence; bind 0.0.0.0 so other nodes (API) can reach it.
apptainer exec --bind "$DATA_DIR/redis:/data" \
  "$REDIS_IMG" redis-server --appendonly yes --dir /data \
  --bind 0.0.0.0 --protected-mode no &
REDIS_PID=$!

# Prometheus — config bind + writable tsdb bind + tmpfs for the rest.
apptainer exec --writable-tmpfs \
  --bind "$PROJECT_ROOT/infra/prometheus.yml:/etc/prometheus/prometheus.yml" \
  --bind "$DATA_DIR/prometheus:/prometheus" \
  "$PROM_IMG" \
  /bin/prometheus --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus &
PROM_PID=$!

# Grafana — writable overlay (dashboards ephemeral; fine for a test stack).
APPTAINERENV_GF_SECURITY_ADMIN_PASSWORD=admin \
apptainer exec --writable-tmpfs "$GRAFANA_IMG" /run.sh &
GRAFANA_PID=$!

echo "[infra] started pids: qdrant=$QDRANT_PID redis=$REDIS_PID prom=$PROM_PID grafana=$GRAFANA_PID"
trap 'kill $QDRANT_PID $REDIS_PID $PROM_PID $GRAFANA_PID 2>/dev/null' EXIT

# Hold the allocation for the job walltime (or until all services exit).
wait
