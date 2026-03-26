# Stack

## Services

| Service | Profile | GPU | URL | Purpose |
|---|---|---|---|---|
| `rag-postgres` | always | — | `localhost:5432` | Structured table storage |
| `qdrant` | always | — | `localhost:7333` | Vector store |
| `rag-pgadmin` | always | — | `localhost:5050` | Postgres UI |
| `lightonocr-vllm` | `ingest` | ~30% | `localhost:8002` | PDF / OCR parsing |
| `qwen35-vllm` | `ingest` | ~60% | `localhost:8000` | Table schema extraction |
| `qwen8b-gen-vllm` | `query` | ~85% | `localhost:8003` | Agent + SQL generation |

Postgres, Qdrant, and pgAdmin run permanently — only the vLLM servers swap between modes.

---

## First-time setup

```bash
cd docker/ingestion-stack
cp .env.example .env   # set HUGGING_FACE_HUB_TOKEN
docker compose up -d   # starts Postgres + Qdrant + pgAdmin only
```

---

## Ingestion mode — PDFs and Excel/CSV

Starts `lightonocr-vllm` (30%) + `qwen35-vllm` (60%) = ~90% VRAM total.

```bash
docker compose --profile ingest up -d
```

Run ingestion:
```bash
python -m src.ingest path/to/file.pdf
python -m src.ingest_tables path/to/file.xlsx
```

Free VRAM when done:
```bash
docker compose --profile ingest down
```

---

## Query mode — agent / chat

Starts `qwen8b-gen-vllm` (~85% VRAM).

```bash
docker compose --profile query up -d
```

Use the agent:
```bash
python -m src.rag_agent --query "your question"
```

Free VRAM when done:
```bash
docker compose --profile query down
```

---

## Switching between modes

Use `stop` (not `down`) to avoid taking Postgres and Qdrant offline during the switch.

```bash
# Ingestion → Query
docker compose stop lightonocr-vllm qwen35-vllm
docker compose up -d qwen8b-gen-vllm

# Query → Ingestion
docker compose stop qwen8b-gen-vllm
docker compose --profile ingest up -d
```

The switch takes ~30–60 seconds for the new vLLM server to load.

---

## Operating Modes (old manual approach — replaced by profiles above)

## Setup

```bash
cd /home/karvanitis/multi-modal-rag/docker/ingestion-stack
cp .env.example .env
# edit .env and set HUGGING_FACE_HUB_TOKEN
```

## Start

```bash
cd /home/karvanitis/multi-modal-rag/docker/ingestion-stack
./up.sh
```

## Run (recommended)

Use ordered startup to avoid vLLM GPU profiling race conditions:

```bash
./up.sh
```

This starts `lightonocr-vllm` first, waits for `:8002`, then starts `qwen35-vllm`.

## Run (plain compose)

```bash
docker compose up -d
```

## Check

```bash
curl -sS http://127.0.0.1:7333/collections
curl -sS http://127.0.0.1:8002/v1/models
curl -sS http://127.0.0.1:8000/v1/models
curl -sS http://127.0.0.1:8003/v1/models
```

## Stop

```bash
docker compose down
```

## Notes

- Defaults are tuned so Qwen and LightOn can run in parallel on one GPU.
- If Qwen fails with memory errors, lower `QWEN_GPU_MEMORY_UTILIZATION` or `QWEN_MAX_MODEL_LEN` in `.env`.
- This stack uses Docker named volumes:
  - `ingestion_stack_qdrant_storage`
  - `ingestion_stack_hf_cache_lightonocr`
  - `ingestion_stack_hf_cache_qwen35`

## Backup / Restore Qdrant

Backup:

```bash
docker run --rm \
  -v ingestion_stack_qdrant_storage:/data \
  -v "$(pwd)":/backup \
  alpine sh -c "cd /data && tar czf /backup/qdrant_backup.tgz ."
```

Restore:

```bash
docker compose down
docker run --rm \
  -v ingestion_stack_qdrant_storage:/data \
  -v "$(pwd)":/backup \
  alpine sh -c "cd /data && rm -rf ./* && tar xzf /backup/qdrant_backup.tgz -C /data"
```
