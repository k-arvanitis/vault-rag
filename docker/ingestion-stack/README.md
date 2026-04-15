# Ingestion Stack

Two services, always running together:

| Service | GPU | URL | Purpose |
|---|---|---|---|
| `qdrant` | — | `localhost:7333` | Vector store (persistent, CPU only) |
| `lightonocr-vllm` | ~30% | `localhost:8002` | LightOn OCR — PDF parsing via vLLM |

Both start with a single command. There are no profiles or mode switches.

---

## Setup

```bash
cd docker/ingestion-stack
cp .env.example .env
# Set HUGGING_FACE_HUB_TOKEN in .env
```

## Start

```bash
./up.sh
```

Or directly:

```bash
docker compose up -d
```

## Check

```bash
curl -sS http://127.0.0.1:7333/collections
curl -sS http://127.0.0.1:8002/v1/models
```

## Stop

```bash
docker compose stop
```

> **Never use `docker compose down`** — it removes the Qdrant volume and deletes all ingested data. Use `stop` to preserve data.

---

## Backup / Restore Qdrant

**Backup:**
```bash
docker run --rm \
  -v ingestion_stack_qdrant_storage:/data \
  -v "$(pwd)":/backup \
  alpine sh -c "cd /data && tar czf /backup/qdrant_backup.tgz ."
```

**Restore:**
```bash
docker compose stop
docker run --rm \
  -v ingestion_stack_qdrant_storage:/data \
  -v "$(pwd)":/backup \
  alpine sh -c "cd /data && rm -rf ./* && tar xzf /backup/qdrant_backup.tgz"
docker compose start
```
