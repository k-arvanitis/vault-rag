#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[stack] starting qdrant + lightonocr-vllm"
docker compose up -d qdrant lightonocr-vllm

echo "[stack] waiting for lightonocr on :8002"
for i in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:8002/v1/models >/dev/null 2>&1; then
    echo "[stack] lightonocr is ready"
    break
  fi
  if (( i == 90 )); then
    echo "[stack] lightonocr did not become ready; showing logs"
    docker compose logs --tail=200 lightonocr-vllm || true
    exit 1
  fi
  sleep 3
done

echo "[stack] starting qwen35-vllm"
docker compose up -d qwen35-vllm

echo "[stack] waiting for qwen on :8000"
for i in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "[stack] qwen is ready"
    break
  fi
  if (( i == 120 )); then
    echo "[stack] qwen did not become ready; showing logs"
    docker compose logs --tail=200 qwen35-vllm || true
    exit 1
  fi
  sleep 3
done

echo "[stack] all services ready"
echo "  - qdrant: http://127.0.0.1:7333"
echo "  - lighton: http://127.0.0.1:8002"
echo "  - qwen:    http://127.0.0.1:8000"
