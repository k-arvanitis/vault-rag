#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[stack] starting qdrant + lightonocr-vllm"
docker compose up -d

echo "[stack] waiting for lightonocr on :8002 (model download may take a few minutes on first run)"
for i in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8002/v1/models >/dev/null 2>&1; then
    echo "[stack] lightonocr is ready"
    break
  fi
  if (( i == 120 )); then
    echo "[stack] lightonocr did not become ready; showing logs"
    docker compose logs --tail=200 lightonocr-vllm || true
    exit 1
  fi
  sleep 3
done

echo "[stack] all services ready"
echo "  - qdrant:    http://127.0.0.1:7333"
echo "  - lightonocr: http://127.0.0.1:8002"
