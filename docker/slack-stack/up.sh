#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  echo "[slack-stack] .env not found — copy .env.example and fill in your tokens"
  echo "  cp docker/slack-stack/.env.example docker/slack-stack/.env"
  exit 1
fi

echo "[slack-stack] building and starting vault-rag-slack"
docker compose up -d --build
echo "[slack-stack] bot is running — check logs with: docker logs -f vault-rag-slack"
