ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: up docker-up docker-up-gpu docker-down api ui slack docker-slack-build docker-slack-up litellm seed eval eval-cross test lint

# Full stack in Docker: Qdrant + Redis + LiteLLM + Ollama + API + UI
docker-up:
	docker compose up -d --build

# Same stack + GPU-backed LightOn OCR (requires NVIDIA container runtime)
docker-up-gpu:
	docker compose --profile gpu up -d --build

docker-down:
	docker compose down

# Ingestion-stack only (Qdrant + LiteLLM + OCR)
up:
	docker compose -f docker/ingestion-stack/docker-compose.yaml up -d

api:
	uv run uvicorn api:app --host :: --port 8001 --reload

ui:
	cd frontend && npm run dev

slack:
	uv run python slack_app.py

docker-slack-build:
	docker build -t vault-rag-slack -f docker/slack-stack/Dockerfile .

docker-slack-up:
	cd docker/slack-stack && ./up.sh

# Start the LiteLLM proxy (Groq primary → OpenRouter fallback)
# Requires: pip install 'litellm[proxy]'
# Reads API keys from .env via litellm_config.yaml (os.environ/ references)
litellm:
	DEBUG= litellm --config litellm_config.yaml --port 4000

seed:
	uv run python scripts/seed.py

eval:
	uv run python eval/run_eval.py

eval-cross:
	uv run python eval/run_eval.py --category cross_document_compare

test:
	uv run pytest tests/ -v --tb=short

# Real end-to-end check against live Qdrant/DuckDB/LLM -- not mocked, not part
# of `make test`. Run before a release or after touching ingestion/retrieval/
# generation code.
smoke:
	uv run python scripts/smoke_test.py

lint:
	uv run ruff check src/ api.py slack_app.py eval/
	uv run ruff format --check src/ api.py slack_app.py eval/
