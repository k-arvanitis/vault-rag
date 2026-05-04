ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: up docker-up docker-up-gpu docker-down app slack docker-slack-build docker-slack-up litellm eval eval-cross test lint

# Full stack in Docker (Qdrant + LiteLLM + Ollama + Streamlit app)
docker-up:
	docker compose up -d --build

# Same stack + GPU-backed LightOn OCR (requires NVIDIA container runtime)
docker-up-gpu:
	docker compose --profile gpu up -d --build

docker-down:
	docker compose down

# Ingestion-stack only (Qdrant + LiteLLM + OCR — no Streamlit container)
up:
	docker compose -f docker/ingestion-stack/docker-compose.yaml up -d

app:
	uv run streamlit run app.py

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

eval:
	uv run python eval/run_eval.py

eval-cross:
	uv run python eval/run_eval.py --category cross_document

test:
	uv run pytest tests/ -v --tb=short

lint:
	uv run ruff check src/ app.py slack_app.py
