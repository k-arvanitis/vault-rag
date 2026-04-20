.PHONY: slack docker-slack-build docker-slack-up test lint

slack:
	uv run python slack_app.py

docker-slack-build:
	docker build -t vault-rag-slack -f docker/slack-stack/Dockerfile .

docker-slack-up:
	cd docker/slack-stack && ./up.sh

test:
	uv run pytest tests/ -v --tb=short

lint:
	uv run ruff check src/ app.py slack_app.py
