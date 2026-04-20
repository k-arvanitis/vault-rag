FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Install dependencies
COPY pyproject.toml ./
RUN uv sync --no-dev

# Copy application code
COPY src/ ./src/
COPY slack_app.py ./

CMD ["uv", "run", "python", "slack_app.py"]
