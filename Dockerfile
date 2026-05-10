FROM python:3.11-slim

WORKDIR /app

# pypdfium2 needs libgl1; fastembed needs libgomp1; unstructured CPU fallback needs tesseract + poppler
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        tesseract-ocr \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Dependency layer first — Docker cache survives code-only changes
COPY pyproject.toml uv.lock ./
# NOTE: image is ~5 GB; torch wheel includes CUDA libs even when RERANKER_DEVICE=cpu
RUN uv sync --no-dev --frozen

# Pre-download reranker so the first query doesn't pay a 30 s HuggingFace download
RUN uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Application code
COPY src/ ./src/
COPY app.py slack_app.py main.py litellm_config.yaml ./
COPY eval/ ./eval/

RUN mkdir -p data/input data/output/processed data/output/lightonocr \
             data/output/pymupdf data/output/chunks eval/results

# Non-root runtime user — owns /app so writes to data/ and eval/results/ work
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
