FROM python:3.11-slim

WORKDIR /app

# poppler-utils + tesseract-ocr: unstructured's CPU OCR fallback (PDF_PARSER=cpu).
# libmagic1: file-type sniffing used by unstructured.
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils tesseract-ocr libmagic1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

# Pre-download the cross-encoder reranker weights at build time so the first
# query doesn't pay the ~30s cold-start download (see README > Known limitations).
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN uv run python -c "\
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
AutoModelForSequenceClassification.from_pretrained('${RERANKER_MODEL}'); \
AutoTokenizer.from_pretrained('${RERANKER_MODEL}')"

COPY src/ ./src/
COPY eval/ ./eval/
COPY api.py ./

EXPOSE 8001

CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
