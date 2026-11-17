# TalentLens backend (FastAPI). Build: docker build -t talentlens .
# Run:  docker run -p 8000:8000 --env-file .env talentlens
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TALENTLENS_ROOT=/app \
    HF_HOME=/app/.hf \
    HF_HUB_DISABLE_XET=1 \
    PORT=8000

WORKDIR /app

# CPU-only PyTorch first: the default wheel bundles CUDA and is several GB larger.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir .

COPY configs ./configs
# data/ is not in git. Either build locally with data/ present (it is copied if it exists),
# or set TALENTLENS_ARTIFACTS_REPO + HF_TOKEN so the server downloads it at startup.
COPY data* ./data/

# Download the embedding model at build time so startup is fast.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["sh", "-c", "uvicorn talentlens.api.main:app --host 0.0.0.0 --port ${PORT}"]
