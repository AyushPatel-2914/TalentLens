# TalentLens backend (FastAPI), built from the local source.
#
#   docker build -t talentlens .
#   docker run -p 7860:7860 --env-file .env -v "%cd%\data:/home/user/app/data" talentlens     (Windows cmd)
#   docker run -p 7860:7860 --env-file .env -v "$(pwd)/data:/home/user/app/data" talentlens   (bash)
#
# data/ is never baked into the image (it contains resume text). Mount it as above, or set
# TALENTLENS_ARTIFACTS_REPO + HF_TOKEN so the container downloads it from a private HF dataset.
# deploy/hf-space/Dockerfile is the same image built from the GitHub repo (for Hugging Face Spaces).
FROM python:3.12-slim

# Hugging Face Spaces run containers as uid 1000; use the same user everywhere.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    TALENTLENS_ROOT=/home/user/app \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860
USER user
WORKDIR /home/user/app

# CPU-only PyTorch first: the default wheel bundles CUDA and is several GB larger.
COPY --chown=user pyproject.toml ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user .
COPY --chown=user configs ./configs
COPY --chown=user frontend/lib/samples.json ./frontend/lib/samples.json

# Bake both models into the image so startup doesn't depend on (slow) model downloads.
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-small-en-v1.5'); \
from transformers import AutoTokenizer as T, AutoModelForSequenceClassification as M; \
T.from_pretrained('BAAI/bge-reranker-base'); M.from_pretrained('BAAI/bge-reranker-base')"

RUN mkdir -p data results
EXPOSE 7860
CMD ["sh", "-c", "python -m uvicorn talentlens.api.main:app --host 0.0.0.0 --port ${PORT}"]
