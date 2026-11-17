# Hugging Face Docker Space for the TalentLens backend.
# Builds from the GitHub repo, so the Space only needs this file + README.md.
# "Factory rebuild" in the Space settings pulls the latest code from GitHub.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    TALENTLENS_ROOT=/home/user/app \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TALENTLENS_PUBLIC_DEMO=1 \
    PORT=7860
USER user
WORKDIR /home/user/app

ARG REPO_URL=https://github.com/AyushPatel-2914/TalentLens.git
ARG REF=main
RUN git clone --depth 1 --branch ${REF} ${REPO_URL} . \
 && pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user .

RUN python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-small-en-v1.5'); \
from transformers import AutoTokenizer as T, AutoModelForSequenceClassification as M; \
T.from_pretrained('BAAI/bge-reranker-base'); M.from_pretrained('BAAI/bge-reranker-base')"

RUN mkdir -p data results
EXPOSE 7860
# On startup the API downloads data/indexes/caches from the private dataset in TALENTLENS_ARTIFACTS_REPO.
CMD ["sh", "-c", "python -m uvicorn talentlens.api.main:app --host 0.0.0.0 --port ${PORT}"]
