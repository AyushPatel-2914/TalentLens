"""Cross-encoder re-ranker (BAAI/bge-reranker-base or the fine-tuned version).

A cross-encoder reads the job and the resume TOGETHER in one input, so attention
can compare every job word with every resume word. That is far more accurate
than comparing two separate embeddings, but needs one model run per pair, so it
only re-orders the top 100 from first-stage retrieval.

The model reads at most 512 tokens for job + resume combined. `pair_texts`
decides what goes in: the job title plus the start of the posting, and the
start of the resume. The SAME function builds the training data, so training
and inference see identical inputs.

Uses transformers directly (not a wrapper) so the raw score is always a logit;
`sigmoid(logit)` gives a 0..1 score for the final match formula.
"""

from __future__ import annotations

import time

import numpy as np


def first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def pair_texts(job_title: str, job_text: str, resume_text: str, job_words: int = 150,
               resume_words: int = 350) -> tuple[str, str]:
    return f"{job_title}. {first_words(job_text, job_words)}", first_words(resume_text, resume_words)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Reranker:
    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 16, device: str | None = None):
        import os

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not os.environ.get("HF_TOKEN"):  # the fine-tuned model is a private Hub repo: pick up the token from .env
            from dotenv import load_dotenv

            from talentlens.config import ROOT
            load_dotenv(ROOT / ".env")
        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.max_length, self.batch_size = max_length, batch_size

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        """Raw logits, one per doc (higher = better fit)."""
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(docs), self.batch_size):
                batch = docs[i : i + self.batch_size]
                enc = self.tokenizer([query] * len(batch), batch, truncation="longest_first",
                                     max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                out.append(self.model(**enc).logits.view(-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype="float32")

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Raw logits for arbitrary (query, doc) pairs, batched. Used for resume -> jobs, where
        every pair has a different job as the query."""
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                enc = self.tokenizer([q for q, _ in batch], [d for _, d in batch], truncation="longest_first",
                                     max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                out.append(self.model(**enc).logits.view(-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype="float32")

    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """candidates: [(doc_id, doc_text)] -> [(doc_id, logit)] sorted best first."""
        if not candidates:
            return []
        scores = self.score(query, [t for _, t in candidates])
        order = np.argsort(-scores, kind="stable")
        return [(candidates[i][0], float(scores[i])) for i in order]


class TimedReranker:
    """Small helper for latency reporting in evaluation."""

    def __init__(self, reranker: Reranker):
        self.reranker, self.seconds = reranker, 0.0

    def rerank(self, query, candidates):
        t0 = time.perf_counter()
        try:
            return self.reranker.rerank(query, candidates)
        finally:
            self.seconds += time.perf_counter() - t0
