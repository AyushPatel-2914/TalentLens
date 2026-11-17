"""Build or load the three retrievers from config."""

from __future__ import annotations

from talentlens.config import resolve
from talentlens.retrieval.bm25 import BM25Index
from talentlens.retrieval.dense import DenseIndex
from talentlens.retrieval.hybrid import HybridRetriever


def index_dir(cfg: dict, corpus: str = "resumes"):
    return resolve(cfg["retrieval"]["index_dir"]) / corpus


def new_dense(cfg: dict) -> DenseIndex:
    d = cfg["retrieval"]["dense"]
    return DenseIndex(d["model"], d["query_instruction"], d["chunk_words"], d["chunk_overlap"], d["batch_size"])


def load_retrievers(cfg: dict, corpus: str = "resumes") -> dict:
    directory = index_dir(cfg, corpus)
    if not (directory / "bm25.pkl").exists():
        raise FileNotFoundError(f"No indexes in {directory}. Run: python scripts/04_build_indexes.py")
    bm25 = BM25Index.load(directory / "bm25.pkl")
    dense = DenseIndex.load(directory)
    r = cfg["retrieval"]
    return {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever(bm25, dense, r["rrf_k"], r["fusion_depth"])}
