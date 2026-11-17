"""Dense retrieval: bi-encoder embeddings in a FAISS index, with chunking.

Resumes average ~800 words, but bge-small reads at most 512 tokens (~380
words), so a single embedding would ignore the second half of most resumes.
Each resume is split into overlapping chunks, every chunk is embedded, and a
resume's score for a query is its BEST chunk's score (max-pooling): one strongly
relevant section is enough to surface the resume.

Vectors are L2-normalised, so inner product = cosine similarity, and
IndexFlatIP does exact search. That's fine for ~10k chunks; see README
section 11 for what changes at 10M.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from talentlens.retrieval.text import chunk_words


class DenseIndex:
    def __init__(self, model_name: str, query_instruction: str = "", chunk_size: int = 300,
                 chunk_overlap: int = 50, batch_size: int = 32, model=None):
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.chunk_size, self.chunk_overlap, self.batch_size = chunk_size, chunk_overlap, batch_size
        self._model = model
        self.index: faiss.Index | None = None
        self.chunk_doc: list[str] = []  # chunk position -> doc_id

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # slow import; only when needed
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=show_progress)
        return np.asarray(vecs, dtype="float32")

    def build(self, doc_ids: list[str], texts: list[str], show_progress: bool = True) -> None:
        chunks, owners = [], []
        for doc_id, text in zip(doc_ids, texts):
            for c in chunk_words(text, self.chunk_size, self.chunk_overlap):
                chunks.append(c)
                owners.append(doc_id)
        vecs = self._embed(chunks, show_progress)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.chunk_doc = owners

    def embed_query(self, query: str) -> np.ndarray:
        # Queries are long job posts: embed the first window only (that's where the role
        # summary and requirements usually are). The instruction prefix is what bge expects.
        first = chunk_words(query, self.chunk_size, self.chunk_overlap)[0]
        return self._embed([self.query_instruction + first])

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        return self.search_vector(self.embed_query(query), k)

    def search_vector(self, qvec: np.ndarray, k: int = 100) -> list[tuple[str, float]]:
        assert self.index is not None, "build() or load() first"
        # Fetch extra chunks: several may belong to the same resume.
        n_chunks = min(self.index.ntotal, k * 8)
        scores, idx = self.index.search(qvec, n_chunks)
        best: dict[str, float] = {}
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            d = self.chunk_doc[i]
            if d not in best:  # results come sorted, so the first chunk seen is the best one
                best[d] = float(s)
        ranked = sorted(best.items(), key=lambda x: -x[1])
        return ranked[:k]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "dense.faiss"))
        meta = {"model_name": self.model_name, "query_instruction": self.query_instruction,
                "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap,
                "batch_size": self.batch_size, "chunk_doc": self.chunk_doc}
        (directory / "dense_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, model=None) -> "DenseIndex":
        meta = json.loads((directory / "dense_meta.json").read_text(encoding="utf-8"))
        obj = cls(meta["model_name"], meta["query_instruction"], meta["chunk_size"],
                  meta["chunk_overlap"], meta["batch_size"], model=model)
        obj.index = faiss.read_index(str(directory / "dense.faiss"))
        obj.chunk_doc = meta["chunk_doc"]
        return obj
