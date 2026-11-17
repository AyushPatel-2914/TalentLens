"""BM25 lexical index.

rank_bm25 builds the statistics (IDF per term, document lengths, average
length). Its own get_scores() loops over every document in Python for every
query word, which takes seconds for a 500-word job post. So scoring here uses an
inverted index (term -> documents that contain it, with term counts) and numpy,
which gives exactly the same scores (tests check this) in milliseconds.
That inverted index is also how search engines like Elasticsearch work.
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from talentlens.retrieval.text import tokenize


class BM25Index:
    def __init__(self, doc_ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.doc_ids = list(doc_ids)
        self.k1, self.b = k1, b
        corpus = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(corpus, k1=k1, b=b)
        self._build_postings()

    def _build_postings(self) -> None:
        lengths = np.asarray(self.bm25.doc_len, dtype="float32")
        self._norm = self.k1 * (1 - self.b + self.b * lengths / self.bm25.avgdl)  # per-document length term
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, freqs in enumerate(self.bm25.doc_freqs):
            for term, tf in freqs.items():
                docs, tfs = postings.setdefault(term, ([], []))
                docs.append(i)
                tfs.append(tf)
        self._postings = {t: (np.asarray(d, dtype="int32"), np.asarray(f, dtype="float32")) for t, (d, f) in postings.items()}

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.doc_ids), dtype="float64")
        for term, qtf in Counter(tokenize(query)).items():  # a repeated query word counts once per repeat
            if term not in self._postings:
                continue
            docs, tf = self._postings[term]
            idf = self.bm25.idf.get(term, 0.0)
            scores[docs] += qtf * idf * tf * (self.k1 + 1) / (tf + self._norm[docs])
        return scores

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        scores = self.scores(query)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.doc_ids[i], float(scores[i])) for i in top]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "BM25Index":
        # Only load index files this project created: pickle can run code from untrusted files.
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not hasattr(obj, "_postings"):  # index saved before the inverted index existed
            obj.k1, obj.b = obj.bm25.k1, obj.bm25.b
            obj._build_postings()
        return obj
