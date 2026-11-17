"""Hybrid retrieval with Reciprocal Rank Fusion (RRF).

    RRF(d) = sum over methods m of 1 / (k + rank_m(d)),   rank starts at 1

Raw BM25 scores (0..50+) and cosine similarities (0..1) are on different scales,
so adding them is meaningless. RRF uses only ranks. A document missing from a
method's list gets nothing from that method.

k controls how much the top positions dominate. Small k: rank 1 is worth far
more than rank 5, so one method's favourite wins. Large k (60 is the usual
default): ranks 1 and 5 are worth almost the same, so being found by BOTH
methods matters more.
"""

from __future__ import annotations

from collections.abc import Sequence


def rrf(rankings: Sequence[Sequence[str]], k: float = 60) -> list[tuple[str, float]]:
    """Fuse ranked lists of doc IDs. Ties are broken by first appearance, so output is deterministic."""
    if k < 0:
        raise ValueError("k must be >= 0")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = order
                order += 1
    return sorted(scores.items(), key=lambda x: (-x[1], first_seen[x[0]]))


class HybridRetriever:
    def __init__(self, bm25, dense, rrf_k: float = 60, fusion_depth: int = 200):
        self.bm25, self.dense = bm25, dense
        self.rrf_k, self.fusion_depth = rrf_k, fusion_depth

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        lexical = [d for d, _ in self.bm25.search(query, self.fusion_depth)]
        semantic = [d for d, _ in self.dense.search(query, self.fusion_depth)]
        return rrf([lexical, semantic], self.rrf_k)[:k]
