"""Ranking metrics: Recall@k, MRR, nDCG@k.

`ranked` is the system's list of doc IDs (best first). `labels` maps doc_id ->
graded label (0 no fit, 1 partial, 2 strong). Unlabelled docs count as 0.
"Relevant" for Recall and MRR means label >= min_relevant (default 1).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(ranked: Sequence[str], labels: Mapping[str, int], k: int, min_relevant: int = 1) -> float | None:
    relevant = {d for d, l in labels.items() if l >= min_relevant}
    if not relevant:
        return None  # undefined for a query with no relevant docs; skipped when averaging
    return len(relevant & set(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], labels: Mapping[str, int], min_relevant: int = 1) -> float:
    for i, d in enumerate(ranked, start=1):
        if labels.get(d, 0) >= min_relevant:
            return 1.0 / i
    return 0.0


def dcg(gains: Sequence[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(ranked: Sequence[str], labels: Mapping[str, int], k: int) -> float | None:
    ideal = dcg(sorted(labels.values(), reverse=True)[:k])
    if ideal == 0:
        return None
    return dcg([labels.get(d, 0) for d in ranked[:k]]) / ideal


def mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
