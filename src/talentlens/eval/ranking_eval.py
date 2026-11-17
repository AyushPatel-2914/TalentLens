"""Run a ranker over labelled queries and compute Recall@100, nDCG@10, MRR and latency.

A "ranker" is any function query_text -> ranked list of doc IDs. BM25, dense,
hybrid and the re-rankers in Stage 5 all go through this same function, so
every row of the ablation table is computed identically.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import pandas as pd

from talentlens.eval.ranking_metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank

Ranker = Callable[[str], list[str]]


def evaluate_ranker(
    name: str,
    ranker: Ranker,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    recall_k: int = 100,
    ndcg_k: int = 10,
    min_relevant: int = 1,
) -> tuple[dict, pd.DataFrame]:
    """Return (summary row, per-query DataFrame). Only queries with labels are evaluated."""
    rows = []
    for qid, labels in qrels.items():
        if qid not in queries:
            continue
        t0 = time.perf_counter()
        ranked = ranker(queries[qid])
        latency_ms = (time.perf_counter() - t0) * 1000
        rows.append({
            "method": name, "query_id": qid, "n_labelled": len(labels),
            "n_relevant": sum(l >= min_relevant for l in labels.values()),
            f"recall@{recall_k}": recall_at_k(ranked, labels, recall_k, min_relevant),
            f"ndcg@{ndcg_k}": ndcg_at_k(ranked, labels, ndcg_k),
            "mrr": reciprocal_rank(ranked, labels, min_relevant),
            "latency_ms": latency_ms,
            "top10": ranked[:10],
        })
    per_query = pd.DataFrame(rows)
    if per_query.empty:
        return {"method": name, "queries": 0}, per_query
    summary = {
        "method": name,
        "queries": len(per_query),
        f"recall@{recall_k}": mean(per_query[f"recall@{recall_k}"].tolist()),
        f"ndcg@{ndcg_k}": mean(per_query[f"ndcg@{ndcg_k}"].tolist()),
        "mrr": mean(per_query["mrr"].tolist()),
        "latency_ms_mean": float(per_query["latency_ms"].mean()),
        "latency_ms_p95": float(per_query["latency_ms"].quantile(0.95)),
    }
    return summary, per_query


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "–"
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join(lines)
