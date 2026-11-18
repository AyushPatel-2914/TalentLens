# Ablation (test split)

Generated 2026-11-17T14:31:58+00:00 by scripts/05b_eval_rerankers.py on the first 50 queries only (partial run).
Re-rankers re-order the hybrid top 100; Recall@100 therefore equals hybrid's. Latency = CPU, per query, including first-stage retrieval. Label source `category` is a PROXY (no human input): a resume counts as relevant when its occupation category equals the job's.

| labels | method | queries | recall@100 | ndcg@10 | mrr | latency_ms_mean | latency_ms_p95 |
|---|---|---|---|---|---|---|---|
| category | bm25 | 50 | 0.226 | 0.354 | 0.529 | 4.388 | 8.397 |
| category | dense | 50 | 0.304 | 0.445 | 0.590 | 151.582 | 108.468 |
| category | hybrid | 50 | 0.291 | 0.428 | 0.602 | 133.280 | 168.988 |
| category | hybrid + bge-reranker-base | 50 | 0.291 | 0.387 | 0.532 | 55610.956 | 109413.482 |
