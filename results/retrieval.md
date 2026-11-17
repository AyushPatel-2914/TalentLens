# First-stage retrieval (test split)

Generated 2026-11-17T13:44:57+00:00 by scripts/04b_evaluate.py. RRF k = 60. Relevant = label >= 1. Label source `category` is a PROXY (no human input): a resume counts as relevant when its occupation category equals the job's, so Recall@100 is capped when a category has more than 100 resumes. Latency = CPU, model loaded, per query.

| labels | method | queries | recall@100 | ndcg@10 | mrr | latency_ms_mean | latency_ms_p95 |
|---|---|---|---|---|---|---|---|
| category | bm25 | 459 | 0.228 | 0.364 | 0.531 | 4.032 | 7.297 |
| category | dense | 459 | 0.315 | 0.466 | 0.618 | 79.263 | 94.327 |
| category | hybrid | 459 | 0.294 | 0.453 | 0.606 | 85.888 | 102.593 |
