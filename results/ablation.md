# Ablation (test split)

Generated 2026-11-18T07:54:21+00:00 by scripts/05b_eval_rerankers.py on the first 50 queries only (partial run).
Re-rankers re-order the hybrid top 100; Recall@100 therefore equals hybrid's. Latency = CPU, per query, including first-stage retrieval. Label source `category` is a PROXY (no human input): a resume counts as relevant when its occupation category equals the job's. The fine-tuned re-ranker was TRAINED on category labels (train/validation jobs only), so its category score is optimistic by construction; judge it on human labels. The `hybrid + bge-reranker-base` row was reused from the previous run (--skip-base), not re-scored.

| labels | method | queries | recall@100 | ndcg@10 | mrr | latency_ms_mean | latency_ms_p95 |
|---|---|---|---|---|---|---|---|
| category | bm25 | 50 | 0.226 | 0.354 | 0.529 | 11.217 | 19.792 |
| category | dense | 50 | 0.304 | 0.445 | 0.590 | 261.856 | 207.555 |
| category | hybrid | 50 | 0.291 | 0.428 | 0.602 | 184.276 | 209.296 |
| category | hybrid + bge-reranker-base | 50 | 0.291 | 0.387 | 0.532 | 55610.956 | 109413.482 |
| category | hybrid + fine-tuned reranker | 50 | 0.291 | 0.831 | 0.922 | 54694.653 | 55244.866 |
