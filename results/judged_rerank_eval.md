# Re-rankers on LLM-judge labels (test split)

Generated 2026-11-18T11:35:41+00:00 by scripts/05c_judged_rerank_eval.py. Judge: `openai/gpt-oss-120b`. **Labels are from an LLM judge and are not human-validated.**

113 judged test pairs (labels {0: 65, 1: 26, 2: 22}); 22 jobs with candidates of different labels. Each scorer orders the same judged candidates of each job.

| Scorer | Jobs | nDCG@4 | Pairwise accuracy | Pairs compared |
|---|---|---|---|---|
| random | 22 | 0.762 | 0.500 | 78 |
| bge-reranker-base | 22 | 0.845 | 0.667 | 78 |
| fine-tuned reranker | 22 | 0.929 | 0.846 | 78 |

Fine-tuned minus base, per job: mean nDCG@4 difference +0.083 (bootstrap 95% CI -0.008 to +0.177); better on 11 jobs, worse on 4, tied on 7.
