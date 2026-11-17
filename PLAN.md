# TalentLens — Build Plan

Status: **all 10 stages coded and tested (127 tests), 2026-11-17.** Real results so far: Stage 1, gold extraction, splits/pools, indexes, fairness (running). Blocked on: hand labels (100 resumes, 200 pairs) and Groq quota (paid tier recommended).

---

## 0. Why this project fits the Elip role

Elip's posting says interns will build **AI agents, job matching systems and semantic search**. TalentLens covers all three, and each one comes with its own measured evidence:

| Elip asks for | TalentLens stage | Evidence you can show in an interview |
|---|---|---|
| Job matching systems | 3, 5, 6 | Graded ground truth, a fine-tuned cross-encoder, nDCG@10 / MRR ablation |
| Semantic search | 4 | BM25 vs dense vs hybrid (RRF), Recall@100, latency per query |
| AI agents | 7 | A hand-written tool loop, human approval gate, task success rate on 20 scripted tasks |
| LLM product work | 2, 6 | Pydantic-validated extraction, field-level F1, a quote-verification failure rate |
| Responsible AI for hiring | 8 | Counterfactual name swap test, before/after mitigation numbers |
| Shipping | 9, 10 | FastAPI + Next.js, pytest, Docker, CI, live URLs |

Recruitment AI is a regulated, high-stakes area (bias, explainability, personal data). The fairness check, evidence verification and PII cleaning are what make this look like a product built by someone who understands the domain.

---

## 1. Repo structure

```
talentlens/
├── README.md                    # Project explanation + real results (filled in as stages finish)
├── PLAN.md                      # This file
├── pyproject.toml               # Dependencies + package config (pip install -e .)
├── .env.example                 # GROQ_API_KEY, HF_TOKEN, ALLOWED_ORIGINS (never commit .env)
├── Makefile                     # make data / extract / eval / test / api / web
│
├── configs/
│   ├── config.yaml              # Paths, model names, k values, RRF k, prices per 1M tokens
│   └── skill_aliases.yaml       # "js" -> "JavaScript", "postgres" -> "PostgreSQL", ...
│
├── data/                        # git-ignored (personal data, large files)
│   ├── raw/                     # Kaggle downloads, untouched
│   ├── processed/               # resumes.parquet, jobs.parquet (cleaned, stable IDs)
│   ├── extracted/               # resumes_struct.parquet, jobs_struct.parquet
│   ├── cache/llm/               # One JSON file per LLM call, keyed by hash -> never pay twice
│   ├── labels/                  # Your hand labels (extraction_gold.jsonl, pairs_human.jsonl)
│   └── indexes/                 # bm25.pkl, faiss.index
│
├── src/talentlens/
│   ├── config.py                # Loads configs/config.yaml + env vars
│   ├── data/                    # Stage 1: load.py, clean.py, pii.py, sample.py
│   ├── llm/                     # Shared Groq client: retries, rate limiting, caching, cost log
│   ├── extraction/              # Stage 2: schemas.py (Pydantic), extract.py, skills.py
│   ├── labeling/                # Stages 2-3: CLI labellers, LLM judge, train/val/test split
│   ├── retrieval/               # Stage 4: bm25.py, dense.py, hybrid.py (RRF)
│   ├── rerank/                  # Stage 5: cross-encoder wrapper (base + fine-tuned)
│   ├── explain/                 # Stage 6: skill/experience fit, quote verification, final score
│   ├── agent/                   # Stage 7: tools.py, loop.py, prompts.py, approval state
│   ├── fairness/                # Stage 8: counterfactual variants, rank-shift metrics
│   ├── eval/                    # metrics.py: nDCG, MRR, Recall@k, kappa (one tested place)
│   └── api/                     # Stage 9: FastAPI app, routes, startup model loading
│
├── scripts/                     # Thin entry points, one per step, numbered in run order
│   ├── 01_prepare_data.py
│   ├── 02_extract.py
│   ├── 02b_label_extraction.py
│   ├── 02c_eval_extraction.py
│   ├── 03_label_pairs.py
│   ├── 03b_judge.py
│   ├── 03c_make_splits.py
│   ├── 04_build_indexes.py
│   ├── 04b_evaluate.py          # = evaluate.py from the brief
│   ├── 05_eval_rerankers.py
│   ├── 06_eval_explanations.py
│   ├── 07_eval_agent.py
│   └── 08_fairness.py
│
├── notebooks/
│   └── finetune_reranker.ipynb  # Stage 5, runs on Colab GPU, pushes model to HF Hub
│
├── results/                     # COMMITTED. Every number in the README comes from here.
│   ├── data_stats.json
│   ├── extraction_eval.csv / .md
│   ├── judge_agreement.json
│   ├── retrieval.csv / .md
│   ├── ablation.csv / .md
│   ├── explanations.json
│   ├── agent_eval.csv / .md
│   ├── fairness_report.md
│   └── llm_costs.csv
│
├── tests/                       # pytest: API, validator, RRF, metrics, quote check, agent loop
├── frontend/                    # Next.js app (Stage 9)
├── Dockerfile                   # Backend image
└── .github/workflows/ci.yml     # Runs tests on every push
```

Design rules behind this layout:

- **One Python package (`src/talentlens`), no microservices.** Scripts only parse arguments and call package functions, so the same code is used by scripts, tests and the API.
- **`data/` is git-ignored; `results/` is committed.** Resumes are personal data and should not go to GitHub. Results are small, and committing them proves where every number came from.
- **All metrics live in `eval/metrics.py`** and are unit-tested against hand-computed examples, so a bug there can't quietly distort every table.

---

## 2. Stage plan

Each stage ends with the same handover: what was built, why, how to run it, expected output, how to verify it, and common errors. **No result is written by hand.** Each one is produced by a script and saved under `results/`.

| # | Stage | Main output | Depends on | Your manual work | Rough effort |
|---|---|---|---|---|---|
| 1 | Data loading & cleaning | `resumes.parquet`, `jobs.parquet`, `data_stats.json` | – | Download 2 Kaggle datasets | 0.5 day |
| 2 | LLM extraction + accuracy | Structured parquet, `extraction_eval.csv`, `llm_costs.csv` | 1 | **Label 100 resumes** (~3–4 h) | 1.5 days |
| 3 | Matching ground truth | `pairs_human.jsonl`, judge labels, `splits.json`, `judge_agreement.json` | 1, 2 | **Label 200 pairs** (~3 h) | 1.5 days |
| 4 | Hybrid retrieval + baseline eval | Indexes, `retrieval.csv/.md` | 3 | – | 1 day |
| 5 | Cross-encoder re-ranker | HF Hub model, `ablation.csv/.md` | 3, 4 | Run Colab notebook | 1.5 days |
| 6 | Explainable matches | Explanation JSON, quote-failure rate, final score | 2, 5 | Review 20 explanations | 1 day |
| 7 | Recruiter agent | Agent loop, `agent_eval.csv/.md` | 4–6 | Write/check 20 tasks | 1.5 days |
| 8 | Fairness check | `fairness_report.md` (before/after) | 5, 6 | – | 1 day |
| 9 | API, frontend, tests | FastAPI, Next.js, pytest suite | all | – | 2–3 days |
| 10 | Docker, CI, deploy, README | Live URLs, final README | 9 | Render/Vercel accounts | 1 day |
| 11 | Interview prep | Mock interview, one question at a time | all | Answer the questions | ongoing |

**Minimum version you can show early:** stages 1 → 5. That gives the core matching story with real numbers (retrieval + re-ranking ablation). Stages 6–8 are the stand-out features. Stages 9–10 make it usable by other people.

---

## 3. Issues to settle before writing code

I found these by checking the brief against the datasets and tools. Each one changes a stage, so it's better to decide them now.

1. **Groq free-tier limits are much lower than this workload.** Stage 2 makes about 5,400 extraction calls (2.4k resumes + 3k jobs) at roughly 1–2k tokens each, so on the order of 5–10M tokens. Stage 3's judge adds about 3,000 more calls. Measured on 2026-11-17: Llama 3.3 is no longer on Groq; the free tier for gpt-oss-20b/120b allows 1,000 requests/day and 8,000 tokens/minute per model, and one resume uses ~2,300 tokens.
   *Proposal:* use `llama-3.3-70b-versatile` for the 100 gold-labelled resumes, the judge and the agent. For bulk extraction, use either the faster 8B model or Groq's paid tier, and **measure the accuracy difference** on your 100 labels rather than assuming it. Model names and prices go in `config.yaml`; check the current numbers on Groq's console.

2. **The resume dataset is not a tech-hiring dataset.** The Kaggle Resume Dataset has 24 broad categories (HR, Chef, Aviation, IT, Finance, ...), mostly US-style resumes, with only about 100–120 per category. A request like *"backend engineers in Bangalore who know Kafka"* will probably return **zero real matches**.
   *Proposal:* write the 20 agent tasks and demo queries around what the data actually contains (Stage 1 stats will show this), and say so openly in the README. An honest "0 results, here's why" also makes a good agent test case.

3. **Evaluating on LLM-judge labels risks circular results.** If the re-ranker is trained on judge labels and tested on judge labels, you are partly measuring how well it agrees with the judge.
   *Proposal:* report test metrics twice: on judge labels (large set) and on **your human-labelled pairs inside the test split** (small set, the trusted one).

4. **Pooling for fair labelling.** If the pairs to label are drawn only from BM25's top results, the evaluation will favour BM25. Stage 3 will sample candidates from the **union** of BM25, dense and random results (the TREC-style "pooling" method), with a mix of likely-strong, borderline and likely-weak pairs, so not every label is obvious.

5. **Unlabelled resumes count as non-relevant.** For each test job, only about 4–60 resumes have labels. The rest are treated as 0 when computing metrics. This is standard practice, but it underestimates every method, so the README will say so.

6. **Dataset licences.** Check both Kaggle licence tabs before making the repo public, and don't commit raw or processed resume text. Only code, aggregate stats and small anonymised examples go to GitHub.

7. **Timeline.** Registration for the Elip drive **closes today (2026-11-17) at 22:00**, so register first. The full plan is about 2–3 weeks of focused work. If shortlisting happens sooner, list TalentLens with only the stages that are actually finished, and use numbers only after they exist in `results/`.

---

## 4. Approval checklist

Reply with OK, or with changes, on:

- [ ] Repo structure above
- [ ] Model choice for bulk extraction (70B paid / 8B free / decide after a 100-resume comparison)
- [ ] Reporting test metrics on both human and judge labels
- [ ] Agent tasks written against the real dataset contents, not the Bangalore/Kafka example

After you approve, I start **Stage 1 only**.
