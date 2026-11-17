# TalentLens: 5-minute interview demo

## Before the interview (T-30 minutes)

1. Plug in your laptop. The re-ranker is CPU-heavy, and battery mode slows it down.
2. Double-click **`start_demo.cmd`** in the project folder. Two windows open (API + frontend), and after about 1 minute the browser opens http://localhost:3000.
   - Run `start_demo.cmd warmup` instead if you changed anything since the last warm-up, or on a new day. It pre-runs every example button, so they load instantly. Results are cached on disk, so a warm-up survives restarts.
3. Click each example button once to check it loads in about a second.
4. Open http://localhost:8000/docs in a second tab (the API docs, for a backend question).
5. Have the README open in VS Code: it has the architecture diagram and the numbers.

If something breaks: close both windows and double-click `start_demo.cmd` again. Cached results come back instantly.

---

## The demo (≈ 5 minutes)

### 1. The problem (20 seconds, no clicking)
> "Recruiters can't trust a black-box match score, and keyword filters miss good candidates. TalentLens ranks candidates in two stages, explains every match with evidence that is verified against the resume, and has an agent that does the searching but never contacts anyone without approval."

### 2. Job → candidates (90 seconds), tab **Job → candidates**
Click **Staff Accountant**, then **Find candidates**.

Point at:
- **"Requirements read from the job (llm)"**: an LLM turned the free-text job into structured requirements (QuickBooks, Microsoft Excel, GAAP; 3+ years), validated with Pydantic.
- **Top result**: a Staff Accountant with 13.2 years. Score bar about 100. **Score parts**: re-ranker 100 · skills 100 · experience 100.
  > "The final score blends the cross-encoder's relevance with hard requirement checks: 60% re-ranker, 30% required-skill coverage, 10% experience. Every part is shown, so a recruiter can see why."
- **Green ✓ / red ✗ chips**: matched and missing required skills, computed in code, not by the LLM.
- **Evidence quotes**, e.g. *"Experience in GAAP accounting and monthly closing processes"*.
  > "The LLM proposes quotes, but every quote must appear word-for-word in the resume. Here one quote was dropped because it wasn't in the resume. I never show unverified evidence."
  (The card says "1 LLM quote(s) dropped".)

Optional: click **Sous Chef** to show that it works across occupations, with honest ✗ chips (e.g. "food safety certification" missing).

### 3. How it works (45 seconds): the architecture diagram in the README
> "Stage one is hybrid retrieval: BM25 for exact terms like QuickBooks, plus bge-small embeddings over resume chunks in FAISS for meaning, fused with Reciprocal Rank Fusion, because their scores are on different scales. Stage two is a cross-encoder that reads job and resume together. That's accurate but slow, so it only re-orders the top candidates. That's the standard retrieve-then-rerank design."

### 4. Recruiter agent (90 seconds), tab **Recruiter agent**
Click the first example (*Find 5 accountants who know QuickBooks, compare the top 3, and draft outreach for the best one*), then **Run agent**.

Point at:
- The answer: 5 real candidate IDs with scores. Then open the **Tool log**: `search_candidates` → `compare_candidates` → `draft_outreach`.
  > "I wrote the tool loop myself, with no framework: the LLM picks a tool, I validate the arguments with Pydantic, run it, and send the result back. There's an 8-step limit, and tool errors go back to the model instead of crashing."
- The **draft** card says **PENDING APPROVAL**. Edit a word, click **Approve**, and it changes to **APPROVED**.
  > "There is no tool that approves or sends. Only this button, a human action through the API, can approve."

Then click **Find backend engineers in Bangalore who know Kafka** → **Run agent**.
> "Honest failure: the data has no Kafka resumes and locations are anonymised, so it says 'no candidates match' and explains why, instead of making something up."

### 5. Reverse direction (20 seconds), tab **Resume → jobs**
Click **Line cook resume**, then **Find jobs**. It returns cook and chef jobs.

### 6. Responsible AI (30 seconds)
> "I ran a counterfactual fairness test: 2,000 name swaps where only the name and gender words change. With the cross-encoder, a name alone changed 27% of top-10 lists, with an average shift of 1.7 places. Redacting names before scoring brought that to zero. I'm also clear about what it doesn't show: proxies like college or employment gaps aren't tested."

---

## Numbers you can quote (all from `results/`)

| Claim | Source |
|---|---|
| 2,481 resumes and 3,000 jobs (from 123,849 postings), 24 categories | `data_stats.json` |
| Removed 36 emails, 155 phones, 201 URLs from resumes before any LLM call; 0 left | `data_stats.json` |
| LLM extraction: 100% valid JSON on 100 gold resumes (97 first try, 3 repaired) | `extraction_run_resume__openai__gpt-oss-120b__gold.json` |
| Rule-based extractor: years for 2,450 / 2,481 resumes | `rules_extraction.json` |
| Fairness (cross-encoder): mean rank shift 1.72, 27.4% of top-10 lists changed → 0 after redaction | `fairness_report.md` |
| Fairness (bi-encoder): mean rank shift 3.36, 44% of top-10 lists changed → 0 after redaction | `fairness_report_dense.md` |
| 139 automated tests; CI on every push | `pytest` |

**Don't claim** nDCG/MRR/ablation numbers yet. They need your 200 labelled pairs. If asked:
> "The evaluation pipeline is built (job-level splits to avoid leakage, pooled candidates, an LLM judge validated with Cohen's kappa). I'm finishing the hand labels now."

---

## Likely questions (short answers; practise the long ones in Phase 11)

- **Why hybrid?** BM25 wins on exact terms and acronyms, and embeddings win on paraphrase. RRF combines their ranks without having to calibrate their scores.
- **Why a cross-encoder only on the top results?** It reads the pair together, so it's more accurate, but it needs one model run per pair. So its cost depends on top-k, not on the size of the pool.
- **Why is the first search slow?** CPU inference of a 278M-parameter cross-encoder. In production: a GPU, a distilled re-ranker, and caching.
- **How do you know the quotes are real?** An exact substring check after whitespace and quote-mark normalisation. Anything that fails is dropped.
- **Why rules for years of experience?** The free LLM tier can't extract 2,481 resumes. The rules parse date ranges, skip the education section, and merge overlapping jobs. They're compared against the LLM and against hand labels.
- **What happens if the LLM quota runs out?** Each free model has its own daily cap. The agent switches models mid-conversation, quotes try 3 models, and matching still works without the LLM.
