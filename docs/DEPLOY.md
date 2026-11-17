# Deploying TalentLens (free): Vercel + Hugging Face Spaces

```
Browser ──► Vercel (Next.js frontend) ──► Hugging Face Space (FastAPI backend, Docker, free CPU)
                                                │ on startup downloads data/indexes/caches from
                                                └► a PRIVATE Hugging Face dataset
```

Total time: about 1 hour, mostly waiting for the first build. You need a **GitHub** account (you have one), a **Hugging Face** account and a **Vercel** account (sign in to Vercel with GitHub).

> Before making the demo public, check the licence on both Kaggle dataset pages. The demo shows short parts of resumes (summaries and evidence quotes). The data was already public on Kaggle and contact details were removed, but it is still personal data, so keep the dataset repo **private**.

---

## Step 0: push the latest code to GitHub
The Space builds from your GitHub repo, so it must have the latest code.
```powershell
cd "C:\Storage\elip project"
git add .
git commit -m "Deployment: HF Space, rate limits, backend status banner"
git push
```

## Step 1: Hugging Face account and token
1. Sign up at https://huggingface.co and note your **username** (used as `<hf-user>` below).
2. Go to https://huggingface.co/settings/tokens → **Create new token** → type **Write** → copy it.
3. Add it to your local `.env` file (never commit this file):
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
   ```

## Step 2: upload the data to a private dataset (about 60 MB)
This uploads the processed data, indexes, rule-based fields and the **warmed-up caches**, so the demo examples are instant on the server and use no LLM quota.
```powershell
.\.venv\Scripts\python.exe scripts\publish_artifacts.py <hf-user>/talentlens-artifacts
```
Check that https://huggingface.co/datasets/<hf-user>/talentlens-artifacts says **Private**.

## Step 3: create the backend Space
1. Go to https://huggingface.co/new-space
   - **Space name:** `talentlens-api`
   - **SDK:** Docker → **Blank**
   - **Hardware:** CPU basic (free)
   - **Visibility:** Public (the frontend needs to reach it; your data stays in the private dataset)
2. In the new Space, open **Files** → **Add file** → **Upload files**, upload the **two** files from `deploy/hf-space/` (`Dockerfile` and `README.md`), then click **Commit**.
3. Open **Settings** → **Variables and secrets**:

   | Type | Name | Value |
   |---|---|---|
   | Secret | `GROQ_API_KEY` | your Groq key |
   | Secret | `HF_TOKEN` | a Hugging Face token with **read** access (you can reuse the write token) |
   | Variable | `TALENTLENS_ARTIFACTS_REPO` | `<hf-user>/talentlens-artifacts` |
   | Variable | `ALLOWED_ORIGIN_REGEX` | `https://talentlens.*\.vercel\.app` |

4. The Space builds automatically (**Logs** tab, about 15–20 minutes the first time: PyTorch + two models). When it shows **Running**, open:
   `https://<hf-user>-talentlens-api.hf.space/health` → should show `"status":"ok"` and `"public_demo":true`.

## Step 4: deploy the frontend on Vercel
1. Go to https://vercel.com/new and import **AyushPatel-2914/TalentLens**.
2. **Root Directory:** click *Edit* → choose `frontend`.
3. **Environment Variables:** `NEXT_PUBLIC_API_URL` = `https://<hf-user>-talentlens-api.hf.space` (no trailing slash).
4. Click **Deploy** (about 1 minute). Vercel gives you a URL like `https://talentlens-xxxx.vercel.app`.
   - If the URL doesn't start with `talentlens`, add it in the Space as a variable `ALLOWED_ORIGINS` = your exact Vercel URL, then **Restart** the Space.

## Step 5: test it
Open your Vercel URL, click **Staff Accountant**, then **Find candidates**. It should answer in about a second (from the cache). Also try the agent tab.

You can also check everything from your laptop:
```powershell
.\.venv\Scripts\python.exe scripts\warmup_demo.py --api https://<hf-user>-talentlens-api.hf.space
```

---

## Before an interview
- **Open the Vercel link 5–10 minutes early.** Free Spaces sleep after a period of inactivity; the page shows "backend is waking up" and retries by itself (1–3 minutes).
- Cached examples are instant. New, pasted job texts take longer on the free CPU (about 30–90 s), because the cross-encoder runs on 2 vCPUs.
- Public visitors are rate-limited to 60 match and 15 agent requests per hour each, so a stranger can't use up your Groq quota.

## Updating after code changes
`git push`, then in the Space **Settings** → **Factory rebuild** (it pulls the latest code from GitHub). Vercel redeploys automatically on every push. If you changed data or warmed new examples, re-run Step 2.

## Common problems
| Symptom | Fix |
|---|---|
| Space log: `Could not find ... bm25.pkl` / `No indexes` | `TALENTLENS_ARTIFACTS_REPO` or `HF_TOKEN` is missing or wrong (Step 3.3), or Step 2 wasn't run |
| Space log: `401` / `Repository not found` while downloading | The HF token can't read the private dataset: use a token from the same account |
| Frontend error `Failed to fetch` | CORS: add your exact Vercel URL as `ALLOWED_ORIGINS` in the Space and restart it; check `NEXT_PUBLIC_API_URL` |
| `rate limit` message | The per-visitor limit was reached; wait, or raise `RATE_LIMIT_*_PER_HOUR` in the Space variables |
| Agent: `LLM quota exhausted` | All free Groq models hit their daily caps; the cached demo requests still work |
| Build fails at `git clone` | The GitHub repo is private: make it public, or change `REPO_URL` in the Space Dockerfile |
