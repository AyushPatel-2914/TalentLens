# Deploying TalentLens (free): Vercel + backend on your laptop via Cloudflare Tunnel

```
Browser ──► Vercel (Next.js frontend, free, stable URL)
              │  link carries ?api=https://<random>.trycloudflare.com
              ▼
         Cloudflare quick tunnel (free, no account) ──► FastAPI backend on this laptop (localhost:8000)
```

**Why this setup:** since mid-2026, new Hugging Face Docker Spaces need a PRO plan ($9/month), and the backend needs about 3 GB of RAM (PyTorch plus two models). That rules out 512 MB free hosts like Render or Koyeb. Your laptop runs it fast and for free. The catch is that the demo only works while the laptop is on and the script is running, which is fine for interviews. (The paid Hugging Face option is kept in [`deploy/hf-space/`](../deploy/hf-space/); see the end of this guide.)

> The demo shows short parts of resumes (summaries and evidence quotes) from public Kaggle datasets, with contact details removed. Check the dataset licences on Kaggle before sharing the link widely.

---

## Step 1: deploy the frontend on Vercel (one time, about 5 minutes)
1. Push the latest code: `git add .`, `git commit -m "..."`, `git push`.
2. Go to https://vercel.com/new, sign in with GitHub and import **AyushPatel-2914/TalentLens**.
3. **Root Directory:** click *Edit* → choose `frontend`.
4. Don't add any environment variables. The backend URL comes from the share link at runtime.
5. Click **Deploy**. Note the URL, e.g. `https://talentlens-xxxx.vercel.app`.
   - The backend's CORS rule allows `https://talentlens*.vercel.app` and `https://talent-lens*.vercel.app`. A different name also works, as long as you set `PUBLIC_FRONTEND_URL` (next step).
6. Add the URL to your local `.env` (no trailing slash):
   ```
   PUBLIC_FRONTEND_URL=https://talent-lens-khaki.vercel.app
   ```

## Step 2: start the public demo (each time you want a live link)
Close any `start_demo` API window first, then double-click **`start_public_demo.cmd`**, or run:
```cmd
cd /d "C:\Storage\elip project"
start_public_demo.cmd
```
It:
1. downloads `cloudflared.exe` to `%USERPROFILE%\tools` the first time (official Cloudflare GitHub release);
2. starts the API in **public-demo mode**, with per-visitor rate limits and CORS for your Vercel URL;
3. opens a free quick tunnel and prints the share link, which is also copied to your clipboard:
   ```
   === Share this link (copied to the clipboard) ===
       https://talentlens-xxxx.vercel.app/?api=https://some-random-words.trycloudflare.com
   ```

Send that link to the interviewer. The browser remembers the backend, so they can reload the page without `?api=`.

**Stop:** press Ctrl+C in the script window, which closes the tunnel, then close the API window.

## Step 3: test it
Open the share link (ideally on your phone over mobile data, to test from outside your network). Click **Staff Accountant** → **Find candidates**; the cached examples answer in about a second. Then try the **Recruiter agent** tab.

Check from the laptop (use your tunnel URL):
```cmd
.venv\Scripts\python.exe scripts\warmup_demo.py --api https://some-random-words.trycloudflare.com
```

---

## Things to know
- **The tunnel URL changes every time you start the script.** Always send the newly printed link. Your Vercel URL stays the same, so it's the one to put on your resume. When the backend is off, that page says so and links to GitHub.
- **Keep the laptop awake and plugged in:** Settings → System → Power → Screen and sleep → *Never* while plugged in.
- **Safety:** only the API port is exposed, not your files. Public visitors are limited to 60 match and 15 agent requests per hour each (identified by Cloudflare's `CF-Connecting-IP`), so nobody can use up your Groq quota. Your keys stay in `.env` on the laptop.
- **Quick tunnels** are meant for testing and demos: they have no uptime guarantee. For a stable custom URL you'd need a Cloudflare account and a domain (a named tunnel).

## Common problems
| Symptom | Fix |
|---|---|
| `A local API ... is already running on port 8000` | Close the `start_demo` API window and run the script again |
| Page says "backend isn't reachable" | The script isn't running, or you opened the Vercel link without `?api=...`; use the newly printed share link |
| Browser console shows a CORS error | Your Vercel URL doesn't start with `talentlens`: set `PUBLIC_FRONTEND_URL` in `.env` and restart the script |
| `No tunnel URL after 2 minutes` | Check `logs\tunnel.log`; usually a firewall or VPN is blocking Cloudflare. Try another network |
| Download of cloudflared fails | Download `cloudflared-windows-amd64.exe` from https://github.com/cloudflare/cloudflared/releases, rename it to `cloudflared.exe` and put it in `%USERPROFILE%\tools` |
| `rate limit` message | That visitor reached the per-hour limit; wait, or raise `RATE_LIMIT_*_PER_HOUR` |
| Agent: `LLM quota exhausted` | All free Groq models hit their daily caps; cached demo requests still work |

---

## Optional: always-on backend (paid)
If you later get Hugging Face PRO ($9/month), the always-on setup is ready:
1. `scripts\publish_artifacts.py <hf-user>/talentlens-artifacts` uploads the data to a **private** dataset (already done for `Anakin2914/talentlens-artifacts`).
2. Create a Docker Space and upload the two files in [`deploy/hf-space/`](../deploy/hf-space/). Add the secrets `GROQ_API_KEY` and `HF_TOKEN`, and the variables `TALENTLENS_ARTIFACTS_REPO` and `ALLOWED_ORIGIN_REGEX=https://talentlens.*\.vercel\.app`.
3. On Vercel, set `NEXT_PUBLIC_API_URL=https://<hf-user>-talentlens-api.hf.space` and redeploy.
