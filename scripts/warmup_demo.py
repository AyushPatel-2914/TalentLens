"""Pre-run every demo example through the running API, so the demo is instant.

Usage (with the API running, e.g. after scripts/start_demo.ps1):
    python scripts/warmup_demo.py
    python scripts/warmup_demo.py --skip-agent      # matching only (no Groq quota used)

Uses the same examples as the UI buttons (frontend/lib/samples.json). Match results
are cached on disk by the service; agent LLM calls are cached by the LLM client, so a
rehearsed agent request replays without using quota. Run it once after any change
to data, weights or the model.
"""

from __future__ import annotations

import argparse
import json
import time

import httpx

from talentlens.config import ROOT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--skip-agent", action="store_true")
    args = p.parse_args()
    samples = json.loads((ROOT / "frontend" / "lib" / "samples.json").read_text(encoding="utf-8"))
    client = httpx.Client(base_url=args.api, timeout=900)

    health = client.get("/health").json()
    print(f"API ok: reranker={health['reranker']}, LLM={'on' if health['llm_enabled'] else 'off'}")

    calls = [("POST", "/match/job", {"job_text": j["text"], "top_k": 10, "with_quotes": True}, j["label"]) for j in samples["jobs"]]
    calls += [("POST", "/match/resume", {"resume_text": r["text"], "top_k": 10}, r["label"]) for r in samples["resumes"]]
    if not args.skip_agent:
        calls += [("POST", "/agent", {"request": a}, a[:50]) for a in samples["agent"]]

    for method, path, body, label in calls:
        for attempt in (1, 2):  # second call proves it's now served from cache
            t0 = time.perf_counter()
            r = client.request(method, path, json=body)
            secs = time.perf_counter() - t0
            status = "OK" if r.status_code == 200 else f"HTTP {r.status_code}: {r.text[:200]}"
            print(f"{path:<14} {label:<52} call {attempt}: {secs:6.1f}s  {status}")
            if r.status_code != 200:
                break
    print("\nWarm-up done. The UI examples now load instantly.")


if __name__ == "__main__":
    main()
