"""Stage 2: extract structured JSON from resumes and/or jobs.

Usage:
    python scripts/02_extract.py --kind resume --gold            # the 100 resumes you will label
    python scripts/02_extract.py --kind resume --gold --model openai/gpt-oss-120b
    python scripts/02_extract.py --kind resume --limit 20         # quick trial
    python scripts/02_extract.py --kind all                       # everything (long on free tier)

Safe to stop and re-run at any time: finished calls come from the cache for free.
If the daily quota runs out, the script saves what it has and tells you to re-run later.

Writes:
    data/extracted/{kind}s__{model}.parquet
    results/extraction_run_{kind}__{model}.json   (status counts, tokens, cost)
    results/llm_costs.csv                          (one row per LLM call)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import pandas as pd
from tqdm import tqdm

from talentlens.config import load_config, resolve
from talentlens.extraction.extract import extract_one
from talentlens.extraction.runner import load_documents, load_gold_ids, make_client, make_normalizer, output_path
from talentlens.llm.client import CacheMiss, LLMError, RateLimitExhausted


def run(cfg: dict, kind: str, model: str, gold: bool, limit: int | None) -> None:
    docs = load_documents(cfg, kind)
    if gold:
        ids = set(load_gold_ids(cfg))
        docs = docs[docs["doc_id"].isin(ids)]
    if limit:
        docs = docs.head(limit)

    client, normalizer = make_client(cfg), make_normalizer(cfg)
    llm, ex = cfg["llm"], cfg["extraction"]
    rows, stopped_early, skipped = [], None, 0
    t0 = time.perf_counter()

    for doc in tqdm(docs.itertuples(index=False), total=len(docs), desc=f"{kind}s with {model}"):
        try:
            r = extract_one(
                client, kind, doc.doc_id, doc.text, model, normalizer, ex["prompt_version"],
                max_repairs=llm["max_repair_attempts"], max_input_chars=llm["max_input_chars"],
            )
        except RateLimitExhausted as e:
            # Quota gone: keep going in cache-only mode to collect answers cached by earlier runs.
            stopped_early = str(e)
            client.cache_only = True
            continue
        except CacheMiss:
            skipped += 1
            continue
        except LLMError as e:
            # One bad document (e.g. too large) must not stop the whole run. It is recorded as
            # failed (not cached), so a re-run tries it again.
            rows.append({"doc_id": doc.doc_id, "status": "failed", "attempts": 1, "error": str(e)[:300],
                         "truncated": False, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                         "fully_cached": False})
            continue
        rows.append({
            "doc_id": r.doc_id, "status": r.status, "attempts": r.attempts, "error": r.error,
            "truncated": r.truncated, "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
            "cost_usd": r.cost_usd, "fully_cached": r.cached_calls == r.attempts,
            **(r.data or {}),
        })

    out = pd.DataFrame(rows)
    path = output_path(cfg, kind, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Merge with earlier runs: this run's rows replace older rows for the same doc, and docs
    # this run didn't reach are kept. A run that stops early can never erase earlier results.
    merged = out
    if path.exists():
        previous = pd.read_parquet(path)
        if len(previous):
            merged = pd.concat([previous[~previous["doc_id"].isin(out["doc_id"])] if len(out) else previous, out])
    if len(merged):
        merged.sort_values("doc_id").to_parquet(path, index=False)

    status = Counter(out["status"]) if len(out) else Counter()
    new_calls = out[~out["fully_cached"]] if len(out) else out
    summary = {
        "kind": kind, "model": model, "gold_only": gold, "requested": len(docs), "processed": len(out),
        "status_counts": dict(status),
        "valid_rate": round((status["ok"] + status["repaired"]) / len(out), 4) if len(out) else None,
        "truncated": int(out["truncated"].sum()) if len(out) else 0,
        "tokens_this_run_uncached": {
            "input": int(new_calls["input_tokens"].sum()) if len(out) else 0,
            "output": int(new_calls["output_tokens"].sum()) if len(out) else 0,
        },
        "tokens_per_doc_mean": {
            "input": round(float(out["input_tokens"].mean()), 1) if len(out) else None,
            "output": round(float(out["output_tokens"].mean()), 1) if len(out) else None,
        },
        "cost_usd_this_run": round(float(new_calls["cost_usd"].sum()), 6) if len(out) else 0.0,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "stopped_early": stopped_early,
        "not_yet_done_waiting_for_quota": skipped,
    }
    results = resolve(cfg["paths"]["results_dir"])
    scope = "gold" if gold else ("limit" if limit else "full")
    summary["rows_in_output_file"] = len(merged)
    with open(results / f"extraction_run_{kind}__{model.replace('/', '__')}__{scope}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    failed = out[out["status"] == "failed"] if len(out) else out
    for row in failed.head(5).itertuples():
        print(f"FAILED {row.doc_id}: {row.error}")
    print(f"Saved {path}")
    if stopped_early:
        print("\nStopped early because of the rate limit. Re-run the same command later to continue.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["resume", "job", "all"], required=True)
    p.add_argument("--model", default=None, help="defaults to llm.models.extraction in config")
    p.add_argument("--gold", action="store_true", help="only the 100 resumes selected for labelling")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    model = args.model or cfg["llm"]["models"]["extraction"]
    for kind in (["resume", "job"] if args.kind == "all" else [args.kind]):
        run(cfg, kind, model, gold=args.gold and kind == "resume", limit=args.limit)


if __name__ == "__main__":
    main()
