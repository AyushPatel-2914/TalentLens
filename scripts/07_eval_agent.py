"""Stage 7: run the 20 scripted recruiter tasks and report success rate, steps and cost.

Usage:
    python scripts/07_eval_agent.py                     # all tasks
    python scripts/07_eval_agent.py --task t05_impossible_kafka_bangalore
    python scripts/07_eval_agent.py --no-reranker       # faster on CPU (retrieval-only relevance)

Each run uses a fresh, temporary outreach store, so drafts from evaluation never
mix with real ones. LLM calls are cached: re-running is free unless prompts change.

Writes results/agent_eval.csv, results/agent_eval.md and results/agent_traces/<task>.json.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from talentlens.agent.evaluate import check_task
from talentlens.agent.loop import run_agent
from talentlens.agent.outreach import OutreachStore
from talentlens.agent.tools import Toolbox
from talentlens.config import ROOT, load_config, resolve
from talentlens.extraction.runner import make_client
from talentlens.service import MatchingService


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default=None)
    p.add_argument("--no-reranker", action="store_true")
    args = p.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = load_config()
    a = cfg["agent"]
    tasks = yaml.safe_load(resolve(a["tasks_file"]).read_text(encoding="utf-8"))
    if args.task:
        tasks = [t for t in tasks if t["id"] == args.task]

    llm = make_client(cfg)
    svc = MatchingService(cfg, use_reranker=not args.no_reranker, llm=llm)
    traces_dir = resolve(cfg["paths"]["results_dir"]) / "agent_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for t in tasks:
        with tempfile.TemporaryDirectory() as tmp:
            toolbox = Toolbox(svc, OutreachStore(Path(tmp) / "outreach.json"), llm, a["model"])
            result = run_agent(t["request"], llm, toolbox, a["model"], a["max_steps"], a["max_tool_result_chars"],
                           a.get("fallback_models"))
        failures = check_task(t["checks"], result, svc)
        if result.stopped_reason == "rate_limit":
            print(f"{t['id']}: stopped by the LLM quota; not counted. Re-run later (finished calls are cached).")
            continue
        rows.append({"task": t["id"], "success": not failures, "llm_calls": result.llm_calls,
                     "tool_calls": len(result.tool_calls), "input_tokens": result.input_tokens,
                     "output_tokens": result.output_tokens, "cost_usd": round(result.cost_usd, 6),
                     "stopped": result.stopped_reason, "failures": "; ".join(failures)})
        (traces_dir / f"{t['id']}.json").write_text(json.dumps({"task": t, "result": result.to_dict(),
                                                                "failures": failures}, indent=2, default=str), encoding="utf-8")
        print(f"{'PASS' if not failures else 'FAIL'} {t['id']} ({result.llm_calls} LLM calls, "
              f"{len(result.tool_calls)} tool calls){'' if not failures else ' -> ' + '; '.join(failures)}")

    if not rows:
        raise SystemExit("No tasks completed.")
    df = pd.DataFrame(rows)
    results = resolve(cfg["paths"]["results_dir"])
    df.to_csv(results / "agent_eval.csv", index=False)
    summary = {"tasks_run": len(df), "tasks_total": len(tasks), "success_rate": float(df["success"].mean()),
               "avg_llm_calls": float(df["llm_calls"].mean()), "avg_tool_calls": float(df["tool_calls"].mean()),
               "avg_cost_usd": float(df["cost_usd"].mean()), "avg_tokens": float((df.input_tokens + df.output_tokens).mean())}
    md = ["# Recruiter agent evaluation", "",
          f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by scripts/07_eval_agent.py. "
          f"Model {a['model']}, max {a['max_steps']} steps, re-ranker {'off' if args.no_reranker else 'on'}. "
          "Cost uses the (unverified) prices in config and is 0 for cached calls.", "",
          f"**Success rate: {summary['success_rate']:.0%}** ({int(df['success'].sum())}/{len(df)} tasks) · "
          f"avg LLM calls {summary['avg_llm_calls']:.1f} · avg tool calls {summary['avg_tool_calls']:.1f} · "
          f"avg tokens {summary['avg_tokens']:.0f} · avg cost ${summary['avg_cost_usd']:.4f}", "",
          "| Task | Result | LLM calls | Tool calls | Failures |", "|---|---|---|---|---|"]
    md += [f"| {r.task} | {'✅' if r.success else '❌'} | {r.llm_calls} | {r.tool_calls} | {r.failures or ''} |" for r in df.itertuples()]
    (results / "agent_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (results / "agent_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(md[:5]))


if __name__ == "__main__":
    main()
