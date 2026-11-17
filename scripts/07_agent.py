"""Run the recruiter agent on one request, with human approval at the end.

Usage:
    python scripts/07_agent.py "Find 3 accountants who know QuickBooks and draft outreach for the best one"
    python scripts/07_agent.py --no-reranker "..."

Prints every step (tool, arguments, result), the final answer, and then asks YOU
to approve or reject each outreach draft. Nothing is approved without your answer.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from talentlens.agent.loop import run_agent
from talentlens.agent.outreach import OutreachStore
from talentlens.agent.tools import Toolbox
from talentlens.config import ROOT, load_config, resolve
from talentlens.extraction.runner import make_client
from talentlens.service import MatchingService


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("request")
    p.add_argument("--no-reranker", action="store_true")
    args = p.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = load_config()
    llm = make_client(cfg)
    store = OutreachStore(resolve(cfg["agent"]["outreach_store"]))
    svc = MatchingService(cfg, use_reranker=not args.no_reranker, llm=llm)
    toolbox = Toolbox(svc, store, llm, cfg["agent"]["model"])
    result = run_agent(args.request, llm, toolbox, cfg["agent"]["model"], cfg["agent"]["max_steps"],
                       cfg["agent"]["max_tool_result_chars"],
                       cfg["agent"].get("fallback_models"))

    for s in result.steps:
        if s.kind == "tool_call":
            print(f"[step {s.step}] {s.tool}({s.arguments})")
            print(f"    -> {'ERROR ' + s.error if s.error else s.result[:300]}")
    print(f"\n=== Final answer ({result.stopped_reason}, {result.llm_calls} LLM calls, "
          f"{result.input_tokens + result.output_tokens} tokens) ===\n{result.final_answer}")

    for d in result.drafts:
        print(f"\n--- Draft {d['id']} to {d['candidate_id']} [{d['status']}] ---\nSubject: {d['subject']}\n\n{d['body']}")
        answer = input("Approve this outreach? (y = approve / n = reject / Enter = leave pending): ").strip().lower()
        if answer in {"y", "n"}:
            print("->", store.decide(d["id"], approve=answer == "y")["status"])


if __name__ == "__main__":
    main()
