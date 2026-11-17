"""Check an agent run against a scripted task's expected outcome (see configs/agent_tasks.yaml)."""

from __future__ import annotations

import re

from talentlens.agent.loop import AgentResult
from talentlens.agent.outreach import PENDING

ID_RE = re.compile(r"\bR\d{5,}\b")


def mentioned_ids(text: str) -> list[str]:
    return list(dict.fromkeys(ID_RE.findall(text)))


def check_task(checks: dict, result: AgentResult, service) -> list[str]:
    """Return a list of failed checks (empty = task passed)."""
    failures = []
    called = [s.tool for s in result.tool_calls]
    ids = mentioned_ids(result.final_answer)

    if result.stopped_reason != "final_answer":
        failures.append(f"no final answer (stopped: {result.stopped_reason})")
    unknown = [i for i in ids if i not in service.resumes.index]
    if unknown:
        failures.append(f"invented candidate IDs: {unknown}")

    for tool in checks.get("tools_called", []):
        if tool not in called:
            failures.append(f"did not call {tool}")
    for tool in checks.get("tools_not_called", []):
        if tool in called:
            failures.append(f"should not call {tool}")
    if "min_ids" in checks and len(ids) < checks["min_ids"]:
        failures.append(f"mentions {len(ids)} candidates, expected >= {checks['min_ids']}")
    if "max_ids" in checks and len(ids) > checks["max_ids"]:
        failures.append(f"mentions {len(ids)} candidates, expected <= {checks['max_ids']}")
    if "ids_satisfy" in checks and ids:
        c = checks["ids_satisfy"]
        ok = set(service.filter_candidates(ids, skills=c.get("skills"), category=c.get("category"),
                                           min_years=c.get("min_years")))
        bad = [i for i in ids if i not in ok]
        if bad:
            failures.append(f"candidates not meeting {c}: {bad}")
    if "compare_count" in checks:
        import json
        counts = [len(json.loads(s.arguments or "{}").get("candidate_ids", []))
                  for s in result.tool_calls if s.tool == "compare_candidates" and not s.error]
        if checks["compare_count"] not in counts:
            failures.append(f"compare_candidates never called with {checks['compare_count']} IDs (calls: {counts})")
    if "drafts" in checks:
        if len(result.drafts) != checks["drafts"]:
            failures.append(f"{len(result.drafts)} drafts, expected {checks['drafts']}")
        if any(d["status"] != PENDING for d in result.drafts):
            failures.append("a draft is not PENDING_APPROVAL")
    if "final_contains_any" in checks:
        low = result.final_answer.lower()
        if not any(p.lower() in low for p in checks["final_contains_any"]):
            failures.append(f"final answer contains none of {checks['final_contains_any']}")
    return failures
