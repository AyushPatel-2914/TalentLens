"""The agent loop, written by hand so every step is visible.

    messages = [system prompt, user request]
    repeat up to max_steps:
        ask the LLM (with tool schemas)
        if it answers with text -> that's the final answer, stop
        if it asks for tools    -> run each one, append the result (or the error), continue
    if max_steps is reached    -> stop and say so (never loop forever)

Every step is logged: tool name, arguments, result summary or error, tokens, latency.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from talentlens.agent.tools import AgentState, ToolError, Toolbox
from talentlens.llm.client import LLMError, RateLimitExhausted

SYSTEM_PROMPT = """You are TalentLens, a recruiter's assistant. You find and compare candidates using tools.

Rules:
- Use the tools. Only mention candidate IDs that a tool returned. Never invent candidates, skills or facts.
- search_candidates finds candidates; filter_candidates applies hard requirements; compare_candidates and
  explain_match give details; draft_outreach writes an email draft.
- Outreach drafts are saved as PENDING_APPROVAL. Never say a message was sent or approved: a human must approve it.
- If no candidate meets the request, say so clearly and explain why (e.g. the data has no locations).
- Final answer: short, list candidate IDs with one-line reasons, and mention any drafts awaiting approval."""


@dataclass
class Step:
    step: int
    kind: str                       # tool_call | final | error | max_steps
    tool: str | None = None
    arguments: str | None = None
    result: str | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class AgentResult:
    request: str
    final_answer: str
    stopped_reason: str             # final_answer | max_steps | rate_limit | llm_error
    steps: list[Step] = field(default_factory=list)
    drafts: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    models_used: list[str] = field(default_factory=list)

    @property
    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "tool_call"]

    def to_dict(self) -> dict:
        return asdict(self) | {"tool_call_count": len(self.tool_calls)}


def _trim(obj, limit: int) -> str:
    text = json.dumps(obj, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 20] + '... [truncated]"'


def run_agent(request: str, llm, toolbox: Toolbox, model: str, max_steps: int = 8,
              max_result_chars: int = 2500, fallback_models: list[str] | None = None) -> AgentResult:
    """`fallback_models`: if the current model's quota runs out, continue with the next one."""
    models = [model, *(fallback_models or [])]
    state = AgentState()
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request}]
    result = AgentResult(request=request, final_answer="", stopped_reason="max_steps")
    schemas = toolbox.schemas()

    for step_no in range(1, max_steps + 1):
        t0 = time.perf_counter()
        try:
            resp = None
            while resp is None:
                try:
                    resp = llm.chat(models[0], messages, json_mode=False, tag="agent", tools=schemas)
                except RateLimitExhausted:
                    if len(models) == 1:
                        raise
                    result.steps.append(Step(step_no, "model_switch", error=f"{models[0]} quota exhausted -> {models[1]}"))
                    models.pop(0)
        except RateLimitExhausted as e:
            result.stopped_reason, result.final_answer = "rate_limit", f"Stopped: LLM quota exhausted ({str(e)[:120]})"
            break
        except LLMError as e:
            if "tool call validation failed" not in str(e):
                result.steps.append(Step(step_no, "error", error=str(e)[:300]))
                result.stopped_reason, result.final_answer = "llm_error", f"Stopped: LLM error ({str(e)[:120]})"
                break
            # The provider rejected the model's tool arguments: tell the model and let it try again.
            result.steps.append(Step(step_no, "error", error=f"invalid tool call: {str(e)[:250]}"))
            messages.append({"role": "user", "content": "Your last tool call was rejected because its arguments did not "
                             f"match the tool schema ({str(e)[60:300]}). Call the tool again with valid arguments: "
                             "use the documented types and leave optional fields out instead of setting them to null."})
            continue
        except Exception as e:  # provider error: stop cleanly with what we have
            result.steps.append(Step(step_no, "error", error=str(e)[:300]))
            result.stopped_reason, result.final_answer = "llm_error", f"Stopped: LLM error ({str(e)[:120]})"
            break
        result.llm_calls += 1
        if models[0] not in result.models_used:
            result.models_used.append(models[0])
        result.input_tokens += resp.input_tokens
        result.output_tokens += resp.output_tokens
        result.cost_usd += resp.cost_usd
        latency = (time.perf_counter() - t0) * 1000

        if not resp.tool_calls:
            result.final_answer, result.stopped_reason = resp.content.strip(), "final_answer"
            result.steps.append(Step(step_no, "final", result=resp.content[:500], input_tokens=resp.input_tokens,
                                     output_tokens=resp.output_tokens, latency_ms=latency))
            break

        messages.append({"role": "assistant", "content": resp.content or "", "tool_calls": resp.tool_calls})
        for call in resp.tool_calls:
            name, args = call["function"]["name"], call["function"].get("arguments") or "{}"
            t1 = time.perf_counter()
            try:
                output, error = toolbox.run(name, args, state), None
                content = _trim(output, max_result_chars)
            except ToolError as e:
                output, error = None, str(e)
                content = json.dumps({"error": error})
            except Exception as e:  # a bug in a tool must not crash the agent
                output, error = None, f"{type(e).__name__}: {e}"[:300]
                content = json.dumps({"error": f"tool failed: {error}"})
            messages.append({"role": "tool", "tool_call_id": call.get("id", name), "content": content})
            result.steps.append(Step(step_no, "tool_call", tool=name, arguments=args, result=content[:400], error=error,
                                     input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                                     latency_ms=latency + (time.perf_counter() - t1) * 1000))
            resp.input_tokens = resp.output_tokens = 0  # count tokens once per LLM call, not per tool
    else:
        result.final_answer = f"Stopped after {max_steps} steps without a final answer."
        result.steps.append(Step(max_steps, "max_steps"))

    result.drafts = state.drafts
    return result
