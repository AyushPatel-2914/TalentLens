import json

import pandas as pd
import pytest

from talentlens.agent.evaluate import check_task, mentioned_ids
from talentlens.agent.loop import run_agent
from talentlens.agent.outreach import APPROVED, PENDING, OutreachStore
from talentlens.agent.tools import Toolbox
from talentlens.llm.client import LLMResponse, RateLimitExhausted


class FakeService:
    """Just enough of MatchingService for the tools."""

    def __init__(self):
        self.jobs = pd.DataFrame({"text": ["Accountant job needs QuickBooks"], "title": ["Accountant"]}, index=["J1"])
        self.resumes = pd.DataFrame({"text": ["QuickBooks accountant", "Chef", "QuickBooks payroll"],
                                     "category": ["ACCOUNTANT", "CHEF", "ACCOUNTANT"]}, index=["R10001", "R10002", "R10003"])

    def _item(self, rid, score):
        return {"resume_id": rid, "score": score, "required_skills": {"matched": ["QuickBooks"], "missing": []},
                "experience": {"summary": "has 5 years"}, "candidate": {"category": "ACCOUNTANT"},
                "nice_to_have_matched": [], "seniority": {}, "location": {}, "evidence_quotes": []}

    def match_job(self, text, top_k=10, job_id=None):
        return {"job": {"required_skills": ["QuickBooks"]}, "results": [self._item("R10001", 0.9), self._item("R10003", 0.8)][:top_k]}

    def filter_candidates(self, ids=None, location=None, min_years=None, skills=None, category=None):
        ids = ids if ids is not None else list(self.resumes.index)
        if location:
            return []
        return [i for i in ids if not skills or "QuickBooks" in self.resumes.loc[i, "text"]]

    def job_struct_for_id(self, job_id):
        return {"title": "Accountant", "required_skills": ["QuickBooks"]}

    def job_struct_for_text(self, text):
        return {"title": "Accountant", "required_skills": ["QuickBooks"]}

    def score_candidates(self, job, text, ids):
        return [self._item(i, 0.9 - n * 0.1) for n, i in enumerate(ids)]

    def candidate_profile(self, rid):
        return {"category": "ACCOUNTANT"}

    def explain(self, job_id, rid, with_quotes=True):
        return {"job_id": job_id, **self._item(rid, 0.9)}


def call(name, args, cid="c1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class ScriptedLLM:
    """Plays back a fixed list of replies: lists of tool calls, final-answer strings, or exceptions."""

    def __init__(self, script):
        self.script, self.seen = list(script), []

    def chat(self, model, messages, json_mode=True, tag="", tools=None):
        self.seen.append([dict(m) for m in messages])
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, list):
            return LLMResponse("", model, 100, 10, 1, 0.001, False, tool_calls=step)
        return LLMResponse(step, model, 100, 10, 1, 0.001, False)


@pytest.fixture
def toolbox(tmp_path):
    return Toolbox(FakeService(), OutreachStore(tmp_path / "o.json"))


def test_tool_loop_runs_tools_then_answers(toolbox):
    llm = ScriptedLLM([[call("search_candidates", {"query": "accountant quickbooks", "top_k": 2})],
                       [call("compare_candidates", {"candidate_ids": ["R10001", "R10003"]}, "c2")],
                       "Best: R10001, then R10003."])
    r = run_agent("find accountants", llm, toolbox, "m", max_steps=5)
    assert r.stopped_reason == "final_answer" and r.final_answer == "Best: R10001, then R10003."
    assert [s.tool for s in r.tool_calls] == ["search_candidates", "compare_candidates"]
    assert r.llm_calls == 3 and r.input_tokens == 300
    # The tool result was sent back to the model as a role=tool message with the right id.
    tool_msgs = [m for m in llm.seen[1] if m["role"] == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "c1" and "R10001" in tool_msgs[0]["content"]


def test_max_steps_stops_the_loop(toolbox):
    llm = ScriptedLLM([[call("search_candidates", {"query": "accountant"})]] * 3)
    r = run_agent("loop forever", llm, toolbox, "m", max_steps=3)
    assert r.stopped_reason == "max_steps" and r.llm_calls == 3 and "3 steps" in r.final_answer


def test_tool_errors_go_back_to_the_model(toolbox):
    llm = ScriptedLLM([[call("no_such_tool", {}), call("compare_candidates", {"candidate_ids": ["R10001"]}, "c2")],
                       "Sorry, fixed."])
    r = run_agent("x", llm, toolbox, "m")
    errors = [s.error for s in r.tool_calls]
    assert "unknown tool" in errors[0] and "invalid arguments" in errors[1]
    assert any("error" in m["content"] for m in llm.seen[1] if m["role"] == "tool")
    assert r.stopped_reason == "final_answer"


def test_unknown_candidate_is_a_tool_error(toolbox):
    llm = ScriptedLLM([[call("explain_match", {"candidate_id": "R99999", "job_id": "J1"})], "done"])
    r = run_agent("x", llm, toolbox, "m")
    assert "unknown candidate_id" in r.tool_calls[0].error


def test_rate_limit_stops_cleanly(toolbox):
    r = run_agent("x", ScriptedLLM([RateLimitExhausted("daily")]), toolbox, "m")
    assert r.stopped_reason == "rate_limit" and r.llm_calls == 0


def test_outreach_is_pending_and_agent_cannot_approve(toolbox):
    llm = ScriptedLLM([[call("draft_outreach", {"candidate_id": "R10001", "job_id": "J1"})], "Draft R10001 is pending approval."])
    r = run_agent("draft and send", llm, toolbox, "m")
    assert len(r.drafts) == 1 and r.drafts[0]["status"] == PENDING
    assert not any("approve" in t["function"]["name"] for t in toolbox.schemas())  # no approval tool exists
    store = toolbox.store
    assert store.decide(r.drafts[0]["id"], approve=True)["status"] == APPROVED
    with pytest.raises(ValueError):
        store.decide(r.drafts[0]["id"], approve=False)  # already decided


def test_outreach_store_persists(tmp_path):
    s = OutreachStore(tmp_path / "o.json")
    d = s.create_draft("R1", "J1", "subj", "body")
    assert OutreachStore(tmp_path / "o.json").get(d["id"])["status"] == PENDING


def test_check_task_flags_invented_ids_and_wrong_drafts(toolbox):
    llm = ScriptedLLM([[call("search_candidates", {"query": "accountant"})], "Top: R10001 and R55555."])
    r = run_agent("x", llm, toolbox, "m")
    failures = check_task({"tools_called": ["search_candidates", "compare_candidates"], "drafts": 1, "min_ids": 1}, r, FakeService())
    text = " ".join(failures)
    assert "invented candidate IDs" in text and "did not call compare_candidates" in text and "0 drafts" in text
    assert mentioned_ids("R10001, R10001 and R10003") == ["R10001", "R10003"]


def test_check_task_passes_good_run(toolbox):
    llm = ScriptedLLM([[call("search_candidates", {"query": "accountant"})], "Top: R10001, R10003."])
    r = run_agent("x", llm, toolbox, "m")
    assert check_task({"tools_called": ["search_candidates"], "min_ids": 2, "ids_satisfy": {"skills": ["QuickBooks"]}}, r, FakeService()) == []


def test_falls_back_to_next_model_when_quota_runs_out(toolbox):
    class QuotaLLM(ScriptedLLM):
        def chat(self, model, messages, json_mode=True, tag="", tools=None):
            if model == "big":
                raise RateLimitExhausted("daily cap")
            return super().chat(model, messages, json_mode, tag, tools)

    r = run_agent("x", QuotaLLM(["Answer from the small model."]), toolbox, "big", fallback_models=["small"])
    assert r.stopped_reason == "final_answer" and r.models_used == ["small"]
    assert any(s.kind == "model_switch" for s in r.steps)


def test_provider_tool_validation_error_is_retried(toolbox):
    from talentlens.llm.client import LLMError
    llm = ScriptedLLM([LLMError('HTTP 400: {"error":{"message":"tool call validation failed: parameters for tool x"}}'),
                       "Recovered answer."])
    r = run_agent("x", llm, toolbox, "m")
    assert r.stopped_reason == "final_answer" and "rejected" in llm.seen[1][-1]["content"]


def test_tool_schemas_are_simple(toolbox):
    import json as _json
    text = _json.dumps(toolbox.schemas())
    assert "anyOf" not in text and '"title"' not in text


VALIDATION_ERR = 'HTTP 400: {"error":{"message":"tool call validation failed: parameters for tool x"}}'


def test_rejected_tool_calls_do_not_use_up_steps(toolbox):
    from talentlens.llm.client import LLMError
    # 3 rejections + 1 real step + final answer fit in max_steps=2, because rejections don't count.
    llm = ScriptedLLM([LLMError(VALIDATION_ERR), [call("filter_candidates", {"skills": ["QuickBooks"]})],
                       LLMError(VALIDATION_ERR), LLMError(VALIDATION_ERR), "R10001 has QuickBooks."])
    r = run_agent("x", llm, toolbox, "m", max_steps=2)
    assert r.stopped_reason == "final_answer"


def test_repeated_rejections_switch_to_fallback_model(toolbox):
    from talentlens.llm.client import LLMError

    class PickyLLM(ScriptedLLM):
        def chat(self, model, messages, json_mode=True, tag="", tools=None):
            if model == "flaky":
                raise LLMError(VALIDATION_ERR)
            return super().chat(model, messages, json_mode, tag, tools)

    r = run_agent("x", PickyLLM(["Answer from the fallback."]), toolbox, "flaky", fallback_models=["steady"])
    assert r.stopped_reason == "final_answer" and r.models_used == ["steady"]
    assert any(s.kind == "model_switch" and "rejected" in (s.error or "") for s in r.steps)


def test_too_many_rejections_stop_the_run(toolbox):
    from talentlens.llm.client import LLMError
    r = run_agent("x", ScriptedLLM([LLMError(VALIDATION_ERR)] * 6), toolbox, "m")
    assert r.stopped_reason == "llm_error"


def test_answer_naming_unfiltered_candidates_is_revised_once(toolbox):
    # R10002 (a chef) did not pass the QuickBooks filter, so the first answer must be revised.
    llm = ScriptedLLM([[call("filter_candidates", {"skills": ["QuickBooks"]})],
                       "R10001 and R10002 both know QuickBooks.",
                       "R10001 knows QuickBooks."])
    r = run_agent("x", llm, toolbox, "m")
    assert r.final_answer == "R10001 knows QuickBooks."
    assert any(s.kind == "revision" for s in r.steps)
    assert "R10002" in llm.seen[-1][-1]["content"]


def test_answer_without_skill_filter_is_not_revised(toolbox):
    llm = ScriptedLLM([[call("search_candidates", {"query": "accountant"})], "R10001 and R10003 look good."])
    r = run_agent("x", llm, toolbox, "m")
    assert r.final_answer == "R10001 and R10003 look good." and not any(s.kind == "revision" for s in r.steps)


def test_unparseable_model_output_is_retried(toolbox):
    from talentlens.llm.client import LLMError
    err = LLMError('HTTP 400: {"error":{"message":"Parsing failed. The model generated output that could not be parsed.",'
                   '"code":"output_parse_failed"}}')
    r = run_agent("x", ScriptedLLM([err, "Recovered answer."]), toolbox, "m")
    assert r.stopped_reason == "final_answer"
