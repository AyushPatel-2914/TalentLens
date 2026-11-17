import json

import pytest
from pydantic import ValidationError

from talentlens.config import resolve
from talentlens.extraction.extract import extract_one, parse_json
from talentlens.extraction.schemas import JobExtraction, ResumeExtraction
from talentlens.extraction.skills import SkillNormalizer
from talentlens.llm.client import LLMError, LLMResponse, RateLimitExhausted

VALID_RESUME = {
    "skills": ["JS", "postgres", "Excel", "javascript"],
    "total_years_experience": 4.5,
    "job_titles": ["Developer"],
    "education_level": "bachelor",
    "location": "City , State",
    "seniority": "mid",
}


class FakeClient:
    """Returns queued replies; an Exception in the queue is raised instead."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.messages_seen = []

    def chat(self, model, messages, json_mode=True, tag=""):
        self.messages_seen.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMResponse(content=r, model=model, input_tokens=10, output_tokens=5,
                           latency_ms=1, cost_usd=0.001, cached=False)


@pytest.fixture(scope="module")
def normalizer():
    return SkillNormalizer.from_yaml(resolve("configs/skill_aliases.yaml"))


def run(client, normalizer, kind="resume"):
    return extract_one(client, kind, "R1", "resume text", "m", normalizer, "v1", max_repairs=2)


# ---------- schemas ----------

def test_valid_resume_and_placeholder_location():
    r = ResumeExtraction.model_validate(VALID_RESUME)
    assert r.location is None  # "City , State" is the dataset's placeholder
    assert r.skills == ["JS", "postgres", "Excel", "javascript"]


@pytest.mark.parametrize("bad", [
    {"seniority": "kinda senior"},
    {"total_years_experience": -1},
    {"education_level": "phd"},
    {"extra_field": 1},
])
def test_invalid_resume_rejected(bad):
    with pytest.raises(ValidationError):
        ResumeExtraction.model_validate({**VALID_RESUME, **bad})


def test_missing_field_rejected():
    data = dict(VALID_RESUME)
    del data["skills"]
    with pytest.raises(ValidationError):
        ResumeExtraction.model_validate(data)


def test_job_schema():
    j = JobExtraction.model_validate({"title": "Cook", "required_skills": ["Food Safety"], "nice_to_have_skills": [],
                                      "min_years_experience": None, "location": "Remote", "seniority": "entry"})
    assert j.location == "Remote"


# ---------- skills ----------

def test_skill_aliases(normalizer):
    assert normalizer.normalize("JS") == "JavaScript"
    assert normalizer.normalize(" postgres ") == "PostgreSQL"
    assert normalizer.normalize("Underwater Welding") == "Underwater Welding"  # unknown: kept as is
    assert normalizer.normalize_list(["JS", "javascript", "Excel", "MS Excel"]) == ["JavaScript", "Microsoft Excel"]


def test_conflicting_alias_rejected():
    with pytest.raises(ValueError):
        SkillNormalizer({"A": ["x"], "B": ["x"]})


# ---------- parsing and repair loop ----------

def test_parse_json_handles_fences_and_extra_text():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here you go: {"a": 1} hope it helps') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("[1, 2]")


def test_ok_first_try_normalises_skills(normalizer):
    r = run(FakeClient([json.dumps(VALID_RESUME)]), normalizer)
    assert r.status == "ok" and r.attempts == 1
    assert r.data["skills"] == ["JavaScript", "PostgreSQL", "Microsoft Excel"]


def test_invalid_then_repaired(normalizer):
    bad = json.dumps({**VALID_RESUME, "seniority": "kinda senior"})
    client = FakeClient(["not json at all", bad, json.dumps(VALID_RESUME)])
    r = run(client, normalizer)
    assert r.status == "repaired" and r.attempts == 3
    # The validation error is sent back to the model.
    assert "seniority" in client.messages_seen[2][-1]["content"]
    assert r.input_tokens == 30 and r.cost_usd == pytest.approx(0.003)


def test_gives_up_after_max_repairs(normalizer):
    r = run(FakeClient(["nope", "nope", "nope"]), normalizer)
    assert r.status == "failed" and r.data is None and r.attempts == 3 and r.error


def test_server_side_json_failure_is_repaired(normalizer):
    client = FakeClient([LLMError('HTTP 400: {"error":{"code":"json_validate_failed"}}'), json.dumps(VALID_RESUME)])
    assert run(client, normalizer).status == "repaired"


def test_other_errors_propagate(normalizer):
    with pytest.raises(RateLimitExhausted):
        run(FakeClient([RateLimitExhausted("daily")]), normalizer)
    with pytest.raises(LLMError):
        run(FakeClient([LLMError("HTTP 401: bad key")]), normalizer)
