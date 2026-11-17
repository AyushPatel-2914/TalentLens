"""LLM extraction with validation and a repair loop.

Flow for one document:
  1. Send the system prompt + document text (JSON mode on).
  2. Parse the reply as JSON and validate it with the Pydantic schema.
  3. If that fails, send the exact error back and ask for corrected JSON
     (up to `max_repairs` times).
  4. Normalise skill names with the alias file.
The result records status (ok / repaired / failed), attempts, tokens and cost,
so failures are counted rather than silently dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from talentlens.extraction.schemas import EDUCATION_HELP, SCHEMAS, SENIORITY_HELP, SKILL_FIELDS
from talentlens.extraction.skills import SkillNormalizer
from talentlens.llm.client import LLMClient, LLMError, RateLimitExhausted

RESUME_PROMPT = f"""You extract structured data from a resume. Reply with ONE JSON object with exactly these keys:

"skills": list of concrete skills, tools, technologies, methods and certifications the candidate has
  (e.g. "Microsoft Excel", "Payroll", "Python", "Food Safety"). Use the resume's wording. Include skills
  from all sections. Do not include personality traits like "hard-working". At most 50.
"total_years_experience": number, total years of professional work experience, computed from the
  date ranges in the work history. These resumes were collected in mid-2021, so "Current" or
  "Present" means June 2021. Do not double-count overlapping jobs.
  Use the stated number if the resume says "X years of experience" and has no dates. null if unknown.
"job_titles": list of job titles the candidate has held, most recent first.
"education_level": highest completed level, one of: {EDUCATION_HELP}.
"location": the candidate's city/region as written, or null. Placeholders like "City , State" mean null.
"seniority": the candidate's current level, one of: {SENIORITY_HELP}.

Only use information in the resume. Do not guess. Reply with JSON only."""

JOB_PROMPT = f"""You extract structured data from a job posting. Reply with ONE JSON object with exactly these keys:

"title": the job title.
"required_skills": list of skills, tools, technologies, certifications or qualifications the posting says
  are required / must-have (e.g. "Microsoft Excel", "Forklift Certification", "Python"). At most 30.
"nice_to_have_skills": list of skills the posting calls preferred, a plus, or nice to have. At most 30.
  If the posting does not separate required from preferred, put everything in required_skills.
"min_years_experience": number, the minimum years of experience required, or null if not stated.
  For a range like "3-5 years" use 3.
"location": the job location as written, or null. Write "Remote" for remote jobs.
"seniority": the level of the role, one of: {SENIORITY_HELP}.

Only use information in the posting. Do not guess. Reply with JSON only."""

PROMPTS = {"resume": RESUME_PROMPT, "job": JOB_PROMPT}

def _strip_fences(text: str) -> str:
    """Remove a leading ```json line and a trailing ``` line, if present."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


@dataclass
class ExtractionResult:
    doc_id: str
    status: str  # ok | repaired | failed
    data: dict[str, Any] | None
    attempts: int
    error: str | None = None
    truncated: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached_calls: int = 0
    errors: list[str] = field(default_factory=list)


def parse_json(content: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating ```json fences or text around the object."""
    text = _strip_fences(content)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    return obj


def format_job_input(title: str, location: str | None, text: str) -> str:
    header = f"Title: {title}\n" + (f"Location: {location}\n" if location else "")
    return header + "\n" + text


def extract_one(
    client: LLMClient,
    kind: str,
    doc_id: str,
    text: str,
    model: str,
    normalizer: SkillNormalizer,
    prompt_version: str,
    max_repairs: int = 2,
    max_input_chars: int = 12000,
) -> ExtractionResult:
    schema = SCHEMAS[kind]
    truncated = len(text) > max_input_chars
    messages = [
        # prompt_version is part of the messages, so it is part of the cache key.
        {"role": "system", "content": f"[prompt {prompt_version}]\n{PROMPTS[kind]}"},
        {"role": "user", "content": text[:max_input_chars]},
    ]
    result = ExtractionResult(doc_id=doc_id, status="failed", data=None, attempts=0, truncated=truncated)

    for attempt in range(1, max_repairs + 2):
        result.attempts = attempt
        try:
            resp = client.chat(model, messages, json_mode=True, tag=f"extract_{kind}")
        except LLMError as e:
            # Groq's JSON mode rejects malformed JSON server-side with HTTP 400
            # "json_validate_failed". Treat that like invalid JSON; re-raise anything else.
            if isinstance(e, RateLimitExhausted) or "json_validate_failed" not in str(e):
                raise
            result.errors.append("server-side JSON validation failed")
            messages = messages + [
                {"role": "user", "content": "Your previous reply was not valid JSON. Reply with one valid JSON object only."}
            ]
            continue
        result.input_tokens += resp.input_tokens
        result.output_tokens += resp.output_tokens
        result.cost_usd += resp.cost_usd
        result.cached_calls += int(resp.cached)
        try:
            data = schema.model_validate(parse_json(resp.content)).model_dump()
        except (ValueError, ValidationError) as e:  # JSONDecodeError is a ValueError
            error = _short_error(e)
            result.errors.append(error)
            messages = messages + [
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": f"That reply was invalid: {error}\nReply again with corrected JSON only."},
            ]
            continue
        for f in SKILL_FIELDS[kind]:
            data[f] = normalizer.normalize_list(data[f])
        result.data = data
        result.status = "ok" if attempt == 1 else "repaired"
        return result

    result.error = result.errors[-1] if result.errors else "unknown error"
    return result


def _short_error(e: Exception) -> str:
    if isinstance(e, ValidationError):
        parts = [f"{'.'.join(map(str, err['loc'])) or 'root'}: {err['msg']}" for err in e.errors()[:5]]
        return "; ".join(parts)
    return f"{type(e).__name__}: {e}"[:300]
