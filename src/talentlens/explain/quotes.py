"""Evidence quotes, verified word-for-word against the resume.

The LLM is asked for short quotes copied exactly from the resume. Each quote is
then checked in code: after normalising whitespace, quote marks and dashes, it
must appear as an exact (case-sensitive) substring of the resume text. Any quote
that doesn't is DROPPED and counted. Nothing unverified is ever shown.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field, ValidationError

from talentlens.extraction.extract import parse_json
from talentlens.llm.client import LLMClient, LLMError, RateLimitExhausted

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_WS = re.compile(r"\s+")

QUOTE_PROMPT = """You find evidence in a resume for a job match.
Return ONE JSON object: {"quotes": [{"quote": "...", "supports": "..."}]}
- 2 or 3 quotes. Each quote must be copied EXACTLY, character for character, from the resume:
  a contiguous phrase or sentence of 5 to 30 words. Do not fix typos, shorten, merge or paraphrase.
- "supports": which job requirement the quote is evidence for (a few words).
- If the resume has no evidence for the job, return {"quotes": []}."""


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    return _WS.sub(" ", text).strip().strip('"').strip()


def verify_quote(quote: str, source: str, min_chars: int = 15) -> bool:
    q = normalize_for_match(quote).rstrip(".")
    return len(q) >= min_chars and q in normalize_for_match(source)


class _Quote(BaseModel):
    quote: str
    supports: str = ""


class _Quotes(BaseModel):
    quotes: list[_Quote] = Field(default_factory=list)


def get_evidence_quotes(client: LLMClient, model: str, job_brief: str, resume_text: str,
                        max_quotes: int = 3, min_chars: int = 15, max_resume_chars: int = 6000) -> dict:
    """Returns {"kept": [...], "dropped": [...], "error": str | None}."""
    messages = [
        {"role": "system", "content": QUOTE_PROMPT},
        {"role": "user", "content": f"JOB:\n{job_brief}\n\nRESUME:\n{resume_text[:max_resume_chars]}"},
    ]
    try:
        resp = client.chat(model, messages, json_mode=True, tag="quotes")
        quotes = _Quotes.model_validate(parse_json(resp.content)).quotes[:max_quotes]
    except RateLimitExhausted:
        raise
    except (LLMError, ValueError, ValidationError) as e:
        return {"kept": [], "dropped": [], "error": str(e)[:200]}
    kept, dropped = [], []
    for q in quotes:
        (kept if verify_quote(q.quote, resume_text, min_chars) else dropped).append(q.model_dump())
    return {"kept": kept, "dropped": dropped, "error": None}
