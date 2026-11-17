"""Rule-based fit checks: skills, experience, seniority, location.

These are computed in code, not by the LLM, so they are deterministic and
always consistent with the structured data shown to the recruiter.

A required skill counts as matched if it is in the resume's extracted skills
OR any known spelling of it appears in the resume text (extraction can miss
skills; the text is the ground truth).
"""

from __future__ import annotations

import re
from functools import lru_cache

from talentlens.extraction.skills import SkillNormalizer, skill_key

SENIORITY_ORDER = ["intern", "entry", "mid", "senior", "lead", "executive"]


@lru_cache(maxsize=4096)
def _phrase_re(phrase: str) -> re.Pattern[str]:
    return re.compile(r"(?<![\w])" + re.escape(phrase) + r"(?![\w])")


# Aliases that are also everyday words or common letter pairs: fine for normalising an
# extracted skill list, but in free text they cause false matches ("go", "word", "ar").
AMBIGUOUS_TEXT_FORMS = frozenset(
    "go ar ap ai ui ux ts js ml word node rest vue mongo recruiting recruitment illustrator outlook".split()
)


# Filler words in requirement phrases: "Food Safety Knowledge" should match a resume that says "food safety".
FILLER_WORDS = {"knowledge", "experience", "skills", "skill", "proficiency", "expertise", "background",
                "familiarity", "understanding", "of", "in", "with",
                # Approximate: "payroll processing" also matches a resume that says "payroll". These matches
                # come from the resume text (via="resume_text"), so the explanation stays transparent.
                "processing", "administration", "certification", "certificate", "certified", "principles", "procedures"}

_EXPERIENCE_REQ_RE = re.compile(r"\b\d+\s*\+?\s*(?:-\s*\d+\s*)?(?:years?|yrs?)\b|^years? of\b", re.IGNORECASE)


def is_experience_requirement(skill: str) -> bool:
    """'3+ years accounting experience' is an experience requirement, not a skill: years are checked separately."""
    return bool(_EXPERIENCE_REQ_RE.search(skill))


def core_phrase(skill: str) -> str:
    words = skill.lower().split()
    while words and words[-1] in FILLER_WORDS:
        words.pop()
    while words and words[0] in FILLER_WORDS:
        words.pop(0)
    return " ".join(words)


def skill_in_text(skill: str, text_lower: str, normalizer: SkillNormalizer) -> bool:
    forms = [f for f in normalizer.forms(skill) if len(f) >= 3 and f not in AMBIGUOUS_TEXT_FORMS]
    core = core_phrase(skill)
    if core and core != skill.lower() and len(core) >= 3 and core not in AMBIGUOUS_TEXT_FORMS:
        forms.append(core)
    return any(_phrase_re(form).search(text_lower) for form in forms)


def match_skills(job_skills: list[str], resume_skills: list[str], resume_text: str,
                 normalizer: SkillNormalizer) -> list[dict]:
    have = {skill_key(normalizer.normalize(s)) for s in resume_skills}
    have |= {core_phrase(k) for k in have}
    text = resume_text.lower()
    out = []
    for s in job_skills:
        canonical = normalizer.normalize(s)
        if skill_key(canonical) in have or core_phrase(canonical) in have:
            via = "extracted"
        elif skill_in_text(canonical, text, normalizer):
            via = "resume_text"
        else:
            via = None
        out.append({"skill": canonical, "matched": via is not None, "via": via})
    return out


def coverage(matches: list[dict]) -> float | None:
    return sum(m["matched"] for m in matches) / len(matches) if matches else None


def experience_fit(has: float | None, needs: float | None) -> tuple[float, str]:
    """1.0 = meets the minimum. Below the minimum, the score falls linearly (2 of 4 years -> 0.5).
    No stated minimum -> 1.0 (nothing to fail). Minimum stated but candidate years unknown -> 0.5."""
    needs = needs if isinstance(needs, (int, float)) and needs == needs else None
    has = has if isinstance(has, (int, float)) and has == has else None
    if not needs:
        return 1.0, "no minimum stated"
    if has is None:
        return 0.5, f"needs {needs:g}+ years; candidate's years unknown"
    if has >= needs:
        return 1.0, f"has {has:g} years, needs {needs:g}+"
    return max(0.0, has / needs), f"has {has:g} years, needs {needs:g}+"


def seniority_fit(job: str | None, candidate: str | None) -> str:
    if job not in SENIORITY_ORDER or candidate not in SENIORITY_ORDER:
        return "unknown"
    diff = SENIORITY_ORDER.index(candidate) - SENIORITY_ORDER.index(job)
    return "match" if diff == 0 else ("above" if diff > 0 else "below")


def location_fit(job: str | None, candidate: str | None) -> str:
    job = job if isinstance(job, str) else None
    candidate = candidate if isinstance(candidate, str) else None
    if job and "remote" in job.lower():
        return "remote"
    if not job or not candidate:
        return "unknown"
    j, c = job.lower(), candidate.lower()
    return "match" if (j in c or c in j or j.split(",")[0].strip() == c.split(",")[0].strip()) else "different"
