"""Pydantic schemas for LLM extraction output.

Every LLM response is parsed into one of these models. Anything that doesn't fit
(missing field, wrong type, unknown enum value, extra field) raises a
ValidationError, and the extractor sends that error back to the LLM to fix.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Seniority = Literal["intern", "entry", "mid", "senior", "lead", "executive", "unknown"]
EducationLevel = Literal["none", "high_school", "associate", "bachelor", "master", "doctorate", "unknown"]

SENIORITY_HELP = (
    "intern (student/internship), entry (0-2 years), mid (2-5 years), senior (5+ years, senior titles), "
    "lead (team lead/manager/supervisor), executive (director, VP, C-level, owner), unknown"
)
EDUCATION_HELP = "none, high_school, associate, bachelor, master, doctorate, unknown (highest level completed)"


def _clean_list(values: list[str]) -> list[str]:
    seen, out = set(), []
    for v in values:
        v = " ".join(str(v).split()).strip(" .,;:")
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def _clean_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    value = " ".join(value.split()).strip(" .,;")
    # The resume dataset replaces real locations with the placeholder "City , State".
    if not value or value.lower().replace(" ", "") in {"city,state", "city", "state", "n/a", "na", "none", "unknown", "null"}:
        return None
    return value


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResumeExtraction(_Base):
    skills: list[str] = Field(max_length=100)
    total_years_experience: float | None = Field(ge=0, le=60)
    job_titles: list[str] = Field(max_length=30)
    education_level: EducationLevel
    location: str | None
    seniority: Seniority

    _lists = field_validator("skills", "job_titles")(_clean_list)
    _loc = field_validator("location")(_clean_optional_str)


class JobExtraction(_Base):
    title: str = Field(min_length=1)
    required_skills: list[str] = Field(max_length=60)
    nice_to_have_skills: list[str] = Field(max_length=60)
    min_years_experience: float | None = Field(ge=0, le=30)
    location: str | None
    seniority: Seniority

    _lists = field_validator("required_skills", "nice_to_have_skills")(_clean_list)
    _loc = field_validator("location")(_clean_optional_str)


SCHEMAS: dict[str, type[_Base]] = {"resume": ResumeExtraction, "job": JobExtraction}
SKILL_FIELDS: dict[str, tuple[str, ...]] = {
    "resume": ("skills",),
    "job": ("required_skills", "nice_to_have_skills"),
}
