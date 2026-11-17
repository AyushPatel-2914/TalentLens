"""Assemble one match explanation from structured data, rules, score and (optional) quotes."""

from __future__ import annotations

from talentlens.explain.fit import coverage, experience_fit, location_fit, match_skills, seniority_fit
from talentlens.explain.scoring import final_score
from talentlens.extraction.skills import SkillNormalizer


def build_explanation(job: dict, resume: dict, resume_text: str, reranker_logit: float | None,
                      normalizer: SkillNormalizer, weights: dict[str, float],
                      quotes: dict | None = None, relevance: float | None = None) -> dict:
    """`job` / `resume` are extraction dicts (JobExtraction / ResumeExtraction fields); missing fields are fine."""
    required = match_skills(job.get("required_skills") or [], resume.get("skills") or [], resume_text, normalizer)
    nice = match_skills(job.get("nice_to_have_skills") or [], resume.get("skills") or [], resume_text, normalizer)
    exp_score, exp_text = experience_fit(resume.get("total_years_experience"), job.get("min_years_experience"))
    cov = coverage(required)
    score, breakdown = final_score(reranker_logit, cov, exp_score, weights, relevance)
    return {
        "score": round(score, 4),
        "score_breakdown": breakdown,
        "required_skills": {
            "matched": [m["skill"] for m in required if m["matched"]],
            "missing": [m["skill"] for m in required if not m["matched"]],
            "coverage": cov,
            "details": required,
        },
        "nice_to_have_matched": [m["skill"] for m in nice if m["matched"]],
        "experience": {"fit": exp_score, "summary": exp_text},
        "seniority": {"job": job.get("seniority"), "candidate": resume.get("seniority"),
                      "fit": seniority_fit(job.get("seniority"), resume.get("seniority"))},
        "location": {"job": job.get("location"), "candidate": resume.get("location"),
                     "fit": location_fit(job.get("location"), resume.get("location"))},
        "evidence_quotes": (quotes or {}).get("kept", []),
        "quotes_dropped_unverified": len((quotes or {}).get("dropped", [])),
    }
