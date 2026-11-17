"""The agent's tools: JSON schemas the LLM sees + the Python functions that run.

Every tool validates its arguments with Pydantic. Invalid arguments raise
ToolError, and the loop sends that message back to the LLM so it can correct
itself instead of crashing.

Tool results are kept compact (IDs, scores, key facts): the LLM only needs
enough to decide the next step, and short results keep each request within the
provider's per-request token limit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from talentlens.agent.outreach import OutreachStore
from talentlens.extraction.extract import parse_json


class ToolError(Exception):
    pass


@dataclass
class AgentState:
    """What the agent has seen so far in one run; tools use it to resolve 'the current job'."""
    current_query: str | None = None
    current_job_id: str | None = None
    seen_candidates: dict[str, dict] = field(default_factory=dict)
    drafts: list[dict] = field(default_factory=list)


class SearchArgs(BaseModel):
    query: str = Field(min_length=3, description="Job description or a short request like 'accountant with QuickBooks'")
    top_k: int = Field(default=10, ge=1, le=25)
    job_id: str | None = Field(default=None, description="Optional dataset job id (e.g. J3884428798)")


class FilterArgs(BaseModel):
    candidate_ids: list[str] | None = Field(default=None, description="Filter these ids; omit to filter all candidates")
    location: str | None = None
    min_years: float | None = Field(default=None, ge=0, le=60)
    skills: list[str] | None = None
    category: str | None = Field(default=None, description="Resume category, e.g. ACCOUNTANT, CHEF, INFORMATION-TECHNOLOGY")


class CompareArgs(BaseModel):
    candidate_ids: list[str] = Field(min_length=2, max_length=5)


class ExplainArgs(BaseModel):
    candidate_id: str
    job_id: str | None = Field(default=None, description="Dataset job id; omit to use the current search")


class DraftArgs(BaseModel):
    candidate_id: str
    job_id: str | None = Field(default=None, description="Dataset job id; omit to use the current search")


DRAFT_PROMPT = """Write a short, friendly recruiting outreach email to a candidate.
Use ONLY the facts given (job title, the candidate's matched skills and experience). Do not invent
company names, salaries, or details about the candidate. Do not use the candidate's name (unknown);
start with "Hi there,". 80-130 words.
Reply with JSON: {"subject": "...", "body": "..."}"""


def simplify_schema(schema: dict) -> dict:
    """Make Pydantic's JSON schema easier for LLMs to follow (and for the provider to validate):
    optional fields become plain types (`anyOf: [string, null]` -> `string`) and are simply left out
    when not needed; titles and `default: null` are removed."""
    def clean(node):
        if isinstance(node, dict):
            if "anyOf" in node:
                options = [o for o in node["anyOf"] if o.get("type") != "null"]
                if len(options) == 1:
                    rest = {k: v for k, v in node.items() if k != "anyOf"}
                    node = {**options[0], **rest}
            return {k: clean(v) for k, v in node.items() if k != "title" and not (k == "default" and v is None)}
        if isinstance(node, list):
            return [clean(x) for x in node]
        return node
    return clean(schema)


def _compact(item: dict) -> dict:
    c = item.get("candidate", {})
    return {"candidate_id": item["resume_id"], "score": round(item["score"], 3), "category": c.get("category"),
            "years_experience": c.get("total_years_experience"), "latest_titles": (c.get("job_titles") or [])[:2],
            "location": c.get("location"), "required_matched": item["required_skills"]["matched"][:8],
            "required_missing": item["required_skills"]["missing"][:8]}


class Toolbox:
    def __init__(self, service, store: OutreachStore, llm=None, draft_model: str | None = None):
        self.svc, self.store, self.llm, self.draft_model = service, store, llm, draft_model
        self.tools: dict[str, tuple[type[BaseModel], Callable, str]] = {
            "search_candidates": (SearchArgs, self.search_candidates,
                                  "Run the full matching pipeline (hybrid retrieval + re-ranker + requirement checks). "
                                  "Returns the best candidates with scores and matched/missing required skills."),
            "filter_candidates": (FilterArgs, self.filter_candidates,
                                  "Keep only candidates meeting hard criteria (location, minimum years, all listed skills, "
                                  "category). Note: most resumes in this dataset have no location."),
            "compare_candidates": (CompareArgs, self.compare_candidates,
                                   "Side-by-side comparison of 2-5 candidates for the current job."),
            "explain_match": (ExplainArgs, self.explain_match,
                              "Detailed explanation of one candidate's fit: skills matched/missing, experience, seniority."),
            "draft_outreach": (DraftArgs, self.draft_outreach,
                               "Draft an outreach email. It is saved as PENDING_APPROVAL; a human must approve it. "
                               "You cannot send or approve it."),
        }

    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": n, "description": d, "parameters": simplify_schema(m.model_json_schema())}}
                for n, (m, _, d) in self.tools.items()]

    def run(self, name: str, raw_args: str | dict, state: AgentState) -> Any:
        if name not in self.tools:
            raise ToolError(f"unknown tool {name!r}; available: {', '.join(self.tools)}")
        model, fn, _ = self.tools[name]
        try:
            args = model.model_validate(json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {}))
        except (json.JSONDecodeError, ValidationError) as e:
            raise ToolError(f"invalid arguments for {name}: {str(e)[:400]}") from e
        return fn(args, state)

    # ---------- tools ----------

    def _job_context(self, job_id: str | None, state: AgentState) -> tuple[str | None, str]:
        if job_id:
            if job_id not in self.svc.jobs.index:
                raise ToolError(f"unknown job_id {job_id}")
            return job_id, self.svc.jobs.loc[job_id, "text"]
        if state.current_job_id:
            return state.current_job_id, self.svc.jobs.loc[state.current_job_id, "text"]
        if state.current_query:
            return None, state.current_query
        raise ToolError("no job context: call search_candidates first or pass job_id")

    def _check_candidate(self, cid: str) -> None:
        if cid not in self.svc.resumes.index:
            raise ToolError(f"unknown candidate_id {cid}")

    def search_candidates(self, a: SearchArgs, state: AgentState) -> dict:
        if a.job_id and a.job_id not in self.svc.jobs.index:
            raise ToolError(f"unknown job_id {a.job_id}")
        text = self.svc.jobs.loc[a.job_id, "text"] if a.job_id else a.query
        out = self.svc.match_job(text, top_k=a.top_k, job_id=a.job_id)
        state.current_query, state.current_job_id = a.query, a.job_id
        results = [_compact(r) for r in out["results"]]
        for r in results:
            state.seen_candidates[r["candidate_id"]] = r
        return {"job_requirements": out["job"].get("required_skills", [])[:12], "candidates": results}

    def filter_candidates(self, a: FilterArgs, state: AgentState) -> dict:
        ids = self.svc.filter_candidates(a.candidate_ids, a.location, a.min_years, a.skills, a.category)
        note = None
        if a.location and not ids:
            note = "No candidates with that location. Most resumes in this dataset have their location removed."
        return {"count": len(ids), "candidate_ids": ids[:25], "truncated": len(ids) > 25, "note": note}

    def compare_candidates(self, a: CompareArgs, state: AgentState) -> dict:
        for cid in a.candidate_ids:
            self._check_candidate(cid)
        job_id, text = self._job_context(None, state) if (state.current_query or state.current_job_id) else (None, None)
        if text is None:
            return {"candidates": [self.svc.candidate_profile(c) for c in a.candidate_ids]}
        job = self.svc.job_struct_for_id(job_id) if job_id else self.svc.job_struct_for_text(text)
        scored = self.svc.score_candidates(job, text, a.candidate_ids)
        for s in scored:
            s["candidate"] = self.svc.candidate_profile(s["resume_id"])
        return {"ranked": [_compact(s) for s in scored]}

    def explain_match(self, a: ExplainArgs, state: AgentState) -> dict:
        self._check_candidate(a.candidate_id)
        job_id, text = self._job_context(a.job_id, state)
        if job_id:
            e = self.svc.explain(job_id, a.candidate_id, with_quotes=self.llm is not None)
        else:
            job = self.svc.job_struct_for_text(text)
            e = self.svc.score_candidates(job, text, [a.candidate_id])[0]
        return {k: e.get(k) for k in ["score", "required_skills", "nice_to_have_matched", "experience",
                                      "seniority", "location", "evidence_quotes"]} | {
            "required_skills": {"matched": e["required_skills"]["matched"], "missing": e["required_skills"]["missing"]}}

    def draft_outreach(self, a: DraftArgs, state: AgentState) -> dict:
        self._check_candidate(a.candidate_id)
        job_id, text = self._job_context(a.job_id, state)
        job = self.svc.job_struct_for_id(job_id) if job_id else self.svc.job_struct_for_text(text)
        fit = self.svc.score_candidates(job, text, [a.candidate_id])[0]
        title = job.get("title") or (state.current_query or "the role")
        facts = {"job_title": title, "matched_skills": fit["required_skills"]["matched"][:6],
                 "experience": fit["experience"]["summary"]}
        subject, body = f"Opportunity: {title}", None
        if self.llm is not None and self.draft_model:
            try:
                resp = self.llm.chat(self.draft_model, [{"role": "system", "content": DRAFT_PROMPT},
                                                        {"role": "user", "content": json.dumps(facts)}], tag="draft")
                d = parse_json(resp.content)
                subject, body = d.get("subject") or subject, d.get("body")
            except Exception:  # fall back to a template rather than failing the whole task
                body = None
        if not body:
            skills = ", ".join(facts["matched_skills"]) or "your background"
            body = (f"Hi there,\n\nI came across your profile and think your experience with {skills} could be a great "
                    f"fit for a {title} role we're hiring for. Would you be open to a short call this week?\n\nBest regards")
        draft = self.store.create_draft(a.candidate_id, job_id or (state.current_query or ""), subject, body)
        state.drafts.append(draft)
        return {"draft_id": draft["id"], "status": draft["status"], "subject": subject, "body": body,
                "note": "Saved as PENDING_APPROVAL. A human must approve it before anything is sent."}
