"""MatchingService: the full pipeline behind the API, the agent and the fairness test.

job text -> hybrid retrieval (top 100) -> cross-encoder re-rank -> rule-based fit
-> final score -> top-k with explanations (+ verified quotes for the top few).

Everything heavy (data, indexes, models) loads ONCE in __init__.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from talentlens.config import load_config, resolve
from talentlens.explain.explain import build_explanation
from talentlens.explain.fit import is_experience_requirement, skill_in_text
from talentlens.explain.quotes import get_evidence_quotes
from talentlens.extraction.extract import extract_one
from talentlens.extraction.runner import make_normalizer
from talentlens.llm.client import RateLimitExhausted
from talentlens.rerank.cross_encoder import first_words, pair_texts
from talentlens.retrieval.load import load_retrievers

_YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|yrs?)", re.IGNORECASE)
# Bump when matching/explanation logic changes, so cached match results are recomputed.
MATCH_LOGIC_VERSION = 3

LINKEDIN_LEVEL_TO_SENIORITY = {"Internship": "intern", "Entry level": "entry", "Associate": "mid",
                               "Mid-Senior level": "senior", "Director": "executive", "Executive": "executive"}


def _rows_to_dicts(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    df = df[df["status"].isin(["ok", "repaired"])]
    def clean(v):
        if hasattr(v, "tolist") and not isinstance(v, str):
            return list(v)
        if isinstance(v, float) and v != v:  # NaN (how parquet/pandas stores a missing value) -> None
            return None
        return v

    return {r["doc_id"]: {k: clean(v) for k, v in r.items()} for r in df.to_dict("records")}


class MatchingService:
    def __init__(self, cfg: dict | None = None, use_reranker: bool = True, llm=None):
        self.cfg = cfg or load_config()
        c = self.cfg
        processed = resolve(c["paths"]["processed_dir"])
        self.jobs = pd.read_parquet(processed / "jobs.parquet").set_index("job_id")
        self.resumes = pd.read_parquet(processed / "resumes.parquet").set_index("resume_id")
        self.normalizer = make_normalizer(c)
        self.llm = llm

        # Structured data: prefer the bulk extraction model, fall back to the strong model's rows.
        ext = resolve("data/extracted")
        self.resume_struct, self.job_struct = {}, {}
        for model in [c["llm"]["models"]["strong"], c["llm"]["models"]["extraction"]]:
            safe = model.replace("/", "__")
            self.resume_struct.update(_rows_to_dicts(ext / f"resumes__{safe}.parquet"))
            self.job_struct.update(_rows_to_dicts(ext / f"jobs__{safe}.parquet"))
        # Vocabulary comes from the LLM's (cleaner) skill lists, before rule-based rows are merged in.
        self.vocabulary = self._build_vocabulary()
        self.resume_struct = self._merge_rules(self.resume_struct, _rows_to_dicts(ext / "resumes__rules.parquet"))

        self.retrievers = load_retrievers(c, "resumes")
        _ = self.retrievers["dense"].model  # load the embedding model now, not on the first request
        try:
            self.job_retrievers = load_retrievers(c, "jobs")
        except FileNotFoundError:
            self.job_retrievers = None

        self.reranker = None
        if use_reranker:
            from talentlens.rerank.cross_encoder import Reranker
            rr = c["rerank"]
            self.reranker = Reranker(rr.get("finetuned_model") or rr["base_model"], rr["max_length"], rr["batch_size"])

        self.weights = self._load_weights()
        # Disk cache for match results: a warmed-up demo query is instant, even after a restart.
        # The version string changes whenever anything that affects results changes.
        self.cache_dir = resolve("data/cache/match")
        self.cache_version = json.dumps([MATCH_LOGIC_VERSION, getattr(self.reranker, "model_name", None), self.weights,
                                         len(self.resume_struct), len(self.job_struct), c["retrieval"], c["rerank"],
                                         c["extraction"]["prompt_version"], llm is not None], sort_keys=True, default=str)

    def _cached(self, kind: str, args: dict, compute):
        key = hashlib.sha256(json.dumps([kind, args, self.cache_version], sort_keys=True, default=str).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            out = json.loads(path.read_text(encoding="utf-8"))
            out["timings_ms"] = {**out.get("timings_ms", {}), "served_from_cache": True}
            return out
        out = compute()
        if out.pop("_incomplete", False):  # e.g. quotes skipped because of the LLM quota: recompute next time
            return out
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
        tmp.replace(path)
        return json.loads(path.read_text(encoding="utf-8"))

    # ---------- setup helpers ----------

    def _load_weights(self) -> dict[str, float]:
        import json
        tuned = resolve(self.cfg["paths"]["results_dir"]) / "score_weights.json"
        if tuned.exists():
            return json.loads(tuned.read_text(encoding="utf-8"))["weights"]
        return dict(self.cfg["scoring"]["weights"])

    @staticmethod
    def _merge_rules(llm_rows: dict[str, dict], rule_rows: dict[str, dict]) -> dict[str, dict]:
        """Every resume gets rule-based fields; LLM fields are layered on where they exist.
        Years and seniority ALWAYS come from the rules, so every candidate is measured the same way
        (and the LLM's v1 prompt had a wrong 'Current' year)."""
        merged = {}
        for rid in set(rule_rows) | set(llm_rows):
            base, llm = rule_rows.get(rid, {}), llm_rows.get(rid)
            if not llm:
                merged[rid] = {**base, "_source": "rules"}
                continue
            row = {**base, **{k: llm[k] for k in ("job_titles", "education_level") if llm.get(k) not in (None, [], "unknown")}}
            row["skills"] = list(dict.fromkeys((llm.get("skills") or []) + (base.get("skills") or [])))
            for k in ("total_years_experience", "seniority"):
                row[k] = base.get(k, llm.get(k))
            row["_source"] = "llm+rules" if base else "llm"
            merged[rid] = row
        return merged

    # Generic words that are in many skill lists but are not useful job requirements.
    GENERIC_SKILLS = frozenset(s.lower() for s in [
        "Management", "Communication", "Communication Skills", "Leadership", "Teamwork", "Problem Solving",
        "Time Management", "Organization", "Organizational Skills", "Multitasking", "Detail Oriented",
        "Interpersonal Skills", "Development", "Training", "Planning", "Research", "Reporting", "Operations"])

    def _build_vocabulary(self) -> list[str]:
        """Skill names used to read requirements from a job when no LLM extraction exists:
        the alias file's canonical names + skills that appear in at least 3 extracted resumes."""
        counts = Counter(s for r in self.resume_struct.values() for s in (r.get("skills") or []))
        frequent = [s for s, n in counts.items() if n >= 3 and 3 <= len(s) <= 40]
        return sorted(set(self.normalizer.canonical_names) | set(frequent))

    # ---------- structured views ----------

    def heuristic_job_struct(self, text: str, title: str = "", location: str | None = None,
                             level: str | None = None) -> dict:
        lower = text.lower()
        skills = [s for s in self.vocabulary
                  if s.lower() not in self.GENERIC_SKILLS and skill_in_text(s, lower, self.normalizer)][:20]
        years = [int(m.group(1)) for m in _YEARS_RE.finditer(text) if int(m.group(1)) <= 20]
        return {"title": title or text.split("\n", 1)[0][:80], "required_skills": skills, "nice_to_have_skills": [],
                "min_years_experience": float(min(years)) if years else None, "location": location,
                "seniority": LINKEDIN_LEVEL_TO_SENIORITY.get(level or "", "unknown"), "_source": "heuristic"}

    @staticmethod
    def _clean_requirements(job: dict) -> dict:
        """Experience phrases the LLM sometimes lists as skills ('3+ years in HR') are dropped:
        years are compared separately via min_years_experience."""
        for key in ("required_skills", "nice_to_have_skills"):
            job[key] = [s for s in (job.get(key) or []) if not is_experience_requirement(s)]
        return job

    def job_struct_for_id(self, job_id: str) -> dict:
        if job_id in self.job_struct:
            return self._clean_requirements({**self.job_struct[job_id], "_source": "llm"})
        j = self.jobs.loc[job_id]
        return self.heuristic_job_struct(j.text, j.title, j.location, j.experience_level)

    def job_struct_for_text(self, text: str) -> dict:
        """New job text (e.g. pasted in the UI): LLM extraction if available, else heuristic."""
        if self.llm is not None:
            try:
                r = extract_one(self.llm, "job", "adhoc", text, self.cfg["llm"]["models"]["extraction"],
                                self.normalizer, self.cfg["extraction"]["prompt_version"],
                                max_repairs=1, max_input_chars=self.cfg["llm"]["max_input_chars"])
                if r.data:
                    return self._clean_requirements({**r.data, "_source": "llm"})
            except Exception:  # LLM down or out of quota: the heuristic still gives a usable answer
                return {**self.heuristic_job_struct(text), "_source": "heuristic (LLM unavailable)"}
        return self.heuristic_job_struct(text)

    def resume_struct_for_id(self, resume_id: str) -> dict:
        return self.resume_struct.get(resume_id, {"skills": [], "total_years_experience": None, "job_titles": [],
                                                  "education_level": "unknown", "location": None, "seniority": "unknown"})

    def candidate_profile(self, resume_id: str) -> dict:
        r = self.resumes.loc[resume_id]
        s = self.resume_struct_for_id(resume_id)
        return {"resume_id": resume_id, "category": r.category, "skills": s.get("skills", [])[:25],
                "total_years_experience": s.get("total_years_experience"), "job_titles": s.get("job_titles", [])[:5],
                "education_level": s.get("education_level"), "location": s.get("location"),
                "seniority": s.get("seniority"), "has_structured_data": resume_id in self.resume_struct,
                "structured_by": s.get("_source", "none"),
                "summary": first_words(r.text, 60)}

    # ---------- matching ----------

    def score_candidates(self, job: dict, job_text: str, resume_ids: list[str],
                         resume_texts: dict[str, str] | None = None) -> list[dict]:
        """Score given candidates for a job. `resume_texts` lets the fairness test pass modified texts."""
        texts = resume_texts or {d: self.resumes.loc[d, "text"] for d in resume_ids}
        if self.reranker is not None:
            rr = self.cfg["rerank"]
            q, _ = pair_texts(job.get("title", ""), job_text, "", rr["job_words"], rr["resume_words"])
            logits = dict(zip(resume_ids, map(float, self.reranker.score(q, [first_words(texts[d], rr["resume_words"]) for d in resume_ids]))))
        else:
            logits = dict.fromkeys(resume_ids)
        out = []
        n = max(len(resume_ids), 1)
        for rank, d in enumerate(resume_ids):
            # Without a re-ranker, retrieval order (best first) supplies the relevance signal.
            relevance = None if logits[d] is not None else 1 - rank / n
            exp = build_explanation(job, self.resume_struct_for_id(d), texts[d], logits[d], self.normalizer,
                                    self.weights, relevance=relevance)
            out.append({"resume_id": d, "reranker_logit": logits[d], **exp})
        return sorted(out, key=lambda x: -x["score"])

    def match_job(self, job_text: str, top_k: int = 10, job_id: str | None = None, with_quotes: bool = False,
                  restrict_to: list[str] | None = None) -> dict:
        args = {"job_text": job_text, "top_k": top_k, "job_id": job_id, "with_quotes": with_quotes, "restrict_to": restrict_to}
        return self._cached("match_job", args, lambda: self._match_job(**args))

    def _match_job(self, job_text: str, top_k: int, job_id: str | None, with_quotes: bool,
                   restrict_to: list[str] | None) -> dict:
        t0 = time.perf_counter()
        job = self.job_struct_for_id(job_id) if job_id else self.job_struct_for_text(job_text)
        title = job.get("title", "")
        query = f"{title}\n{job_text}" if job_id else job_text
        depth = self.cfg["retrieval"]["depth"]
        candidates = [d for d, _ in self.retrievers["hybrid"].search(query, depth)]
        if self.reranker is not None:  # the cross-encoder is the slow part: only re-rank the top few on CPU
            candidates = candidates[: max(top_k, self.cfg["rerank"].get("interactive_depth", depth))]
        if restrict_to is not None:
            allowed = set(restrict_to)
            candidates = [d for d in candidates if d in allowed] + [d for d in restrict_to if d not in set(candidates)]
        t_retrieval = time.perf_counter()
        scored = self.score_candidates(job, job_text, candidates)[:top_k]
        t_rank = time.perf_counter()
        incomplete = job.get("_source") == "heuristic (LLM unavailable)"
        if with_quotes and self.llm is not None:
            brief = f"{title}. Required: {', '.join(job.get('required_skills', [])[:12])}"
            for item in scored[:3]:
                q = self.quotes_for(brief, item["resume_id"])
                if q is None:
                    item["quotes_note"] = "Evidence quotes unavailable right now (LLM quota)."
                    incomplete = True
                    continue
                item["evidence_quotes"], item["quotes_dropped_unverified"] = q["kept"], len(q["dropped"])
        for item in scored:
            item["candidate"] = self.candidate_profile(item["resume_id"])
        return {"job": job, "results": scored, "_incomplete": incomplete,
                "timings_ms": {"retrieval": round((t_retrieval - t0) * 1000, 1),
                               "rerank_and_explain": round((t_rank - t_retrieval) * 1000, 1),
                               "total": round((time.perf_counter() - t0) * 1000, 1)}}

    def match_resume(self, resume_text: str, top_k: int = 10) -> dict:
        """Reverse direction: resume -> best-fit jobs."""
        if self.job_retrievers is None:
            raise RuntimeError("Job indexes missing. Run: python scripts/04_build_indexes.py --corpus jobs")
        return self._cached("match_resume", {"resume_text": resume_text, "top_k": top_k},
                            lambda: self._match_resume(resume_text, top_k))

    def _match_resume(self, resume_text: str, top_k: int) -> dict:
        if self.job_retrievers is None:
            raise RuntimeError("Job indexes missing. Run: python scripts/04_build_indexes.py --corpus jobs")
        t0 = time.perf_counter()
        job_ids = [d for d, _ in self.job_retrievers["hybrid"].search(resume_text, self.cfg["retrieval"]["depth"])]
        rr = self.cfg["rerank"]
        resume_doc = first_words(resume_text, rr["resume_words"])
        logits = {}
        if self.reranker is not None:  # batched, and only the top few on CPU (like match_job)
            job_ids = job_ids[: max(top_k, rr.get("interactive_depth", len(job_ids)))]
            pairs = [(pair_texts(self.jobs.loc[j, "title"], self.jobs.loc[j, "text"], "", rr["job_words"],
                                 rr["resume_words"])[0], resume_doc) for j in job_ids]
            logits = dict(zip(job_ids, map(float, self.reranker.score_pairs(pairs))))
        resume = {"skills": [s for s in self.vocabulary if skill_in_text(s, resume_text.lower(), self.normalizer)],
                  "total_years_experience": None, "seniority": "unknown", "location": None}
        results = []
        for jid in job_ids:
            job = self.job_struct_for_id(jid)
            exp = build_explanation(job, resume, resume_text, logits.get(jid), self.normalizer, self.weights)
            j = self.jobs.loc[jid]
            results.append({"job_id": jid, "title": j.title, "company": j.company_name, "location": j.location,
                            "category": j.matched_category, **exp})
        results.sort(key=lambda x: -x["score"])
        return {"results": results[:top_k], "timings_ms": {"total": round((time.perf_counter() - t0) * 1000, 1)}}

    def explain(self, job_id: str, resume_id: str, with_quotes: bool = True) -> dict:
        job = self.job_struct_for_id(job_id)
        j = self.jobs.loc[job_id]
        item = self.score_candidates(job, j.text, [resume_id])[0]
        if with_quotes and self.llm is not None:
            brief = f"{job.get('title', '')}. Required: {', '.join(job.get('required_skills', [])[:12])}"
            q = self.quotes_for(brief, resume_id)
            if q is None:
                item["quotes_note"] = "Evidence quotes unavailable right now (LLM quota)."
            else:
                item["evidence_quotes"], item["quotes_dropped_unverified"] = q["kept"], len(q["dropped"])
        return {"job_id": job_id, "job_title": j.title, **item}

    def quotes_for(self, job_brief: str, resume_id: str) -> dict | None:
        """Verified evidence quotes, trying each configured model in turn (free-tier quotas are per model).
        Returns None if every model is out of quota, so the match is still shown, just without quotes."""
        sc = self.cfg["scoring"]
        for model in sc.get("quotes_models") or [sc["quotes_model"]]:
            try:
                return get_evidence_quotes(self.llm, model, job_brief, self.resumes.loc[resume_id, "text"],
                                           sc["max_quotes"], sc["min_quote_chars"])
            except RateLimitExhausted:
                continue
        return None

    def filter_candidates(self, resume_ids: list[str] | None = None, location: str | None = None,
                          min_years: float | None = None, skills: list[str] | None = None,
                          category: str | None = None) -> list[str]:
        ids = resume_ids if resume_ids is not None else list(self.resumes.index)
        out = []
        for d in ids:
            if d not in self.resumes.index:
                continue
            s = self.resume_struct_for_id(d)
            text = self.resumes.loc[d, "text"].lower()
            if category and self.resumes.loc[d, "category"].lower() != category.lower():
                continue
            if location and not (s.get("location") and location.lower() in s["location"].lower()):
                continue
            if min_years is not None and not (s.get("total_years_experience") is not None and s["total_years_experience"] >= min_years):
                continue
            if skills and not all(skill_in_text(self.normalizer.normalize(k), text, self.normalizer)
                                  or self.normalizer.normalize(k).lower() in {x.lower() for x in s.get("skills", [])}
                                  for k in skills):
                continue
            out.append(d)
        return out
