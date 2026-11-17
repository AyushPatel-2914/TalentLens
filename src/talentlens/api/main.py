"""TalentLens API.

Run locally:
    uvicorn talentlens.api.main:app --reload          # http://localhost:8000/docs

Environment variables:
    GROQ_API_KEY            enables LLM features (job extraction for pasted text, quotes, agent)
    ALLOWED_ORIGINS         comma-separated CORS origins (e.g. https://talentlens.vercel.app)
    TALENTLENS_NO_RERANKER  set to 1 on small servers: skips the 1.1 GB cross-encoder
    ALLOWED_ORIGIN_REGEX    optional regex for extra CORS origins (e.g. Vercel preview URLs)
    TALENTLENS_PUBLIC_DEMO  set to 1 on a public server: per-visitor rate limits protect the LLM quota
    RATE_LIMIT_MATCH_PER_HOUR / RATE_LIMIT_AGENT_PER_HOUR   limits per visitor (defaults 60 / 15)

All models load ONCE at startup (lifespan). Repeated identical match requests are
served from an in-memory LRU cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from talentlens.agent.loop import run_agent
from talentlens.agent.outreach import OutreachStore
from talentlens.agent.tools import Toolbox
from talentlens.llm.client import LLMError, RateLimitExhausted


class MatchJobRequest(BaseModel):
    job_text: str = Field(min_length=30, max_length=20000, description="The job description")
    top_k: int = Field(default=10, ge=1, le=25)
    with_quotes: bool = Field(default=False, description="Ask the LLM for evidence quotes for the top 3 (verified)")


class MatchResumeRequest(BaseModel):
    resume_text: str = Field(min_length=30, max_length=30000)
    top_k: int = Field(default=10, ge=1, le=25)


class AgentRequest(BaseModel):
    request: str = Field(min_length=5, max_length=2000, examples=["Find 3 accountants who know QuickBooks and draft outreach for the best one."])


class DecisionRequest(BaseModel):
    edited_body: str | None = Field(default=None, max_length=5000)


class LRUCache:
    def __init__(self, size: int = 256):
        self.size, self.data = size, OrderedDict()
        self.hits = self.misses = 0

    @staticmethod
    def key(*parts: Any) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def get(self, key: str):
        if key in self.data:
            self.data.move_to_end(key)
            self.hits += 1
            return self.data[key]
        self.misses += 1
        return None

    def put(self, key: str, value) -> None:
        self.data[key] = value
        self.data.move_to_end(key)
        while len(self.data) > self.size:
            self.data.popitem(last=False)


class RateLimiter:
    """Per-visitor sliding-window limit (in memory). On a public demo, anyone could otherwise
    use up the free LLM quota; cached demo examples are still counted, but the limits are generous."""

    def __init__(self, per_hour: int, clock=time.monotonic):
        self.per_hour, self.clock, self.hits = per_hour, clock, {}

    def allow(self, key: str) -> tuple[bool, int]:
        now = self.clock()
        recent = [t for t in self.hits.get(key, []) if now - t < 3600]
        if len(recent) >= self.per_hour:
            self.hits[key] = recent
            return False, int(3600 - (now - recent[0])) + 1
        recent.append(now)
        self.hits[key] = recent
        return True, 0


def client_key(request: Request) -> str:
    # Behind a Cloudflare tunnel the visitor's IP is in CF-Connecting-IP; behind other proxies
    # (Hugging Face Spaces, Render) it is the first X-Forwarded-For entry.
    cf_ip = request.headers.get("cf-connecting-ip", "").strip()
    if cf_ip:
        return cf_ip
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _default_components():
    from dotenv import load_dotenv

    from talentlens.config import ROOT, load_config, resolve
    from talentlens.extraction.runner import make_client
    from talentlens.service import MatchingService

    from talentlens.artifacts import ensure_artifacts

    load_dotenv(ROOT / ".env")
    ensure_artifacts()  # downloads data/indexes on a fresh server if TALENTLENS_ARTIFACTS_REPO is set
    cfg = load_config()
    llm = make_client(cfg) if os.environ.get("GROQ_API_KEY") else None
    service = MatchingService(cfg, use_reranker=os.environ.get("TALENTLENS_NO_RERANKER") != "1", llm=llm)
    store = OutreachStore(resolve(cfg["agent"]["outreach_store"]))
    return cfg, service, llm, store


def create_app(components=None) -> FastAPI:
    """`components` = (cfg, service, llm, store). Tests pass fakes; production loads the real ones at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t0 = time.perf_counter()
        app.state.cfg, app.state.service, app.state.llm, app.state.store = components or _default_components()
        app.state.cache = LRUCache()
        app.state.startup_seconds = round(time.perf_counter() - t0, 1)
        yield

    app = FastAPI(title="TalentLens API", version="0.1.0", lifespan=lifespan,
                  description="Explainable job matching and a recruiter agent with human approval.")
    origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_origin_regex=os.environ.get("ALLOWED_ORIGIN_REGEX") or None,
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    public = os.environ.get("TALENTLENS_PUBLIC_DEMO") == "1"
    limits = {"match": RateLimiter(int(os.environ.get("RATE_LIMIT_MATCH_PER_HOUR", "60"))),
              "agent": RateLimiter(int(os.environ.get("RATE_LIMIT_AGENT_PER_HOUR", "15")))}

    def check_rate(request: Request, kind: str) -> None:
        if not public:
            return
        ok, retry = limits[kind].allow(client_key(request))
        if not ok:
            raise HTTPException(429, {"error": "rate limit", "detail": f"This public demo allows {limits[kind].per_hour} "
                                      f"{kind} requests per hour per visitor. Try again in {retry // 60 + 1} min."})

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=detail)

    def service(request: Request):
        svc = getattr(request.app.state, "service", None)
        if svc is None:
            raise HTTPException(503, {"error": "service not ready", "detail": "models are still loading"})
        return svc

    @app.get("/health")
    def health(request: Request) -> dict:
        svc = service(request)
        reranker = getattr(svc, "reranker", None)
        cache = request.app.state.cache
        return {"status": "ok", "public_demo": public, "startup_seconds": request.app.state.startup_seconds,
                "reranker": getattr(reranker, "model_name", None), "llm_enabled": request.app.state.llm is not None,
                "resumes": len(svc.resumes), "jobs": len(svc.jobs),
                "resumes_with_structured_data": len(getattr(svc, "resume_struct", {})),
                "cache": {"entries": len(cache.data), "hits": cache.hits, "misses": cache.misses}}

    @app.post("/match/job")
    def match_job(body: MatchJobRequest, request: Request) -> dict:
        check_rate(request, "match")
        svc, cache = service(request), request.app.state.cache
        key = cache.key("job", body.job_text, body.top_k, body.with_quotes)
        if (hit := cache.get(key)) is not None:
            return {**hit, "cached": True}
        try:
            out = svc.match_job(body.job_text, top_k=body.top_k, with_quotes=body.with_quotes)
        except RateLimitExhausted as e:
            raise HTTPException(429, {"error": "LLM quota exhausted", "detail": str(e)[:300]}) from e
        cache.put(key, out)
        return {**out, "cached": False}

    @app.post("/match/resume")
    def match_resume(body: MatchResumeRequest, request: Request) -> dict:
        check_rate(request, "match")
        svc, cache = service(request), request.app.state.cache
        key = cache.key("resume", body.resume_text, body.top_k)
        if (hit := cache.get(key)) is not None:
            return {**hit, "cached": True}
        try:
            out = svc.match_resume(body.resume_text, top_k=body.top_k)
        except RuntimeError as e:
            raise HTTPException(503, {"error": "reverse matching unavailable", "detail": str(e)}) from e
        cache.put(key, out)
        return {**out, "cached": False}

    @app.post("/agent")
    def agent(body: AgentRequest, request: Request) -> dict:
        check_rate(request, "agent")
        svc, llm, cfg = service(request), request.app.state.llm, request.app.state.cfg
        if llm is None:
            raise HTTPException(503, {"error": "agent unavailable", "detail": "GROQ_API_KEY is not set on the server"})
        a = cfg["agent"]
        toolbox = Toolbox(svc, request.app.state.store, llm, a["model"])
        result = run_agent(body.request, llm, toolbox, a["model"], a["max_steps"], a["max_tool_result_chars"],
                           a.get("fallback_models"))
        if result.stopped_reason == "rate_limit":
            raise HTTPException(429, {"error": "LLM quota exhausted", "detail": result.final_answer})
        if result.stopped_reason == "llm_error":
            raise HTTPException(502, {"error": "LLM provider error", "detail": result.final_answer})
        return result.to_dict()

    @app.get("/outreach")
    def list_outreach(request: Request, status: str | None = None) -> list[dict]:
        return request.app.state.store.list(status)

    def decide(request: Request, draft_id: str, approve: bool, body: DecisionRequest | None) -> dict:
        try:
            return request.app.state.store.decide(draft_id, approve, body.edited_body if body else None)
        except KeyError as e:
            raise HTTPException(404, {"error": "draft not found", "detail": draft_id}) from e
        except ValueError as e:
            raise HTTPException(409, {"error": "already decided", "detail": str(e)}) from e

    @app.post("/outreach/{draft_id}/approve")
    def approve(draft_id: str, request: Request, body: DecisionRequest | None = None) -> dict:
        """The ONLY way a draft becomes APPROVED: a human calling this (the UI's Approve button)."""
        return decide(request, draft_id, True, body)

    @app.post("/outreach/{draft_id}/reject")
    def reject(draft_id: str, request: Request) -> dict:
        return decide(request, draft_id, False, None)

    return app


app = create_app()
