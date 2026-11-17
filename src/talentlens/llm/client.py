"""A small Groq chat client with caching, rate limiting, retries and cost logging.

Plain HTTP (httpx) against Groq's OpenAI-compatible endpoint, so every part of a
call is visible: request body, rate-limit headers, token usage.

- Cache: a response is stored under a hash of everything that affects it (model,
  messages, settings). The same call twice costs nothing the second time, and
  changing the prompt automatically misses the cache.
- Rate limiting: a 60-second sliding window over requests and tokens keeps us
  under the per-minute limits; HTTP 429 responses are retried after the
  server's `retry-after`.
- Daily limit: if the server asks us to wait longer than `max_wait_seconds`,
  we raise RateLimitExhausted instead of sleeping for hours. Scripts save what
  they have and can be re-run later; finished calls come from the cache.
- Cost log: every call (including cache hits, at cost 0) is appended to a CSV.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
COST_LOG_FIELDS = [
    "timestamp", "tag", "model", "cached", "input_tokens", "output_tokens",
    "latency_ms", "cost_usd",
]


class LLMError(RuntimeError):
    pass


class RateLimitExhausted(LLMError):
    """The provider wants us to wait too long (usually the daily quota)."""


class CacheMiss(LLMError):
    """Raised in cache-only mode when a call is not cached (no request is sent)."""


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    cached: bool
    tool_calls: list[dict] | None = None  # OpenAI-format tool calls, when tools were offered


class SlidingWindowLimiter:
    """Blocks until a new request fits within per-minute request/token budgets."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int, clock=time.monotonic, sleep=time.sleep):
        self.rpm, self.tpm = requests_per_minute, tokens_per_minute
        self.clock, self.sleep = clock, sleep
        self.events: deque[tuple[float, int]] = deque()  # (time, tokens)

    def _prune(self, now: float) -> None:
        while self.events and now - self.events[0][0] >= 60:
            self.events.popleft()

    def wait(self, estimated_tokens: int) -> None:
        estimated_tokens = min(estimated_tokens, self.tpm)  # a single huge call must still be allowed
        while True:
            now = self.clock()
            self._prune(now)
            used = sum(t for _, t in self.events)
            if len(self.events) < self.rpm and used + estimated_tokens <= self.tpm:
                return
            # Wait until the oldest event leaves the window.
            self.sleep(max(0.05, 60 - (now - self.events[0][0])))

    def record(self, tokens: int) -> None:
        self.events.append((self.clock(), tokens))


def message_chars(m: dict[str, Any]) -> int:
    """Size of one message for token estimates (tool-call messages have no plain content)."""
    content = m.get("content")
    return len(content) if isinstance(content, str) and "tool_calls" not in m else len(json.dumps(m))


def cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class LLMClient:
    def __init__(
        self,
        llm_cfg: dict[str, Any],
        api_key: str | None = None,
        cache_dir: Path | None = None,
        cost_log: Path | None = None,
        http: httpx.Client | None = None,
        max_wait_seconds: float = 120,
        sleep=time.sleep,
    ):
        self.cfg = llm_cfg
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.cache_dir = cache_dir
        self.cost_log = cost_log
        self.http = http or httpx.Client(base_url=llm_cfg["base_url"], timeout=120)
        self.max_wait_seconds = max_wait_seconds
        self.sleep = sleep
        # After the daily quota runs out, scripts switch this on so they can still
        # collect every answer that is already cached, without sending requests.
        self.cache_only = False
        limits = llm_cfg["rate_limits"]
        self._limiters: dict[str, SlidingWindowLimiter] = {}  # Groq limits are per model
        self._limits = (limits["requests_per_minute"], limits["tokens_per_minute"])

    # ---------- public ----------

    def chat(self, model: str, messages: list[dict[str, Any]], json_mode: bool = True, tag: str = "",
             tools: list[dict] | None = None) -> LLMResponse:
        """One chat completion. With `tools`, the reply may contain tool_calls instead of content."""
        body = self._build_body(model, messages, json_mode)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        key = cache_key(body)

        cached = self._cache_get(key)
        if cached is not None:
            resp = LLMResponse(**{**cached, "latency_ms": 0.0, "cost_usd": 0.0, "cached": True})
            self._log(tag, resp)
            return resp

        if self.cache_only:
            raise CacheMiss("not cached (cache-only mode)")
        if not self.api_key:
            raise LLMError("GROQ_API_KEY is not set. Put it in .env (see .env.example).")

        data, latency_ms = self._post_with_retries(model, body)
        usage = data.get("usage", {})
        message = data["choices"][0]["message"]
        resp = LLMResponse(
            content=message.get("content") or "",
            tool_calls=message.get("tool_calls") or None,
            model=model,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            cost_usd=self.cost(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
            cached=False,
        )
        self._cache_put(key, resp)
        self._log(tag, resp)
        return resp

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price = self.cfg.get("prices_per_million", {}).get(model)
        if not price:
            return 0.0
        return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000

    # ---------- internals ----------

    def _build_body(self, model: str, messages: list[dict[str, str]], json_mode: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.cfg.get("temperature", 0),
            "max_completion_tokens": self._max_output(messages),
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if model.startswith("openai/gpt-oss") and self.cfg.get("reasoning_effort"):
            body["reasoning_effort"] = self.cfg["reasoning_effort"]
            body["include_reasoning"] = False  # reasoning is still used, just not returned
        return body

    def _max_output(self, messages: list[dict[str, str]]) -> int:
        """Groq counts input + max_completion_tokens against the per-minute token limit,
        and refuses (HTTP 413) a single request bigger than that limit. For long inputs,
        lower the output cap so the request fits. Short inputs keep the configured cap,
        so their cache keys don't change."""
        max_out = self.cfg.get("max_output_tokens", 2048)
        budget = self.cfg["rate_limits"]["tokens_per_minute"] - 200
        est_input = int(sum(message_chars(m) for m in messages) / 3.2) + 50  # conservative chars->tokens
        if est_input + max_out > budget:
            max_out = max(512, budget - est_input)
        return max_out

    def _limiter(self, model: str) -> SlidingWindowLimiter:
        if model not in self._limiters:
            self._limiters[model] = SlidingWindowLimiter(*self._limits, sleep=self.sleep)
        return self._limiters[model]

    def _post_with_retries(self, model: str, body: dict[str, Any], max_attempts: int = 6) -> tuple[dict, float]:
        limiter = self._limiter(model)
        est_tokens = sum(message_chars(m) for m in body["messages"]) // 4 + 800 + len(json.dumps(body.get("tools", []))) // 4
        for attempt in range(1, max_attempts + 1):
            limiter.wait(est_tokens)
            t0 = time.perf_counter()
            try:
                r = self.http.post(
                    "/chat/completions", json=body, headers={"Authorization": f"Bearer {self.api_key}"}
                )
            except httpx.TransportError as e:
                limiter.record(0)
                if attempt == max_attempts:
                    raise LLMError(f"network error: {e}") from e
                self.sleep(min(2**attempt, 30))
                continue
            latency_ms = (time.perf_counter() - t0) * 1000

            if r.status_code == 200:
                data = r.json()
                limiter.record(int(data.get("usage", {}).get("total_tokens", est_tokens)))
                return data, latency_ms

            limiter.record(est_tokens if r.status_code == 429 else 0)
            if r.status_code not in RETRYABLE_STATUS or attempt == max_attempts:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
            wait = self._retry_after(r) or min(2**attempt, 30)
            if wait > self.max_wait_seconds:
                raise RateLimitExhausted(
                    f"Provider asks to wait {wait:.0f}s (likely the daily quota). "
                    f"Re-run later; finished calls are cached. Details: {r.text[:300]}"
                )
            self.sleep(wait)
        raise LLMError("unreachable")

    @staticmethod
    def _retry_after(r: httpx.Response) -> float | None:
        value = r.headers.get("retry-after")
        try:
            return float(value) if value else None
        except ValueError:
            return None

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / key[:2] / f"{key}.json" if self.cache_dir else None

    def _cache_get(self, key: str) -> dict | None:
        path = self._cache_path(key)
        if path and path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _cache_put(self, key: str, resp: LLMResponse) -> None:
        path = self._cache_path(key)
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "content": resp.content, "model": resp.model,
            "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
        }
        if resp.tool_calls:
            record["tool_calls"] = resp.tool_calls
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
        tmp.replace(path)  # atomic: an interrupted run never leaves a half-written cache file

    def _log(self, tag: str, resp: LLMResponse) -> None:
        if not self.cost_log:
            return
        self.cost_log.parent.mkdir(parents=True, exist_ok=True)
        new = not self.cost_log.exists()
        with open(self.cost_log, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COST_LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tag": tag, "model": resp.model, "cached": resp.cached,
                "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                "latency_ms": round(resp.latency_ms, 1), "cost_usd": round(resp.cost_usd, 8),
            })
