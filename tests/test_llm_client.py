import csv
import json

import httpx
import pytest

from talentlens.llm.client import LLMClient, LLMError, RateLimitExhausted, SlidingWindowLimiter

CFG = {
    "base_url": "https://fake.local/v1",
    "temperature": 0,
    "max_output_tokens": 100,
    "reasoning_effort": "low",
    "rate_limits": {"requests_per_minute": 1000, "tokens_per_minute": 10**9},
    "prices_per_million": {"m": {"input": 1.0, "output": 2.0}},
}
OK_BODY = {"choices": [{"message": {"content": '{"a": 1}'}}],
           "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}}


def make_client(responses, tmp_path, **kw):
    calls = []

    def handler(request: httpx.Request):
        calls.append(json.loads(request.content))
        status, body, headers = responses.pop(0)
        return httpx.Response(status, json=body, headers=headers)

    http = httpx.Client(base_url=CFG["base_url"], transport=httpx.MockTransport(handler))
    client = LLMClient(CFG, api_key="test", cache_dir=tmp_path / "cache", cost_log=tmp_path / "costs.csv",
                       http=http, sleep=lambda s: None, **kw)
    return client, calls


MSGS = [{"role": "user", "content": "hello"}]


def test_success_cost_and_log(tmp_path):
    client, calls = make_client([(200, OK_BODY, {})], tmp_path)
    r = client.chat("m", MSGS)
    assert r.content == '{"a": 1}' and not r.cached
    assert r.cost_usd == pytest.approx((1000 * 1 + 500 * 2) / 1e6)
    assert calls[0]["response_format"] == {"type": "json_object"}
    rows = list(csv.DictReader(open(tmp_path / "costs.csv")))
    assert rows[0]["input_tokens"] == "1000" and rows[0]["cached"] == "False"


def test_second_identical_call_is_cached_and_free(tmp_path):
    client, calls = make_client([(200, OK_BODY, {})], tmp_path)
    client.chat("m", MSGS)
    r2 = client.chat("m", MSGS)  # no response queued: would fail if it hit the network
    assert r2.cached and r2.cost_usd == 0 and r2.content == '{"a": 1}'
    assert len(calls) == 1


def test_different_messages_miss_cache(tmp_path):
    client, calls = make_client([(200, OK_BODY, {}), (200, OK_BODY, {})], tmp_path)
    client.chat("m", MSGS)
    client.chat("m", [{"role": "user", "content": "different"}])
    assert len(calls) == 2


def test_retries_429_then_succeeds(tmp_path):
    client, calls = make_client([(429, {"error": "slow down"}, {"retry-after": "2"}), (200, OK_BODY, {})], tmp_path)
    assert client.chat("m", MSGS).content == '{"a": 1}'
    assert len(calls) == 2


def test_long_retry_after_raises_exhausted(tmp_path):
    client, _ = make_client([(429, {"error": "daily limit"}, {"retry-after": "3600"})], tmp_path)
    with pytest.raises(RateLimitExhausted):
        client.chat("m", MSGS)


def test_non_retryable_error(tmp_path):
    client, calls = make_client([(400, {"error": "bad request"}, {})], tmp_path)
    with pytest.raises(LLMError, match="HTTP 400"):
        client.chat("m", MSGS)
    assert len(calls) == 1


def test_reasoning_params_only_for_gpt_oss(tmp_path):
    client, calls = make_client([(200, OK_BODY, {}), (200, OK_BODY, {})], tmp_path)
    client.chat("m", MSGS)
    client.chat("openai/gpt-oss-20b", MSGS)
    assert "reasoning_effort" not in calls[0]
    assert calls[1]["reasoning_effort"] == "low"


def test_missing_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = LLMClient(CFG, api_key="", cache_dir=None, cost_log=None)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        client.chat("m", MSGS)


def test_limiter_waits_when_token_budget_full():
    t = [0.0]
    slept = []
    lim = SlidingWindowLimiter(100, 1000, clock=lambda: t[0], sleep=lambda s: (slept.append(s), t.__setitem__(0, t[0] + s)))
    lim.wait(600); lim.record(600)
    t[0] = 10
    lim.wait(600)  # 600 + 600 > 1000 -> must wait until the first event is 60 s old
    assert slept and t[0] >= 60


def test_output_cap_shrinks_only_for_long_inputs(tmp_path):
    cfg = {**CFG, "max_output_tokens": 4096, "rate_limits": {"requests_per_minute": 30, "tokens_per_minute": 8000}}
    client = LLMClient(cfg, api_key="x", cache_dir=None, cost_log=None)
    short = [{"role": "user", "content": "a" * 4000}]     # ~1.3k tokens: keeps full cap
    long = [{"role": "user", "content": "a" * 16000}]     # ~5k tokens: cap reduced to fit 8000
    assert client._build_body("m", short, True)["max_completion_tokens"] == 4096
    capped = client._build_body("m", long, True)["max_completion_tokens"]
    assert capped < 4096 and 16000 / 3.2 + capped <= 8000


def test_tool_calls_returned_and_cached(tmp_path):
    tool_body = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"q": "x"}'}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    client, calls = make_client([(200, tool_body, {})], tmp_path)
    tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    r = client.chat("m", MSGS, json_mode=False, tools=tools)
    assert r.tool_calls[0]["function"]["name"] == "search" and r.content == ""
    assert calls[0]["tools"] == tools and calls[0]["tool_choice"] == "auto"
    again = client.chat("m", MSGS, json_mode=False, tools=tools)
    assert again.cached and again.tool_calls == r.tool_calls
