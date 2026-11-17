import pytest
from fastapi.testclient import TestClient

from talentlens.agent.outreach import OutreachStore
from talentlens.api.main import create_app
from tests.test_agent import FakeService, ScriptedLLM, call

CFG = {"agent": {"model": "m", "max_steps": 5, "max_tool_result_chars": 2000}}
JOB = "Accountant needed. Must know QuickBooks and month-end close. 3+ years of experience."


class CountingService(FakeService):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.resume_struct = {}

    def match_job(self, text, top_k=10, job_id=None, with_quotes=False):
        self.calls += 1
        return super().match_job(text, top_k, job_id)

    def match_resume(self, text, top_k=10):
        return {"results": [{"job_id": "J1", "score": 0.7}]}


def client_with(llm=None, tmp_path=None):
    svc = CountingService()
    app = create_app((CFG, svc, llm, OutreachStore(tmp_path / "o.json" if tmp_path else None)))
    return TestClient(app), svc


@pytest.fixture
def client(tmp_path):
    with client_with(tmp_path=tmp_path)[0] as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["resumes"] == 3


def test_match_job_and_cache(tmp_path):
    c, svc = client_with(tmp_path=tmp_path)
    with c:
        first = c.post("/match/job", json={"job_text": JOB, "top_k": 2})
        second = c.post("/match/job", json={"job_text": JOB, "top_k": 2})
    assert first.status_code == 200 and first.json()["cached"] is False
    assert first.json()["results"][0]["resume_id"] == "R10001"
    assert second.json()["cached"] is True and svc.calls == 1  # second answer came from the cache


def test_validation_errors(client):
    assert client.post("/match/job", json={"job_text": "too short"}).status_code == 422
    assert client.post("/match/job", json={"job_text": JOB, "top_k": 500}).status_code == 422
    assert client.post("/agent", json={}).status_code == 422


def test_match_resume(client):
    r = client.post("/match/resume", json={"resume_text": "Accountant with QuickBooks and payroll experience, 5 years."})
    assert r.status_code == 200 and r.json()["results"][0]["job_id"] == "J1"


def test_agent_without_llm_is_503(client):
    r = client.post("/agent", json={"request": "Find accountants"})
    assert r.status_code == 503 and r.json()["error"] == "agent unavailable"


def test_agent_draft_then_human_approval(tmp_path):
    llm = ScriptedLLM([[call("draft_outreach", {"candidate_id": "R10001", "job_id": "J1"})],
                       '{"subject": "Accountant role", "body": "Hi there, ..."}',  # the draft tool's own LLM call
                       "Draft is pending approval."])
    c, _ = client_with(llm=llm, tmp_path=tmp_path)
    with c:
        r = c.post("/agent", json={"request": "Draft outreach to R10001 for J1"})
        assert r.status_code == 200
        draft = r.json()["drafts"][0]
        assert draft["status"] == "PENDING_APPROVAL"
        assert c.get("/outreach", params={"status": "PENDING_APPROVAL"}).json()[0]["id"] == draft["id"]
        ok = c.post(f"/outreach/{draft['id']}/approve", json={"edited_body": "Edited by a human"})
        assert ok.status_code == 200 and ok.json()["status"] == "APPROVED" and ok.json()["body"] == "Edited by a human"
        assert c.post(f"/outreach/{draft['id']}/approve").status_code == 409  # can't approve twice
        assert c.post("/outreach/nope/approve").status_code == 404


def test_rate_limiter_window():
    from talentlens.api.main import RateLimiter
    t = [0.0]
    lim = RateLimiter(2, clock=lambda: t[0])
    assert lim.allow("a")[0] and lim.allow("a")[0]
    ok, retry = lim.allow("a")
    assert not ok and retry > 0
    assert lim.allow("b")[0]          # limits are per visitor
    t[0] = 3601
    assert lim.allow("a")[0]          # window has passed


def test_public_demo_rate_limits_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("TALENTLENS_PUBLIC_DEMO", "1")
    monkeypatch.setenv("RATE_LIMIT_MATCH_PER_HOUR", "2")
    c, _ = client_with(tmp_path=tmp_path)
    with c:
        codes = [c.post("/match/job", json={"job_text": JOB}, headers={"X-Forwarded-For": "1.2.3.4"}).status_code for _ in range(3)]
        other = c.post("/match/job", json={"job_text": JOB}, headers={"X-Forwarded-For": "5.6.7.8"}).status_code
        assert c.get("/health").json()["public_demo"] is True
    assert codes == [200, 200, 429] and other == 200


def test_cors_origin_regex(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGIN_REGEX", r"https://talentlens-.*\.vercel\.app")
    c, _ = client_with(tmp_path=tmp_path)
    with c:
        r = c.options("/match/job", headers={"Origin": "https://talentlens-git-main.vercel.app", "Access-Control-Request-Method": "POST"})
        bad = c.options("/match/job", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "https://talentlens-git-main.vercel.app"
    assert "access-control-allow-origin" not in bad.headers
