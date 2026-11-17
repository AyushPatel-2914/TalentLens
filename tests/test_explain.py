import json

import pytest

from talentlens.config import resolve
from talentlens.explain.explain import build_explanation
from talentlens.explain.fit import experience_fit, location_fit, match_skills, seniority_fit
from talentlens.explain.quotes import get_evidence_quotes, normalize_for_match, verify_quote
from talentlens.explain.scoring import final_score, sigmoid, weight_grid
from talentlens.extraction.skills import SkillNormalizer
from talentlens.llm.client import LLMResponse

RESUME = ("Senior accountant with 6 years of experience. Managed accounts payable and month-end close "
          "using QuickBooks and Excel. Prepared “quarterly” tax filings — on time.")


@pytest.fixture(scope="module")
def norm():
    return SkillNormalizer.from_yaml(resolve("configs/skill_aliases.yaml"))


# ---------- quote verification ----------

def test_exact_quote_verified():
    assert verify_quote("Managed accounts payable and month-end close", RESUME)


def test_quote_verification_normalises_whitespace_quotes_and_dashes():
    assert verify_quote('Prepared "quarterly"   tax filings - on time', RESUME)


def test_paraphrase_or_changed_word_rejected():
    assert not verify_quote("Managed accounts receivable and month-end close", RESUME)
    assert not verify_quote("Handled month-end close using QuickBooks", RESUME)


def test_quote_is_case_sensitive_and_needs_minimum_length():
    assert not verify_quote("managed accounts payable and month-end close", RESUME)
    assert not verify_quote("Excel", RESUME)  # too short to count as evidence


def test_normalize_for_match():
    assert normalize_for_match("  a’b \n\t c—d ") == "a'b c-d"


class FakeClient:
    def __init__(self, content):
        self.content = content

    def chat(self, model, messages, json_mode=True, tag=""):
        return LLMResponse(self.content, model, 1, 1, 1, 0, False)


def test_get_evidence_quotes_drops_hallucinations():
    reply = json.dumps({"quotes": [
        {"quote": "Managed accounts payable and month-end close", "supports": "AP"},
        {"quote": "Led a team of 40 auditors at a Big Four firm", "supports": "leadership"},  # invented
    ]})
    out = get_evidence_quotes(FakeClient(reply), "m", "Accountant", RESUME)
    assert [q["quote"] for q in out["kept"]] == ["Managed accounts payable and month-end close"]
    assert len(out["dropped"]) == 1 and out["error"] is None


def test_get_evidence_quotes_bad_json_is_an_error_not_a_crash():
    out = get_evidence_quotes(FakeClient("not json"), "m", "Accountant", RESUME)
    assert out["kept"] == [] and out["error"]


# ---------- fit rules ----------

def test_match_skills_uses_extracted_then_text(norm):
    m = match_skills(["Excel", "QuickBooks", "SAP", "Go"], ["MS Excel"], RESUME, norm)
    by = {x["skill"]: x for x in m}
    assert by["Microsoft Excel"]["via"] == "extracted"
    assert by["QuickBooks"]["via"] == "resume_text"
    assert not by["SAP"]["matched"]
    assert not by["Go"]["matched"]  # "go" alone is too ambiguous to find in free text


def test_experience_fit():
    assert experience_fit(6, 3) == (1.0, "has 6 years, needs 3+")
    assert experience_fit(2, 4)[0] == 0.5
    assert experience_fit(None, 4)[0] == 0.5
    assert experience_fit(1, None)[0] == 1.0


def test_seniority_and_location():
    assert seniority_fit("senior", "senior") == "match"
    assert seniority_fit("mid", "lead") == "above"
    assert seniority_fit("senior", "unknown") == "unknown"
    assert location_fit("Remote", None) == "remote"
    assert location_fit("Austin, TX", "Austin, Texas") == "match"
    assert location_fit("Austin, TX", None) == "unknown"


# ---------- scoring ----------

def test_final_score_formula_and_reweighting():
    w = {"reranker": 0.6, "required_skills": 0.3, "experience": 0.1}
    score, b = final_score(0.0, 0.5, 1.0, w)
    assert score == pytest.approx(0.6 * 0.5 + 0.3 * 0.5 + 0.1 * 1.0)
    # No required skills listed: their weight is spread over the rest.
    score2, b2 = final_score(0.0, None, 1.0, w)
    assert score2 == pytest.approx((0.6 * 0.5 + 0.1 * 1.0) / 0.7)
    assert sum(b2["weights_used"].values()) == pytest.approx(1.0)


def test_weight_grid_sums_to_one():
    grid = weight_grid(0.25)
    assert all(sum(g.values()) == pytest.approx(1) for g in grid)
    assert all(g["reranker"] > 0 for g in grid)
    assert sigmoid(0) == 0.5


def test_build_explanation(norm):
    job = {"title": "Accountant", "required_skills": ["QuickBooks", "SAP"], "nice_to_have_skills": ["Excel"],
           "min_years_experience": 3, "location": "Remote", "seniority": "senior"}
    resume = {"skills": ["Excel"], "total_years_experience": 6, "seniority": "senior", "location": None}
    e = build_explanation(job, resume, RESUME, 1.0, norm, {"reranker": 0.6, "required_skills": 0.3, "experience": 0.1})
    assert e["required_skills"]["matched"] == ["QuickBooks"] and e["required_skills"]["missing"] == ["SAP"]
    assert e["nice_to_have_matched"] == ["Microsoft Excel"]
    assert e["experience"]["fit"] == 1.0 and e["seniority"]["fit"] == "match" and e["location"]["fit"] == "remote"
    assert 0 < e["score"] < 1


def test_filler_words_in_requirements(norm):
    text = "Line cook. Certified in food safety and sanitation."
    m = match_skills(["Food Safety Knowledge", "Experience with Inventory Control"], ["Inventory Control"], text, norm)
    assert [x["matched"] for x in m] == [True, True]


def test_experience_phrases_are_not_skills():
    from talentlens.explain.fit import is_experience_requirement
    assert is_experience_requirement("3+ years accounting experience")
    assert is_experience_requirement("2+ years in HR") and is_experience_requirement("4-6 yrs in sales")
    assert not is_experience_requirement("QuickBooks") and not is_experience_requirement("Payroll Processing")


def test_payroll_processing_matches_payroll(norm):
    assert match_skills(["payroll processing"], [], "Ran weekly payroll for 300 staff.", norm)[0]["matched"]
