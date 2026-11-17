import pandas as pd
import pytest

from talentlens.config import load_config
from talentlens.data.prepare import prepare_jobs, prepare_resumes
from talentlens.data.sample import assign_category, balanced_allocation, compile_rules, sample_by_category


@pytest.fixture(scope="module")
def rules():
    return compile_rules(load_config()["data"]["category_rules"])


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Senior Software Engineer", "INFORMATION-TECHNOLOGY"),  # IT rule before ENGINEERING
        ("Mechanical Engineer", "ENGINEERING"),
        ("IT Support Specialist", "INFORMATION-TECHNOLOGY"),
        ("Sous Chef", "CHEF"),
        ("Sales Consultant", "SALES"),                           # SALES before CONSULTANT
        ("HR Generalist", "HR"),
        ("Registered Nurse - ICU", "HEALTHCARE"),
        ("Make It Happen Coordinator", None),                    # "It" is not the acronym IT
        ("Warehouse Associate", None),
    ],
)
def test_assign_category(rules, title, expected):
    assert assign_category(title, rules) == expected


def test_all_24_categories_configured():
    assert len(load_config()["data"]["category_rules"]) == 24


def test_balanced_allocation_water_fills():
    assert balanced_allocation({"A": 10, "B": 1000, "C": 1000}, 300) == {"A": 10, "B": 145, "C": 145}


def test_balanced_allocation_never_exceeds_available():
    alloc = balanced_allocation({"A": 3, "B": 4}, 100)
    assert alloc == {"A": 3, "B": 4}


def test_sample_is_reproducible():
    df = pd.DataFrame({"id": range(100), "cat": ["x"] * 50 + ["y"] * 50})
    a = sample_by_category(df, "cat", target=20, seed=42)
    b = sample_by_category(df, "cat", target=20, seed=42)
    assert a["id"].tolist() == b["id"].tolist()
    assert a["cat"].value_counts().to_dict() == {"x": 10, "y": 10}


def test_prepare_resumes_end_to_end():
    long = "Experienced accountant skilled in Excel and SAP. " * 10
    raw = pd.DataFrame(
        {
            "ID": [2, 1, 3, 4],
            "Resume_str": [long + " Email: a@b.com", long + " Call 555-123-4567", "too short", long + " Email: a@b.com"],
            "Category": ["ACCOUNTANT"] * 4,
        }
    )
    df, stats = prepare_resumes(raw, min_chars=200)
    assert df["resume_id"].tolist() == ["R1", "R2"]  # sorted; R3 too short; R4 duplicate of R2
    assert stats["dropped_too_short"] == 1
    assert stats["dropped_duplicate_text"] == 1
    assert "[EMAIL]" in df.loc[df.resume_id == "R2", "text"].item()
    assert "[PHONE]" in df.loc[df.resume_id == "R1", "text"].item()


def test_prepare_jobs_end_to_end():
    desc = "We need a cook with 2 years of kitchen experience. Contact hr@diner.com. " * 6
    raw = pd.DataFrame(
        {
            "job_id": ["10", "11", "12", "13"],
            "title": ["Line Cook", "Warehouse Associate", "Line Cook", "Sous Chef"],
            "company_name": ["Diner"] * 4,
            "location": ["Austin, TX"] * 4,
            "description": [desc, desc, desc, desc + " Weekends."],
            "skills_desc": [None] * 4,
            "formatted_experience_level": [None] * 4,
            "formatted_work_type": ["Full-time"] * 4,
            "remote_allowed": [None] * 4,
        }
    )
    rule_list = [{"category": "CHEF", "keywords": ["chef", "cook"]}]
    df, stats = prepare_jobs(raw, rule_list, target=10, seed=42, min_chars=100)
    assert stats["title_matched_rows"] == 3          # warehouse job not matched
    assert stats["dropped_duplicate_text"] == 1      # job 12 repeats job 10
    assert df["job_id"].tolist() == ["J10", "J13"]
    assert df["text"].str.contains("[EMAIL]", regex=False).all()
