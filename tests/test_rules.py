import pytest

from talentlens.extraction.rules import (
    education_level, extract_resume, headline_title, merged_years, seniority, sections, skills_section_items,
    work_intervals,
)

RESUME = ("HR MANAGER Summary Experienced HR leader. Skills Payroll, Recruiting, HRIS, Million "
          "Experience HR Manager 01/2015 to Current Company Name - City , State Led hiring. "
          "HR Generalist Jan 2010 to Mar 2015 Company Name Training and development for staff. "
          "Recruiter 2008 - 2010 Agency "
          "Education Bachelor of Science : Business 2004 - 2008 State University")


def test_sections_found():
    names = [h for h, _, _ in sections(RESUME)]
    assert names[:4] == ["Header", "Summary", "Skills", "Experience"] and "Education" in names


def test_work_intervals_skip_education_and_merge_overlaps():
    ivs = work_intervals(RESUME, current=(2021, 6))
    years = {(s // 12, e // 12) for s, e in ivs}
    assert (2004, 2008) not in years                     # the degree dates are not work experience
    assert (2015, 2021) in years and (2010, 2015) in years and (2008, 2010) in years
    # 2008-01 .. 2021-05 continuous after merging: 13.4 years, not the double-counted sum
    assert merged_years(ivs) == pytest.approx(13.4, abs=0.1)


def test_in_sentence_training_is_not_a_section_break():
    # "Training and development" appears inside the 2010-2015 job; that job must still count.
    assert any(s // 12 == 2010 for s, _ in work_intervals(RESUME, current=(2021, 6)))


def test_merged_years_handles_overlap_and_empty():
    assert merged_years([(0, 24), (12, 36)]) == 3.0  # overlapping jobs count once
    assert merged_years([]) is None


def test_title_seniority_education():
    assert headline_title(RESUME) == "Hr Manager"
    assert seniority("Hr Manager", 13) == "lead"
    assert seniority("Accountant", 1) == "entry" and seniority("Accountant", 7) == "senior"
    assert seniority("Vice President of Sales", 3) == "executive"
    assert seniority(None, None) == "unknown"
    assert education_level(RESUME) == "bachelor"
    assert education_level("Education MBA : Finance") == "master"


def test_skills_section_and_extract():
    assert skills_section_items(RESUME)[:3] == ["Payroll", "Recruiting", "HRIS"]
    out = extract_resume(RESUME, lambda t: ["Payroll"], frequent_section_skills={"payroll", "hris"})
    assert out["skills"] == ["Payroll", "HRIS"]           # "Million" filtered (not frequent across resumes)
    assert out["location"] is None and out["education_level"] == "bachelor"
    assert out["total_years_experience"] == pytest.approx(13.4, abs=0.1)
