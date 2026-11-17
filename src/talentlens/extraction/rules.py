"""Rule-based resume extraction: free, instant and deterministic.

Used for ALL resumes (the free LLM tier can't extract 2,481 of them), and as a
baseline the LLM is compared against on the hand-labelled gold set.

- Years of experience: parse work date ranges in three formats ("01/2015 to
  Current", "Jan 2015 to Mar 2018", "2015 - 2018"), skip ranges inside
  Education/Certification sections, MERGE overlapping ranges (two jobs at once
  don't count double), and sum. "Current" = the dataset's collection date.
- Seniority: title keywords first (Director -> executive, Manager -> lead, ...),
  otherwise from years.
- Education: highest degree keyword, searched in the Education section.
- Skills: the resume's own "Skills" section items + known skill names found in the text.
- Location: the dataset anonymised it ("City , State"), so always None.
"""

from __future__ import annotations

import re
from collections import Counter

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
CURRENT = (2021, 6)  # resumes were collected around mid-2021 (latest common year in the data)

_END = r"(?:Current|Present|Now|Till Date|Today)"
_NUM_RANGE = re.compile(rf"\b(\d{{1,2}})/(\d{{4}})\s*(?:to|-|–)\s*(?:(\d{{1,2}})/(\d{{4}})|({_END}))", re.IGNORECASE)
_MON_RANGE = re.compile(
    rf"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{{4}})\s*(?:to|-|–)\s*"
    rf"(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{{4}})|({_END}))", re.IGNORECASE)
_YEAR_RANGE = re.compile(rf"\b((?:19|20)\d{{2}})\s*(?:to|-|–)\s*((?:19|20)\d{{2}}|{_END})\b", re.IGNORECASE)

HEADINGS = ["Professional Experience", "Work History", "Work Experience", "Experience", "Education and Training",
            "Education", "Certifications", "Licenses", "Skills", "Skill Highlights", "Highlights", "Summary",
            "Accomplishments", "Additional Information", "Affiliations", "Qualifications", "Core Qualifications",
            "Interests", "Languages", "Professional Summary", "Career Overview", "Executive Profile", "Training"]
_HEADING_RE = re.compile(r"\b(" + "|".join(map(re.escape, sorted(HEADINGS, key=len, reverse=True))) + r")\b")
# Only clear section names: words like "Training" also appear inside sentences in this flattened text,
# and treating them as headings would wrongly skip real work dates.
NON_WORK_SECTIONS = {"Education", "Education and Training", "Certifications"}

DEGREES = [  # highest first
    ("doctorate", r"\b(?:Ph\.?\s?D|Doctor of|Doctorate|Ed\.D|D\.B\.A|Juris Doctor|J\.D\.|M\.D\.)"),
    ("master", r"\b(?:Master'?s?|M\.S\.|M\.A\.|MBA|M\.B\.A|M\.Ed|MSc|M\.Sc|MPA|MSW)\b"),
    ("bachelor", r"\b(?:Bachelor'?s?|B\.S\.|B\.A\.|BSc|B\.Sc|BBA|B\.B\.A|B\.Ed|B\.Tech|BS in|BA in)"),
    ("associate", r"\b(?:Associate of|Associate'?s Degree|Associates? Degree|Associate in|A\.A\.S|A\.A\.|A\.S\.)"),
    ("high_school", r"\b(?:High School|GED|H\.S\. Diploma)\b"),
]

SENIORITY_TITLES = [
    ("intern", r"\b(?:intern|internship|trainee|student)\b"),
    ("executive", r"\b(?:director|vice president|vp|chief|ceo|cfo|coo|cto|president|owner|founder|partner|principal)\b"),
    ("lead", r"\b(?:manager|lead|supervisor|head|team leader|superintendent|coordinator)\b"),
    ("senior", r"\b(?:senior|sr\.?)\b"),
]

# Skills-section items that are words, not skills.
_SKILL_STOP = {"and", "sales and", "million", "skills", "skill", "management", "excellent", "strong", "ability", "time",
               "detail oriented", "hard working", "team player", "self-motivated", "organized", "highly organized"}


def sections(text: str) -> list[tuple[str, int, int]]:
    """[(heading, start, end)] in order of appearance; text before the first heading is 'Header'."""
    marks = [(m.group(1), m.start()) for m in _HEADING_RE.finditer(text)]
    if not marks or marks[0][1] > 0:
        marks.insert(0, ("Header", 0))
    return [(h, s, marks[i + 1][1] if i + 1 < len(marks) else len(text)) for i, (h, s) in enumerate(marks)]


def _section_at(pos: int, secs: list[tuple[str, int, int]]) -> str:
    for h, s, e in secs:
        if s <= pos < e:
            return h
    return "Header"


def _to_month(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def work_intervals(text: str, current: tuple[int, int] = CURRENT) -> list[tuple[int, int]]:
    secs = sections(text)
    now = _to_month(*current)
    found, used = [], []

    def add(start: int, end: int, pos: int, span: tuple[int, int]):
        used.append(span)
        if _section_at(pos, secs) in NON_WORK_SECTIONS:
            return
        if _to_month(1960, 1) <= start <= end <= now + 1 and end - start <= 50 * 12:
            found.append((start, end))

    for m in _NUM_RANGE.finditer(text):
        mo, y = int(m.group(1)), int(m.group(2))
        if not 1 <= mo <= 12:
            continue
        end = now if m.group(5) else _to_month(int(m.group(4)), min(max(int(m.group(3)), 1), 12))
        add(_to_month(y, mo), end, m.start(), m.span())
    for m in _MON_RANGE.finditer(text):
        end = now if m.group(5) else _to_month(int(m.group(4)), MONTHS[m.group(3).lower()])
        add(_to_month(int(m.group(2)), MONTHS[m.group(1).lower()]), end, m.start(), m.span())
    for m in _YEAR_RANGE.finditer(text):
        if any(s <= m.start() < e for s, e in used):  # already parsed as a month-level range
            continue
        end_raw = m.group(2)
        end = now if not end_raw[:1].isdigit() else _to_month(int(end_raw), 12)
        add(_to_month(int(m.group(1)), 1), end, m.start(), m.span())
    return found


def merged_years(intervals: list[tuple[int, int]]) -> float | None:
    """Union of month intervals in years: overlapping jobs are counted once."""
    if not intervals:
        return None
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return round(total / 12, 1)


def headline_title(text: str) -> str | None:
    """Resumes in this dataset start with the current title in capitals: 'PUBLIC RELATIONS DIRECTOR Summary ...'."""
    m = re.match(r"\s*([A-Z][A-Z0-9&/,.'()\- ]{2,80}?)(?=\s+(?:[A-Z][a-z]|$))", text)
    if not m:
        return None
    title = m.group(1).strip(" ,-/")
    return title.title() if len(title) >= 3 else None


def seniority(title: str | None, years: float | None) -> str:
    low = (title or "").lower()
    for level, pattern in SENIORITY_TITLES:
        if re.search(pattern, low):
            return level
    if years is None:
        return "unknown"
    return "entry" if years < 2 else ("mid" if years < 5 else "senior")


def education_level(text: str) -> str:
    edu = " ".join(text[s:e] for h, s, e in sections(text) if h in {"Education", "Education and Training"}) or text
    for level, pattern in DEGREES:
        if re.search(pattern, edu):
            return level
    return "unknown"


def skills_section_items(text: str) -> list[str]:
    out = []
    for h, s, e in sections(text):
        if h in {"Skills", "Skill Highlights", "Highlights", "Core Qualifications"}:
            for item in re.split(r"[,;•|]", text[s + len(h):e]):
                item = " ".join(item.split()).strip(" .:-")
                if 3 <= len(item) <= 40 and len(item.split()) <= 5 and item.lower() not in _SKILL_STOP:
                    out.append(item)
    return out


def extract_resume(text: str, vocabulary_match, frequent_section_skills: set[str] | None = None) -> dict:
    """`vocabulary_match(text) -> list[str]` finds known skills in text; section items are kept only if
    they are frequent across the corpus (filters one-off junk like 'Million')."""
    title = headline_title(text)
    years = merged_years(work_intervals(text))
    section = [s for s in skills_section_items(text) if frequent_section_skills is None or s.lower() in frequent_section_skills]
    skills = list(dict.fromkeys(vocabulary_match(text) + section))[:50]
    return {"skills": skills, "total_years_experience": years, "job_titles": [title] if title else [],
            "education_level": education_level(text), "location": None, "seniority": seniority(title, years)}


def frequent_section_items(texts, min_count: int = 5) -> set[str]:
    counts = Counter(s.lower() for t in texts for s in set(skills_section_items(t)))
    return {s for s, n in counts.items() if n >= min_count}
