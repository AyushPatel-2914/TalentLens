"""Rule-based extraction for ALL resumes (free, ~1 minute). See src/talentlens/extraction/rules.py.

Usage:
    python scripts/02d_rules_extract.py

Writes data/extracted/resumes__rules.parquet (same columns as the LLM output, so
scripts/02c_eval_extraction.py scores it against your labels automatically as model "rules")
and results/rules_extraction.json (coverage stats + agreement with the LLM on the gold set).
"""

from __future__ import annotations

import json

import pandas as pd

from talentlens.config import load_config, resolve
from talentlens.explain.fit import skill_in_text
from talentlens.extraction.rules import extract_resume, frequent_section_items
from talentlens.extraction.runner import make_normalizer


def main() -> None:
    cfg = load_config()
    resumes = pd.read_parquet(resolve(cfg["paths"]["processed_dir"]) / "resumes.parquet")
    normalizer = make_normalizer(cfg)

    # Vocabulary: the alias file's canonical skills + skills the LLM found in >= 2 gold resumes.
    vocab = set(normalizer.canonical_names)
    llm_path = resolve("data/extracted") / f"resumes__{cfg['llm']['models']['strong'].replace('/', '__')}.parquet"
    llm = pd.read_parquet(llm_path) if llm_path.exists() else pd.DataFrame()
    if len(llm):
        counts = pd.Series([s for lst in llm["skills"].dropna() for s in lst]).value_counts()
        vocab |= {s for s, n in counts.items() if n >= 2 and 3 <= len(s) <= 40}
    vocab = sorted(vocab)

    def vocabulary_match(text: str) -> list[str]:
        low = text.lower()
        return [s for s in vocab if skill_in_text(s, low, normalizer)]

    frequent = frequent_section_items(resumes["text"])
    rows = []
    for r in resumes.itertuples(index=False):
        d = extract_resume(r.text, vocabulary_match, frequent)
        d["skills"] = normalizer.normalize_list(d["skills"])
        rows.append({"doc_id": r.resume_id, "status": "ok", "attempts": 0, "error": None, "truncated": False,
                     "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "fully_cached": True, **d})
    out = pd.DataFrame(rows)
    path = resolve("data/extracted") / "resumes__rules.parquet"
    out.to_parquet(path, index=False)

    stats = {
        "resumes": len(out),
        "with_years": int(out["total_years_experience"].notna().sum()),
        "with_title": int(out["job_titles"].map(len).gt(0).sum()),
        "with_education": int((out["education_level"] != "unknown").sum()),
        "seniority": out["seniority"].value_counts().to_dict(),
        "mean_skills_per_resume": round(float(out["skills"].map(len).mean()), 1),
        "vocabulary_size": len(vocab),
        "years_mean": round(float(out["total_years_experience"].mean()), 1),
    }
    if len(llm):  # agreement with the LLM on the gold resumes (NOT accuracy: that needs your labels)
        m = out.merge(llm[["doc_id", "total_years_experience", "education_level"]], on="doc_id", suffixes=("_rules", "_llm"))
        both = m.dropna(subset=["total_years_experience_rules", "total_years_experience_llm"])
        stats["agreement_with_llm_on_gold"] = {
            "resumes": len(m),
            "years_both_present": len(both),
            "years_within_1": round(float(((both.total_years_experience_rules - both.total_years_experience_llm).abs() <= 1).mean()), 3) if len(both) else None,
            "years_within_3": round(float(((both.total_years_experience_rules - both.total_years_experience_llm).abs() <= 3).mean()), 3) if len(both) else None,
            "education_exact": round(float((m.education_level_rules == m.education_level_llm).mean()), 3),
            "years_within_1_no_current_job": None,
            "note": "Gold LLM rows were made with prompt v1, which treated 'Current' as 2026 (rules: mid-2021), "
                    "so resumes with a current job differ by ~5 years. The no-current-job number is the fair comparison.",
        }
    if len(llm) and stats["agreement_with_llm_on_gold"]["years_both_present"]:
        texts = resumes.set_index("resume_id")["text"]
        nc = both[~both["doc_id"].map(texts).str.contains("to Current|to Present", case=False)]
        stats["agreement_with_llm_on_gold"]["years_within_1_no_current_job"] = (
            round(float(((nc.total_years_experience_rules - nc.total_years_experience_llm).abs() <= 1).mean()), 3) if len(nc) else None)
        stats["agreement_with_llm_on_gold"]["no_current_job_resumes"] = len(nc)
    (resolve(cfg["paths"]["results_dir"]) / "rules_extraction.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
