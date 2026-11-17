"""Stage 1: load, clean, de-identify and sample the datasets.

Usage:
    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py --job-target 3000 --config configs/config.yaml

Writes:
    data/processed/resumes.parquet
    data/processed/jobs.parquet
    results/data_stats.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd

from talentlens.config import load_config, resolve
from talentlens.data.load import find_file, load_jobs, load_resumes
from talentlens.data.pii import EMAIL_RE, PHONE_RE
from talentlens.data.prepare import prepare_jobs, prepare_resumes


def print_category_table(resume_counts: dict, job_counts: dict) -> None:
    table = pd.DataFrame({"resumes": resume_counts, "jobs": job_counts}).fillna(0).astype(int)
    table.loc["TOTAL"] = table.sum()
    print(table.to_string())


def leftover_pii(df: pd.DataFrame) -> dict[str, int]:
    """Verification: after redaction, no email or phone pattern should remain."""
    return {
        "email": int(df["text"].str.contains(EMAIL_RE).sum()),
        "phone": int(df["text"].str.contains(PHONE_RE).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--job-target", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, dcfg = cfg["paths"], cfg["data"]
    raw_dir = resolve(paths["raw_dir"])
    out_dir = resolve(paths["processed_dir"])
    results_dir = resolve(paths["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    job_target = args.job_target or dcfg["job_target"]

    t0 = time.perf_counter()
    resume_path = find_file(resolve(paths["raw_resumes"]), raw_dir)
    jobs_path = find_file(resolve(paths["raw_jobs"]), raw_dir)

    print(f"Loading resumes from {resume_path}")
    resumes, resume_stats = prepare_resumes(load_resumes(resume_path), dcfg["resume_min_chars"])

    print(f"Loading jobs from {jobs_path} (large file, this can take a minute)")
    jobs, job_stats = prepare_jobs(
        load_jobs(jobs_path),
        dcfg["category_rules"],
        target=job_target,
        seed=dcfg["seed"],
        min_chars=dcfg["job_min_chars"],
    )

    # Hard checks: a failure here means something upstream is wrong.
    assert resumes["resume_id"].is_unique, "resume IDs are not unique"
    assert jobs["job_id"].is_unique, "job IDs are not unique"
    resume_stats["leftover_pii_after_redaction"] = leftover_pii(resumes)
    job_stats["leftover_pii_after_redaction"] = leftover_pii(jobs)

    resumes.to_parquet(out_dir / "resumes.parquet", index=False)
    jobs.to_parquet(out_dir / "jobs.parquet", index=False)

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"seed": dcfg["seed"], "job_target": job_target,
                   "resume_min_chars": dcfg["resume_min_chars"], "job_min_chars": dcfg["job_min_chars"]},
        "sources": {"resumes": resume_path.name, "jobs": jobs_path.name},
        "resumes": resume_stats,
        "jobs": job_stats,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    with open(results_dir / "data_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n=== Counts per category (resumes vs sampled jobs) ===")
    print_category_table(resume_stats["per_category"], job_stats["per_category"])
    for name, s in (("Resumes", resume_stats), ("Jobs", job_stats)):
        print(f"\n=== {name} ===")
        print(f"rows: raw {s['raw_rows']} -> final {s['final_rows']}")
        print(f"dropped: too short {s['dropped_too_short']}, duplicate text {s['dropped_duplicate_text']}")
        print(f"length: mean {s['length']['mean_chars']} chars, median {s['length']['median_chars']} chars, "
              f"mean {s['length']['mean_words']} words")
        pii = s.get("pii_removed") or s.get("pii_removed_before_sampling")
        print(f"PII removed: {pii}; leftover after redaction: {s['leftover_pii_after_redaction']}")
    print(f"\nJobs whose title matched a category: {job_stats['title_matched_rows']} "
          f"(eligible after cleaning: {job_stats['eligible_rows']})")
    print(f"\nSaved {out_dir / 'resumes.parquet'}, {out_dir / 'jobs.parquet'}, {results_dir / 'data_stats.json'}")
    print(f"Done in {stats['runtime_seconds']} s")


if __name__ == "__main__":
    main()
