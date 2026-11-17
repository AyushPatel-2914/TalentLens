"""Stage 3, step 1: split jobs into train/validation/test BY JOB, then choose the pairs to label.

Usage:
    python scripts/03_make_splits_and_pools.py

Needs the Stage 4 indexes (python scripts/04_build_indexes.py), because candidates
are pooled from BM25, dense and hybrid results plus random resumes.

Writes:
    data/labels/splits.json        job_id -> train | validation | test
    data/labels/pools.parquet      one row per (job, resume) pair to label, with source and split
    results/splits.json            counts per split
"""

from __future__ import annotations

import json
import random

import pandas as pd
from tqdm import tqdm

from talentlens.config import load_config, resolve
from talentlens.data.sample import sample_by_category
from talentlens.labeling.pools import build_pool, pick_for_human
from talentlens.labeling.splits import split_jobs
from talentlens.retrieval.load import load_retrievers


def job_query(row) -> str:
    return f"{row.title}\n{row.text}"


def main() -> None:
    cfg = load_config()
    lab, seed = cfg["labels"], cfg["data"]["seed"]
    processed = resolve(cfg["paths"]["processed_dir"])
    jobs = pd.read_parquet(processed / "jobs.parquet")
    resume_ids = pd.read_parquet(processed / "resumes.parquet")["resume_id"].tolist()

    # 1. Split all 3,000 jobs by job.
    splits = split_jobs(jobs, lab["splits"], seed)
    jobs["split"] = jobs["job_id"].map(splits)
    splits_path = resolve(lab["splits_file"])
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits, indent=0), encoding="utf-8")

    # 2. Choose which jobs get labelled. Human jobs come from TEST, so the trusted
    #    human labels can be used to report test metrics without relying on the judge.
    test = jobs[jobs["split"] == "test"]
    human_jobs = sample_by_category(test, "matched_category", lab["human_jobs"], seed)
    remaining = jobs.drop(human_jobs.index)
    judge_parts = []
    for name, frac in lab["splits"].items():
        part = remaining[remaining["split"] == name]
        judge_parts.append(sample_by_category(part, "matched_category", round(lab["judge_jobs"] * frac), seed))
    judge_jobs = pd.concat(judge_parts)
    selected = pd.concat([human_jobs.assign(job_set="human"), judge_jobs.assign(job_set="judge")])

    # 3. Pool candidates for each selected job.
    retrievers = load_retrievers(cfg, "resumes")
    rows = []
    for job in tqdm(selected.itertuples(index=False), total=len(selected), desc="pooling"):
        q = job_query(job)
        rankings = {
            "bm25": [d for d, _ in retrievers["bm25"].search(q, 20)],
            "dense": [d for d, _ in retrievers["dense"].search(q, 20)],
            "hybrid": [d for d, _ in retrievers["hybrid"].search(q, 100)],
        }
        rng = random.Random(f"{seed}-{job.job_id}")  # per-job seed: adding jobs never changes old pools
        pool = build_pool(rankings, resume_ids, rng)
        human = set(pick_for_human(pool, lab["human_per_job"], rng)) if job.job_set == "human" else set()
        for resume_id, source in pool:
            rows.append({"pair_id": f"{job.job_id}__{resume_id}", "job_id": job.job_id, "resume_id": resume_id,
                         "source": source, "split": job.split, "job_set": job.job_set,
                         "for_human": resume_id in human})

    pools = pd.DataFrame(rows)
    pools.to_parquet(resolve(lab["pools_file"]), index=False)

    # Leakage check: no job may appear in two splits.
    assert pools.groupby("job_id")["split"].nunique().max() == 1

    summary = {
        "jobs_per_split": jobs["split"].value_counts().to_dict(),
        "labelled_jobs_per_split": selected["split"].value_counts().to_dict(),
        "pairs_total": len(pools),
        "pairs_per_split": pools["split"].value_counts().to_dict(),
        "pairs_for_human": int(pools["for_human"].sum()),
        "pairs_per_source": pools["source"].value_counts().to_dict(),
        "human_pairs_per_source": pools[pools["for_human"]]["source"].value_counts().to_dict(),
    }
    (resolve(cfg["paths"]["results_dir"]) / "splits.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {splits_path} and {resolve(lab['pools_file'])}")
    print("Next: python scripts/03b_label_pairs.py")


if __name__ == "__main__":
    main()
