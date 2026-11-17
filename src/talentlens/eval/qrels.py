"""Relevance labels ("qrels") for ranking evaluation.

Three label sources:
  human     your 200 hand labels (test split only): small, trusted
  judge     LLM-judge labels on all pool pairs: large
  combined  judge labels, with your label replacing the judge's where both exist

Returns {job_id: {resume_id: label}} for one split.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from talentlens.config import resolve


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["pair_id", "label"])
    return pd.read_json(path, lines=True).drop_duplicates("pair_id", keep="last")


def judge_labels_path(cfg: dict, model: str | None = None) -> Path:
    model = model or cfg["llm"]["models"]["strong"]
    return resolve(cfg["labels"]["judge_pairs_dir"]) / f"{model.replace('/', '__')}.jsonl"


def load_qrels(cfg: dict, source: str, split: str, judge_model: str | None = None) -> dict[str, dict[str, int]]:
    pools = pd.read_parquet(resolve(cfg["labels"]["pools_file"]))
    pools = pools[pools["split"] == split][["pair_id", "job_id", "resume_id"]]
    human = _read_jsonl(resolve(cfg["labels"]["human_pairs_file"]))[["pair_id", "label"]]
    judge = _read_jsonl(judge_labels_path(cfg, judge_model))[["pair_id", "label"]].dropna()

    if source == "category":
        return category_qrels(cfg, split)
    if source == "human":
        labels = human
    elif source == "judge":
        labels = judge
    elif source == "combined":
        labels = pd.concat([judge[~judge["pair_id"].isin(human["pair_id"])], human])
    else:
        raise ValueError(f"unknown label source {source!r}")

    df = pools.merge(labels, on="pair_id")
    qrels: dict[str, dict[str, int]] = {}
    for row in df.itertuples(index=False):
        qrels.setdefault(row.job_id, {})[row.resume_id] = int(row.label)
    return qrels


def category_qrels(cfg: dict, split: str) -> dict[str, dict[str, int]]:
    """Proxy labels, no human input: a resume is relevant (1) to a job when both belong to the
    same occupation category (the resume's dataset category vs the job's title-rule category).

    Coarse on purpose and reported as a proxy: it can't tell a strong fit from a weak one inside
    a category, and a mis-categorised job or a cross-category fit counts against every method.
    It covers every job in the split, not only pooled pairs.
    """
    processed = resolve(cfg["paths"]["processed_dir"])
    jobs = pd.read_parquet(processed / "jobs.parquet", columns=["job_id", "matched_category"])
    resumes = pd.read_parquet(processed / "resumes.parquet", columns=["resume_id", "category"])
    splits = load_splits(cfg)
    by_category: dict[str, dict[str, int]] = {}
    for row in resumes.itertuples(index=False):
        by_category.setdefault(row.category, {})[row.resume_id] = 1
    return {j.job_id: dict(by_category[j.matched_category]) for j in jobs.itertuples(index=False)
            if splits.get(j.job_id) == split and j.matched_category in by_category}


def available_sources(cfg: dict, proxy: bool = False) -> list[str]:
    """Label sources that have at least one label on disk (plus the category proxy if asked)."""
    out = []
    if len(_read_jsonl(resolve(cfg["labels"]["human_pairs_file"]))):
        out.append("human")
    if len(_read_jsonl(judge_labels_path(cfg))):
        out += ["judge", "combined"]
    if proxy:
        out.append("category")
    return out


def load_splits(cfg: dict) -> dict[str, str]:
    return json.loads(resolve(cfg["labels"]["splits_file"]).read_text(encoding="utf-8"))
