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


def available_sources(cfg: dict) -> list[str]:
    """Label sources that have at least one label on disk."""
    out = []
    if len(_read_jsonl(resolve(cfg["labels"]["human_pairs_file"]))):
        out.append("human")
    if len(_read_jsonl(judge_labels_path(cfg))):
        out += ["judge", "combined"]
    return out


def load_splits(cfg: dict) -> dict[str, str]:
    return json.loads(resolve(cfg["labels"]["splits_file"]).read_text(encoding="utf-8"))
