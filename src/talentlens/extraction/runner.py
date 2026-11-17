"""Shared setup for extraction scripts: config -> client, normaliser, documents."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from talentlens.config import ROOT, load_config, resolve
from talentlens.extraction.extract import format_job_input
from talentlens.extraction.skills import SkillNormalizer
from talentlens.llm.client import LLMClient


def make_client(cfg: dict) -> LLMClient:
    load_dotenv(ROOT / ".env")
    llm = cfg["llm"]
    return LLMClient(llm, cache_dir=resolve(llm["cache_dir"]), cost_log=resolve(llm["cost_log"]))


def make_normalizer(cfg: dict) -> SkillNormalizer:
    return SkillNormalizer.from_yaml(resolve(cfg["extraction"]["skill_aliases"]))


def load_documents(cfg: dict, kind: str) -> pd.DataFrame:
    """Return columns doc_id, text (the exact text sent to the LLM), category."""
    processed = resolve(cfg["paths"]["processed_dir"])
    if kind == "resume":
        df = pd.read_parquet(processed / "resumes.parquet")
        return pd.DataFrame({"doc_id": df["resume_id"], "text": df["text"], "category": df["category"]})
    df = pd.read_parquet(processed / "jobs.parquet")
    texts = [format_job_input(t, l, x) for t, l, x in zip(df["title"], df["location"], df["text"])]
    return pd.DataFrame({"doc_id": df["job_id"], "text": texts, "category": df["matched_category"]})


def output_path(cfg: dict, kind: str, model: str) -> Path:
    safe_model = model.replace("/", "__")
    return resolve("data/extracted") / f"{kind}s__{safe_model}.parquet"


def gold_ids_path() -> Path:
    return resolve("data/labels/gold_ids.txt")


def load_gold_ids(cfg: dict) -> list[str]:
    """100 resumes for hand-labelling, spread across categories. Created once, then reused."""
    path = gold_ids_path()
    if path.exists():
        return path.read_text(encoding="utf-8").split()
    docs = load_documents(cfg, "resume")
    n = cfg["extraction"]["gold_sample_size"]
    seed = cfg["data"]["seed"]
    per_cat = max(1, n // docs["category"].nunique())
    picked = docs.groupby("category", group_keys=False).apply(
        lambda g: g.sample(n=min(per_cat, len(g)), random_state=seed)
    )
    rest = docs.drop(picked.index)
    if len(picked) < n:
        picked = pd.concat([picked, rest.sample(n=n - len(picked), random_state=seed)])
    ids = sorted(picked["doc_id"].tolist())[:n]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return ids


__all__ = ["load_config", "make_client", "make_normalizer", "load_documents", "output_path", "load_gold_ids"]
