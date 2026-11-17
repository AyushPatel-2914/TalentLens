import json
import random

import pandas as pd
import pytest

from talentlens.labeling.judge import agreement, judge_pair
from talentlens.labeling.pools import build_pool, pick_for_human
from talentlens.labeling.splits import split_jobs
from talentlens.llm.client import LLMResponse


def make_jobs(n_per_cat=20, cats=("A", "B", "C")):
    return pd.DataFrame([{"job_id": f"J{c}{i}", "matched_category": c} for c in cats for i in range(n_per_cat)])


def test_split_is_by_job_complete_and_deterministic():
    jobs = make_jobs()
    s1 = split_jobs(jobs, {"train": 0.7, "validation": 0.15, "test": 0.15}, seed=42)
    s2 = split_jobs(jobs, {"train": 0.7, "validation": 0.15, "test": 0.15}, seed=42)
    assert s1 == s2
    assert set(s1) == set(jobs["job_id"])          # every job in exactly one split
    counts = pd.Series(s1).value_counts().to_dict()
    assert counts == {"train": 42, "validation": 9, "test": 9}


def test_split_is_stratified():
    jobs = make_jobs()
    s = split_jobs(jobs, {"train": 0.7, "validation": 0.15, "test": 0.15}, seed=1)
    jobs["split"] = jobs["job_id"].map(s)
    per_cat_test = jobs[jobs.split == "test"].groupby("matched_category").size()
    assert per_cat_test.min() >= 2  # each category represented in test


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        split_jobs(make_jobs(), {"train": 0.5, "test": 0.3}, seed=1)


def test_pool_sources_unique_and_windows():
    rankings = {"bm25": [f"b{i}" for i in range(30)], "dense": ["b0", "b1"] + [f"d{i}" for i in range(30)],
                "hybrid": [f"h{i}" for i in range(100)]}
    all_ids = [f"r{i}" for i in range(50)] + rankings["bm25"] + rankings["hybrid"]
    pool = build_pool(rankings, all_ids, random.Random(0))
    ids = [d for d, _ in pool]
    assert len(ids) == len(set(ids)) == 10
    hybrid_picks = [d for d, s in pool if s == "hybrid"]
    assert all(rankings["hybrid"].index(d) >= 20 for d in hybrid_picks)  # middle of the list only
    assert build_pool(rankings, all_ids, random.Random(0)) == pool        # deterministic


def test_human_picks_cover_every_source():
    pool = [("a", "bm25"), ("b", "bm25"), ("c", "dense"), ("d", "hybrid"), ("e", "random"), ("f", "random")]
    chosen = pick_for_human(pool, 4, random.Random(0))
    sources = {dict(pool)[d] for d in chosen}
    assert len(chosen) == 4 and sources == {"bm25", "dense", "hybrid", "random"}


def test_agreement_metrics():
    perfect = agreement([0, 1, 2, 2], [0, 1, 2, 2])
    assert perfect["cohen_kappa"] == pytest.approx(1.0)
    assert perfect["confusion_matrix"]["rows_human_cols_judge"] == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
    # Off-by-one disagreements are punished less by the weighted kappa than 0-vs-2 ones.
    near = agreement([0, 1, 2, 0, 1, 2], [1, 1, 2, 0, 2, 2])
    far = agreement([0, 1, 2, 0, 1, 2], [2, 1, 2, 0, 1, 0])
    assert near["weighted_kappa_quadratic"] > far["weighted_kappa_quadratic"]
    with pytest.raises(ValueError):
        agreement([], [])


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, model, messages, json_mode=True, tag=""):
        return LLMResponse(self.replies.pop(0), model, 1, 1, 1, 0, False)


def test_judge_parses_and_repairs():
    out, err = judge_pair(FakeClient([json.dumps({"reason": "fits", "label": 2})]), "m", "job", "cv", 100, 100)
    assert out.label == 2 and err is None
    out, err = judge_pair(FakeClient(['{"reason": "x", "label": 5}', '{"reason": "x", "label": 1}']),
                          "m", "job", "cv", 100, 100)
    assert out.label == 1
    out, err = judge_pair(FakeClient(["bad", "bad"]), "m", "job", "cv", 100, 100)
    assert out is None and err
