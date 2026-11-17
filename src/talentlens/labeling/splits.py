"""Train / validation / test split BY JOB.

Every pair for a job goes into that job's split. If a job appeared in both
train and test, the re-ranker could memorise that job's wording and its good
candidates during training and "recognise" them at test time. That inflates
the scores without the model learning anything general (data leakage).
Splitting by job tests what a real system must do: rank candidates for a job
it has never seen.

The split is stratified by category, so every split has a similar mix of
occupations, and it is seeded, so it is the same on every run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_jobs(jobs: pd.DataFrame, fractions: dict[str, float], seed: int,
               id_col: str = "job_id", strata_col: str = "matched_category") -> dict[str, str]:
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    rng = np.random.default_rng(seed)
    names = list(fractions)
    assignment: dict[str, str] = {}
    for _, group in jobs.sort_values(id_col).groupby(strata_col, sort=True):
        ids = group[id_col].to_numpy().copy()
        rng.shuffle(ids)
        # Cumulative cut points, rounded, so small categories still get split sensibly.
        cuts = np.round(np.cumsum([fractions[n] for n in names]) * len(ids)).astype(int)
        start = 0
        for name, end in zip(names, cuts):
            for job_id in ids[start:end]:
                assignment[str(job_id)] = name
            start = end
    return assignment
