"""Candidate pools for labelling (TREC-style pooling).

For each job we pick candidates from several sources, so labels don't favour
any one retrieval method and not every pair is obvious:

  bm25      top of the BM25 list           (lexically similar: often fits, sometimes keyword traps)
  dense     top of the embedding list      (semantically similar)
  hybrid    middle of the hybrid list      (borderline: the hard cases)
  random    random resumes                 (mostly clear non-fits, keeps the label mix honest)

The source is recorded with every pair, but hidden from you while labelling.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

# (source, how many, which ranks to draw from: (start, end), 0-based, end exclusive)
POOL_PLAN = [
    ("bm25", 3, (0, 20)),
    ("dense", 3, (0, 20)),
    ("hybrid", 2, (20, 100)),
    ("random", 2, None),
]


def build_pool(
    rankings: dict[str, Sequence[str]],
    all_ids: Sequence[str],
    rng: random.Random,
    plan=POOL_PLAN,
) -> list[tuple[str, str]]:
    """Return [(resume_id, source), ...] with no duplicates, in pick order."""
    picked: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source, n, window in plan:
        if source == "random":
            candidates = [d for d in all_ids if d not in seen]
        else:
            start, end = window
            candidates = [d for d in rankings[source][start:end] if d not in seen]
        for d in rng.sample(candidates, min(n, len(candidates))):
            picked.append((d, source))
            seen.add(d)
    return picked


def pick_for_human(pool: list[tuple[str, str]], n: int, rng: random.Random) -> list[str]:
    """Pick n pairs for hand-labelling, one per source first, so you see easy AND hard cases."""
    by_source: dict[str, list[str]] = {}
    for d, s in pool:
        by_source.setdefault(s, []).append(d)
    chosen = [rng.choice(ids) for ids in by_source.values()][:n]
    rest = [d for d, _ in pool if d not in chosen]
    chosen += rng.sample(rest, max(0, min(n - len(chosen), len(rest))))
    return chosen
