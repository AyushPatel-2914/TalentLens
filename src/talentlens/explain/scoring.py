"""Final match score: re-ranker relevance combined with hard requirements.

    final = w_r * sigmoid(reranker_logit) + w_s * required_skill_coverage + w_e * experience_fit

- sigmoid(logit) turns the re-ranker's raw score into 0..1 ("how relevant overall").
- required_skill_coverage = matched required skills / total required skills (0..1).
- experience_fit = 1 if years >= minimum, else years / minimum (see fit.py).

Why add the rule-based parts? The re-ranker sees only the first ~500 tokens and
has no notion of "this skill is mandatory". Coverage and experience are exactly
the checks a recruiter does, and each one can be shown and explained.

If a component is unavailable (e.g. the job lists no required skills), its
weight is redistributed over the others, so scores stay on the same 0..1 scale.
Weights are tuned on VALIDATION (scripts/06_explanations.py --tune), never on test.
"""

from __future__ import annotations

import math


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def final_score(reranker_logit: float | None, skill_coverage: float | None, experience: float | None,
                weights: dict[str, float], relevance: float | None = None) -> tuple[float, dict]:
    """`relevance` (0..1) is used in place of the re-ranker when no re-ranker is loaded
    (e.g. a small server): then the first-stage retrieval rank supplies the relevance signal."""
    if reranker_logit is not None:
        relevance = sigmoid(reranker_logit)
    parts = {
        "reranker": relevance,
        "required_skills": skill_coverage,
        "experience": experience,
    }
    used = {k: weights.get(k, 0.0) for k, v in parts.items() if v is not None and weights.get(k, 0.0) > 0}
    total = sum(used.values())
    if total == 0:
        return 0.0, {"components": parts, "weights_used": {}}
    score = sum(parts[k] * w for k, w in used.items()) / total
    return score, {"components": parts, "weights_used": {k: w / total for k, w in used.items()}}


def weight_grid(step: float = 0.1) -> list[dict[str, float]]:
    """All weight combinations on a simplex grid (they sum to 1), reranker weight > 0."""
    n = round(1 / step)
    grid = []
    for r in range(1, n + 1):
        for s in range(0, n - r + 1):
            e = n - r - s
            grid.append({"reranker": r / n, "required_skills": s / n, "experience": e / n})
    return grid
