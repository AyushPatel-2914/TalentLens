"""Counterfactual variants, redaction (the mitigation) and rank-shift metrics.

A variant changes ONLY identity signals: a name line at the top of the resume
(the dataset has names removed; real resumes start with one) and gender-coded
words. Skills, experience and everything else stay identical, so any score
change is caused by the identity signal alone.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

import numpy as np

NAME_LINE_RE = re.compile(r"^\s*Name:[^\n]*\n?", re.IGNORECASE)


def _swap_words(text: str, mapping: dict[str, str]) -> str:
    """Replace whole words, keeping capitalisation ("He" -> "She")."""
    if not mapping:
        return text
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(mapping, key=len, reverse=True))) + r")\b", re.IGNORECASE)

    def repl(m: re.Match) -> str:
        word = m.group(0)
        new = mapping[word.lower()]
        return new.capitalize() if word[0].isupper() else new

    return pattern.sub(repl, text)


def make_variant(text: str, name: str, gender: str, gendered_words: dict[str, str]) -> str:
    """Add a name line and make gender-coded words match `gender`."""
    if gender == "female":
        mapping = dict(gendered_words)
    else:
        mapping = {f: m for m, f in gendered_words.items() if f not in {"her"}}
        mapping["her"] = "his"  # "her" is ambiguous (his/him); "his" is the more common resume use
    return f"Name: {name}\n" + _swap_words(text, mapping)


def redact_identity(text: str, gendered_words: dict[str, str], extra_names: Iterable[str] = ()) -> str:
    """Mitigation: remove the name line and known names, and neutralise gender-coded words before scoring."""
    text = NAME_LINE_RE.sub("", text)
    for name in extra_names:
        for part in name.split():
            text = re.sub(rf"\b{re.escape(part)}\b", "", text)
    neutral = {w: "they" for w in ["he", "she"]} | {w: "them" for w in ["him"]} | {w: "their" for w in ["his", "her"]}
    neutral |= {w: "themself" for w in ["himself", "herself"]}
    for m, f in gendered_words.items():
        if m not in neutral:
            neutral[m] = neutral.get(m, m)       # keep the male form as a neutral base ...
            neutral[f] = neutral.get(f, m)       # ... and map the female form onto it
    return re.sub(r"[ \t]{2,}", " ", _swap_words(text, neutral)).strip()


def ranks(scores: dict[str, float]) -> dict[str, int]:
    """1-based rank by score (higher first); ties broken by ID for determinism."""
    order = sorted(scores, key=lambda d: (-scores[d], d))
    return {d: i for i, d in enumerate(order, start=1)}


def counterfactual_shift(baseline: dict[str, float], candidate: str, new_score: float, top_n: int = 10) -> dict:
    """Swap ONE candidate's score, keep everyone else at baseline, and measure what changes."""
    before = ranks(baseline)
    after_scores = dict(baseline)
    after_scores[candidate] = new_score
    after = ranks(after_scores)
    top_before = sorted(before, key=before.get)[:top_n]
    top_after = sorted(after, key=after.get)[:top_n]
    return {"rank_shift": after[candidate] - before[candidate], "score_diff": new_score - baseline[candidate],
            "top_changed": top_before != top_after}


def summarize(records: list[dict]) -> dict:
    if not records:
        return {}
    shifts = np.array([r["rank_shift"] for r in records], dtype=float)
    diffs = np.array([r["score_diff"] for r in records], dtype=float)
    return {"counterfactuals": len(records),
            "mean_abs_rank_shift": float(np.abs(shifts).mean()),
            "max_abs_rank_shift": int(np.abs(shifts).max()),
            "pct_top10_changed": float(np.mean([r["top_changed"] for r in records])),
            "mean_abs_score_diff": float(np.abs(diffs).mean()),
            "max_abs_score_diff": float(np.abs(diffs).max())}


def by_group(records: list[dict], key: str = "group") -> dict[str, dict]:
    """Signed mean score difference per group: shows the DIRECTION of any bias."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r[key], []).append(r)
    return {g: {"mean_score_diff": float(np.mean([r["score_diff"] for r in rs])),
                "mean_rank_shift": float(np.mean([r["rank_shift"] for r in rs])), "n": len(rs)}
            for g, rs in sorted(groups.items())}


ScoreFn = Callable[[str, dict[str, str]], dict[str, float]]
