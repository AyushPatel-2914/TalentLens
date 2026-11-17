"""Field-level precision / recall / F1 for extraction.

List fields (skills, job_titles): micro-averaged over all labelled resumes.
  TP = items in both gold and prediction, FP = predicted but not in gold,
  FN = in gold but not predicted. Items are compared case-insensitively after
  skill normalisation, so this is a STRICT match ("Excel" vs "Excel VBA" counts
  as a miss).

Scalar fields (years, education, seniority, location): per resume,
  "predicted" = the model gave a value (not null/unknown),
  "relevant"  = the gold label has a value,
  "correct"   = both have a value and they match.
  precision = correct / predicted, recall = correct / relevant.
  Accuracy (including "both empty" as agreement) is reported too.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class FieldScore:
    field: str
    n: int
    precision: float | None
    recall: float | None
    f1: float | None
    accuracy: float | None = None
    tp: int | None = None
    fp: int | None = None
    fn: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None or p + r == 0:
        return 0.0 if (p == 0 or r == 0) else None
    return 2 * p * r / (p + r)


def list_field_score(name: str, gold: Iterable[set[str]], pred: Iterable[set[str]]) -> FieldScore:
    tp = fp = fn = n = 0
    for g, p in zip(gold, pred, strict=True):
        n += 1
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return FieldScore(name, n, precision, recall, _f1(precision, recall), tp=tp, fp=fp, fn=fn)


def scalar_field_score(
    name: str,
    gold: Iterable[Any],
    pred: Iterable[Any],
    is_empty: Callable[[Any], bool],
    matches: Callable[[Any, Any], bool],
) -> FieldScore:
    n = predicted = relevant = correct = agree = 0
    for g, p in zip(gold, pred, strict=True):
        n += 1
        g_empty, p_empty = is_empty(g), is_empty(p)
        predicted += not p_empty
        relevant += not g_empty
        hit = not g_empty and not p_empty and matches(g, p)
        correct += hit
        agree += hit or (g_empty and p_empty)
    precision, recall = _ratio(correct, predicted), _ratio(correct, relevant)
    return FieldScore(name, n, precision, recall, _f1(precision, recall), accuracy=_ratio(agree, n))
