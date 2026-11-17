import pytest

from talentlens.eval.extraction_metrics import list_field_score, scalar_field_score


def test_list_field_micro_prf():
    gold = [{"python", "sql", "excel"}, {"payroll"}]
    pred = [{"python", "sql", "java"}, set()]
    s = list_field_score("skills", gold, pred)
    # TP = 2, FP = 1 (java), FN = 2 (excel, payroll)
    assert (s.tp, s.fp, s.fn) == (2, 1, 2)
    assert s.precision == pytest.approx(2 / 3)
    assert s.recall == pytest.approx(2 / 4)
    assert s.f1 == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))


def test_scalar_field_prf_and_accuracy():
    empty = lambda v: v is None
    gold = [5, 3, None, 10]
    pred = [5.5, None, None, 2]   # hit (within 1), miss (no prediction), both empty, wrong
    s = scalar_field_score("years", gold, pred, empty, lambda g, p: abs(g - p) <= 1)
    assert s.precision == pytest.approx(1 / 2)   # 2 predicted, 1 correct
    assert s.recall == pytest.approx(1 / 3)      # 3 gold values, 1 correct
    assert s.accuracy == pytest.approx(2 / 4)    # hit + both-empty


def test_lengths_must_match():
    with pytest.raises(ValueError):
        list_field_score("x", [set()], [])
