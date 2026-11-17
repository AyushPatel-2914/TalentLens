import pytest

from talentlens.fairness.counterfactual import (
    by_group, counterfactual_shift, make_variant, ranks, redact_identity, summarize,
)

WORDS = {"he": "she", "his": "her", "him": "her", "chairman": "chairwoman", "salesman": "saleswoman"}
TEXT = "Experienced salesman. He grew his territory by 30%. Elected chairman of the sales club."


def test_variant_changes_only_identity_signals():
    f = make_variant(TEXT, "Priya Sharma", "female", WORDS)
    assert f.startswith("Name: Priya Sharma\n")
    assert "saleswoman" in f and "She grew her territory by 30%" in f and "chairwoman" in f
    m = make_variant(f.split("\n", 1)[1], "Rahul Sharma", "male", WORDS)
    assert "He grew his territory by 30%" in m and "salesman" in m and "chairman" in m
    # Everything except the swapped words is identical.
    assert f.replace("saleswoman", "salesman").replace("She", "He").replace("her", "his").replace("chairwoman", "chairman") \
        .split("\n", 1)[1] == TEXT


def test_redaction_makes_variants_identical():
    names = ["Priya Sharma", "Rahul Sharma"]
    f = redact_identity(make_variant(TEXT, "Priya Sharma", "female", WORDS), WORDS, names)
    m = redact_identity(make_variant(TEXT, "Rahul Sharma", "male", WORDS), WORDS, names)
    assert f == m
    assert "Priya" not in f and "Name:" not in f and "They grew their territory" in f


def test_ranks_and_counterfactual_shift():
    base = {f"r{i}": 1.0 - i * 0.1 for i in range(12)}  # r0 best ... r11 worst
    assert ranks(base)["r0"] == 1 and ranks(base)["r11"] == 12
    up = counterfactual_shift(base, "r11", 0.95)        # worst candidate jumps to rank 2
    assert up["rank_shift"] == -10 and up["top_changed"] and up["score_diff"] == pytest.approx(0.95 - (1.0 - 1.1))
    same = counterfactual_shift(base, "r3", base["r3"])
    assert same == {"rank_shift": 0, "score_diff": 0.0, "top_changed": False}


def test_summary_and_direction():
    recs = [{"rank_shift": 2, "score_diff": -0.02, "top_changed": True, "group": "a"},
            {"rank_shift": 0, "score_diff": 0.0, "top_changed": False, "group": "b"}]
    s = summarize(recs)
    assert s["mean_abs_rank_shift"] == 1 and s["pct_top10_changed"] == 0.5 and s["max_abs_score_diff"] == pytest.approx(0.02)
    assert by_group(recs)["a"]["mean_score_diff"] == pytest.approx(-0.02)
