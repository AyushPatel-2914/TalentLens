import numpy as np
import pytest

from talentlens.eval.ranking_metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank
from talentlens.retrieval.bm25 import BM25Index
from talentlens.retrieval.dense import DenseIndex
from talentlens.retrieval.hybrid import HybridRetriever, rrf
from talentlens.retrieval.text import chunk_words, tokenize

# ---------- RRF ----------

def test_rrf_readme_example():
    # README 4.4: A is rank 1 in BM25 and rank 10 in dense; B is rank 3 in both. B must win.
    bm25 = ["A", "x1", "B"] + [f"y{i}" for i in range(10)]
    dense = ["x2", "x3", "B"] + [f"z{i}" for i in range(6)] + ["A"]
    fused = dict(rrf([bm25, dense], k=60))
    assert fused["A"] == pytest.approx(1 / 61 + 1 / 70)
    assert fused["B"] == pytest.approx(2 / 63)
    assert fused["B"] > fused["A"]


def test_rrf_small_k_favours_single_top_rank():
    lists = [["A", "x", "B"], ["C", "y", "B"]]  # A first in one list; B third in both
    assert rrf(lists, k=0)[0][0] == "A"   # k=0: A = 1/1 = 1.0 beats B = 1/3 + 1/3 = 0.67
    assert rrf(lists, k=60)[0][0] == "B"  # k=60: B = 2/63 = 0.032 beats A = 1/61 = 0.016


def test_rrf_doc_in_one_list_only_and_determinism():
    out = rrf([["A"], ["B"]], k=60)
    assert [d for d, _ in out] == ["A", "B"]  # equal score -> first seen first
    assert out == rrf([["A"], ["B"]], k=60)


def test_rrf_rejects_negative_k():
    with pytest.raises(ValueError):
        rrf([["A"]], k=-1)


# ---------- metrics (README section 5 worked examples) ----------

def test_ndcg_readme_example():
    labels = {"a": 2, "b": 0, "c": 1}
    assert ndcg_at_k(["a", "b", "c"], labels, k=10) == pytest.approx(3.5 / (3 + 1 / np.log2(3)))
    assert ndcg_at_k(["a", "c", "b"], labels, k=10) == pytest.approx(1.0)


def test_ndcg_unlabelled_counts_as_zero_and_none_when_no_relevant():
    assert ndcg_at_k(["u", "a"], {"a": 2}, k=10) == pytest.approx(3 / np.log2(3) / 3)
    assert ndcg_at_k(["a"], {"a": 0}, k=10) is None


def test_mrr_and_recall():
    labels = {"a": 0, "b": 1, "c": 2, "d": 2}
    ranked = ["a", "x", "b", "c"]
    assert reciprocal_rank(ranked, labels) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, labels, min_relevant=2) == pytest.approx(1 / 4)
    assert recall_at_k(ranked, labels, k=4) == pytest.approx(2 / 3)  # b, c found; d missed
    assert recall_at_k(ranked, {"a": 0}, k=4) is None
    assert mean([1.0, None, 0.0]) == 0.5


# ---------- text ----------

def test_tokenize_keeps_tech_terms_and_drops_stopwords():
    assert tokenize("Experience with C++, C#, .NET and Node.js in the IT team") == \
        ["experience", "c++", "c#", "net", "node.js", "it", "team"]


def test_chunk_words():
    words = " ".join(str(i) for i in range(10))
    assert chunk_words(words, 4, 1) == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]
    assert chunk_words("one two", 4, 1) == ["one two"]
    with pytest.raises(ValueError):
        chunk_words("x", 2, 2)


# ---------- BM25 / dense / hybrid ----------

DOCS = {
    "r1": "accountant with quickbooks payroll and tax experience",
    "r2": "line cook and sous chef in a busy kitchen",
    "r3": "software engineer python kafka backend services",
}


def test_bm25_ranks_exact_terms_first(tmp_path):
    idx = BM25Index(list(DOCS), list(DOCS.values()))
    assert idx.search("kafka backend engineer", k=3)[0][0] == "r3"
    idx.save(tmp_path / "bm25.pkl")
    assert BM25Index.load(tmp_path / "bm25.pkl").search("payroll tax", k=1)[0][0] == "r1"


class FakeEncoder:
    """Bag-of-keywords 'embedding' so the dense index can be tested without downloading a model."""
    VOCAB = ["accountant", "cook", "software", "finance", "kitchen", "backend"]

    def encode(self, texts, **kw):
        v = np.array([[t.lower().count(w) for w in self.VOCAB] for t in texts], dtype="float32") + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_dense_max_pools_chunks_and_round_trips(tmp_path):
    long_resume = "filler " * 40 + "backend software"  # relevant part only in the last chunk
    idx = DenseIndex("fake", chunk_size=20, chunk_overlap=5, model=FakeEncoder())
    idx.build(["r1", "r2", "r3"], [DOCS["r1"], DOCS["r2"], long_resume], show_progress=False)
    assert idx.index.ntotal > 3  # r3 was chunked
    hits = idx.search("backend software role", k=3)
    assert hits[0][0] == "r3"
    assert len({d for d, _ in hits}) == len(hits)  # one entry per resume, not per chunk
    idx.save(tmp_path)
    again = DenseIndex.load(tmp_path, model=FakeEncoder())
    assert again.search("kitchen cook", k=1)[0][0] == "r2"


def test_hybrid_returns_fused_unique_ids():
    bm25 = BM25Index(list(DOCS), list(DOCS.values()))
    dense = DenseIndex("fake", model=FakeEncoder())
    dense.build(list(DOCS), list(DOCS.values()), show_progress=False)
    hits = HybridRetriever(bm25, dense, rrf_k=60, fusion_depth=3).search("backend software kafka", k=2)
    assert hits[0][0] == "r3" and len(hits) == 2


def test_fast_bm25_scores_equal_rank_bm25():
    texts = list(DOCS.values()) + ["payroll payroll tax accountant", "kitchen kitchen cook python"]
    idx = BM25Index([f"d{i}" for i in range(len(texts))], texts)
    for q in ["payroll tax accountant", "kitchen cook cook", "python kafka unknownword", "the and of"]:
        from talentlens.retrieval.text import tokenize
        expected = idx.bm25.get_scores(tokenize(q))
        assert np.allclose(idx.scores(q), expected), q
