"""Stage 4: evaluate BM25, dense and hybrid retrieval on the TEST split (= evaluate.py in the plan).

Usage:
    python scripts/04b_evaluate.py              # every label source that exists
    python scripts/04b_evaluate.py --sweep-k    # also tune the RRF k on VALIDATION

The RRF k used for TEST is fixed before looking at test results: it comes from
config, or from the validation sweep if you run it and then update config.

Writes:
    results/retrieval.csv / retrieval.md        one row per method x label source
    results/retrieval_examples.md                3 real test queries where methods disagree
    results/rrf_k_sweep.csv                      (with --sweep-k)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from talentlens.config import load_config, resolve
from talentlens.eval.qrels import available_sources, load_qrels
from talentlens.eval.ranking_eval import evaluate_ranker, markdown_table
from talentlens.retrieval.hybrid import HybridRetriever
from talentlens.retrieval.load import load_retrievers

COLUMNS = ["labels", "method", "queries", "recall@100", "ndcg@10", "mrr", "latency_ms_mean", "latency_ms_p95"]


def job_queries(cfg: dict) -> dict[str, str]:
    jobs = pd.read_parquet(resolve(cfg["paths"]["processed_dir"]) / "jobs.parquet")
    return {j.job_id: f"{j.title}\n{j.text}" for j in jobs.itertuples(index=False)}


def rankers(retrievers: dict, depth: int) -> dict:
    return {name: (lambda q, r=r: [d for d, _ in r.search(q, depth)]) for name, r in retrievers.items()}


def sweep_k(cfg, retrievers, queries) -> None:
    real = available_sources(cfg)
    qrels = load_qrels(cfg, "combined" if "combined" in real else ("human" if real else "category"), "validation")
    if not qrels:
        print("No validation labels yet (needs judge labels): skipping the k sweep.")
        return
    rows = []
    for k in [1, 5, 10, 30, 60, 100, 200]:
        h = HybridRetriever(retrievers["bm25"], retrievers["dense"], rrf_k=k, fusion_depth=cfg["retrieval"]["fusion_depth"])
        s, _ = evaluate_ranker(f"hybrid k={k}", lambda q: [d for d, _ in h.search(q, 100)], queries, qrels)
        rows.append({"rrf_k": k, **s})
    out = pd.DataFrame(rows)
    out.to_csv(resolve(cfg["paths"]["results_dir"]) / "rrf_k_sweep.csv", index=False)
    print(out[["rrf_k", "recall@100", "ndcg@10", "mrr"]].to_string(index=False))
    print("If a k is clearly better on validation, set retrieval.rrf_k in config, then re-run without --sweep-k.")


def examples(cfg, per_query: pd.DataFrame, qrels, queries) -> str:
    """Pick 3 test queries where BM25 and dense disagree most, and show their top-3 with labels."""
    resumes = pd.read_parquet(resolve(cfg["paths"]["processed_dir"]) / "resumes.parquet").set_index("resume_id")
    wide = per_query.pivot_table(index="query_id", columns="method", values="ndcg@10")
    if not {"bm25", "dense"} <= set(wide.columns):
        return ""
    wide["gap"] = (wide["bm25"] - wide["dense"]).abs()
    picks = wide.dropna().sort_values("gap", ascending=False).head(3).index
    md = ["# Retrieval examples (test split)", "",
          "Three test jobs where BM25 and dense disagree most. Label: 2 strong, 1 partial, 0 no fit, ? unlabelled.", ""]
    for qid in picks:
        title = queries[qid].split("\n", 1)[0]
        md.append(f"## {qid}: {title}")
        md.append("")
        for method in ["bm25", "dense", "hybrid"]:
            row = per_query[(per_query.query_id == qid) & (per_query.method == method)].iloc[0]
            items = [f"{d} ({resumes.loc[d, 'category']}, label {qrels[qid].get(d, '?')})" for d in row["top10"][:3]]
            md.append(f"- **{method}** nDCG@10 {row['ndcg@10']:.3f}: " + "; ".join(items))
        md.append("")
    return "\n".join(md)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-k", action="store_true")
    args = p.parse_args()
    cfg = load_config()
    sources = available_sources(cfg, proxy=True)
    if not sources:
        raise SystemExit("No labels yet. Label pairs (scripts/03b_label_pairs.py) and/or run the judge first.")

    retrievers, queries = load_retrievers(cfg), job_queries(cfg)
    depth = cfg["retrieval"]["depth"]
    if args.sweep_k:
        sweep_k(cfg, retrievers, queries)

    summaries, example_md = [], ""
    for source in sources:
        qrels = load_qrels(cfg, source, "test")
        per_query_all = []
        for name, ranker in rankers(retrievers, depth).items():
            s, pq = evaluate_ranker(name, ranker, queries, qrels)
            summaries.append({"labels": source, **s})
            per_query_all.append(pq)
            print(f"[{source}] {name}: " + ", ".join(f"{k}={v:.3f}" for k, v in s.items() if isinstance(v, float)))
        if source == sources[-1]:
            example_md = examples(cfg, pd.concat(per_query_all), qrels, queries)

    results = resolve(cfg["paths"]["results_dir"])
    pd.DataFrame(summaries).to_csv(results / "retrieval.csv", index=False)
    md = ["# First-stage retrieval (test split)", "",
          f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by scripts/04b_evaluate.py. "
          f"RRF k = {cfg['retrieval']['rrf_k']}. Relevant = label >= 1. Label source `category` is a PROXY (no human input): a resume counts as relevant when its occupation category equals the job's, so Recall@100 is capped when a category has more than 100 resumes. Latency = CPU, model loaded, per query.", "",
          markdown_table(summaries, COLUMNS)]
    (results / "retrieval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if example_md:
        (results / "retrieval_examples.md").write_text(example_md + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
