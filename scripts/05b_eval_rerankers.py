"""Stage 5: ablation on the TEST split.

    BM25 | dense | hybrid | hybrid + bge-reranker-base | hybrid + fine-tuned re-ranker

Usage:
    python scripts/05b_eval_rerankers.py                  # all label sources that exist
    python scripts/05b_eval_rerankers.py --max-queries 10 # quick check

The fine-tuned row appears once rerank.finetuned_model is set in config.
Re-ranking 100 candidates with a base-size cross-encoder on CPU takes seconds
per query; the latency column shows that cost honestly.

Writes results/ablation.csv and results/ablation.md.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from talentlens.config import ROOT, load_config, resolve
from talentlens.eval.qrels import available_sources, load_qrels
from talentlens.eval.ranking_eval import evaluate_ranker, markdown_table
from talentlens.rerank.cross_encoder import Reranker, first_words
from talentlens.retrieval.load import load_retrievers

COLUMNS = ["labels", "method", "queries", "recall@100", "ndcg@10", "mrr", "latency_ms_mean", "latency_ms_p95"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--skip-base", action="store_true",
                   help="reuse the base re-ranker row from results/ablation.csv (same labels and query count) "
                        "instead of re-scoring it (~1 min per query on CPU)")
    p.add_argument("--labels", nargs="+", default=None,
                   help="label sources to evaluate (e.g. category human judge); default: all that exist")
    args = p.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = load_config()
    rr = cfg["rerank"]
    sources = available_sources(cfg, proxy=True)
    if args.labels:
        missing = set(args.labels) - set(sources)
        if missing:
            raise SystemExit(f"label sources not available: {sorted(missing)} (have {sources})")
        sources = [s for s in sources if s in args.labels]
    if not sources:
        raise SystemExit("No labels yet.")

    processed = resolve(cfg["paths"]["processed_dir"])
    jobs = pd.read_parquet(processed / "jobs.parquet").set_index("job_id")
    resume_text = pd.read_parquet(processed / "resumes.parquet").set_index("resume_id")["text"].to_dict()
    queries = {jid: f"{j.title}\n{j.text}" for jid, j in jobs.iterrows()}
    retrievers = load_retrievers(cfg)
    depth = cfg["retrieval"]["depth"]

    def first_stage(name):
        return lambda q: [d for d, _ in retrievers[name].search(q, depth)]

    def reranked(model: Reranker):
        def run(q: str) -> list[str]:
            title, body = q.split("\n", 1)
            rq = f"{title}. {first_words(body, rr['job_words'])}"
            cands = [(d, first_words(resume_text[d], rr["resume_words"])) for d, _ in retrievers["hybrid"].search(q, rr["depth"])]
            return [d for d, _ in model.rerank(rq, cands)]
        return run

    BASE = "hybrid + bge-reranker-base"
    previous = pd.DataFrame()
    if args.skip_base:
        prev_path = resolve(cfg["paths"]["results_dir"]) / "ablation.csv"
        if not prev_path.exists():
            raise SystemExit("--skip-base needs an earlier results/ablation.csv")
        previous = pd.read_csv(prev_path)
        previous = previous[previous["method"] == BASE]
    methods = {n: first_stage(n) for n in ["bm25", "dense", "hybrid"]}
    if not args.skip_base:
        print(f"Loading {rr['base_model']} ...")
        methods[BASE] = reranked(Reranker(rr["base_model"], rr["max_length"], rr["batch_size"]))
    if rr.get("finetuned_model"):
        print(f"Loading {rr['finetuned_model']} ...")
        if not os.environ.get("HF_TOKEN"):
            print("  (no HF_TOKEN in .env: loading a private Hub repo will fail)")
        methods["hybrid + fine-tuned reranker"] = reranked(Reranker(rr["finetuned_model"], rr["max_length"], rr["batch_size"]))
    else:
        print("rerank.finetuned_model is not set: the fine-tuned row is skipped.")

    rows = []
    for source in sources:
        qrels = load_qrels(cfg, source, "test")
        if args.max_queries:
            qrels = dict(list(sorted(qrels.items()))[: args.max_queries])
        for name, ranker in methods.items():
            s, _ = evaluate_ranker(name, ranker, queries, qrels)
            rows.append({"labels": source, **s})
            print(f"[{source}] {name}: ndcg@10={s.get('ndcg@10')}, mrr={s.get('mrr')}, latency={s.get('latency_ms_mean')}")
            if name == "hybrid" and args.skip_base:
                old = previous[(previous["labels"] == source) & (previous["queries"] == len(qrels))]
                if old.empty:
                    raise SystemExit(f"--skip-base: no earlier base row for labels={source}, queries={len(qrels)}")
                rows.append(old.iloc[0].to_dict())
                print(f"[{source}] {BASE}: reused from the earlier results/ablation.csv")

    results = resolve(cfg["paths"]["results_dir"])
    pd.DataFrame(rows).to_csv(results / "ablation.csv", index=False)
    md = ["# Ablation (test split)", "",
          f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by scripts/05b_eval_rerankers.py"
          + (f" on the first {args.max_queries} queries only (partial run)" if args.max_queries else "") + ".",
          f"Re-rankers re-order the hybrid top {rr['depth']}; Recall@100 therefore equals hybrid's. "
          "Latency = CPU, per query, including first-stage retrieval. "
          "Label source `category` is a PROXY (no human input): a resume counts as relevant when its occupation "
          "category equals the job's."
          + (" The fine-tuned re-ranker was TRAINED on category labels (train/validation jobs only), so its "
             "category score is optimistic by construction; judge it on human labels." if rr.get("finetuned_model") else "")
          + (f" The `{BASE}` row was reused from the previous run (--skip-base), not re-scored." if args.skip_base else ""),
          "",
          markdown_table(rows, COLUMNS)]
    (results / "ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
