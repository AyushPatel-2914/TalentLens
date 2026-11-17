"""Stage 4: build BM25 and dense (FAISS) indexes over resumes and over jobs.

Usage:
    python scripts/04_build_indexes.py                 # both corpora
    python scripts/04_build_indexes.py --corpus resumes

Resumes are indexed for job -> candidates search; jobs are indexed for the
reverse direction (resume -> best-fit jobs, Stage 9).
The first run downloads BAAI/bge-small-en-v1.5 (~130 MB) from Hugging Face.

Writes data/indexes/{resumes,jobs}/ and results/index_build.json.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from talentlens.config import load_config, resolve
from talentlens.retrieval.bm25 import BM25Index
from talentlens.retrieval.load import index_dir, load_retrievers, new_dense


def corpus_frame(cfg: dict, corpus: str) -> pd.DataFrame:
    processed = resolve(cfg["paths"]["processed_dir"])
    if corpus == "resumes":
        df = pd.read_parquet(processed / "resumes.parquet")
        return pd.DataFrame({"doc_id": df["resume_id"], "text": df["text"]})
    df = pd.read_parquet(processed / "jobs.parquet")
    return pd.DataFrame({"doc_id": df["job_id"], "text": df["title"] + "\n" + df["text"]})


def build(cfg: dict, corpus: str) -> dict:
    df = corpus_frame(cfg, corpus)
    out = index_dir(cfg, corpus)
    r = cfg["retrieval"]

    t0 = time.perf_counter()
    bm25 = BM25Index(df["doc_id"].tolist(), df["text"].tolist(), **r["bm25"])
    bm25.save(out / "bm25.pkl")
    t_bm25 = time.perf_counter() - t0

    t0 = time.perf_counter()
    dense = new_dense(cfg)
    print(f"Embedding {corpus} with {dense.model_name} (CPU; a few minutes)...")
    dense.build(df["doc_id"].tolist(), df["text"].tolist())
    dense.save(out)
    t_dense = time.perf_counter() - t0

    return {"corpus": corpus, "docs": len(df), "dense_chunks": dense.index.ntotal,
            "chunks_per_doc": round(dense.index.ntotal / len(df), 2),
            "bm25_build_seconds": round(t_bm25, 1), "dense_build_seconds": round(t_dense, 1)}


def smoke_test(cfg: dict) -> None:
    """One real query, so you can eyeball that results make sense."""
    retrievers = load_retrievers(cfg, "resumes")
    jobs = pd.read_parquet(resolve(cfg["paths"]["processed_dir"]) / "jobs.parquet")
    resumes = pd.read_parquet(resolve(cfg["paths"]["processed_dir"]) / "resumes.parquet").set_index("resume_id")
    job = jobs[jobs["matched_category"] == "ACCOUNTANT"].iloc[0]
    query = job["title"] + "\n" + job["text"]
    print(f"\nSmoke test query: {job['job_id']} '{job['title']}' (category ACCOUNTANT)")
    for name, ret in retrievers.items():
        hits = ret.search(query, k=10)
        cats = [resumes.loc[d, "category"] for d, _ in hits]
        share = sum(c == "ACCOUNTANT" for c in cats) / len(cats)
        print(f"  {name:>6}: top-10 categories -> {share:.0%} ACCOUNTANT | {', '.join(cats[:5])} ...")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", choices=["resumes", "jobs", "both"], default="both")
    args = p.parse_args()
    cfg = load_config()

    stats = [build(cfg, c) for c in (["resumes", "jobs"] if args.corpus == "both" else [args.corpus])]
    results = resolve(cfg["paths"]["results_dir"])
    (results / "index_build.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    if any(s["corpus"] == "resumes" for s in stats):
        smoke_test(cfg)


if __name__ == "__main__":
    main()
