"""Stage 5: export labelled TRAIN and VALIDATION pairs for fine-tuning on Colab.

Usage:
    python scripts/05_export_training_data.py                      # judge labels (after the kappa check)
    python scripts/05_export_training_data.py --labels category    # weak labels, free: no LLM, no human

--labels category (weak supervision): for every TRAIN / VALIDATION job, take the hybrid top N
resumes and label each 2 if its occupation category equals the job's, else 0. Free and large
(about 40k pairs), but noisy: a same-category resume can still be a weak fit. Because the model
is trained on this proxy, evaluate it on YOUR human labels (test split) for the honest number;
its score on category labels is optimistic by construction. Test jobs are never exported.

Uses judge labels (your own labels are all in the test split, so they are never
trained on). Upload the two files to the Colab notebook notebooks/finetune_reranker.ipynb.

Hard negatives: the pools already contain them. Pairs found by BM25/dense top-20
or the middle of the hybrid list that were labelled 0 look relevant but are not,
which is exactly what teaches the re-ranker fine distinctions. Random pairs labelled
0 are kept too, as easy negatives. Unlabelled resumes are NOT used as negatives:
some of them are real matches nobody labelled (false negatives would teach the
model the wrong thing).

Writes data/training/{train,validation}.jsonl and results/training_data.json.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from tqdm import tqdm

from talentlens.config import load_config, resolve
from talentlens.eval.qrels import judge_labels_path
from talentlens.rerank.cross_encoder import pair_texts


def category_pairs(cfg, jobs: pd.DataFrame, resumes: pd.DataFrame, per_job: int) -> pd.DataFrame:
    from talentlens.eval.qrels import load_splits
    from talentlens.retrieval.load import load_retrievers
    hybrid = load_retrievers(cfg)["hybrid"]
    splits = load_splits(cfg)
    rows = []
    todo = [j for j in jobs.index if splits.get(j) in ("train", "validation")]
    for jid in tqdm(todo, desc="hybrid top-N"):
        job = jobs.loc[jid]
        for rid, _ in hybrid.search(f"{job.title}\n{job.text}", per_job):  # same query text as evaluation
            same = resumes.loc[rid, "category"] == job.matched_category
            rows.append({"pair_id": f"{jid}__{rid}", "job_id": jid, "resume_id": rid, "split": splits[jid],
                         "source": "hybrid_top", "label": 2 if same else 0})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", choices=["judge", "category"], default="judge")
    ap.add_argument("--per-job", type=int, default=20, help="with --labels category: hybrid top N per job")
    args = ap.parse_args()
    cfg = load_config()
    rr = cfg["rerank"]
    processed = resolve(cfg["paths"]["processed_dir"])
    jobs = pd.read_parquet(processed / "jobs.parquet").set_index("job_id")
    resumes = pd.read_parquet(processed / "resumes.parquet").set_index("resume_id")
    pools = pd.read_parquet(resolve(cfg["labels"]["pools_file"]))

    if args.labels == "category":
        df = category_pairs(cfg, jobs, resumes, args.per_job)
        assert not set(df["split"]) - {"train", "validation"}, "test jobs must never be exported"
    else:
        path = judge_labels_path(cfg)
        if not path.exists():
            raise SystemExit(f"No judge labels at {path}. Run scripts/03c_judge.py --label-all first.")
        judge = pd.read_json(path, lines=True).drop_duplicates("pair_id", keep="last").dropna(subset=["label"])
        df = pools.merge(judge[["pair_id", "label"]], on="pair_id")

    out_dir = resolve("data/training")
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"label_source": args.labels, **({"per_job": args.per_job} if args.labels == "category" else {})}
    for split in ["train", "validation"]:
        part = df[df["split"] == split]
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in part.itertuples(index=False):
                job = jobs.loc[r.job_id]
                q, d = pair_texts(job.title, job.text, resumes.loc[r.resume_id, "text"], rr["job_words"], rr["resume_words"])
                f.write(json.dumps({"job_id": r.job_id, "resume_id": r.resume_id, "source": r.source,
                                    "query": q, "doc": d, "label": int(r.label)}) + "\n")
        stats[split] = {
            "pairs": len(part), "jobs": int(part["job_id"].nunique()),
            "labels": part["label"].astype(int).value_counts().sort_index().to_dict(),
            "hard_negatives": int(((part["label"] == 0) & (part["source"] != "random")).sum()),
        }
    (resolve(cfg["paths"]["results_dir"]) / "training_data.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Upload {out_dir / 'train.jsonl'} and {out_dir / 'validation.jsonl'} to the Colab notebook.")


if __name__ == "__main__":
    main()
