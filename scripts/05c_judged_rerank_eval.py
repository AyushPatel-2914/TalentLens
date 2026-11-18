"""Stage 5: re-ranker comparison on LLM-JUDGE labels, on the same judged candidates per job.

Usage:
    python scripts/05c_judged_rerank_eval.py                  # judge = llm.models.strong
    python scripts/05c_judged_rerank_eval.py --model openai/gpt-oss-20b

Why a separate evaluation: the judge labelled only 4 candidates per test job (the set
drawn for human labelling). Standard nDCG@10 over a full ranking would count every
unlabelled candidate as irrelevant and favour the retrieval methods that happened to
contribute the labelled ones (pool bias). Here every scorer orders the SAME judged
candidates of each job ("condensed list" evaluation), so the comparison is fair:

    nDCG@4             graded (0/1/2) ranking quality of the judged candidates
    pairwise accuracy  for pairs of candidates with different labels, % ordered correctly
    random             expected values for a random order (computed exactly)

Labels come from an LLM judge and are NOT human-validated. Test split only; the
fine-tuned model never saw these jobs.

Writes results/judged_rerank_eval.json and .md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from talentlens.config import ROOT, load_config, resolve
from talentlens.eval.qrels import judge_labels_path
from talentlens.rerank.cross_encoder import Reranker, pair_texts


def ndcg(labels_in_order: list[int]) -> float:
    def dcg(ls):
        return sum((2 ** l - 1) / math.log2(i + 2) for i, l in enumerate(ls))
    ideal = dcg(sorted(labels_in_order, reverse=True))
    return dcg(labels_in_order) / ideal if ideal > 0 else float("nan")


def pairwise(labels_in_order: list[int]) -> tuple[int, int]:
    """(correctly ordered, total) over pairs with different labels."""
    ok = tot = 0
    for i, j in itertools.combinations(range(len(labels_in_order)), 2):
        a, b = labels_in_order[i], labels_in_order[j]
        if a != b:
            tot += 1
            ok += a > b
    return ok, tot


def random_ndcg(labels: list[int]) -> float:
    perms = list(itertools.permutations(labels))
    return float(np.mean([ndcg(list(p)) for p in perms]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="judge model whose labels to use")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = load_config()
    rr = cfg["rerank"]
    judge_model = args.model or cfg["llm"]["models"]["strong"]

    pools = pd.read_parquet(resolve(cfg["labels"]["pools_file"]))
    judge = pd.read_json(judge_labels_path(cfg, judge_model), lines=True).drop_duplicates("pair_id", keep="last")
    df = pools[pools["split"] == "test"].merge(judge[["pair_id", "label"]].dropna(), on="pair_id")
    df["label"] = df["label"].astype(int)
    processed = resolve(cfg["paths"]["processed_dir"])
    jobs = pd.read_parquet(processed / "jobs.parquet").set_index("job_id")
    resumes = pd.read_parquet(processed / "resumes.parquet").set_index("resume_id")

    groups = {jid: g for jid, g in df.groupby("job_id") if len(g) >= 2 and g["label"].nunique() > 1}
    print(f"{len(df)} judged test pairs, {df['job_id'].nunique()} jobs; "
          f"{len(groups)} jobs have >= 2 candidates with different labels (used).")
    if not groups:
        raise SystemExit("Not enough judged pairs yet.")

    scorers = {"bge-reranker-base": rr["base_model"]}
    if rr.get("finetuned_model"):
        scorers["fine-tuned reranker"] = rr["finetuned_model"]

    per_job = {name: [] for name in ["random", *scorers]}
    for jid, g in groups.items():
        labels = g["label"].tolist()
        ok, tot = pairwise(labels)
        per_job["random"].append({"job_id": jid, "ndcg": random_ndcg(labels), "pair_ok": tot / 2, "pair_total": tot})
    for name, model_id in scorers.items():
        print(f"Scoring with {model_id} ...")
        model = Reranker(model_id, rr["max_length"], rr["batch_size"])
        for jid, g in groups.items():
            job = jobs.loc[jid]
            texts = [pair_texts(job.title, job.text, resumes.loc[r, "text"], rr["job_words"], rr["resume_words"])
                     for r in g["resume_id"]]
            scores = model.score(texts[0][0], [d for _, d in texts])
            order = np.argsort(-np.asarray(scores), kind="stable")
            ranked = [int(g["label"].iloc[i]) for i in order]
            ok, tot = pairwise(ranked)
            per_job[name].append({"job_id": jid, "ndcg": ndcg(ranked), "pair_ok": ok, "pair_total": tot})

    rows = []
    for name, items in per_job.items():
        d = pd.DataFrame(items)
        rows.append({"scorer": name, "jobs": len(d), "ndcg@4": float(d["ndcg"].mean()),
                     "pairwise_accuracy": float(d["pair_ok"].sum() / d["pair_total"].sum()),
                     "pairs_compared": int(d["pair_total"].sum())})
    # paired comparison fine-tuned vs base: per-job nDCG wins/losses
    paired = None
    if "fine-tuned reranker" in per_job:
        a = pd.DataFrame(per_job["fine-tuned reranker"]).set_index("job_id")["ndcg"]
        b = pd.DataFrame(per_job["bge-reranker-base"]).set_index("job_id")["ndcg"]
        diff = (a - b).dropna()
        rng = np.random.default_rng(42)
        boots = [float(rng.choice(diff.values, len(diff)).mean()) for _ in range(5000)]
        paired = {"mean_ndcg_diff": float(diff.mean()), "jobs_better": int((diff > 1e-9).sum()),
                  "jobs_worse": int((diff < -1e-9).sum()), "jobs_tied": int((diff.abs() <= 1e-9).sum()),
                  "bootstrap_95ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "judge_model": judge_model,
           "labels": "LLM judge, not human-validated", "split": "test",
           "judged_pairs": len(df), "label_distribution": df["label"].value_counts().sort_index().to_dict(),
           "rows": rows, "finetuned_vs_base": paired}
    results = resolve(cfg["paths"]["results_dir"])
    (results / "judged_rerank_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = ["# Re-rankers on LLM-judge labels (test split)", "",
          f"Generated {out['generated_at']} by scripts/05c_judged_rerank_eval.py. Judge: `{judge_model}`. "
          "**Labels are from an LLM judge and are not human-validated.**", "",
          f"{len(df)} judged test pairs (labels {out['label_distribution']}); {len(groups)} jobs with candidates of "
          "different labels. Each scorer orders the same judged candidates of each job.", "",
          "| Scorer | Jobs | nDCG@4 | Pairwise accuracy | Pairs compared |", "|---|---|---|---|---|",
          *[f"| {r['scorer']} | {r['jobs']} | {r['ndcg@4']:.3f} | {r['pairwise_accuracy']:.3f} | {r['pairs_compared']} |"
            for r in rows]]
    if paired:
        md += ["", f"Fine-tuned minus base, per job: mean nDCG@4 difference {paired['mean_ndcg_diff']:+.3f} "
               f"(bootstrap 95% CI {paired['bootstrap_95ci'][0]:+.3f} to {paired['bootstrap_95ci'][1]:+.3f}); "
               f"better on {paired['jobs_better']} jobs, worse on {paired['jobs_worse']}, tied on {paired['jobs_tied']}."]
    (results / "judged_rerank_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
