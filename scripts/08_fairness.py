"""Stage 8: counterfactual fairness test, before and after mitigation.

Usage:
    python scripts/08_fairness.py                     # final score with the cross-encoder (slow on CPU, ~20-40 min)
    python scripts/08_fairness.py --scorer dense      # bi-encoder cosine only (fast)
    python scripts/08_fairness.py --jobs 4 --per-job 10   # quick check

Method: 10 test jobs x their hybrid top-20 candidates = 200 candidate slots. For each
candidate and each of 10 name groups, ONLY that candidate's resume gets the name +
gender-coded words; everyone else stays original. We re-score that one candidate and
measure its rank shift, whether the top-10 changes, and the score difference.
Then we repeat with the mitigation (redact names + neutralise gendered words before
scoring) and compare.

Writes results/fairness.json and results/fairness_report.md.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone

import numpy as np
import yaml
from tqdm import tqdm

from talentlens.config import load_config, resolve
from talentlens.data.sample import sample_by_category
from talentlens.eval.qrels import load_splits
from talentlens.fairness.counterfactual import by_group, counterfactual_shift, make_variant, redact_identity, summarize
from talentlens.retrieval.text import chunk_words
from talentlens.service import MatchingService


def make_scorer(svc: MatchingService, kind: str):
    if kind == "final":
        def score(job_id: str, texts: dict[str, str]) -> dict[str, float]:
            j = svc.jobs.loc[job_id]
            out = svc.score_candidates(svc.job_struct_for_id(job_id), j.text, list(texts), resume_texts=texts)
            return {o["resume_id"]: o["score"] for o in out}
        return score

    dense = svc.retrievers["dense"]

    def score(job_id: str, texts: dict[str, str]) -> dict[str, float]:
        j = svc.jobs.loc[job_id]
        q = dense.embed_query(f"{j.title}\n{j.text}")[0]
        out = {}
        for rid, t in texts.items():
            chunks = dense._embed(chunk_words(t, dense.chunk_size, dense.chunk_overlap))
            out[rid] = float((chunks @ q).max())  # same max-over-chunks scoring as the index
        return out
    return score


def run(svc, scorer, jobs, per_job, groups, words, mitigate: bool) -> list[dict]:
    all_names = [n for g in groups.values() for n in g["names"]]
    prep = (lambda t: redact_identity(t, words, all_names)) if mitigate else (lambda t: t)
    records = []
    for job_id in tqdm(jobs, desc="mitigated" if mitigate else "baseline"):
        j = svc.jobs.loc[job_id]
        cands = [d for d, _ in svc.retrievers["hybrid"].search(f"{j.title}\n{j.text}", per_job)]
        original = {d: svc.resumes.loc[d, "text"] for d in cands}
        baseline = scorer(job_id, {d: prep(t) for d, t in original.items()})
        for i, d in enumerate(cands):
            for gname, g in groups.items():
                name = g["names"][i % len(g["names"])]
                variant = prep(make_variant(original[d], name, g["gender"], words))
                new = scorer(job_id, {d: variant})[d]
                records.append({"job_id": job_id, "resume_id": d, "group": gname, "gender": g["gender"],
                                **counterfactual_shift(baseline, d, new)})
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scorer", choices=["final", "dense"], default="final")
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--per-job", type=int, default=20)
    args = p.parse_args()
    cfg = load_config()
    fair = yaml.safe_load(resolve("configs/fairness_names.yaml").read_text(encoding="utf-8"))
    groups, words = fair["groups"], fair["gendered_words"]

    svc = MatchingService(cfg, use_reranker=args.scorer == "final")
    splits = load_splits(cfg)
    test_jobs = svc.jobs[svc.jobs.index.map(lambda j: splits.get(j) == "test")].reset_index()
    jobs = sample_by_category(test_jobs, "matched_category", args.jobs, cfg["data"]["seed"])["job_id"].tolist()
    random.Random(cfg["data"]["seed"]).shuffle(jobs)
    scorer = make_scorer(svc, args.scorer)

    before = run(svc, scorer, jobs, args.per_job, groups, words, mitigate=False)
    after = run(svc, scorer, jobs, args.per_job, groups, words, mitigate=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scorer": "final score (cross-encoder + requirement checks)" if args.scorer == "final" else "bi-encoder cosine",
        "reranker": svc.reranker.model_name if svc.reranker else None,
        "jobs": jobs, "candidates_per_job": args.per_job, "candidate_slots": len(jobs) * args.per_job,
        "before": summarize(before), "after": summarize(after),
        "before_by_group": by_group(before), "after_by_group": by_group(after),
        "before_by_gender": by_group(before, "gender"), "after_by_gender": by_group(after, "gender"),
    }
    results = resolve(cfg["paths"]["results_dir"])
    suffix = "" if args.scorer == "final" else "_dense"
    (results / f"fairness{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    b, a = report["before"], report["after"]
    md = [f"# Fairness report ({report['scorer']})", "",
          f"Generated {report['generated_at']} by scripts/08_fairness.py. {len(jobs)} test jobs x top-{args.per_job} "
          f"candidates = {report['candidate_slots']} candidate slots x {len(groups)} name groups "
          f"= {b['counterfactuals']} counterfactuals per condition.", "",
          "| | Mean abs rank shift | Max abs rank shift | % top-10 lists changed | Mean abs score diff | Max abs score diff |",
          "|---|---|---|---|---|---|",
          *[f"| {label} | {s['mean_abs_rank_shift']:.3f} | {s['max_abs_rank_shift']} | {s['pct_top10_changed']:.1%} | "
            f"{s['mean_abs_score_diff']:.5f} | {s['max_abs_score_diff']:.5f} |" for label, s in [("Before mitigation", b), ("After mitigation (redaction)", a)]],
          "", "## Direction by group (before mitigation)", "",
          "Signed mean score difference vs. the original resume (positive = the name helped).", "",
          "| Group | Mean score diff | Mean rank shift |", "|---|---|---|",
          *[f"| {g} | {v['mean_score_diff']:+.5f} | {v['mean_rank_shift']:+.3f} |" for g, v in report["before_by_group"].items()],
          "", "## What this test does and does not show", "",
          "- It shows how much the ranking reacts to a name line and gender-coded words **with everything else held equal**.",
          "- It does **not** test proxies for identity: college names, gaps in employment, clubs, hobbies, language style.",
          "- Names are an imperfect stand-in for gender or region, and 30 names cannot represent whole groups.",
          "- It measures the second stage (scores of already-retrieved candidates). Names could also change which "
          "candidates first-stage retrieval finds; that would need re-indexing every variant.",
          "- Bias can also come from the training labels (human or LLM judge), which this test does not measure.",
          "- After mitigation the name signal is removed by construction, so a near-zero shift there is expected. "
          "It shows the redaction works, not that the model is unbiased.",
          "- A small shift on 200 slots means no large effect was detected by this test, not that the system is fair."]
    (results / f"fairness_report{suffix}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:8]))


if __name__ == "__main__":
    main()
