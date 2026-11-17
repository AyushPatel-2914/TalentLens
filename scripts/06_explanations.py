"""Stage 6: tune the final-score weights and measure evidence-quote verification.

Usage:
    python scripts/06_explanations.py --tune          # weights on VALIDATION, then report on TEST
    python scripts/06_explanations.py --quotes 30     # quotes for the top-3 of 30 test jobs (LLM calls)

--tune writes results/score_weights.json and results/final_score.md
--quotes writes results/explanations.json and results/explanation_examples.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from dotenv import load_dotenv

from talentlens.config import ROOT, load_config, resolve
from talentlens.eval.qrels import available_sources, load_qrels
from talentlens.eval.ranking_metrics import mean, ndcg_at_k, reciprocal_rank
from talentlens.explain.quotes import get_evidence_quotes
from talentlens.explain.scoring import final_score, weight_grid
from talentlens.extraction.runner import make_client
from talentlens.llm.client import RateLimitExhausted
from talentlens.service import MatchingService


def components_for_split(svc: MatchingService, qrels: dict) -> dict[str, list[tuple[str, dict]]]:
    """For each labelled job: hybrid top-100 candidates with their score components (computed once)."""
    out = {}
    for job_id in qrels:
        j = svc.jobs.loc[job_id]
        job = svc.job_struct_for_id(job_id)
        cands = [d for d, _ in svc.retrievers["hybrid"].search(f"{j.title}\n{j.text}", svc.cfg["retrieval"]["depth"])]
        scored = svc.score_candidates(job, j.text, cands)
        out[job_id] = [(s["resume_id"], s["score_breakdown"]["components"] | {"logit": s["reranker_logit"]}) for s in scored]
    return out


def rank_with(weights: dict, comps: list[tuple[str, dict]]) -> list[str]:
    scored = [(d, final_score(c["logit"], c["required_skills"], c["experience"], weights)[0]) for d, c in comps]
    return [d for d, _ in sorted(scored, key=lambda x: -x[1])]


def evaluate(weights, comps_by_job, qrels) -> dict:
    ranked = {j: rank_with(weights, c) for j, c in comps_by_job.items()}
    return {"ndcg@10": mean(ndcg_at_k(ranked[j], qrels[j], 10) for j in ranked),
            "mrr": mean(reciprocal_rank(ranked[j], qrels[j]) for j in ranked), "queries": len(ranked)}


def tune(cfg, svc: MatchingService) -> None:
    sources = available_sources(cfg)
    val_source = "combined" if "combined" in sources else None
    if not val_source:
        raise SystemExit("Tuning needs validation labels from the judge (scripts/03c_judge.py --label-all).")
    val_qrels = load_qrels(cfg, val_source, "validation")
    print(f"Scoring {len(val_qrels)} validation jobs ...")
    val_comps = components_for_split(svc, val_qrels)

    reranker_only = {"reranker": 1.0, "required_skills": 0.0, "experience": 0.0}
    trials = [(w, evaluate(w, val_comps, val_qrels)) for w in weight_grid(0.1)]
    best_w, best_val = max(trials, key=lambda t: (t[1]["ndcg@10"] or 0, t[0]["reranker"]))
    base_val = evaluate(reranker_only, val_comps, val_qrels)

    rows = [{"split": "validation", "labels": val_source, "scoring": "reranker only", **base_val},
            {"split": "validation", "labels": val_source, "scoring": "tuned final score", **best_val}]
    for source in sources:
        test_qrels = load_qrels(cfg, source, "test")
        test_comps = components_for_split(svc, test_qrels)
        rows.append({"split": "test", "labels": source, "scoring": "reranker only", **evaluate(reranker_only, test_comps, test_qrels)})
        rows.append({"split": "test", "labels": source, "scoring": "tuned final score", **evaluate(best_w, test_comps, test_qrels)})

    results = resolve(cfg["paths"]["results_dir"])
    (results / "score_weights.json").write_text(json.dumps(
        {"weights": best_w, "tuned_on": f"validation ({val_source} labels)", "validation_ndcg@10": best_val["ndcg@10"],
         "reranker": svc.reranker.model_name if svc.reranker else None,
         "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    md = ["# Final match score", "", f"Weights tuned on validation: `{json.dumps(best_w)}`", "",
          "| Split | Labels | Scoring | nDCG@10 | MRR | Queries |", "|---|---|---|---|---|---|"]
    md += [f"| {r['split']} | {r['labels']} | {r['scoring']} | {r['ndcg@10']:.3f} | {r['mrr']:.3f} | {r['queries']} |" for r in rows]
    (results / "final_score.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def quotes(cfg, svc: MatchingService, n_jobs: int) -> None:
    client = make_client(cfg)
    sources = available_sources(cfg)
    qrels = load_qrels(cfg, sources[0], "test") if sources else {}
    job_ids = sorted(qrels)[:n_jobs] or sorted(svc.jobs.index)[:n_jobs]
    total = kept = dropped = errors = 0
    examples = []
    for job_id in job_ids:
        j = svc.jobs.loc[job_id]
        job = svc.job_struct_for_id(job_id)
        top = svc.score_candidates(job, j.text, [d for d, _ in svc.retrievers["hybrid"].search(f"{j.title}\n{j.text}", 100)])[:3]
        brief = f"{j.title}. Required: {', '.join(job.get('required_skills', [])[:12])}"
        for item in top:
            try:
                q = get_evidence_quotes(client, cfg["scoring"]["quotes_model"], brief, svc.resumes.loc[item["resume_id"], "text"],
                                        cfg["scoring"]["max_quotes"], cfg["scoring"]["min_quote_chars"])
            except RateLimitExhausted as e:
                print(f"Stopped at the rate limit: {e}")
                break
            errors += q["error"] is not None
            kept += len(q["kept"]); dropped += len(q["dropped"]); total += len(q["kept"]) + len(q["dropped"])
            if len(examples) < 5 and (q["kept"] or q["dropped"]):
                examples.append({"job_id": job_id, "job_title": j.title, "resume_id": item["resume_id"],
                                 "score": item["score"], "required_matched": item["required_skills"]["matched"],
                                 "required_missing": item["required_skills"]["missing"],
                                 "experience": item["experience"]["summary"], "kept": q["kept"], "dropped": q["dropped"]})
    stats = {"pairs": len(job_ids) * 3, "quotes_generated": total, "kept_verified": kept, "dropped_unverified": dropped,
             "failure_rate": round(dropped / total, 4) if total else None, "llm_errors": errors,
             "model": cfg["scoring"]["quotes_model"], "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    results = resolve(cfg["paths"]["results_dir"])
    (results / "explanations.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    md = ["# Explanation examples", ""]
    for e in examples:
        md += [f"## {e['job_id']} {e['job_title']} -> {e['resume_id']} (score {e['score']:.3f})",
               f"- Required matched: {', '.join(e['required_matched']) or '–'}",
               f"- Required missing: {', '.join(e['required_missing']) or '–'}",
               f"- Experience: {e['experience']}",
               *[f"- ✅ \"{q['quote']}\" ({q['supports']})" for q in e["kept"]],
               *[f"- ❌ dropped, not found in resume: \"{q['quote']}\"" for q in e["dropped"]], ""]
    (results / "explanation_examples.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--quotes", type=int, default=0, metavar="N_JOBS")
    args = p.parse_args()
    if not args.tune and not args.quotes:
        p.error("choose --tune and/or --quotes N")
    load_dotenv(ROOT / ".env")
    cfg = load_config()
    svc = MatchingService(cfg, use_reranker=True)
    if args.tune:
        tune(cfg, svc)
    if args.quotes:
        quotes(cfg, svc, args.quotes)


if __name__ == "__main__":
    main()
