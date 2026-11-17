"""Stage 3, step 3: LLM judge.

Usage:
    # 1. Label YOUR 200 pairs with the judge and measure agreement with you:
    python scripts/03c_judge.py --agreement --model openai/gpt-oss-120b
    python scripts/03c_judge.py --agreement --model openai/gpt-oss-20b      # compare models
    # 2. Only if weighted kappa >= threshold, label all ~3,000 pool pairs:
    python scripts/03c_judge.py --label-all --model openai/gpt-oss-120b

Safe to stop and re-run: calls are cached. --label-all stops cleanly at the daily quota.

Writes:
    data/labels/judge/{model}.jsonl                 judge labels
    results/judge_agreement__{model}.json / .md     kappa, confusion matrix
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from talentlens.config import load_config, resolve
from talentlens.extraction.runner import make_client
from talentlens.labeling.judge import agreement, judge_pair
from talentlens.llm.client import LLMError, RateLimitExhausted


def safe(model: str) -> str:
    return model.replace("/", "__")


def run_judge(cfg, model: str, pairs: pd.DataFrame) -> pd.DataFrame:
    lab = cfg["labels"]
    processed = resolve(cfg["paths"]["processed_dir"])
    jobs = pd.read_parquet(processed / "jobs.parquet").set_index("job_id")
    resumes = pd.read_parquet(processed / "resumes.parquet").set_index("resume_id")
    client = make_client(cfg)

    out_path = resolve(lab["judge_pairs_dir"]) / f"{safe(model)}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_json(out_path, lines=True) if out_path.exists() and out_path.stat().st_size else pd.DataFrame()
    done = set(existing["pair_id"]) if len(existing) else set()

    todo = pairs[~pairs["pair_id"].isin(done)]
    with open(out_path, "a", encoding="utf-8") as f:
        for p in tqdm(todo.itertuples(index=False), total=len(todo), desc=f"judge {model}"):
            job = jobs.loc[p.job_id]
            job_text = f"Title: {job.title}\nLocation: {job.location}\n\n{job.text}"
            try:
                out, err = judge_pair(client, model, job_text, resumes.loc[p.resume_id, "text"],
                                      lab["judge_max_job_chars"], lab["judge_max_resume_chars"])
            except RateLimitExhausted as e:
                print(f"\nStopped at the rate limit: {e}\nRe-run later to continue.")
                break
            except LLMError as e:
                out, err = None, str(e)[:200]
            f.write(json.dumps({"pair_id": p.pair_id, "label": out.label if out else None,
                                "reason": out.reason if out else None, "error": err}) + "\n")
            f.flush()
    return pd.read_json(out_path, lines=True).drop_duplicates("pair_id", keep="last")


def write_agreement(cfg, model: str, stats: dict) -> None:
    results = resolve(cfg["paths"]["results_dir"])
    stats = {"model": model, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "threshold": cfg["labels"]["kappa_threshold"], **stats}
    stats["passes_threshold"] = stats["weighted_kappa_quadratic"] >= stats["threshold"]
    (results / f"judge_agreement__{safe(model)}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    cm = stats["confusion_matrix"]["rows_human_cols_judge"]
    md = [
        f"# LLM judge agreement: {model}", "",
        f"Pairs: {stats['n']} (unparseable judge replies excluded: {stats['judge_failures']})", "",
        "| Metric | Value |", "|---|---|",
        f"| Accuracy | {stats['accuracy']:.3f} |",
        f"| Cohen's kappa | {stats['cohen_kappa']:.3f} |",
        f"| Quadratic weighted kappa | {stats['weighted_kappa_quadratic']:.3f} |",
        f"| Threshold | {stats['threshold']} -> {'PASS' if stats['passes_threshold'] else 'FAIL'} |", "",
        "Confusion matrix (rows = your label, columns = judge label):", "",
        "| | judge 0 | judge 1 | judge 2 |", "|---|---|---|---|",
        *[f"| you {i} | {r[0]} | {r[1]} | {r[2]} |" for i, r in enumerate(cm)],
    ]
    (results / f"judge_agreement__{safe(model)}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--agreement", action="store_true")
    mode.add_argument("--label-all", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--force", action="store_true", help="label all even if agreement failed (not recommended)")
    args = p.parse_args()

    cfg = load_config()
    model = args.model or cfg["llm"]["models"]["strong"]
    pools = pd.read_parquet(resolve(cfg["labels"]["pools_file"]))

    if args.agreement:
        human_path = resolve(cfg["labels"]["human_pairs_file"])
        if not human_path.exists():
            raise SystemExit("No human labels yet. Run scripts/03b_label_pairs.py first.")
        human = pd.read_json(human_path, lines=True).drop_duplicates("pair_id", keep="last")
        judged = run_judge(cfg, model, pools[pools["pair_id"].isin(human["pair_id"])])
        merged = human.merge(judged[["pair_id", "label"]], on="pair_id", suffixes=("_human", "_judge"))
        failures = int(merged["label_judge"].isna().sum())
        merged = merged.dropna(subset=["label_judge"])
        stats = agreement(merged["label_human"].astype(int).tolist(), merged["label_judge"].astype(int).tolist())
        write_agreement(cfg, model, {**stats, "judge_failures": failures})
        return

    agreement_file = resolve(cfg["paths"]["results_dir"]) / f"judge_agreement__{safe(model)}.json"
    if not agreement_file.exists():
        raise SystemExit(f"Run --agreement with {model} first.")
    stats = json.loads(agreement_file.read_text(encoding="utf-8"))
    if not stats["passes_threshold"] and not args.force:
        raise SystemExit(f"Weighted kappa {stats['weighted_kappa_quadratic']:.3f} < {stats['threshold']}. "
                         "Improve the rubric/prompt (bump the judge prompt version) and re-measure first.")
    judged = run_judge(cfg, model, pools)
    print(f"Judge labels: {judged['label'].notna().sum()}/{len(pools)} pairs. "
          f"Distribution: {judged['label'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
