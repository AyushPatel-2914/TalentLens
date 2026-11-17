"""Stage 2: score each model's extraction against your hand labels.

Usage:
    python scripts/02c_eval_extraction.py

Reads data/labels/extraction_gold.jsonl and every data/extracted/resumes__*.parquet.
Writes results/extraction_eval.csv and results/extraction_eval.md.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from talentlens.config import load_config, resolve
from talentlens.eval.extraction_metrics import list_field_score, scalar_field_score
from talentlens.extraction.runner import make_normalizer
from talentlens.extraction.skills import skill_key


def _as_list(v) -> list[str]:
    return [] if v is None or (isinstance(v, float) and pd.isna(v)) else list(v)


def _empty(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or v == "unknown" or v == ""


def _norm_text(v) -> str:
    return " ".join(str(v).lower().replace(",", " ").split())


def score_model(gold: pd.DataFrame, pred: pd.DataFrame, normalizer) -> list[dict]:
    df = gold.merge(pred, on="doc_id", how="left", suffixes=("_gold", "_pred"))

    def skill_sets(col):
        return [{skill_key(s) for s in normalizer.normalize_list(_as_list(v))} for v in df[col]]

    def title_sets(col):
        return [{_norm_text(t) for t in _as_list(v)} for v in df[col]]

    scores = [
        list_field_score("skills", skill_sets("skills_gold"), skill_sets("skills_pred")),
        list_field_score("job_titles", title_sets("job_titles_gold"), title_sets("job_titles_pred")),
        scalar_field_score("total_years_experience (±1y)", df["total_years_experience_gold"],
                           df["total_years_experience_pred"], _empty, lambda g, p: abs(float(g) - float(p)) <= 1),
        scalar_field_score("education_level", df["education_level_gold"], df["education_level_pred"],
                           _empty, lambda g, p: g == p),
        scalar_field_score("seniority", df["seniority_gold"], df["seniority_pred"], _empty, lambda g, p: g == p),
        scalar_field_score("location", df["location_gold"], df["location_pred"], _empty,
                           lambda g, p: _norm_text(g) == _norm_text(p)),
    ]
    return [s.as_dict() for s in scores]


def main() -> None:
    cfg = load_config()
    gold_path = resolve(cfg["extraction"]["gold_file"])
    if not gold_path.exists():
        raise SystemExit(f"No labels yet at {gold_path}. Run scripts/02b_label_extraction.py first.")
    gold = pd.read_json(gold_path, lines=True).drop_duplicates("doc_id", keep="last")
    normalizer = make_normalizer(cfg)

    rows = []
    for path in sorted(resolve("data/extracted").glob("resumes__*.parquet")):
        model = path.stem.split("__", 1)[1].replace("__", "/")
        pred = pd.read_parquet(path)
        covered = gold["doc_id"].isin(pred["doc_id"]).sum()
        if covered < len(gold):
            print(f"Skipping {model}: predictions cover only {covered}/{len(gold)} labelled resumes. "
                  f"Run: python scripts/02_extract.py --kind resume --gold --model {model}")
            continue
        pred = pred[pred["doc_id"].isin(gold["doc_id"])]
        valid_rate = pred["status"].isin(["ok", "repaired"]).mean()
        for s in score_model(gold, pred, normalizer):
            rows.append({"model": model, **s, "valid_json_rate": round(float(valid_rate), 4),
                         "mean_tokens_per_resume": round(float((pred.input_tokens + pred.output_tokens).mean()), 1)})

    if not rows:
        raise SystemExit("No model has predictions for all labelled resumes.")
    out = pd.DataFrame(rows)
    results = resolve(cfg["paths"]["results_dir"])
    out.to_csv(results / "extraction_eval.csv", index=False)

    modes = gold["mode"].value_counts().to_dict()
    fmt = lambda x: "–" if pd.isna(x) else f"{x:.3f}"
    lines = [
        "# Extraction accuracy",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by scripts/02c_eval_extraction.py.",
        f"Labelled resumes: {len(gold)} (label modes: {json.dumps(modes)}).",
        "List fields use strict case-insensitive matching after skill normalisation.",
        "",
        "| Model | Field | Precision | Recall | F1 | Accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for r in out.itertuples():
        lines.append(f"| {r.model} | {r.field} | {fmt(r.precision)} | {fmt(r.recall)} | {fmt(r.f1)} | {fmt(r.accuracy)} |")
    (results / "extraction_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
