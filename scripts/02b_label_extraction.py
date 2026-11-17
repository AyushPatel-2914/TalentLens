"""Hand-label the 100 gold resumes, one at a time. Progress is saved after every resume.

Usage:
    python scripts/02b_label_extraction.py              # blind: you type every field (recommended)
    python scripts/02b_label_extraction.py --assist     # start from the 120b model's answer and correct it

Tip: the current resume is also written to data/labels/_current_resume.txt.
Open that file in VS Code next to the terminal; it's easier to read than the terminal.

Blind mode gives the most trustworthy labels. Assisted mode is faster but tends
to pull your labels toward the model's answer, which makes the model look better
than it is. Each label records which mode was used, and the evaluation reports it.

Years of experience: these resumes were collected in mid-2021, so count "Current" as June 2021,
and don't double-count overlapping jobs.

Lists: separate items with commas (or semicolons if an item contains a comma).
Leave a number blank for "unknown". Press Ctrl+C any time; finished resumes are kept.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime, timezone
from typing import get_args

import pandas as pd

from talentlens.config import load_config, resolve
from talentlens.extraction.runner import load_documents, load_gold_ids, output_path
from talentlens.extraction.schemas import EducationLevel, ResumeExtraction, Seniority

EDU = list(get_args(EducationLevel))
SEN = list(get_args(Seniority))


def split_items(raw: str) -> list[str]:
    sep = ";" if ";" in raw else ","
    return [x.strip() for x in raw.split(sep) if x.strip()]


def ask_list(label: str, current: list[str] | None) -> list[str]:
    if current is None:
        return split_items(input(f"{label} (comma-separated): "))
    items = list(current)
    while True:
        print(f"\n{label}:")
        for i, s in enumerate(items, 1):
            print(f"  {i:>2}. {s}")
        cmd = input("Enter = keep | -3,7 remove | +A, B add: ").strip()
        if not cmd:
            return items
        if cmd.startswith("-"):
            drop = {int(x) for x in cmd[1:].replace(" ", "").split(",") if x.isdigit()}
            items = [s for i, s in enumerate(items, 1) if i not in drop]
        elif cmd.startswith("+"):
            items += split_items(cmd[1:])


def ask_number(label: str, current: float | None) -> float | None:
    hint = f" [Enter = {current}]" if current is not None else ""
    while True:
        raw = input(f"{label} (number, blank = unknown){hint}: ").strip()
        if not raw:
            return current
        if raw.lower() in {"none", "null", "-"}:
            return None
        try:
            return float(raw)
        except ValueError:
            print("  Please type a number.")


def ask_choice(label: str, options: list[str], current: str | None) -> str:
    print(f"{label}: " + "  ".join(f"{i}={o}" for i, o in enumerate(options, 1)))
    hint = f" [Enter = {current}]" if current else " [Enter = unknown]"
    while True:
        raw = input(f"  choose 1-{len(options)}{hint}: ").strip()
        if not raw:
            return current or "unknown"
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("  Invalid choice.")


def ask_text(label: str, current: str | None) -> str | None:
    hint = f" [Enter = {current}]" if current else ""
    raw = input(f"{label} (blank = none){hint}: ").strip()
    return raw or current


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assist", action="store_true")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    gold_file = resolve(cfg["extraction"]["gold_file"])
    gold_file.parent.mkdir(parents=True, exist_ok=True)
    current_file = gold_file.parent / "_current_resume.txt"

    done = set()
    if gold_file.exists():
        done = {json.loads(line)["doc_id"] for line in gold_file.read_text(encoding="utf-8").splitlines() if line}
    ids = [i for i in load_gold_ids(cfg) if i not in done]
    docs = load_documents(cfg, "resume").set_index("doc_id")

    preds = None
    if args.assist:
        path = output_path(cfg, "resume", cfg["llm"]["models"]["strong"])
        if not path.exists():
            raise SystemExit(f"--assist needs {path}. Run: python scripts/02_extract.py --kind resume --gold "
                             f"--model {cfg['llm']['models']['strong']}")
        preds = pd.read_parquet(path).set_index("doc_id")

    print(f"{len(done)} labelled, {len(ids)} to go. Mode: {'assisted' if args.assist else 'blind'}.")
    for n, doc_id in enumerate(ids, len(done) + 1):
        text = docs.loc[doc_id, "text"]
        current_file.write_text(text, encoding="utf-8")
        print("\n" + "=" * 100)
        print(f"[{n}/{len(done) + len(ids)}] {doc_id}  category: {docs.loc[doc_id, 'category']}")
        print("=" * 100)
        print(textwrap.fill(text, width=100))
        print("=" * 100)

        pr = preds.loc[doc_id].to_dict() if preds is not None and doc_id in preds.index else {}
        def cur(key):
            v = pr.get(key)
            return list(v) if hasattr(v, "__len__") and not isinstance(v, str) else v

        while True:
            label = {
                "skills": ask_list("Skills", cur("skills") if args.assist else None),
                "total_years_experience": ask_number("Total years of experience ('Current' = mid-2021)", cur("total_years_experience")),
                "job_titles": ask_list("Job titles, most recent first", cur("job_titles") if args.assist else None),
                "education_level": ask_choice("Education", EDU, cur("education_level")),
                "location": ask_text("Location", cur("location")),
                "seniority": ask_choice("Seniority", SEN, cur("seniority")),
            }
            try:
                label = ResumeExtraction.model_validate(label).model_dump()
            except Exception as e:  # show the problem and let the labeller redo it
                print(f"Invalid label: {e}")
                continue
            print(json.dumps(label, indent=2))
            choice = input("(s)ave / (r)edo / (q)uit without saving this one [s]: ").strip().lower() or "s"
            if choice == "r":
                continue
            if choice == "q":
                print("Stopped. Run again to continue.")
                return
            break

        record = {"doc_id": doc_id, "mode": "assisted" if args.assist else "blind",
                  "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **label}
        with open(gold_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved ({n}).")

    print("All gold resumes labelled. Next: python scripts/02c_eval_extraction.py")


if __name__ == "__main__":
    main()
