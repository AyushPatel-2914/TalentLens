"""Stage 3, step 2: hand-label the 200 job-resume pairs (0 / 1 / 2).

Usage:
    python scripts/03b_label_pairs.py

The job and resume are also written to data/labels/_current_job.txt and
data/labels/_current_resume.txt: open both in VS Code side by side.

Pairs come in a shuffled order and you never see which retrieval method found
the candidate, so your labels can't be biased toward one method.
Progress is saved after every pair. Keys: 0 / 1 / 2, s = skip for now, q = quit.
"""

from __future__ import annotations

import json
import random
import textwrap
from datetime import datetime, timezone

import pandas as pd

from talentlens.config import load_config, resolve
from talentlens.labeling.rubric import RUBRIC


def main() -> None:
    cfg = load_config()
    lab = cfg["labels"]
    processed = resolve(cfg["paths"]["processed_dir"])
    pools = pd.read_parquet(resolve(lab["pools_file"]))
    jobs = pd.read_parquet(processed / "jobs.parquet").set_index("job_id")
    resumes = pd.read_parquet(processed / "resumes.parquet").set_index("resume_id")

    out = resolve(lab["human_pairs_file"])
    done = set()
    if out.exists():
        done = {json.loads(l)["pair_id"] for l in out.read_text(encoding="utf-8").splitlines() if l}
    todo = pools[pools["for_human"] & ~pools["pair_id"].isin(done)]["pair_id"].tolist()
    random.Random(cfg["data"]["seed"]).shuffle(todo)
    pairs = pools.set_index("pair_id")
    total = int(pools["for_human"].sum())

    print(RUBRIC)
    print(f"\n{len(done)} labelled, {len(todo)} to go.")
    job_file, resume_file = out.parent / "_current_job.txt", out.parent / "_current_resume.txt"

    for pair_id in todo:
        p = pairs.loc[pair_id]
        job, resume = jobs.loc[p.job_id], resumes.loc[p.resume_id]
        job_text = f"{job.title}\nLocation: {job.location}\n\n{job.text}"
        job_file.write_text(job_text, encoding="utf-8")
        resume_file.write_text(resume.text, encoding="utf-8")

        print("\n" + "#" * 100)
        print(f"[{len(done) + 1}/{total}]  JOB {p.job_id}: {job.title}")
        print("#" * 100)
        print(textwrap.fill(job.text[:2500], 100) + (" ..." if len(job.text) > 2500 else ""))
        print("-" * 100 + f"\nCANDIDATE {p.resume_id}\n" + "-" * 100)
        print(textwrap.fill(resume.text[:3500], 100) + (" ... (full text in _current_resume.txt)" if len(resume.text) > 3500 else ""))

        while True:
            key = input("\nLabel 0 = no fit, 1 = partial, 2 = strong | s = skip | q = quit: ").strip().lower()
            if key in {"0", "1", "2", "s", "q"}:
                break
        if key == "q":
            print("Stopped. Run again to continue.")
            return
        if key == "s":
            continue
        record = {"pair_id": pair_id, "job_id": p.job_id, "resume_id": p.resume_id, "label": int(key),
                  "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        done.add(pair_id)

    print(f"Done: {len(done)}/{total} labelled. Next: python scripts/03c_judge.py --agreement")


if __name__ == "__main__":
    main()
