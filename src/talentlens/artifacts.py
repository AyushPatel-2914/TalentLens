"""Ship the data the API needs (parquet, indexes, extractions) without committing it to git.

`data/` is git-ignored because it contains resume text. For deployment, the
artifacts are uploaded once to a PRIVATE Hugging Face dataset repo
(scripts/publish_artifacts.py), and the server downloads them at startup when
TALENTLENS_ARTIFACTS_REPO is set (needs HF_TOKEN with read access).
"""

from __future__ import annotations

import os

from talentlens.config import ROOT

ARTIFACT_PATTERNS = ["data/processed/*", "data/indexes/**", "data/extracted/*", "results/score_weights.json"]


def ensure_artifacts() -> None:
    repo = os.environ.get("TALENTLENS_ARTIFACTS_REPO")
    if not repo or (ROOT / "data" / "indexes" / "resumes" / "bm25.pkl").exists():
        return
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(ROOT), allow_patterns=ARTIFACT_PATTERNS,
                      token=os.environ.get("HF_TOKEN"))
