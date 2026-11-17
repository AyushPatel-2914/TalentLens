"""Upload the API's data artifacts to a PRIVATE Hugging Face dataset repo (for deployment).

Usage:
    python scripts/publish_artifacts.py <hf-username>/talentlens-artifacts

Needs HF_TOKEN (write access) in .env. The repo is created private. Check the
Kaggle dataset licences before sharing it with anyone, and never make it public:
it contains (de-identified) resume text.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from talentlens.artifacts import ARTIFACT_PATTERNS
from talentlens.config import ROOT


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    load_dotenv(ROOT / ".env")
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    repo = sys.argv[1]
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(ROOT), repo_id=repo, repo_type="dataset", allow_patterns=ARTIFACT_PATTERNS)
    print(f"Uploaded to https://huggingface.co/datasets/{repo} (private). "
          f"Set TALENTLENS_ARTIFACTS_REPO={repo} and HF_TOKEN on the server.")


if __name__ == "__main__":
    main()
