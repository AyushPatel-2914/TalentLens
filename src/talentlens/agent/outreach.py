"""Outreach drafts with a human-approval gate.

The agent can only CREATE drafts (status PENDING_APPROVAL). Changing a draft to
APPROVED or REJECTED is a separate method that only the API endpoint / UI button
calls; there is no agent tool for it. So the agent cannot send or approve anything.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

PENDING, APPROVED, REJECTED = "PENDING_APPROVAL", "APPROVED", "REJECTED"


class OutreachStore:
    def __init__(self, path: Path | None = None):
        self.path = path
        self._lock = threading.Lock()
        self._drafts: dict[str, dict] = {}
        if path and path.exists():
            self._drafts = json.loads(path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._drafts, indent=2), encoding="utf-8")

    def create_draft(self, candidate_id: str, job_ref: str, subject: str, body: str) -> dict:
        with self._lock:
            draft = {"id": uuid.uuid4().hex[:10], "candidate_id": candidate_id, "job_ref": job_ref,
                     "subject": subject, "body": body, "status": PENDING,
                     "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "decided_at": None}
            self._drafts[draft["id"]] = draft
            self._save()
            return dict(draft)

    def decide(self, draft_id: str, approve: bool, edited_body: str | None = None) -> dict:
        """Human action only (API / UI). Raises KeyError for unknown ids, ValueError if already decided."""
        with self._lock:
            draft = self._drafts[draft_id]
            if draft["status"] != PENDING:
                raise ValueError(f"draft {draft_id} is already {draft['status']}")
            if edited_body:
                draft["body"] = edited_body
            draft["status"] = APPROVED if approve else REJECTED
            draft["decided_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._save()
            return dict(draft)

    def get(self, draft_id: str) -> dict:
        return dict(self._drafts[draft_id])

    def list(self, status: str | None = None) -> list[dict]:
        return [dict(d) for d in self._drafts.values() if status is None or d["status"] == status]
