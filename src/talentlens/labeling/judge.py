"""LLM judge: labels a (job, resume) pair 0/1/2 with the same rubric you use.

The judge writes a one-sentence reason BEFORE the label: making the model state
its evidence first tends to give more consistent labels than asking for a bare
number. Agreement with your labels is measured with Cohen's kappa, and the
judge is only used for more pairs if quadratic-weighted kappa reaches the threshold.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from talentlens.extraction.extract import parse_json
from talentlens.labeling.rubric import RUBRIC
from talentlens.llm.client import LLMClient, LLMError, RateLimitExhausted

JUDGE_PROMPT = f"""You are an experienced recruiter labelling job-candidate fit for an evaluation dataset.

{RUBRIC}

Reply with ONE JSON object: {{"reason": "<one sentence citing the key evidence>", "label": 0 | 1 | 2}}"""


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    reason: str
    label: Literal[0, 1, 2]


def judge_pair(client: LLMClient, model: str, job_text: str, resume_text: str,
               max_job_chars: int, max_resume_chars: int, prompt_version: str = "j1",
               max_repairs: int = 1) -> tuple[JudgeOutput | None, str | None]:
    messages = [
        {"role": "system", "content": f"[judge {prompt_version}]\n{JUDGE_PROMPT}"},
        {"role": "user", "content": f"JOB:\n{job_text[:max_job_chars]}\n\nCANDIDATE RESUME:\n{resume_text[:max_resume_chars]}"},
    ]
    error = None
    for _ in range(max_repairs + 1):
        try:
            resp = client.chat(model, messages, json_mode=True, tag="judge")
        except RateLimitExhausted:
            raise
        except LLMError as e:
            if "json_validate_failed" not in str(e):
                raise
            error = "server-side JSON validation failed"
            continue
        try:
            return JudgeOutput.model_validate(parse_json(resp.content)), None
        except (ValueError, ValidationError) as e:
            error = str(e)[:200]
            messages = messages + [
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": f"Invalid reply: {error}. Reply with the JSON object only."},
            ]
    return None, error


def agreement(human: list[int], judge: list[int]) -> dict:
    """Cohen's kappa (plain and quadratic-weighted), accuracy and a 3x3 confusion matrix."""
    if len(human) != len(judge) or not human:
        raise ValueError("need two equal-length, non-empty label lists")
    cm = confusion_matrix(human, judge, labels=[0, 1, 2])
    return {
        "n": len(human),
        "accuracy": float(sum(h == j for h, j in zip(human, judge)) / len(human)),
        "cohen_kappa": float(cohen_kappa_score(human, judge, labels=[0, 1, 2])),
        "weighted_kappa_quadratic": float(cohen_kappa_score(human, judge, labels=[0, 1, 2], weights="quadratic")),
        "confusion_matrix": {"rows_human_cols_judge": cm.tolist(), "labels": [0, 1, 2]},
        "label_distribution": {
            "human": {str(k): int(sum(h == k for h in human)) for k in (0, 1, 2)},
            "judge": {str(k): int(sum(j == k for j in judge)) for k in (0, 1, 2)},
        },
    }
