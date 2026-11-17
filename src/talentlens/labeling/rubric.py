"""The single labelling rubric, shown to you in the CLI and given to the LLM judge.

Using the exact same text for both is what makes the agreement number meaningful.
"""

RUBRIC = """Label how well the CANDIDATE fits the JOB.

2 = STRONG FIT. Same occupation/field AND meets most of the core requirements
    (key skills, roughly the required experience and level). A recruiter would
    happily put this person on the shortlist.

1 = PARTIAL FIT. Related field or transferable background, but clear gaps:
    missing some key skills, noticeably less (or much more) experience, or a
    different level. Worth a look, not a top pick.

0 = NO FIT. Different occupation or field, or missing most core requirements.
    A recruiter would not contact this person for this job.

Rules:
- Judge skills and experience only. Ignore names, gender, age, nationality,
  and anything else not related to doing the job.
- Missing location information is not a reason to lower the label.
- Unclear between two labels? Choose the lower one."""
