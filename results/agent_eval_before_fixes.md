# Recruiter agent evaluation

Generated 2026-11-17T14:54:30+00:00 by scripts/07_eval_agent.py. Model qwen/qwen3.8-27b, max 8 steps, re-ranker on. Cost uses the (unverified) prices in config and is 0 for cached calls.

**Success rate: 90%** (18/20 tasks) · avg LLM calls 3.4 · avg tool calls 2.6 · avg tokens 5977 · avg cost $0.0002

| Task | Result | LLM calls | Tool calls | Failures |
|---|---|---|---|---|
| t01_accountant_quickbooks | ✅ | 2 | 1 |  |
| t02_hr_payroll_compare | ✅ | 3 | 2 |  |
| t03_chef_food_safety_outreach | ✅ | 4 | 3 |  |
| t04_it_sql | ✅ | 4 | 3 |  |
| t05_impossible_kafka_bangalore | ✅ | 5 | 4 |  |
| t06_teacher_two_skills | ✅ | 2 | 1 |  |
| t07_designer_explain | ❌ | 4 | 3 | candidates not meeting {'skills': ['Adobe Photoshop']}: ['R39252859'] |
| t08_healthcare_emr | ✅ | 3 | 4 |  |
| t09_construction_autocad_compare | ✅ | 6 | 5 |  |
| t10_direct_draft | ✅ | 2 | 1 |  |
| t11_refuse_to_approve | ✅ | 2 | 1 |  |
| t12_pilots_faa | ✅ | 3 | 4 |  |
| t13_seo | ✅ | 3 | 2 |  |
| t14_java_explain_draft | ❌ | 5 | 5 | no final answer (stopped: max_steps) |
| t15_rare_skill | ✅ | 4 | 3 |  |
| t16_compare_given_ids | ✅ | 2 | 1 |  |
| t17_location_unavailable | ✅ | 3 | 2 |  |
| t18_pr_social_top5 | ✅ | 3 | 2 |  |
| t19_sales_salesforce_two_drafts | ✅ | 5 | 4 |  |
| t20_banking_min_years | ✅ | 3 | 2 |  |
