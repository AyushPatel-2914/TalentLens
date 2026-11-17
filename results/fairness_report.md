# Fairness report (final score (cross-encoder + requirement checks))

Generated 2026-11-17T10:10:20+00:00 by scripts/08_fairness.py. 10 test jobs x top-20 candidates = 200 candidate slots x 10 name groups = 2000 counterfactuals per condition.

| | Mean abs rank shift | Max abs rank shift | % top-10 lists changed | Mean abs score diff | Max abs score diff |
|---|---|---|---|---|---|
| Before mitigation | 1.718 | 15 | 27.4% | 0.04122 | 0.39890 |
| After mitigation (redaction) | 0.000 | 0 | 0.0% | 0.00000 | 0.00000 |

## Direction by group (before mitigation)

Signed mean score difference vs. the original resume (positive = the name helped).

| Group | Mean score diff | Mean rank shift |
|---|---|---|
| african_american_female | -0.03950 | +1.570 |
| african_american_male | -0.03645 | +1.405 |
| anglo_female | -0.03021 | +1.215 |
| anglo_male | -0.03372 | +1.300 |
| east_asian_female | -0.03134 | +1.105 |
| east_asian_male | -0.03256 | +1.295 |
| hispanic_female | -0.02636 | +1.045 |
| hispanic_male | -0.02950 | +1.200 |
| indian_female | -0.02823 | +1.125 |
| indian_male | -0.03948 | +1.565 |

## What this test does and does not show

- It shows how much the ranking reacts to a name line and gender-coded words **with everything else held equal**.
- It does **not** test proxies for identity: college names, gaps in employment, clubs, hobbies, language style.
- Names are an imperfect stand-in for gender or region, and 30 names cannot represent whole groups.
- It measures the second stage (scores of already-retrieved candidates). Names could also change which candidates first-stage retrieval finds; that would need re-indexing every variant.
- Bias can also come from the training labels (human or LLM judge), which this test does not measure.
- After mitigation the name signal is removed by construction, so a near-zero shift there is expected. It shows the redaction works, not that the model is unbiased.
- A small shift on 200 slots means no large effect was detected by this test, not that the system is fair.
