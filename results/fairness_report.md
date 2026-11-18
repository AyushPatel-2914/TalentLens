# Fairness report (final score (cross-encoder + requirement checks))

Generated 2026-11-18T08:26:01+00:00 by scripts/08_fairness.py. 10 test jobs x top-20 candidates = 200 candidate slots x 10 name groups = 2000 counterfactuals per condition.

| | Mean abs rank shift | Max abs rank shift | % top-10 lists changed | Mean abs score diff | Max abs score diff |
|---|---|---|---|---|---|
| Before mitigation | 0.919 | 17 | 19.2% | 0.04227 | 0.85690 |
| After mitigation (redaction) | 0.000 | 0 | 0.0% | 0.00000 | 0.00000 |

## Direction by group (before mitigation)

Signed mean score difference vs. the original resume (positive = the name helped).

| Group | Mean score diff | Mean rank shift |
|---|---|---|
| african_american_female | -0.00424 | +0.105 |
| african_american_male | -0.00514 | +0.020 |
| anglo_female | -0.00164 | +0.020 |
| anglo_male | -0.00834 | +0.170 |
| east_asian_female | -0.01346 | +0.400 |
| east_asian_male | -0.02578 | +0.510 |
| hispanic_female | +0.00509 | -0.055 |
| hispanic_male | +0.00458 | -0.010 |
| indian_female | +0.00463 | +0.005 |
| indian_male | +0.00596 | -0.050 |

## What this test does and does not show

- It shows how much the ranking reacts to a name line and gender-coded words **with everything else held equal**.
- It does **not** test proxies for identity: college names, gaps in employment, clubs, hobbies, language style.
- Names are an imperfect stand-in for gender or region, and 30 names cannot represent whole groups.
- It measures the second stage (scores of already-retrieved candidates). Names could also change which candidates first-stage retrieval finds; that would need re-indexing every variant.
- Bias can also come from the training labels (human or LLM judge), which this test does not measure.
- After mitigation the name signal is removed by construction, so a near-zero shift there is expected. It shows the redaction works, not that the model is unbiased.
- A small shift on 200 slots means no large effect was detected by this test, not that the system is fair.
