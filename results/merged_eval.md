# Merged-dataset evaluation (synthetic_india_v2)

**Everything in this report is SYNTHETIC.** The dataset is a third-party
research prototype with assumption-driven parameters and no supplied
generator; it is not calibrated to Indian AMI field data. No Indian field
performance is claimed.

## Track A — inspection-supervised (headline)

Models never see hidden truth. Scoring is the inherited M0 own-history
residual ranking; evidence is the quarantined inspection log only.

- Meters scored: 1800 (40 DTs x 60 days)
- theft_found inspections: 3 of the 6 dataset-wide after quarantine (8 inconsistent rows removed, never relabelled)
- no_theft controls: 131; not_accessible excluded from negatives: 12
- Ranks of the 3 theft_found meters (of 1800): {'M000790': 127, 'M001576': 505, 'M001951': 185}
- In top-50: 0 (chance expectation 0.08); top-25: 0
- Mann-Whitney U (found > no_theft), p = 0.011773477177486948

Dataset-wide only 6 theft_found inspections exist after quarantine (3 of them in this cohort); they are sparse, selected and biased. Enrichment is indicative only.

## Track B — oracle evaluation (declared)

Hidden truth used for evaluation only. This is a synthetic-oracle diagnostic,
not inspection-supervised performance and not field performance.

- Theft prevalence: 0.094
- M0 residual ROC-AUC 0.908, PR-AUC 0.674
- DT-balance vs direct-hooking DTs: AUC 0.542 (hooking is unmetered upstream load; DT balance is the only honest signal)
- Per-family presence in the top-50: {"collusion": {"positives": 24, "in_top50": 0, "median_rank": 98.5}, "direct_hooking": {"positives": 18, "in_top50": 1, "median_rank": 880.0}, "fractional_underrecord": {"positives": 18, "in_top50": 2, "median_rank": 92.0}, "frozen_meter": {"positives": 18, "in_top50": 17, "median_rank": 19.5}, "partial_bypass": {"positives": 23, "in_top50": 5, "median_rank": 70.0}, "tamper_assisted": {"positives": 21, "in_top50": 3, "median_rank": 83.0}, "tariff_misuse": {"positives": 17, "in_top50": 1, "median_rank": 141.0}, "time_suppression": {"positives": 31, "in_top50": 3, "median_rank": 136.0}}

## Protocol
Event flags excluded from predictors; hidden truth never a predictor;
feeder/pole identifiers never a predictor; 8 inconsistent inspections
quarantined; uncovered intervals never treated as zeros (eval cohort coverage
is complete by construction).
