# Track C — real CEEW telemetry with injected theft labels

**REAL telemetry (CEEW Mathura/Bareilly, CC0) with SYNTHETIC injected labels (Zidi et al. 2023 + our DT-hook extension). NOT real theft labels. Virtual DTs, star topology, single-phase, no weather. No field-performance claim.**

## Setup

- Dataset: CEEW/Harvard Dataverse doi:10.7910/DVN/GOCHJH (Kaggle mirror, CC0); 15 min (resampled from 3 min); districts ['BR', 'MH']
- Injection: Zidi et al. (2023) six-type generator, prevalence 9%, seed 42; DT-hook = our extension
- Injection window: assessment days 7-13 only; baseline days 0-6 stay clean so own-history features keep their reference (full injection self-cancels every ratio feature)
- Windows: 35 valid 14-day monthly windows (skipped: 0)
- Models: frozen served artifacts (M0, M1, M2, M3, M4); features unchanged (pinned build.py)
- Physics rule: dt input from TRUE loads + sweep loss; theft touches reported readings only

## Pooled metrics (meter-window instances)

| Model | ROC-AUC | PR-AUC | MW p | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|---|---|---|---|
| M0 | 0.749 | 0.324 | 5.89e-12 | 0.30 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| M1 | 0.648 | 0.177 | 2.77e-05 | 0.80 | 1.00 | 0.29 | 0.06 | 0.00 | 0.11 |
| M2 | 0.749 | 0.401 | 2.44e-12 | 0.60 | 0.07 | 0.71 | 1.00 | 0.00 | 0.11 |
| M3 | 0.733 | 0.236 | 8.93e-11 | 0.60 | 0.00 | 0.71 | 1.00 | 0.00 | 0.11 |
| M4 | 0.777 | 0.382 | 2.00e-14 | 0.90 | 0.67 | 0.86 | 0.67 | 0.00 | 0.11 |

(T1–T6 = per-type recall at top-20% inspection budget; T6 is the type Zidi et al.'s RF caught at only 3.1%.)

## DT-hook detection (our extension)

- Windows with hook: 12
- Hooked DT ranked #1 by balance residual: 1.0
- Mean balance AUC (hooked vs clean DTs): 0.9583333333333334

## Periodicity validation (Zheng et al. IEEE TII lens, full history)

- Daily-total WoW-PCC: meters 84; mean clean -0.004984055742053644 vs injected -0.004896128603000317; share > 0.8 clean 0.0 vs injected 0.0
- Slot-weekly WoW-PCC: mean clean 0.22826381563073533 vs injected 0.21811395047643037; share > 0.8 clean 0.0 vs injected 0.0
- Slot-weekly per-type (clean -> injected): theft_1 0.024->0.008 (n=1), theft_2 0.205->0.159 (n=2), theft_3 0.338->0.218 (n=2), theft_4 0.128->-0.002 (n=1), theft_5 0.323->-0.048 (n=1), theft_6 0.286->0.283 (n=1)
- Note: WoW-PCC over full history, clean vs injected. Zheng et al. thresholds (normal > 0.8, thieves < 0.7) quoted for reference; transfer to this cohort is an empirical question (clean meters already sit far below 0.8). WoW-PCC is invariant to within-day reversal (T6) by construction: the same permutation applies to both correlated weeks.

## Limitations

Injected labels are synthetic (not real theft); virtual DTs with star topology;
single-phase; neutral temperature (no weather); meters repeat across windows
(pooled instances are not independent); small cohort.
