# Track D — shadow-augmented retraining study (served vs shadow)

**SHADOW models — NOT served; synthetic data + synthetic labels (Zidi family). All published Track A/B/C numbers refer to the ORIGINAL served models and are unchanged. Generator-signature caveat applies (train/eval salts separated).**

## Protocol

- Salts: train 4101, eval 7717, Track C seed 42; demo∩eval disjoint: True
- Shadow train pack: 36 injected meters + 2 DT-hooks (factor 1.15) — DT-hooks live in the shadow TRAIN pack only; the Track D eval bed is meter-level
- Track D eval bed: eval cohort, truth-clean meters, assessment window only, EVAL_SALT (147 meters, 317 positives of 1800)
- Config hashes: served `6e60f7bd16aff2ef` vs shadow `258a4ba4f71fff26`; served artifacts untouched: True

## Test bed 1 — Track B replay (original oracle truth; prevalence 0.0944)

| Model | served ROC | shadow ROC | ΔROC | served PR | shadow PR | ΔPR |
|---|---|---|---|---|---|---|
| M0 | 0.908 | 0.908 | 0.000 | 0.674 | 0.674 | 0.000 |
| M1 | 0.867 | 0.860 | -0.006 | 0.492 | 0.438 | -0.054 |
| M2 | 0.945 | 0.952 | 0.007 | 0.884 | 0.895 | 0.011 |
| M3 | 0.944 | 0.944 | -0.000 | 0.878 | 0.895 | 0.017 |
| M4 | 0.589 | 0.599 | 0.009 | 0.138 | 0.149 | 0.011 |

## Test bed 2 — Track C replay (real CEEW telemetry, injected labels)

Pooled over 35 of 35 windows (skipped 0).

| Model | served ROC | shadow ROC | ΔROC | served PR | shadow PR | ΔPR |
|---|---|---|---|---|---|---|
| M0 | 0.749 | 0.749 | 0.000 | 0.324 | 0.324 | 0.000 |
| M1 | 0.648 | 0.570 | -0.078 | 0.177 | 0.135 | -0.043 |
| M2 | 0.749 | 0.775 | 0.026 | 0.401 | 0.401 | 0.000 |
| M3 | 0.733 | 0.668 | -0.065 | 0.236 | 0.195 | -0.041 |
| M4 | 0.777 | 0.825 | 0.047 | 0.382 | 0.345 | -0.038 |

## Test bed 3 — Track D (eval cohort + eval-time Zidi injection)

Main (oracle OR injected labels):

| Model | served ROC | shadow ROC | ΔROC | served PR | shadow PR | ΔPR |
|---|---|---|---|---|---|---|
| M0 | 0.900 | 0.900 | 0.000 | 0.733 | 0.733 | 0.000 |
| M1 | 0.894 | 0.900 | 0.006 | 0.674 | 0.649 | -0.025 |
| M2 | 0.888 | 0.973 | 0.085 | 0.834 | 0.953 | 0.118 |
| M3 | 0.892 | 0.970 | 0.078 | 0.844 | 0.953 | 0.109 |
| M4 | 0.603 | 0.617 | 0.015 | 0.258 | 0.268 | 0.010 |

Clean subset (oracle labels, non-injected meters only):

| Model | served ROC | shadow ROC | ΔROC | served PR | shadow PR | ΔPR |
|---|---|---|---|---|---|---|
| M0 | 0.934 | 0.934 | 0.000 | 0.692 | 0.692 | 0.000 |
| M1 | 0.866 | 0.860 | -0.006 | 0.504 | 0.453 | -0.051 |
| M2 | 0.944 | 0.951 | 0.006 | 0.886 | 0.897 | 0.011 |
| M3 | 0.940 | 0.946 | 0.005 | 0.882 | 0.899 | 0.016 |
| M4 | 0.619 | 0.632 | 0.013 | 0.156 | 0.167 | 0.011 |

## Gates (pre-registered)

| Gate | Verdict | Inputs |
|---|---|---|
| G1 regression guard (≤0.02 drop, M2/M3/M4, both metrics) | True | max PR drop -0.011, max ROC drop 0.000 |
| G2 blind-spot lift (T5/T6 ≥0.50, any shadow model) | False | best recall 0.40 |
| G3 in-domain sanity (shadow PR > served, M3 and M4) | True | M3 Δ 0.109, M4 Δ 0.010 |

G1 per-model drops (served − shadow, positive = regression):

| Model | ΔPR | ΔROC |
|---|---|---|
| M2 | 0.011 | 0.007 |
| M3 | 0.017 | -0.000 |
| M4 | 0.011 | 0.009 |

Track C shadow per-type recall@20% (T5/T6):

| Model (shadow) | T5 | T6 |
|---|---|---|
| M0 | 0.00 | 0.00 |
| M1 | 0.10 | 0.11 |
| M2 | 0.00 | 0.00 |
| M3 | 0.40 | 0.11 |
| M4 | 0.20 | 0.11 |

## M0 soft comparison

- Track B PR: served 0.674 vs shadow 0.674; ROC: served 0.908 vs shadow 0.908; probability correlation 0.853
- M0 rule formula is identical by construction; only the Platt calibration was refit on shadow training data.

## Limitations

- Shadow models are offline only: never served; published Track A/B/C numbers unchanged.
- All labels are synthetic (generator archetypes / Zidi injection); no real theft labels are involved.
- Generator-signature caveat: train and eval salts are separated, but the generator family is shared; a family-level match is not proof of real-theft detection.
- Track C bed inherits virtual DTs, star topology, single-phase, no weather; meters repeat across windows (pooled instances are not independent).
- M0 is a soft comparison row: the rule is identical, only the calibration differs, so probability agreement is below 1 by design.
- Track D bed main labels are oracle OR eval-time injection; the clean-subset block uses oracle labels on non-injected meters only.
