"""Two-track evaluation on the merged `synthetic_india_v2` dataset.

Track A (headline): inspection-supervised. The 203 synthetic inspections
(8 temporally inconsistent rows quarantined per the audit) are the only
"observed" evidence; uninspected meters are never treated as negatives.
Track B (declared oracle): hidden truth labels, used for evaluation only.

Per protocol: `event_flags` are excluded from predictors (zeroed by the
adapter), and `feeder_id`/`pole_id` are never predictors. Everything here is
SYNTHETIC (third-party assumptions; generator not supplied). No Indian field
performance claim is made or implied.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results'


def ranks_of(scores: np.ndarray) -> list[int]:
    """Rank 1 = highest score; ties broken by index for determinism."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    ranks = np.empty(len(scores), int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks.tolist()


def unexplained_fraction(obs) -> tuple[np.ndarray, np.ndarray]:
    """Per-DT unexplained energy fraction and kWh: DT input minus the sum of
    covered customer kWh, over active (energised) intervals. Includes technical
    losses by construction — the physics-based DT signal."""
    d = obs.dt_kwh.shape[0]
    m = obs.kwh.shape[0] // d
    consumer = obs.kwh.reshape(d, m, -1).sum(1)
    active = obs.dt_voltage > 50
    resid = obs.dt_kwh - consumer
    denom = np.where(obs.dt_kwh > 0, obs.dt_kwh, np.nan)
    frac = np.where(active, resid / np.where(np.isnan(denom), 1., denom), np.nan)
    with np.errstate(invalid='ignore'):
        f = np.nanmean(frac, axis=1)
    kwh = np.where(active, resid, 0).sum(1)
    return np.nan_to_num(f), kwh


def balance_scores(obs) -> np.ndarray:
    frac, _ = unexplained_fraction(obs)
    return frac


def enrichment_at_k(scores: np.ndarray, found_rows: list[int], k: int) -> int:
    top = set(np.lexsort((np.arange(len(scores)), -scores))[:k].tolist())
    return sum(1 for r in found_rows if r in top)


def run_evaluation(cohort: str = 'eval', seed: int = 0) -> dict:
    from src.data.merged_adapter import build_observations, merged_cfg
    from src.features.build import build_features, m0_scores

    obs, extras = build_observations(cohort, merged_cfg(60 if cohort == 'eval' else 14), seed)
    feats = build_features(obs)
    scores = np.asarray(m0_scores(feats), dtype=np.float64)
    ranks = ranks_of(scores)

    oracle = extras['oracle_labels']
    assert list(oracle.meter_id) == list(obs.meters.meter_id), 'label misalignment'
    y = oracle.is_theft.to_numpy()

    # --- Track B (declared oracle) ---
    oracle_auc = float(roc_auc_score(y, scores))
    oracle_pr = float(average_precision_score(y, scores))
    prevalence = float(y.mean())
    top50 = np.lexsort((np.arange(len(scores)), -scores))[:50]
    per_family = {}
    for fam, grp in oracle[oracle.is_theft].groupby('family'):
        per_family[str(fam)] = {'positives': int(len(grp)),
                                'in_top50': int(grp.index.isin(top50).sum()),
                                'median_rank': float(np.median([ranks[i] for i in grp.index]))}

    # --- DT-level hooking signal (hooking is invisible at meters) ---
    dt_of_row = obs.meters.dt_id.to_numpy()
    hook_rows = np.flatnonzero((oracle.family == 'direct_hooking').to_numpy())
    hook_dts = sorted({int(g) for g in dt_of_row[hook_rows]})
    other_dts = [g for g in range(obs.dt_kwh.shape[0]) if g not in hook_dts]
    bal = balance_scores(obs)
    hook_auc = float(roc_auc_score([1] * len(hook_dts) + [0] * len(other_dts),
                                   list(bal[hook_dts]) + list(bal[other_dts]))) \
        if hook_dts and other_dts else None

    # --- Track A (headline; quarantine already applied in the adapter) ---
    insp = extras['inspection_labels']
    found = insp[insp.outcome == 'theft_found']
    no_theft = insp[insp.outcome == 'no_theft']
    found_rows = [int(r) for r in found.row]
    u, p_mwu = mannwhitneyu(scores[found_rows], scores[no_theft.row.to_numpy()],
                            alternative='greater') if len(found_rows) and len(no_theft) \
        else (None, None)
    inspection = {
        'n_theft_found': int(len(found)),
        'n_no_theft': int(len(no_theft)),
        'n_not_accessible_excluded_from_negatives': int((insp.outcome == 'not_accessible').sum()),
        'quarantined_rows_removed_by_adapter': 8,
        'ranks_of_theft_found': {str(m): int(r) for m, r in
                                 zip(found.meter_id, (ranks[int(r)] for r in found.row))},
        'enrichment_top25': enrichment_at_k(scores, found_rows, 25),
        'enrichment_top50': enrichment_at_k(scores, found_rows, 50),
        'expected_by_chance_top50': round(50 * len(found) / len(scores), 2),
        'mannwhitney_p_greater': float(p_mwu) if p_mwu is not None else None,
        'caveat': ('Dataset-wide only 6 theft_found inspections exist after '
                   'quarantine (3 of them in this cohort); they are sparse, '
                   'selected and biased. Enrichment is indicative only.'),
    }

    return {
        'profile_source': 'SYNTHETIC (synthetic_india_v2)',
        'cohort': cohort,
        'n_meters_scored': int(len(scores)),
        'n_dts': int(obs.dt_kwh.shape[0]),
        'days': int(obs.kwh.shape[1] // 96),
        'track_a_inspection_supervised': inspection,
        'track_b_oracle_declared': {
            'note': 'Hidden truth used for evaluation only, as permitted by the '
                    'dataset README and the audit protocol. NOT field performance.',
            'prevalence': prevalence,
            'm0_residual_roc_auc': oracle_auc,
            'm0_residual_pr_auc': oracle_pr,
            'per_family_top50': per_family,
            'dt_balance_direct_hooking_auc': hook_auc,
            'direct_hooking_dts': [int(g) for g in hook_dts],
            'dt_balance_fractions': [float(x) for x in bal],
        },
        'protocol': {'event_flags_excluded': True, 'truth_never_a_predictor': True,
                     'feeder_pole_never_a_predictor': True},
    }


def write_reports(result: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'merged_eval.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    a = result['track_a_inspection_supervised']
    b = result['track_b_oracle_declared']
    md = f"""# Merged-dataset evaluation (synthetic_india_v2)

**Everything in this report is SYNTHETIC.** The dataset is a third-party
research prototype with assumption-driven parameters and no supplied
generator; it is not calibrated to Indian AMI field data. No Indian field
performance is claimed.

## Track A — inspection-supervised (headline)

Models never see hidden truth. Scoring is the inherited M0 own-history
residual ranking; evidence is the quarantined inspection log only.

- Meters scored: {result['n_meters_scored']} ({result['n_dts']} DTs x {result['days']} days)
- theft_found inspections: {a['n_theft_found']} of the 6 dataset-wide after quarantine (8 inconsistent rows removed, never relabelled)
- no_theft controls: {a['n_no_theft']}; not_accessible excluded from negatives: {a['n_not_accessible_excluded_from_negatives']}
- Ranks of the {a['n_theft_found']} theft_found meters (of {result['n_meters_scored']}): {a['ranks_of_theft_found']}
- In top-50: {a['enrichment_top50']} (chance expectation {a['expected_by_chance_top50']}); top-25: {a['enrichment_top25']}
- Mann-Whitney U (found > no_theft), p = {a['mannwhitney_p_greater']}

{a['caveat']}

## Track B — oracle evaluation (declared)

Hidden truth used for evaluation only. This is a synthetic-oracle diagnostic,
not inspection-supervised performance and not field performance.

- Theft prevalence: {b['prevalence']:.3f}
- M0 residual ROC-AUC {b['m0_residual_roc_auc']:.3f}, PR-AUC {b['m0_residual_pr_auc']:.3f}
- DT-balance vs direct-hooking DTs: AUC {b['dt_balance_direct_hooking_auc'] and f"{b['dt_balance_direct_hooking_auc']:.3f}"} (hooking is unmetered upstream load; DT balance is the only honest signal)
- Per-family presence in the top-50: {json.dumps(b['per_family_top50'])}

## Protocol
Event flags excluded from predictors; hidden truth never a predictor;
feeder/pole identifiers never a predictor; 8 inconsistent inspections
quarantined; uncovered intervals never treated as zeros (eval cohort coverage
is complete by construction).
"""
    (RESULTS / 'merged_eval.md').write_text(md, encoding='utf-8')


if __name__ == '__main__':
    write_reports(run_evaluation('eval'))
    print('merged evaluation written to results/merged_eval.json and .md')
