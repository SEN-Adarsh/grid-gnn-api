"""Builds and serves the merged-dataset (`synthetic_india_v2`) demo context.

The inherited `Scorer` (app/scoring.py, hash-pinned) is fully parametrised by
GRID_CONFIG: it loads `demo_context.joblib`, `models.joblib` and
`selected_model.json` from the configured artifacts_dir. This module builds
those artifacts from the merged demo cohort WITHOUT hidden truth:

  - M0: the inherited own-history residual rule (unsupervised);
  - M1: IsolationForest on own-history features fitted on demo-cohort meters
    (unsupervised);
  - both Platt-calibrated on the QUARANTINED-CLEAN inspection labels only
    (theft_found / no_theft; not_accessible never used as a negative);
  - operating thresholds computed on those same inspected meters, which will
    typically report `validation_target_met: false` — the app then disables
    inspection flags and says so. That is the honest behaviour with 6
    dataset-wide positives.

Run `python -m app.merged_context build` to (re)build artifacts, then start
the API with GRID_CONFIG=configs/merged_demo.yaml.

NOTE: the pinned Scorer labels its responses `profile_source:
synthetic_fallback` (a hard-coded cosmetic string); the true provenance of
this context is `synthetic_india_v2` and is recorded in the audit log,
selected_model.json and project docs.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from app.scoring import Scorer  # noqa: E402  (deferred import inside smoke_score)
from src.common import config_hash, load_config  # noqa: E402
from src.data.merged_adapter import build_observations, merged_cfg  # noqa: E402
from src.features.build import build_features  # noqa: E402
from src.models.baselines import Baseline, Calibrator, operating_threshold  # noqa: E402


def _inspection_targets(extras) -> tuple[np.ndarray, np.ndarray]:
    """(row_indices, y) over inspected meters only; not_accessible excluded.

    Uninspected meters are deliberately absent: per the audit protocol they
    are not verified negatives."""
    insp = extras['inspection_labels']
    keep = insp.outcome.isin(['theft_found', 'no_theft'])
    rows = insp.loc[keep, 'row'].to_numpy(int)
    y = (insp.loc[keep, 'outcome'] == 'theft_found').to_numpy(int)
    return rows, y


def build_merged_artifacts() -> dict:
    cfg = load_config(ROOT / 'configs' / 'merged_demo.yaml')
    artifacts = ROOT / cfg['artifacts_dir']
    artifacts.mkdir(parents=True, exist_ok=True)

    obs, extras = build_observations('demo', cfg)
    feats = build_features(obs)
    rows, y = _inspection_targets(extras)

    m0 = Baseline('M0')
    m1 = Baseline('M1')
    m1.scaler = StandardScaler().fit(feats.own)
    m1.model = IsolationForest(n_estimators=cfg['rf_trees'], max_samples=256,
                               contamination='auto', random_state=0, n_jobs=2)
    m1.model.fit(m1.scaler.transform(feats.own))
    m1.model.n_jobs = 1

    target = str(cfg['target_precision'])
    for model in (m0, m1):
        raw = model.raw(feats)
        model.calibrator = Calibrator().fit(raw[rows], y)
        _, prob = model.predict(feats)
        model.thresholds[target] = operating_threshold(
            y, prob[rows], cfg['target_precision'], cfg['min_validation_flags'])

    joblib.dump(obs, artifacts / 'demo_context.joblib')
    joblib.dump({'M0': m0, 'M1': m1}, artifacts / 'models.joblib')
    selection = {
        'model': 'M0',
        'config_hash': config_hash(cfg),
        'profile_source': 'synthetic_india_v2',
        'dataset': 'merged demo cohort: 500 meters, 10 DTs, 14 days '
                   '(demo_only_14d partition of the audit pack)',
        'training_data': 'unsupervised residual/isolation features; Platt '
                         'calibration on quarantined-clean inspection labels '
                         f'({int(y.sum())} theft_found, {int((y == 0).sum())} no_theft)',
        'models_available': ['M0', 'M1'],
        'gnn_global_permutation_importance': [],
    }
    (artifacts / 'selected_model.json').write_text(json.dumps(selection, indent=2),
                                                   encoding='utf-8')
    return {'config_hash': selection['config_hash'], 'meters': int(len(obs.meters)),
            'dts': int(obs.dt_kwh.shape[0]), 'inspected': int(len(rows)),
            'positives': int(y.sum())}


def smoke_score() -> dict:
    """Instantiate the pinned Scorer against the merged artifacts and score
    the full demo prefix for one stored meter (no scenario)."""
    os.environ['GRID_CONFIG'] = 'configs/merged_demo.yaml'
    from app.scoring import Scorer
    service = Scorer()
    cfg = service.cfg
    end = (cfg['baseline_days'] + 7) * 96
    out = service.score({'as_of_interval': end, 'scenario': 'none',
                         'meter_id': str(service.context.meters.meter_id.iloc[0])},
                        write_audit=False)
    return {'model': out['model'], 'meters': len(out['meters']), 'dts': len(out['dts']),
            'config_hash': out['config_hash'],
            'flags_disabled_note': any('unattainable' in n for n in out['notes'])}


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'build'
    if action == 'build':
        print(json.dumps(build_merged_artifacts(), indent=2))
    elif action == 'smoke':
        print(json.dumps(smoke_score(), indent=2))
