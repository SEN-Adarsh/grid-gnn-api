"""Builds and serves the merged-dataset (`synthetic_india_v2`) demo context.

The inherited `Scorer` (app/scoring.py, hash-pinned) is fully parametrised by
GRID_CONFIG: it loads `demo_context.joblib`, `models.joblib` and
`selected_model.json` from the configured artifacts_dir. This module builds
those artifacts from the merged demo cohort:

  - M0: the inherited own-history residual rule (unsupervised);
  - M1: IsolationForest on own-history features fitted on demo-cohort meters
    (unsupervised);
  - both Platt-calibrated on the QUARANTINED-CLEAN inspection labels only
    (theft_found / no_theft; not_accessible never used as a negative);
  - M2/M3/M4: TRAINED ON HIDDEN TRUTH — DECLARED ORACLE-TRACK demo models
    (DT-disjoint 7/3 split, calibrated on the oracle validation split). They
    exist so reviewers can run the full ladder live; they are labelled as
    oracle-trained in selected_model.json, /health and the UI, and are never
    the default. The default M0 never touches hidden truth.
  - operating thresholds computed per model on their own calibration split;
    M0/M1 typically report `validation_target_met: false` — the app then
    disables inspection flags and says so. That is the honest behaviour with
    6 dataset-wide positives.

Run `python -m app.merged_context build` to (re)build artifacts, then start
the API with GRID_CONFIG=configs/merged_demo.yaml.

NOTE: the pinned Scorer labels its responses `profile_source` from
selected_model.json (`synthetic_india_v2`).
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
from sklearn.ensemble import IsolationForest, RandomForestClassifier  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from app.scoring import Scorer  # noqa: E402  (deferred import inside smoke_score)
from src.common import config_hash, load_config  # noqa: E402
from src.data.merged_adapter import build_observations, merged_cfg  # noqa: E402
from src.features.build import build_features  # noqa: E402
from src.models.baselines import Baseline, Calibrator, finish_calibration, operating_threshold  # noqa: E402
from src.models.gnn import fit_gnn, permutation_groups  # noqa: E402


def _inspection_targets(extras) -> tuple[np.ndarray, np.ndarray]:
    """(row_indices, y) over inspected meters only; not_accessible excluded.

    Uninspected meters are deliberately absent: per the audit protocol they
    are not verified negatives."""
    insp = extras['inspection_labels']
    keep = insp.outcome.isin(['theft_found', 'no_theft'])
    rows = insp.loc[keep, 'row'].to_numpy(int)
    y = (insp.loc[keep, 'outcome'] == 'theft_found').to_numpy(int)
    return rows, y


def _oracle_targets(extras, meters) -> np.ndarray:
    """Hidden-truth labels aligned to the demo meters. DECLARED ORACLE USE:
    training the optional M2/M3/M4 demo models only; never a model input."""
    oracle = extras['oracle_labels']
    assert list(oracle.meter_id) == list(meters.meter_id), 'label misalignment'
    return oracle.is_theft.to_numpy(int)


def _builder_config_path() -> Path:
    """GRID_CONFIG selects the build config (shadow builds); default preserved."""
    return ROOT / os.environ.get('GRID_CONFIG', 'configs/merged_demo.yaml')


def build_merged_artifacts() -> dict:
    cfg = load_config(_builder_config_path())
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

    # --- M2/M3/M4: hidden-truth-trained, DECLARED ORACLE-TRACK demo models ---
    y_or = _oracle_targets(extras, obs.meters)
    n_dts = int(cfg['n_dts'])
    split = np.where(np.arange(n_dts) < 7, 'train', 'val')  # DT-disjoint, deterministic
    train_mask = split[feats.dt_index] == 'train'
    val_mask = split[feats.dt_index] == 'val'
    if int(y_or[train_mask].sum()) < 8 or int(y_or[val_mask].sum()) < 3:
        raise RuntimeError('demo cohort lacks enough oracle positives for a safe train/val split')

    trained = {}
    for name in ('M2', 'M3'):
        x = feats.own if name == 'M2' else feats.full
        item = Baseline(name)
        item.explanation_reference = np.median(x[train_mask], axis=0)
        best = -np.inf
        for h in ({'max_depth': 8, 'min_samples_leaf': 3},
                  {'max_depth': None, 'min_samples_leaf': 2}):
            rf = RandomForestClassifier(n_estimators=cfg['rf_trees'],
                                       class_weight='balanced_subsample',
                                       random_state=0, n_jobs=2, max_features=.8, **h)
            rf.fit(x[train_mask], y_or[train_mask])
            rf.n_jobs = 1  # deterministic serial probability reduction
            ap = average_precision_score(y_or[val_mask], rf.predict_proba(x[val_mask])[:, 1])
            if ap > best:
                item.model, best = rf, ap
        finish_calibration(item, feats, y_or, val_mask, cfg)
        trained[name] = item

    m4, _m4log = fit_gnn(feats, y_or, feats.dt_estimate, split, cfg, seed=0)
    trained['M4'] = m4
    gnn_perm = permutation_groups(m4, feats, y_or, val_mask, seed=802)

    joblib.dump(obs, artifacts / 'demo_context.joblib')
    joblib.dump({'M0': m0, 'M1': m1, **trained}, artifacts / 'models.joblib')
    selection = {
        'model': 'M0',
        'config_hash': config_hash(cfg),
        'profile_source': 'synthetic_india_v2',
        'dataset': 'merged demo cohort: 500 meters, 10 DTs, 14 days '
                   '(demo_only_14d partition of the audit pack)',
        'training_data': 'M0/M1: unsupervised residual/isolation features, '
                         'Platt calibration on quarantined-clean inspection labels '
                         f'({int(y.sum())} theft_found, {int((y == 0).sum())} no_theft); '
                         'M2/M3/M4: TRAINED ON HIDDEN TRUTH — declared oracle-track '
                         'demo models only (DT-disjoint 7/3 split), never the default',
        'models_available': ['M0', 'M1', 'M2', 'M3', 'M4'],
        'models_training': {
            'M0': 'untrained rule; Platt on inspection labels (Track A compliant)',
            'M1': 'IsolationForest unsupervised; Platt on inspection labels',
            'M2': 'RandomForest on hidden truth — DECLARED ORACLE-TRACK demo model',
            'M3': 'RandomForest (own+context) on hidden truth — DECLARED ORACLE-TRACK demo model',
            'M4': 'TemporalGraphNet on hidden truth — DECLARED ORACLE-TRACK demo model',
        },
        'oracle_split': {'train_dts': list(range(7)), 'val_dts': list(range(7, n_dts)),
                         'positives_train': int(y_or[train_mask].sum()),
                         'positives_val': int(y_or[val_mask].sum())},
        'gnn_global_permutation_importance': gnn_perm,
    }
    (artifacts / 'selected_model.json').write_text(json.dumps(selection, indent=2),
                                                   encoding='utf-8')
    return {'config_hash': selection['config_hash'], 'meters': int(len(obs.meters)),
            'dts': int(obs.dt_kwh.shape[0]), 'inspected': int(len(rows)),
            'positives': int(y.sum()),
            'models': sorted(['M0', 'M1', *trained])}


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
