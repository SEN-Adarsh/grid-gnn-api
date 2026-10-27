import json
from pathlib import Path

ROOT_ARTIFACTS = Path(__file__).resolve().parents[1] / 'artifacts_merged_demo'


def test_build_and_smoke():
    from app.merged_context import build_merged_artifacts, smoke_score
    summary = build_merged_artifacts()
    assert summary['meters'] == 440 and summary['dts'] == 10
    assert summary['inspected'] >= 1 and summary['config_hash']
    smoke = smoke_score()
    assert smoke['model'] == 'M0' and smoke['meters'] == 440 and smoke['dts'] == 10
    assert smoke['config_hash'] == summary['config_hash']
    # with 6 dataset-wide positives the honest outcome is disabled flags
    assert smoke['flags_disabled_note'] is True
    sel = json.loads((ROOT_ARTIFACTS / 'selected_model.json').read_text(encoding='utf-8'))
    assert sel['profile_source'] == 'synthetic_india_v2'
    assert sel['models_available'] == ['M0', 'M1', 'M2', 'M3', 'M4']
    assert 'DECLARED ORACLE-TRACK' in sel['models_training']['M4']


def test_oracle_models_score():
    import os
    os.environ['GRID_CONFIG'] = 'configs/merged_demo.yaml'
    from app.scoring import Scorer
    service = Scorer()
    end = (service.cfg['baseline_days'] + 7) * 96
    for name in ('M2', 'M3', 'M4'):
        out = service.score({'model': name, 'as_of_interval': end, 'scenario': 'none',
                             'meter_id': str(service.context.meters.meter_id.iloc[0])},
                            write_audit=False)
        assert out['model'] == name and len(out['meters']) == 440
        assert len(out['dts']) == 10
