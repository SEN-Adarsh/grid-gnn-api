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
