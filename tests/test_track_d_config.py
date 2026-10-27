from pathlib import Path

from app.merged_context import ROOT, _builder_config_path


def test_default_config_path_unchanged(monkeypatch):
    monkeypatch.delenv('GRID_CONFIG', raising=False)
    assert _builder_config_path() == ROOT / 'configs' / 'merged_demo.yaml'


def test_grid_config_env_overrides(monkeypatch):
    monkeypatch.setenv('GRID_CONFIG', 'configs/shadow_injected.yaml')
    assert _builder_config_path() == ROOT / 'configs' / 'shadow_injected.yaml'


def test_shadow_config_exists_and_points_at_shadow_artifacts():
    import yaml
    cfg = yaml.safe_load((ROOT / 'configs' / 'shadow_injected.yaml').read_text())
    assert cfg['artifacts_dir'] == 'artifacts_shadow'
    assert cfg['n_dts'] == 10 and cfg['meters_per_dt'] == 44 and cfg['days'] == 14
