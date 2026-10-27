import os


def test_ies_endpoints_and_health_block():
    os.environ['GRID_CONFIG'] = 'configs/merged_demo.yaml'
    from app import api
    api.scorer.cache_clear()
    h = api.health()
    assert h['ies_v0_4_alignment']['data_residency'] == 'local_discom_retained_no_central_db'
    assert len(h['ies_v0_4_alignment']['building_blocks_used']) == 4
    assert {'status', 'model', 'models', 'config_hash', 'ies_v0_4_alignment'} <= set(h)
    assert h['models'] == ['M0', 'M1', 'M2', 'M3', 'M4']  # Task 3 artifacts
    m = api.ies_manifest()
    assert m['standard'] == 'India Energy Stack (IES) v0.4'
    assert m['central_database'] is False
    assert {b['id'] for b in m['building_blocks']} == {'BB1', 'BB3', 'BB4', 'BB5'}
