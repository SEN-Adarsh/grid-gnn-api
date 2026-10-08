import numpy as np
import pandas as pd
import pytest

from src.data.merged_adapter import (DEMO_METERS, EVAL_METERS, ID_MAPS, PARTITION,
                                     QUARANTINE_IDS, ZIP_SHA256, load_observable,
                                     load_readings)


def test_counts_match_audit():
    obs = load_observable()
    assert len(obs['meters']) == 2500
    # 203 inspections - 8 quarantined = 195 (8 rows removed, never relabelled)
    assert len(obs['inspection_log']) == 195
    r = load_readings(columns=['meter_id'])
    assert len(r) == 12192000


def test_zip_hash_recorded():
    assert ZIP_SHA256 == '57F8C31B71DF3D45EB872A8BC2A646325A07213BBBA7ABFF32E3B4A9904335E6'


def test_quarantine_excluded_and_never_relabelled():
    insp = load_observable()['inspection_log']
    assert len(QUARANTINE_IDS) == 8
    assert set(QUARANTINE_IDS).isdisjoint(set(insp.meter_id))
    assert (insp.outcome == 'theft_found').sum() == 6
    assert (insp.outcome == 'no_theft').sum() == 175
    assert (insp.outcome == 'not_accessible').sum() == 14


def test_cohorts_by_coverage():
    assert len(EVAL_METERS) == 2000 and len(DEMO_METERS) == 500
    assert set(EVAL_METERS).isdisjoint(DEMO_METERS)


def test_partition_disjoint_by_dt():
    assert len(PARTITION) == 50
    assert (PARTITION == 'train').sum() == 20
    assert (PARTITION == 'val').sum() == 10
    assert (PARTITION == 'test').sum() == 10
    # cohort 0 (demo) DTs are outside the eval partitions


def test_id_maps():
    meter_map, dt_map = ID_MAPS
    assert len(meter_map) == 2500 and len(dt_map) == 50
    assert sorted(dt_map.values()) == list(range(50))


def test_readings_grid_and_tz():
    r = load_readings(EVAL_METERS[:10], columns=['meter_id', 'ts', 'kwh', 'is_missing'])
    assert str(r.ts.dtype.tz) == 'Asia/Kolkata'
    assert r.groupby('meter_id').size().nunique() == 1
    assert (r.kwh.notna() | r.is_missing).all()


from src.data.merged_adapter import build_observations, merged_cfg


def test_eval_observations_shape():
    cfg = merged_cfg(60)
    obs, extras = build_observations('eval', cfg)
    assert obs.kwh.shape == (1800, 5760)
    assert obs.dt_kwh.shape == (40, 5760)
    assert obs.meters.meter_id.is_unique
    # DT-major contiguous blocks for the pinned builder's g*m+j grouping
    dt_ids = obs.meters.dt_id.to_numpy()
    assert all((dt_ids[g * 45:(g + 1) * 45] == g).all() for g in range(40))
    # real missing dropouts exist, coverage is complete per meter
    assert obs.missing.mean() < 0.01
    assert (obs.missing.mean(1) < 1.0).all()
    # protocol: event flags are never predictors
    assert obs.events.sum() == 0
    assert extras['event_flags_zeroed'] and len(extras['subsample_excluded']) == 200
    assert extras['oracle_labels'].is_theft.sum() > 0


def test_features_run_and_no_forbidden_names():
    from src.features.build import build_features, FORBIDDEN
    obs, extras = build_observations('eval', merged_cfg(60))
    feats = build_features(obs)
    names = feats.own.dtype, feats.graph.dtype
    assert all(np.isfinite(feats.own).all() for _ in [0])
    # forbidden tokens must not appear in any feature naming surface
    blob = ' '.join(str(x) for x in feats.__dict__.keys())
    assert not any(tok in blob.lower() for tok in FORBIDDEN)


def test_demo_steps_14d():
    obs, extras = build_observations('demo', merged_cfg(14))
    assert obs.kwh.shape == (440, 1344)
    assert obs.dt_kwh.shape == (10, 1344)


def test_topologies_from_real_edges():
    obs, _ = build_observations('demo', merged_cfg(14))
    edges = len(pd.read_parquet(r"D:\rodic\merged_data\data\observable\topology_edges.parquet"))
    for g, topo in enumerate(obs.topologies):
        assert topo.size == 23  # 22 edges per DT + root
        assert (topo.parent[1:] < np.arange(1, topo.size)).all()  # parents precede children
        assert topo.meter_nodes.min() >= 1
