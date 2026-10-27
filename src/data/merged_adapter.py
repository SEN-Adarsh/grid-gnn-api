"""Adapter for the `synthetic_india_v2` merged dataset (the project's final
primary synthetic dataset).

Applies the curation protocol from D:/rodic/merged_audit/Merged_Dataset_Audit_Pack:
  - the 8 temporally inconsistent theft_found inspections are quarantined
    (removed, never relabelled);
  - the original 500-meter / 14-day cohort is demo-only; the 2,000-meter /
    60-day added cohort is the evaluation benchmark;
  - the audit's whole-cohort split proposal is adopted as-is (batches 1+2
    train, batch 3 validation, batch 4 test); it is a proposal, not a
    previously trained or evaluated split;
  - truth/* is oracle-evaluation only and is never a model input.

Every parameter and label in this dataset is SYNTHETIC (third-party
assumptions, generator not supplied). No Indian field-calibration claim is
made anywhere downstream.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MERGED_BASE = Path(os.environ.get('MERGED_DATA_DIR', 'D:/rodic/merged_data/data'))
AUDIT_DIR = Path(os.environ.get('MERGED_AUDIT_DIR', 'D:/rodic/merged_audit/Merged_Dataset_Audit_Pack'))
ZIP_SHA256 = '57F8C31B71DF3D45EB872A8BC2A646325A07213BBBA7ABFF32E3B4A9904335E6'

_METER_CACHE: dict[str, pd.DataFrame] = {}


def _load_quarantine() -> frozenset[str]:
    ids = set()
    with open(AUDIT_DIR / 'inspection_rows_to_quarantine.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ids.add(row['meter_id'])
    return frozenset(ids)


# Import-safe: the audit pack and merged data live on the dev machine. On a
# deployment box these are empty and the loaders below raise a clear error;
# serving uses the prebuilt artifacts in artifacts_merged_demo instead.
QUARANTINE_IDS: frozenset[str] = _load_quarantine() if AUDIT_DIR.exists() else frozenset()

_METER_CACHE['quarantine'] = None  # placeholder to keep cache dict typed


def _meters_partition_frame() -> pd.DataFrame:
    """Audit split table keyed by meter (meter_id, cohort, partition, days_available)."""
    if _METER_CACHE.get('splits') is None:
        _METER_CACHE['splits'] = pd.read_csv(AUDIT_DIR / 'recommended_meter_splits.csv')
    return _METER_CACHE['splits']


def load_observable() -> dict[str, pd.DataFrame]:
    """Observable-side tables with the audit protocol applied.

    inspection_log has the 8 quarantined rows REMOVED (asserted), so
    theft_found==6, no_theft==175, not_accessible==14.
    """
    if not MERGED_BASE.exists() or not AUDIT_DIR.exists():
        raise FileNotFoundError(
            'merged dataset / audit pack not present; serving uses '
            'artifacts_merged_demo (see app/merged_context.py)')
    meters = pd.read_parquet(MERGED_BASE / 'observable' / 'meters.parquet')
    insp = pd.read_parquet(MERGED_BASE / 'observable' / 'inspection_log.parquet')
    before = len(insp)
    insp = insp[~insp.meter_id.isin(QUARANTINE_IDS)].reset_index(drop=True)
    assert before - len(insp) == 8, 'quarantine mismatch'
    assert (insp.outcome == 'theft_found').sum() == 6
    assert (insp.outcome == 'no_theft').sum() == 175
    assert (insp.outcome == 'not_accessible').sum() == 14
    return {
        'meters': meters,
        'inspection_log': insp,
        'calendar': pd.read_parquet(MERGED_BASE / 'observable' / 'calendar.parquet'),
        'weather': pd.read_parquet(MERGED_BASE / 'observable' / 'weather.parquet'),
        'outage_log_observed': pd.read_parquet(MERGED_BASE / 'observable' / 'outage_log_observed.parquet'),
        'topology_edges': pd.read_parquet(MERGED_BASE / 'observable' / 'topology_edges.parquet'),
    }


def load_readings(meter_ids: list[str] | None = None,
                  columns: list[str] | None = None) -> pd.DataFrame:
    """Long-format readings; filters by meter at the parquet level."""
    path = pq.ParquetFile(MERGED_BASE / 'observable' / 'readings.parquet')
    if columns is not None:
        read = path.read(columns=columns)
    else:
        read = path.read()
    df = read.to_pandas()
    if meter_ids is not None:
        df = df[df.meter_id.isin(set(meter_ids))].reset_index(drop=True)
    return df


def load_truth() -> dict[str, pd.DataFrame]:
    """ORACLE ONLY. Never a model input (audit protocol)."""
    return {
        'truth_meters': pd.read_parquet(MERGED_BASE / 'truth' / 'truth_meters.parquet'),
        'truth_dt': pd.read_parquet(MERGED_BASE / 'truth' / 'truth_dt.parquet'),
        'truth_outages': pd.read_parquet(MERGED_BASE / 'truth' / 'truth_outages.parquet'),
    }


def _cohort_frames() -> tuple[list[str], list[str], pd.DataFrame]:
    s = _meters_partition_frame()
    ev = s[s.days_available == 60]
    demo = s[s.days_available == 14]
    assert len(ev) == 2000 and len(demo) == 500
    return list(ev.meter_id), list(demo.meter_id), s


EVAL_METERS, DEMO_METERS, _SPLITS = ([''], [''], None) if not AUDIT_DIR.exists() else _cohort_frames()

_DT_PARTITION: dict[str, str] | None = None


def _dt_partitions() -> dict[str, str]:
    global _DT_PARTITION
    if _DT_PARTITION is None:
        s = _meters_partition_frame()
        ev = s[s.days_available == 60]
        _DT_PARTITION = {str(k): {'validation': 'val'}.get(str(v), str(v))
                         for k, v in ev.groupby('gis_dt_id').partition.first().items()}
        assert len(_DT_PARTITION) == 40
        assert sum(1 for v in _DT_PARTITION.values() if v == 'train') == 20
        assert sum(1 for v in _DT_PARTITION.values() if v == 'val') == 10
        assert sum(1 for v in _DT_PARTITION.values() if v == 'test') == 10
    return _DT_PARTITION


def load_topology_edges() -> pd.DataFrame:
    return pd.read_parquet(MERGED_BASE / 'observable' / 'topology_edges.parquet')


def build_id_maps() -> tuple[dict[str, int], dict[str, int]]:
    meters = pd.read_parquet(MERGED_BASE / 'observable' / 'meters.parquet')
    meter_map = {mid: i for i, mid in enumerate(meters.meter_id)}
    dt_map = {d: i for i, d in enumerate(sorted(meters.gis_dt_id.unique()))}
    return meter_map, dt_map


ID_MAPS = build_id_maps() if MERGED_BASE.exists() else ({}, {})

_PART = _dt_partitions() if AUDIT_DIR.exists() else {}
PARTITION = np.array([_PART.get(d, 'demo') for d in sorted(ID_MAPS[1])], dtype='<U5')

# ---- Task 2: inherited Observations bridge -------------------------------

import sys  # noqa: E402
from ..sim.physics import Topology  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PHASE_MAP = {'A': 0, 'B': 1, 'C': 2}
# agricultural_pumpset is the merged dataset's pumpset category; the inherited
# feature builder flags 'agricultural'. small_industrial is kept distinct so it
# forms its own peer group rather than being folded into another class.
CATEGORY_MAP = {'agricultural_pumpset': 'agricultural'}
ORIGIN = pd.Timestamp('2025-10-01 00:00', tz='Asia/Kolkata')
METERS_PER_DT = {'eval': 45, 'demo': 44}   # min over the cohort's DTs; uniform m
                                           # is a hard requirement of the pinned
                                           # build_features (g*m+j grouping)
AREA_TYPE = 'synthetic_mixed'              # not present in merged meters; constant


def merged_cfg(days: int) -> dict:
    from src.common import load_config
    cfg = load_config(ROOT / 'configs' / 'final.yaml')
    cohort = 'eval' if days > 14 else 'demo'
    dts = 40 if cohort == 'eval' else 10
    cfg.update(n_dts=dts, meters_per_dt=METERS_PER_DT[cohort], days=days,
               interval_minutes=15)
    # The demo cohort has only 14 days; scoring.py requires a prefix of at
    # least (baseline_days + 7) days, so cap the baseline window at 7 days.
    if cohort == 'demo':
        cfg['baseline_days'] = min(cfg['baseline_days'], 7)
    return cfg


def _uniform_subsample(cohort: str, seed: int) -> tuple[list[str], list[str]]:
    """Fixed-seed uniform per-DT subsample to satisfy the pinned builder.

    Returns (kept_ids, excluded_ids); kept is DT-major sorted."""
    meters_all = pd.read_parquet(MERGED_BASE / 'observable' / 'meters.parquet')
    cohort_ids = set(EVAL_METERS if cohort == 'eval' else DEMO_METERS)
    m = METERS_PER_DT[cohort]
    rng = np.random.default_rng(seed)
    kept, excluded = [], []
    dts = sorted(meters_all[meters_all.meter_id.isin(cohort_ids)].gis_dt_id.unique())
    for dt in dts:
        pool = sorted(set(meters_all[meters_all.gis_dt_id == dt].meter_id) & cohort_ids)
        assert len(pool) >= m, f'{dt}: {len(pool)} < {m}'
        pick = rng.choice(len(pool), m, replace=False)
        chosen = {pool[i] for i in pick}
        kept.extend(sorted(chosen))
        excluded.extend(sorted(set(pool) - chosen))
    return kept, excluded


def _scatter_matrix(n: int, steps: int, row_map: dict[str, int],
                    values: np.ndarray, rows: np.ndarray, times: np.ndarray,
                    fill=np.nan) -> np.ndarray:
    out = np.full((n, steps), fill, dtype=np.float32)
    r = rows.map(row_map).to_numpy()
    out[r, times] = values
    return out


def _topology_for_dt(dt: str, edges: pd.DataFrame, meter_poles: list[str]) -> tuple[Topology, int]:
    sub = edges[edges.from_node.str.startswith(dt + '_')]
    adjacency: dict[str, list] = {}
    for row in sub.itertuples():
        adjacency.setdefault(row.from_node, []).append(
            (row.to_node, row.r_ohm, row.x_ohm, row.length_m))
    root = dt + '_bus'
    assert root in adjacency, f'{root} missing from topology edges'
    order = [root]
    node_idx = {root: 0}
    parents, r, x, lengths = [-1], [0.], [0.], [0.]
    for node in order:
        for child, rr, xx, ll in sorted(adjacency.get(node, [])):
            if child in node_idx:
                continue
            node_idx[child] = len(order)
            parents.append(node_idx[node])
            r.append(rr)
            x.append(xx)
            lengths.append(ll)
            order.append(child)
    assert len(node_idx) == len(parents)

    resolved: list[int] = []
    attached_foreign = 0
    for pole in meter_poles:
        if pole in node_idx:
            resolved.append(node_idx[pole])
            continue
        # GIS mapping error (206 meters dataset-wide): the pole reference
        # belongs to another DT. Re-seat on the declared DT's tree at the same
        # feeder/position when it exists, else on the first feeder head.
        # Documented assumption; pole_id itself is never a model predictor.
        suffix = pole[len(pole.split('_', 1)[0]) + 1:]
        cand = f'{dt}_{suffix}'
        if cand in node_idx:
            resolved.append(node_idx[cand])
        else:
            head = f'{dt}_F00'
            assert head in node_idx, f'{dt}: no feeder head for fallback'
            resolved.append(node_idx[head])
        attached_foreign += 1
    topo = Topology(np.array(parents), np.array(r), np.array(x), np.array(lengths),
                    len(order) - 1, np.array(resolved), 'synthetic_india_v2_edges')
    return topo, attached_foreign


def build_observations(cohort: str, cfg: dict, seed: int = 0):
    """cohort 'eval' (2,000-meter/60-day -> uniform 45/DT = 1,800 kept) or
    'demo' (500-meter/14-day -> uniform 44/DT = 440 kept).

    Returns (Observations, extras). extras carries ORACLE labels (never a
    model input), the quarantined-clean inspection labels, the excluded
    subsample ids, and coverage/step facts. event_flags are zeroed by the
    audit protocol (they are an artificial shortcut, excluded from predictors).
    """
    from src.sim.simulator import Observations

    days = cfg['days']
    steps = days * 96
    src = load_observable()
    meters_src = src['meters'].set_index('meter_id')
    kept, excluded = _uniform_subsample(cohort, seed)
    dt_ids = sorted(meters_src.loc[kept].gis_dt_id.unique())
    g_map = ID_MAPS[1]                     # global dt idx (PARTITION lookup)
    d_map = {dt: i for i, dt in enumerate(dt_ids)}   # local obs idx 0..d-1
    kept_set = set(kept)

    meters = meters_src.loc[kept].reset_index()
    meters['dt_id'] = meters.gis_dt_id.map(d_map)
    meters['phase'] = meters.phase_recorded.map(PHASE_MAP)
    meters['category'] = meters.category.replace(CATEGORY_MAP)
    meters['area_type'] = AREA_TYPE
    meters = meters.sort_values(['dt_id', 'meter_id']).reset_index(drop=True)
    row_map = {mid: i for i, mid in enumerate(meters.meter_id)}
    n, d = len(meters), len(dt_ids)
    assert n == cfg['n_dts'] * cfg['meters_per_dt']

    reads = load_readings(list(kept_set),
                          columns=['meter_id', 'ts', 'kwh', 'kvarh',
                                   'voltage_v', 'is_missing'])
    ti = ((reads.ts - ORIGIN) / pd.Timedelta(minutes=15)).astype('int64').to_numpy()
    per_meter = reads.groupby('meter_id').size()
    expected_steps = steps if cohort == 'eval' else 14 * 96
    assert (per_meter == expected_steps).all(), 'unexpected coverage'
    assert int(ti.max()) < expected_steps

    kwh = _scatter_matrix(n, expected_steps, row_map, reads.kwh.to_numpy(np.float32),
                          reads.meter_id, ti)
    kvarh = _scatter_matrix(n, expected_steps, row_map, reads.kvarh.to_numpy(np.float32),
                            reads.meter_id, ti)
    voltage = _scatter_matrix(n, expected_steps, row_map, reads.voltage_v.to_numpy(np.float32),
                              reads.meter_id, ti)
    missing = ~np.isfinite(kwh)
    kwh = np.nan_to_num(kwh).astype(np.float32)
    kvarh = np.nan_to_num(kvarh).astype(np.float32)
    voltage = np.nan_to_num(voltage).astype(np.float32)
    events = np.zeros((n, expected_steps), dtype=np.uint8)  # protocol exclusion

    dtr = pd.read_parquet(MERGED_BASE / 'observable' / 'dt_readings.parquet')
    dtr = dtr[dtr.dt_id.isin(dt_ids)].copy()
    d_map_dt = {dt: i for i, dt in enumerate(sorted(dt_ids))}
    dti = ((dtr.ts - ORIGIN) / pd.Timedelta(minutes=15)).astype('int64').to_numpy()
    dt_kwh = _scatter_matrix(d, expected_steps, d_map_dt, dtr.kwh_in.to_numpy(np.float32),
                             dtr.dt_id, dti)
    dt_kvarh = _scatter_matrix(d, expected_steps, d_map_dt, dtr.kvarh_in.to_numpy(np.float32),
                               dtr.dt_id, dti)
    dt_voltage = _scatter_matrix(d, expected_steps, d_map_dt, dtr.v_lv.to_numpy(np.float32),
                                 dtr.dt_id, dti)
    dt_currents = np.stack([
        _scatter_matrix(d, expected_steps, d_map_dt, dtr[c].to_numpy(np.float32),
                        dtr.dt_id, dti)
        for c in ('i_a', 'i_b', 'i_c')], axis=2)
    # Outage intervals report null DT input; the simulator convention is zero
    # true and reported load during outages.
    dt_kwh = np.nan_to_num(dt_kwh).astype(np.float32)
    dt_kvarh = np.nan_to_num(dt_kvarh).astype(np.float32)
    dt_voltage = np.nan_to_num(dt_voltage).astype(np.float32)
    dt_currents = np.nan_to_num(dt_currents).astype(np.float32)

    weather = src['weather'].sort_values('date').temperature_c.to_numpy(np.float32)
    temperature = np.repeat(weather[:days], 96)

    edges = src['topology_edges']
    topologies = []
    foreign_attached = 0
    for g, dt in enumerate(sorted(dt_ids)):
        block = meters.iloc[g * cfg['meters_per_dt']:(g + 1) * cfg['meters_per_dt']]
        topo, attached = _topology_for_dt(dt, edges, list(block.pole_id))
        topologies.append(topo)
        foreign_attached += attached

    if cohort == 'eval':
        split = PARTITION[[g_map[dt] for dt in sorted(dt_ids)]]
    else:
        split = np.full(d, 'demo', dtype='<U5')

    obs = Observations(meters=meters, kwh=kwh, kvarh=kvarh, voltage=voltage,
                       events=events, missing=missing,
                       dt_kwh=dt_kwh, dt_kvarh=dt_kvarh, dt_voltage=dt_voltage,
                       dt_currents=dt_currents, temperature=temperature,
                       topologies=topologies, split=split, cfg=cfg, seed=seed)

    truth = load_truth()['truth_meters'].set_index('meter_id')
    oracle = truth.loc[kept].reset_index()
    oracle = oracle.rename(columns={'archetype': 'family', 'start_date': 'theft_start'})
    insp = src['inspection_log']
    insp_kept = insp[insp.meter_id.isin(kept_set)].reset_index(drop=True)
    insp_kept['row'] = insp_kept.meter_id.map(row_map)
    extras = {
        'oracle_labels': oracle,
        'inspection_labels': insp_kept,
        'subsample_excluded': excluded,
        'gis_pole_mismatch_attached': foreign_attached,
        'steps': expected_steps,
        'coverage_complete': bool(cohort == 'eval' or expected_steps == 14 * 96),
        'event_flags_zeroed': True,
    }
    return obs, extras
