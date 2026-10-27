"""Derives the Track D shadow data pack from synthetic_india_v2.

Writes D:/rodic/merged_data_shadow/ -- a structure-compatible, DEMO-ONLY pack
with Zidi theft injected into train-DT meters (assessment days only) and
DT-hooks on 2 train DTs. The original merged_data is never written. Run
WITHOUT MERGED_DATA_DIR set (reads the original); the shadow build then runs
WITH MERGED_DATA_DIR pointing at the pack.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.merged_adapter import (MERGED_BASE, _uniform_subsample,  # noqa: E402
                                     load_observable, load_readings, load_truth)
from src.data.theft_inject import apply_theft, assign_types, meter_rng  # noqa: E402

TZ = 'Asia/Kolkata'
ASSESS_START = pd.Timestamp('2025-10-08 00:00', tz=TZ)
ASSESS_END = pd.Timestamp('2025-10-15 00:00', tz=TZ)      # exclusive
TRAIN_SALT = 4101
N_INJECTED = 36                                           # 6 types x 6 meters
HOOK_FACTOR = 1.15
N_HOOK_DTS = 2
SHADOW_DIR = Path('D:/rodic/merged_data_shadow')
OBS_FILES = ('meters', 'inspection_log', 'calendar', 'weather',
             'outage_log_observed', 'topology_edges')


def select_eligible(kept, meters_src, truth):
    """(eligible ids, assign_types map, hooked local dt indices 0..6)."""
    dts = sorted(meters_src.loc[kept].gis_dt_id.unique())
    train_dts = set(dts[:7])                              # mirrors merged_context split
    clean = set(truth.loc[~truth.is_theft, 'meter_id'])
    eligible = sorted(m for m in kept
                      if meters_src.loc[m].gis_dt_id in train_dts and m in clean)
    types = assign_types(eligible, prevalence=N_INJECTED / len(eligible))
    hook_rng = np.random.default_rng([42, TRAIN_SALT, 999])
    hooked = sorted(int(i) for i in hook_rng.choice(7, N_HOOK_DTS, replace=False))
    return eligible, types, hooked


def inject_readings_frame(reads, selected):
    """Assessment-day kwh edits for selected meters; NaN slots preserved;
    baseline days and all other rows untouched."""
    out = reads.copy()
    for mid, t in selected.items():
        if t is None:
            continue
        m = ((out.meter_id.astype(str) == mid)
             & (out.ts >= ASSESS_START) & (out.ts < ASSESS_END))
        g = out.loc[m].sort_values('ts')
        assert len(g) == 7 * 96, f'{mid}: expected 672 assessment rows, got {len(g)}'
        kwh = g.kwh.to_numpy(np.float64)
        nan_mask = ~np.isfinite(kwh)
        filled = np.nan_to_num(kwh)
        rng = meter_rng(mid, salt=TRAIN_SALT)
        new = filled.reshape(7, 96).copy()
        for d in range(7):
            new[d] = apply_theft(new[d], t, rng)
        new = new.reshape(-1)
        new[nan_mask] = np.nan
        out.loc[g.index, 'kwh'] = new
    return out


def hook_dt_frame(dtr, hooked_dt_ids, factor=HOOK_FACTOR):
    """Scale kwh_in/kvarh_in of hooked DTs (local indices into sorted dt_id)
    over assessment days only."""
    out = dtr.copy()
    dt_ids = sorted(out.dt_id.unique())
    hooked = [dt_ids[i] for i in hooked_dt_ids]
    m = out.dt_id.isin(hooked) & (out.ts >= ASSESS_START) & (out.ts < ASSESS_END)
    out.loc[m, 'kwh_in'] = out.loc[m, 'kwh_in'] * factor
    out.loc[m, 'kvarh_in'] = out.loc[m, 'kvarh_in'] * factor
    return out


def flip_truth_frame(truth, selected):
    """Mark injected meters as theft with a distinct archetype; others untouched."""
    out = truth.copy()
    for mid, t in selected.items():
        if t is None:
            continue
        m = out.meter_id == mid
        out.loc[m, 'is_theft'] = True
        out.loc[m, 'archetype'] = f'zidi_t{t}_injected'
        out.loc[m, 'start_date'] = ASSESS_START
        out.loc[m, 'intensity'] = 0.45
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def derive_shadow(out_dir: Path = SHADOW_DIR, force: bool = False) -> dict:
    assert os.environ.get('MERGED_DATA_DIR') is None, \
        'derive must read the ORIGINAL dataset (unset MERGED_DATA_DIR)'
    if out_dir.exists():
        if not force:
            raise FileExistsError(f'{out_dir} exists; rerun with --force to rebuild')
        shutil.rmtree(out_dir)
    (out_dir / 'observable').mkdir(parents=True)
    (out_dir / 'truth').mkdir()

    kept, _excluded = _uniform_subsample('demo', 0)
    src = load_observable()
    meters_src = src['meters'].set_index('meter_id')
    truth = load_truth()['truth_meters']
    eligible, types, hooked = select_eligible(kept, meters_src, truth)
    selected = {m: t for m, t in types.items() if t is not None}
    assert len(selected) == N_INJECTED, f'expected {N_INJECTED} thieves, got {len(selected)}'

    reads = load_readings(kept, columns=['meter_id', 'ts', 'kwh', 'kvarh',
                                         'voltage_v', 'is_missing'])
    reads_i = inject_readings_frame(reads, selected)

    demo_dts = sorted(meters_src.loc[kept].gis_dt_id.unique())
    dtr = pd.read_parquet(MERGED_BASE / 'observable' / 'dt_readings.parquet')
    dtr = dtr[dtr.dt_id.isin(demo_dts)].reset_index(drop=True)
    dtr_i = hook_dt_frame(dtr, hooked)

    truth_i = flip_truth_frame(truth, selected)

    reads_i.to_parquet(out_dir / 'observable' / 'readings.parquet', index=False)
    dtr_i.to_parquet(out_dir / 'observable' / 'dt_readings.parquet', index=False)
    truth_i.to_parquet(out_dir / 'truth' / 'truth_meters.parquet', index=False)
    for name in OBS_FILES:
        shutil.copy2(MERGED_BASE / 'observable' / f'{name}.parquet',
                     out_dir / 'observable' / f'{name}.parquet')
    shutil.copy2(MERGED_BASE / 'truth' / 'truth_outages.parquet',
                 out_dir / 'truth' / 'truth_outages.parquet')
    tdt = pd.read_parquet(MERGED_BASE / 'truth' / 'truth_dt.parquet')
    tdt[tdt.dt_id.isin(demo_dts)].to_parquet(out_dir / 'truth' / 'truth_dt.parquet',
                                             index=False)

    manifest = {
        'source': 'synthetic_india_v2',
        'train_salt': TRAIN_SALT, 'n_injected': len(selected),
        'per_type': {f'theft_{t}': sum(1 for v in selected.values() if v == t)
                     for t in range(1, 7)},
        'selected': selected, 'hooked_local_dts': hooked,
        'hook_factor': HOOK_FACTOR,
        'eligible_count': len(eligible),
        'assessment_window': [str(ASSESS_START), str(ASSESS_END)],
        'source_sha256': {
            'readings': _sha256(MERGED_BASE / 'observable' / 'readings.parquet'),
            'truth_meters': _sha256(MERGED_BASE / 'truth' / 'truth_meters.parquet')},
        'output_sha256': {
            'readings': _sha256(out_dir / 'observable' / 'readings.parquet'),
            'truth_meters': _sha256(out_dir / 'truth' / 'truth_meters.parquet')},
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2),
                                           encoding='utf-8')
    return manifest


if __name__ == '__main__':
    print(json.dumps(derive_shadow(force='--force' in sys.argv), indent=2))
