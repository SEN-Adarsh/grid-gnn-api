"""Track C adapter: real CEEW Mathura/Bareilly telemetry (CC0, Harvard Dataverse
doi:10.7910/DVN/GOCHJH, Kaggle mirror) -> the synthetic_india_v2 observable schema
with per-window virtual DTs.

Theft is injected ONLY into reported meter readings; dt_readings are built from
TRUE loads before injection (physics-consistent injection rule). See
docs/superpowers/specs/2026-10-27-track-c-ceew-injection-design.md.
"""
from __future__ import annotations

import json
import os
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

CEEW_BASE = Path(os.environ.get('CEEW_DATA_DIR', 'D:/rodic/ceew_data'))
RAW = CEEW_BASE / 'raw'
STAGE = CEEW_BASE / 'data' / 'observable'
WINDOWS_JSON = CEEW_BASE / 'windows.json'
TZ = 'Asia/Kolkata'
WINDOW_DAYS = 14
WINDOW_STEPS = WINDOW_DAYS * 96

DISTRICT_FILES = {
    'BR': ['SM Cleaned Data BR2019.csv',
           'CEEW - Smart meter data Bareilly 2020.csv',
           'CEEW - Smart meter data Bareilly 2021.csv'],
    'MH': ['CEEW - Smart meter data Mathura 2019.csv',
           'CEEW - Smart meter data Mathura 2020.csv',
           'SM Cleaned Data MH2021.csv'],
}
RAW_COLS = ['x_Timestamp', 't_kWh', 'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)',
            'y_Freq (Hz)', 'meter']


def meter_rng(meter_id: str, salt: int = 0) -> np.random.Generator:
    """Order-independent deterministic RNG per meter (seed 42 + crc32 + salt)."""
    return np.random.default_rng([42, zlib.crc32(meter_id.encode()), salt])


def resample_15min(df: pd.DataFrame) -> pd.DataFrame:
    """3-min -> 15-min: kwh summed, V/I/F averaged. Zero-voltage rows (real
    outages) stay 0. Absent slots simply produce no row (NaN downstream)."""
    d = df.copy()
    ts = pd.to_datetime(d['x_Timestamp'])
    d['ts'] = ts.dt.tz_localize(TZ) if ts.dt.tz is None else ts
    d = d.set_index('ts')
    g = d.groupby([d['meter'], pd.Grouper(freq='15min')])
    out = g.agg(kwh=('t_kWh', 'sum'),
                voltage_v=('z_Avg Voltage (Volt)', 'mean'),
                current_a=('z_Avg Current (Amp)', 'mean'),
                freq_hz=('y_Freq (Hz)', 'mean')).reset_index()
    out = out.rename(columns={'meter': 'meter_id'})
    # Categorical keeps ~46 unique strings instead of one per row (memory).
    out['meter_id'] = pd.Categorical('CEEW-' + out['meter_id'].astype(str))
    return out[['meter_id', 'ts', 'kwh', 'voltage_v', 'current_a', 'freq_hz']]


@lru_cache(maxsize=4)
def load_district(district: str) -> pd.DataFrame:
    """Read + concatenate + resample one district (cached: build_window runs
    per window over the same multi-hundred-MB CSVs)."""
    frames = []
    for name in DISTRICT_FILES[district]:
        path = RAW / name
        if not path.exists():
            raise FileNotFoundError(f'raw file missing: {path} (run the staging copy step)')
        df = pd.read_csv(path, usecols=RAW_COLS)
        frames.append(resample_15min(df))
    base = pd.concat(frames, ignore_index=True)
    return base.sort_values(['meter_id', 'ts']).reset_index(drop=True)


def discover_windows(district: str, base: pd.DataFrame,
                     min_coverage: float = 0.95, target_m: int = 11) -> tuple[list[dict], list[dict]]:
    """One 14-day window per calendar month. Uniform n_dts x meters_per_dt grid
    (build_features hard-requires g*m+j grouping); remainder meters dropped.

    Returns (windows, skipped); a meter is valid in a window only if it covers
    >= min_coverage of the 1344 slots.
    """
    windows, skipped = [], []
    starts = pd.date_range(base.ts.min().normalize(),
                           base.ts.max().normalize(), freq='MS')
    for start in starts:
        end = start + pd.Timedelta(days=WINDOW_DAYS)
        if start >= base.ts.max():
            break
        w = base[(base.ts >= start) & (base.ts < end)]
        if w.empty:
            continue
        counts = w.groupby('meter_id', observed=True).size()
        valid = sorted(str(x) for x in counts[counts >= min_coverage * WINDOW_STEPS].index)
        if len(valid) < 10:
            skipped.append({'window_id': f'{district}-{start:%Y-%m}',
                            'reason': 'too_few_meters', 'n_valid': len(valid)})
            continue
        n_dts = max(1, len(valid) // target_m)
        m = len(valid) // n_dts
        kept = valid[:n_dts * m]
        dropped = [x for x in valid if x not in kept]
        windows.append({'window_id': f'{district}-{start:%Y-%m}', 'district': district,
                        'start': start.strftime('%Y-%m-%d'), 'end': end.strftime('%Y-%m-%d'),
                        'n_dts': n_dts, 'meters_per_dt': m,
                        'kept_meters': kept, 'dropped_meters': dropped,
                        'n_valid_before_drop': len(valid)})
    return windows, skipped
