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


# ---- Stage 2: virtual DTs + physics-consistent dt input --------------------
from ..common import load_config                                   # noqa: E402
from ..sim.physics import (Topology, meter_power_to_nodes, sweep,  # noqa: E402
                           transformer_loss)
from ..features.build import causal_impute                         # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs' / 'final.yaml'


def star_topology(n_meters: int) -> Topology:
    """Honest minimal LV structure: wiring unknown, root feeds each meter
    directly with nominal cable impedance."""
    parents = np.array([-1] + [0] * n_meters, dtype=int)
    r = np.array([0.0] + [0.01] * n_meters, dtype=float)
    x = np.array([0.0] + [0.001] * n_meters, dtype=float)
    lengths = np.array([0] + [50] * n_meters, dtype=int)
    return Topology(parents, r, x, lengths, n_meters,
                    np.arange(1, n_meters + 1), 'ceew_star')


def trackc_cfg(n_dts: int, meters_per_dt: int) -> dict:
    cfg = load_config(CONFIG)
    cfg.update(n_dts=n_dts, meters_per_dt=meters_per_dt, days=WINDOW_DAYS,
               interval_minutes=15, baseline_days=7, profile_source='ceew_real_injected')
    return cfg


def _dt_input_from_true(true_kwh: np.ndarray, topo: Topology, cfg: dict,
                        dt_voltage: np.ndarray) -> np.ndarray:
    """dt_readings.kwh_in = sum(true) + technical loss computed with the SAME
    physics math as build.py:186-197, applied to TRUE loads. This makes the
    clean residual about 0 so injected theft alone creates unexplained energy."""
    qratio = np.tan(np.arccos(cfg['power_factor']))
    phase = np.zeros(true_kwh.shape[0], dtype=int)
    power = true_kwh / (cfg['interval_minutes'] / 60)
    pp, qq = meter_power_to_nodes(topo, power, phase, qratio)
    root = np.maximum(dt_voltage, 0)
    _, line_loss, _ = sweep(topo, pp, qq, root, cfg['solver_iterations'])
    dt_loss = transformer_loss(pp.sum((0, 2)), qq.sum((0, 2)), cfg) * (root > 50)
    loss = (line_loss + dt_loss) * (cfg['interval_minutes'] / 60) * cfg['loss_estimator_multiplier']
    return true_kwh.sum(axis=0) + np.maximum(loss, 0)


def build_window(district: str, win: dict, cfg: dict | None = None):
    """Build Observations for one 14-day window with TRUE readings (pre-injection).

    Missing slots stay NaN in obs.kwh (causal_impute convention); obs.missing
    mirrors ~isfinite. dt arrays are zero-filled (serving convention).
    """
    from ..sim.simulator import Observations

    base = load_district(district)
    start = pd.Timestamp(win['start'], tz=TZ)
    end = pd.Timestamp(win['end'], tz=TZ)
    kept = list(win['kept_meters'])
    d, m = win['n_dts'], win['meters_per_dt']
    assert len(kept) == d * m
    if cfg is None:
        cfg = trackc_cfg(d, m)

    w = base[(base.ts >= start) & (base.ts < end) & base.meter_id.isin(kept)].copy()
    w['meter_id'] = w['meter_id'].astype(str)
    idx = pd.MultiIndex.from_product(
        [kept, pd.date_range(start, end, freq='15min', inclusive='left')],
        names=['meter_id', 'ts'])
    piv = w.set_index(['meter_id', 'ts']).reindex(idx)
    kwh = piv.kwh.to_numpy(np.float64).reshape(len(kept), WINDOW_STEPS)
    voltage = piv.voltage_v.to_numpy(np.float64).reshape(len(kept), WINDOW_STEPS)
    current = piv.current_a.to_numpy(np.float64).reshape(len(kept), WINDOW_STEPS)
    missing = ~np.isfinite(kwh)

    # meters sorted by (dt_id, meter_id): dt = original position // m
    meters = pd.DataFrame({
        'meter_id': kept,
        'dt_id': np.repeat(np.arange(d), m),
        'phase': 0,
        'category': 'residential',
        'area_type': 'real_india_urban',
        'sanctioned_load_kw': 5.0,
    })
    sort_key = np.lexsort((np.array(kept), meters.dt_id.to_numpy()))
    kept_sorted = [kept[i] for i in sort_key]
    meters = meters.iloc[sort_key].reset_index(drop=True)
    kwh, voltage, current, missing = (a[sort_key] for a in (kwh, voltage, current, missing))

    # neutral temperature (no weather in CEEW; seasonal_adjust degenerates to 1)
    temperature = np.full(WINDOW_STEPS, 25.0, np.float32)

    # DT-level measurements from member meters (mean V / mean I; 0 when all missing)
    dt_voltage = np.zeros((d, WINDOW_STEPS), np.float64)
    dt_current = np.zeros((d, WINDOW_STEPS), np.float64)
    for g in range(d):
        blk_v = voltage[g * m:(g + 1) * m]
        blk_i = current[g * m:(g + 1) * m]
        with np.errstate(invalid='ignore'):
            dt_voltage[g] = np.nanmean(np.where(np.isfinite(blk_v), blk_v, np.nan), axis=0)
            dt_current[g] = np.nanmean(np.where(np.isfinite(blk_i), blk_i, np.nan), axis=0)
    dt_voltage = np.nan_to_num(dt_voltage)
    dt_current = np.nan_to_num(dt_current)

    # TRUE dt input: sum of causally-imputed true loads + physics loss on true loads
    true_for_loss = causal_impute(kwh.copy(), dt_voltage,
                                  meters.dt_id.to_numpy(), True)
    dt_kwh = np.zeros((d, WINDOW_STEPS), np.float64)
    for g in range(d):
        topo = star_topology(m)
        dt_kwh[g] = _dt_input_from_true(true_for_loss[g * m:(g + 1) * m],
                                        topo, cfg, dt_voltage[g])

    topologies = [star_topology(m) for _ in range(d)]
    obs = Observations(
        meters=meters,
        kwh=kwh.astype(np.float32),                 # NaN preserved
        kvarh=np.where(missing, np.nan, 0.0).astype(np.float32),
        voltage=voltage.astype(np.float32),
        events=np.zeros((d * m, WINDOW_STEPS), np.uint8),
        missing=missing,
        dt_kwh=dt_kwh.astype(np.float32),
        dt_kvarh=np.zeros((d, WINDOW_STEPS), np.float32),
        dt_voltage=dt_voltage.astype(np.float32),
        dt_currents=np.stack([dt_current, np.zeros_like(dt_current),
                              np.zeros_like(dt_current)], axis=2).astype(np.float32),
        temperature=temperature,
        topologies=topologies,
        split=np.full(d, 'trackc'),
        cfg=cfg,
        seed=42,
    )
    info = {'meters': kept_sorted,
            'dt_of_meter': {mid: i // m for i, mid in enumerate(kept_sorted)}}
    return obs, info


# ---- Stage 3: theft injection (Zidi et al. 2023, Sec. 3.2) + DT-hook -------
MANIFEST_PATH = CEEW_BASE / 'manifest.json'
PREVALENCE = 0.09
HOOK_U = (0.05, 0.15)


def assign_types(meter_ids: list[str], prevalence: float = PREVALENCE) -> dict:
    """Type fixed per meter across all windows (seed 42, order-independent)."""
    types = {}
    for mid in sorted(meter_ids):
        rng = meter_rng(mid, salt=1)
        types[mid] = int(rng.integers(1, 7)) if rng.random() < prevalence else None
    return types


def apply_theft(day: np.ndarray, theft_type: int, rng: np.random.Generator) -> np.ndarray:
    """Zidi et al. (2023) Sec. 3.2 six-type generator on one day vector."""
    y = np.asarray(day, dtype=np.float64).copy()
    L = len(y)
    hour_slots = max(L // 24, 1)
    if theft_type == 1:
        y = y * rng.uniform(0.1, 0.8)
    elif theft_type == 2:
        min_slots = 4 * hour_slots                    # >= 4 hours
        if L > min_slots:
            dur = int(rng.integers(min_slots, L + 1))
            start = int(rng.integers(0, L - dur + 1))
            y[start:start + dur] = 0.0
        else:
            y[:] = 0.0
    elif theft_type == 3:
        y = y * rng.uniform(0.1, 0.8, size=L)
    elif theft_type == 4:
        y = rng.uniform(0.1, 0.8, size=L) * np.nanmean(y)
    elif theft_type == 5:
        y[:] = np.nanmean(y)
    elif theft_type == 6:
        y = y[::-1].copy()
    else:
        raise ValueError(f'unknown theft type {theft_type}')
    return y


def inject_window(obs, types: dict, win: dict, win_index: int):
    """Return (copy of obs with reported kwh injected + optional DT-hook, manifest row).

    Only complete days are injected. DT-hook (our extension beyond Zidi) adds
    energy to dt_kwh only -- meter readings untouched.
    """
    import copy as _copy
    rep = _copy.deepcopy(obs)
    d = obs.cfg['n_dts']
    positives, injected_days_total = [], 0
    for i, mid in enumerate(rep.meters.meter_id):
        t = types.get(mid)
        if t is None:
            continue
        kwh_row = rep.kwh[i]
        complete = np.isfinite(kwh_row).reshape(WINDOW_DAYS, 96).all(axis=1)
        n_days = 0
        rng = meter_rng(mid, salt=win_index + 100)
        for day in np.flatnonzero(complete):
            sl = slice(day * 96, (day + 1) * 96)
            kwh_row[sl] = apply_theft(kwh_row[sl].astype(np.float64), t, rng)
            n_days += 1
        rep.missing[i] = ~np.isfinite(kwh_row)
        if n_days > 0:
            removed = float(np.nansum(obs.kwh[i]) - np.nansum(kwh_row))
            positives.append({'meter_id': mid, 'type': f'theft_{t}',
                              'injected_days': n_days, 'kwh_removed': round(removed, 3)})
            injected_days_total += n_days
    # DT-hook extension: inflate dt input on a few DTs (never on meter rows)
    hooks = []
    n_hook = min(2, max(0, d - 1))
    if n_hook:
        hrng = meter_rng(f"hook:{win['window_id']}", salt=win_index)
        hooked = sorted(hrng.choice(d, size=n_hook, replace=False).tolist())
        for g in hooked:
            u_draws = hrng.uniform(*HOOK_U, size=WINDOW_DAYS)
            for day in range(WINDOW_DAYS):
                sl = slice(day * 96, (day + 1) * 96)
                mean_load = float(np.mean(rep.dt_kwh[g, sl]))
                if mean_load > 0:
                    rep.dt_kwh[g, sl] += u_draws[day] * mean_load
            hooks.append({'dt_id': int(g), 'u_mean': round(float(u_draws.mean()), 4)})
    row = {'window_id': win['window_id'], 'district': win['district'],
           'positives': positives, 'hooks': hooks,
           'injected_days_total': injected_days_total}
    return rep, row


def write_manifest(meter_types: dict, window_rows: list[dict],
                   skipped_windows: list[dict]) -> None:
    manifest = {'seed': 42, 'prevalence': PREVALENCE,
                'dt_hook': 'our extension beyond Zidi et al. 2023',
                'meter_types': {k: (f'theft_{v}' if v else None)
                                for k, v in sorted(meter_types.items())},
                'windows': window_rows, 'skipped_windows': skipped_windows}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def read_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
