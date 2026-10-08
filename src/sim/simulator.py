from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .physics import make_topology, sweep, meter_power_to_nodes, transformer_loss


@dataclass
class Observations:
    """Public telemetry only. No true load, label, physical DT or true loss."""
    meters: pd.DataFrame
    kwh: np.ndarray
    kvarh: np.ndarray
    voltage: np.ndarray
    events: np.ndarray
    missing: np.ndarray
    dt_kwh: np.ndarray
    dt_kvarh: np.ndarray
    dt_voltage: np.ndarray
    dt_currents: np.ndarray
    temperature: np.ndarray
    topologies: list
    split: np.ndarray
    cfg: dict
    seed: int


@dataclass
class HiddenLabels:
    labels_meter: pd.DataFrame
    ntl: np.ndarray
    technical: np.ndarray
    visible: np.ndarray
    invisible: np.ndarray
    theft_kwh: np.ndarray
    true_dt: np.ndarray
    clean_balance_error: float


def dt_splits(n, rng):
    order = rng.permutation(n)
    split = np.full(n, 'train', dtype='<U5')
    v = max(4, int(round(n * .2)))
    te = max(4, int(round(n * .2)))
    split[order[-te:]] = 'test'
    split[order[-te-v:-te]] = 'val'
    return split


def simulate(cfg, seed, clean=False):
    rng = np.random.default_rng(seed)
    d, m, days = cfg['n_dts'], cfg['meters_per_dt'], cfg['days']
    n, steps = d * m, days * 96
    base = cfg['baseline_days'] * 96
    dt_hours = cfg['interval_minutes'] / 60
    if cfg['interval_minutes'] != 15:
        raise ValueError('This generator implements quarter-hour intervals only.')
    split = dt_splits(d, rng)
    true_dt = np.repeat(np.arange(d), m)
    phase = rng.integers(0, 3, n)
    late_phase = phase.copy()
    phase_changes = rng.random(n) < cfg['phase_shift_fraction']
    late_phase[phase_changes] = (phase[phase_changes] + 1) % 3
    if clean:
        late_phase = phase.copy()
    category = rng.choice(['domestic', 'small_commercial', 'agricultural'], n, p=[.85, .10, .05])
    area_dt = rng.choice(['urban', 'semi_urban', 'rural'], d, p=[.4, .35, .25])
    level = np.clip(rng.lognormal(-.55, .55, n), .08, 2.)
    hour = (np.arange(steps) % 96) / 4
    day = np.arange(steps) // 96
    temperature = (31 + 4 * np.sin(2 * np.pi * day / 45) +
                   3 * np.sin(2 * np.pi * (hour - 9) / 24) +
                   rng.normal(0, .8, steps)).astype('float32')
    morning = np.exp(-((hour - 8) / 2.5) ** 2)
    evening = np.exp(-((hour - 20) / 3.) ** 2)
    domestic = .3 + .3 * morning + .7 * evening
    shapes = np.broadcast_to(domestic, (n, steps)).copy()
    shapes[category == 'small_commercial'] = .12 + .85 * ((hour >= 9) & (hour < 21))
    shapes[category == 'agricultural'] = .2 + .6 * ((hour >= 5) & (hour < 12))
    shapes *= (1 + .12 * (day % 7 >= 5))
    cooling = rng.uniform(.0, .055, n)[:, None] * np.maximum(temperature - 29, 0)
    daily_jitter = rng.lognormal(-.5 * .18**2, .18, (n, days))
    p = ((shapes + cooling) * level[:, None] * np.repeat(daily_jitter, 96, axis=1)).astype('float32')
    p *= rng.lognormal(-.5 * .10**2, .10, (n, steps)).astype('float32')
    # Unlabelled gradual changes prevent a perfectly stable synthetic baseline.
    p *= 1 + rng.uniform(-.15, .20, n)[:, None] * np.linspace(0, 1, steps)
    archetype = np.full(n, 'honest', dtype='<U24')
    is_theft = np.zeros(n, dtype=bool)
    start = np.full(n, -1, dtype=int)
    params = [{} for _ in range(n)]
    # Half of DTs in EACH preassigned split carry theft; prevalence is exact
    # within rounding. Graph train/validation/test never share meters.
    if not clean:
        for split_name in ['train', 'val', 'test']:
            ids = np.flatnonzero(split == split_name)
            risk_dt = rng.choice(ids, max(1, len(ids) // 2), replace=False)
            candidates = np.flatnonzero(np.isin(true_dt, risk_dt))
            count = int(round(cfg['prevalence'] * len(ids) * m))
            theft_ids = rng.choice(candidates, count, replace=False)
            is_theft[theft_ids] = True
        theft_modes = ['partial_bypass', 'slowing', 'frozen', 'intermittent']
        archetype[is_theft] = rng.choice(theft_modes, is_theft.sum())
        free = np.flatnonzero(~is_theft)
        rng.shuffle(free)
        pos = 0
        for name, fraction in [('vacant', cfg['vacancy_fraction'] / 2),
                               ('seasonal', cfg['vacancy_fraction'] / 2),
                               ('solar', cfg['solar_fraction']), ('ev_onset', cfg['ev_fraction']),
                               ('honest_meter_fault', cfg['honest_meter_fault_fraction'])]:
            count = int(round(n * fraction))
            archetype[free[pos:pos+count]] = name
            pos += count
    for i in range(n):
        a = archetype[i]
        if a == 'honest':
            continue
        onset = int(rng.integers(base, max(base + 1, min(steps - 7*96, base + 15*96))))
        start[i] = onset
        if a in ('vacant', 'seasonal'):
            fraction = rng.uniform(.02, .22) if a == 'vacant' else rng.uniform(.08, .42)
            p[i, onset:] *= fraction
            if a == 'seasonal':
                # Some migration windows end; others remain indistinguishable
                # from persistent vacancy during the observation horizon.
                for j in range(onset, steps, 7*96):
                    if rng.random() < .4:
                        p[i, j:j+48] /= fraction
            params[i] = {'remaining_fraction': float(fraction)}
        elif a == 'solar':
            solar = np.maximum(np.sin(np.pi * (hour - 6) / 12), 0)
            capacity = rng.uniform(.4, 2.)
            p[i, onset:] -= capacity * solar[onset:]
            params[i] = {'solar_peak_kw': float(capacity)}
        elif a == 'ev_onset':
            p[i, onset:] += rng.uniform(1.2, 3.2) * ((hour[onset:] >= 22) | (hour[onset:] < 1))
    outage = np.zeros((d, steps), bool)
    if not clean:
        for g in range(d):
            for day_id in range(days):
                if day_id % cfg['planned_outage_days'] == g % cfg['planned_outage_days']:
                    outage[g, day_id*96+44:day_id*96+48] = True
                if rng.random() < cfg['unplanned_outage_probability_per_day']:
                    at = day_id*96+int(rng.integers(0, 90))
                    outage[g, at:at+int(rng.integers(2, 10))] = True
        p[outage[true_dt]] = 0
    metered_p = p.copy()
    for i in range(n):
        a = archetype[i]
        onset = start[i]
        if a not in ('partial_bypass', 'slowing', 'frozen', 'intermittent', 'honest_meter_fault'):
            continue
        if is_theft[i] and rng.random() < .15:
            onset = 0  # Persistent theft defeats own-history-only expectations.
            start[i] = 0
        shifted = cfg['parameter_set'] == 'B'
        remaining = rng.uniform(.35, .80) if shifted else rng.uniform(.10, .60)
        if a == 'partial_bypass':
            metered_p[i, onset:] *= remaining
        elif a == 'slowing':
            factors = np.clip(remaining + .12*np.sin(np.arange(steps-onset)/96), .03, .95)
            metered_p[i, onset:] *= factors
        elif a in ('frozen', 'honest_meter_fault'):
            const = 0 if rng.random() < .5 else float(np.quantile(p[i], .1) * .2)
            metered_p[i, onset:] = np.minimum(p[i, onset:], const)
        elif a == 'intermittent':
            window = ((hour >= 11) & (hour < 18)) if shifted else ((hour >= 18) | (hour < 2))
            active = window & (np.arange(steps) >= onset)
            metered_p[i, active] *= remaining
        params[i] = {'remaining_fraction': float(remaining), 'parameter_set': cfg['parameter_set']}
    visible_meter = np.maximum(p - metered_p, 0) * dt_hours
    theft_kwh = visible_meter * is_theft[:, None]
    visible = visible_meter.reshape(d, m, steps).sum(1)
    invisible = np.zeros((d, steps), np.float32)
    hook_power = np.zeros((d, steps), np.float32)
    if not clean:
        hook_dts = rng.choice(d, int(round(d * cfg['hooking_dt_fraction'])), replace=False)
        for g in hook_dts:
            hook_power[g, base:] = rng.uniform(.3, 2.) * (.4 + evening[base:])
            hook_power[g, outage[g]] = 0
    invisible[:] = hook_power * dt_hours
    technical = np.zeros((d, steps), np.float32)
    dt_input = np.zeros_like(technical)
    dt_q = np.zeros_like(technical)
    v = np.zeros((n, steps), np.float32)
    dt_v = np.zeros((d, steps), np.float32)
    currents = np.zeros((d, steps, 3), np.float32)
    topologies = []
    q_ratio = np.tan(np.arccos(cfg['power_factor']))
    for g in range(d):
        topo = make_topology(m, rng, cfg['topology_source'])
        topologies.append(topo)
        ids = slice(g*m, (g+1)*m)
        pp, qq = meter_power_to_nodes(topo, p[ids], phase[ids], q_ratio,
                                      late_phase[ids], base + (steps-base)//2)
        # Hooking is a pole load, never assigned a consumer meter label.
        hook_node = max(1, topo.pole_count//2)
        pp[hook_node, :, g % 3] += hook_power[g]
        qq[hook_node, :, g % 3] += hook_power[g] * q_ratio
        root_v = np.full(steps, cfg['nominal_voltage_v'], np.float64)
        if not clean:
            root_v += 1.5*np.sin(np.arange(steps)/100) + rng.normal(0, .5, steps)
        root_v[outage[g]] = 0
        vv, line_loss, ii = sweep(topo, pp, qq, root_v, cfg['solver_iterations'])
        load_p = pp.sum((0, 2))
        load_q = qq.sum((0, 2))
        dt_loss = transformer_loss(load_p, load_q, cfg) * ~outage[g]
        technical[g] = (line_loss + dt_loss) * dt_hours
        dt_input[g] = load_p*dt_hours + technical[g]
        dt_q[g] = load_q*dt_hours  # reactive line loss deliberately omitted
        currents[g] = np.abs(ii)
        dt_v[g] = root_v
        for j, node in enumerate(topo.meter_nodes):
            switch = base + (steps-base)//2
            v[g*m+j, :switch] = vv[node, :switch, phase[g*m+j]]
            v[g*m+j, switch:] = vv[node, switch:, late_phase[g*m+j]]
    balance_error = float(np.max(np.abs(dt_input - (metered_p.reshape(d,m,steps).sum(1)*dt_hours + technical + visible + invisible))))
    kwh = metered_p * dt_hours
    kvarh = np.maximum(metered_p, 0) * q_ratio * dt_hours
    events = np.zeros((n, steps), np.uint8)
    missing = np.zeros((n, steps), bool)
    if not clean:
        kwh *= (1 + rng.normal(0, cfg['energy_noise_fraction'], (n, 1))).astype('float32')
        v += rng.normal(0, cfg['voltage_noise_v'], v.shape).astype('float32')
        resolution = cfg['meter_voltage_resolution_v']
        v = np.round(v / resolution) * resolution
        v[outage[true_dt]] = 0
        for g in range(d):
            offset = rng.uniform(-cfg['dt_skew_minutes'], cfg['dt_skew_minutes']) / cfg['interval_minutes']
            dt_input[g] = np.interp(np.arange(steps)+offset, np.arange(steps), dt_input[g])
            dt_input[g] *= 1 + rng.uniform(-cfg['dt_accuracy_fraction'], cfg['dt_accuracy_fraction'])
        dt_v += rng.normal(0, cfg['dt_voltage_noise_v'], dt_v.shape)
        dt_v[outage] = 0
        if cfg['events']:
            prob = np.full((n, 1), cfg['event_false_positive_per_day'] / 96)
            prob[is_theft] = cfg['event_sensitivity_per_day'] / 96
            event_draw = rng.random((n, steps)) < prob
            active = np.arange(steps)[None, :] >= start[:, None]
            event_draw[is_theft] &= active[is_theft]
            events[event_draw] = 1  # generic tamper-like event, imperfect
            events[outage[true_dt]] |= 2  # power-off event
        missing = rng.random((n, steps)) < cfg['dropout']
        # Disable bursts at zero dropout so the zero stress point is truly zero.
        if cfg['dropout'] > 0:
            for i in np.flatnonzero(rng.random(n) < cfg['burst_dropout_fraction']):
                at = int(rng.integers(base, steps - 96))
                missing[i, at:at+int(rng.integers(24, 3*96))] = True
        kwh[missing] = np.nan
        kvarh[missing] = np.nan
        v[missing] = np.nan
        events[missing] = 0
    # Swap complete meter records between observed DT slots, within the same
    # partition. Physical topology remains hidden. No cross-split edge exists.
    observed_order = np.arange(n)
    if not clean:
        for s in ['train', 'val', 'test']:
            ids = np.flatnonzero(np.isin(true_dt, np.flatnonzero(split == s)))
            chosen = rng.choice(ids, int(round(cfg['mapping_error']*len(ids))), replace=False)
            # Only actual cross-DT swaps count; achieved rate is reported.
            if len(chosen) > 1:
                observed_order[chosen] = np.roll(chosen, 1)
    orig = observed_order
    mapped_dt = np.repeat(np.arange(d), m)
    meters = pd.DataFrame({
        'meter_id': [f'M{i:05d}' for i in orig],
        'dt_id': mapped_dt, 'mapped_dt_id': mapped_dt,
        'feeder_id': [f'DT{g:03d}-LV' for g in mapped_dt],
        'pole_id': [f'DT{g:03d}-P{topologies[g].parent[topologies[g].meter_nodes[j]]:02d}' for g in range(d) for j in range(m)],
        'phase': phase[orig], 'category': category[orig],
        'sanctioned_load_kw': np.ceil(np.maximum(level[orig]*3, 1)),
        'lat': 17.38 + mapped_dt*.0003 + rng.normal(0, .0001, n),
        'lon': 78.48 + np.tile(np.arange(m), d)*.00006,
        'has_event_log': bool(cfg['events']),
        'area_type': area_dt[true_dt[orig]],
    })
    label_frame = pd.DataFrame({'meter_id': meters.meter_id, 'archetype': archetype[orig],
                                'is_theft': is_theft[orig], 'theft_start': start[orig],
                                'theft_params': [json.dumps(params[i]) for i in orig],
                                'physical_dt_id': true_dt[orig]})
    obs = Observations(meters, kwh[orig], kvarh[orig], v[orig], events[orig], missing[orig],
                       dt_input, dt_q, dt_v, currents, temperature, topologies, split,
                       dict(cfg), seed)
    hidden = HiddenLabels(label_frame, visible+invisible, technical, visible, invisible,
                          theft_kwh[orig], true_dt[orig], balance_error)
    return obs, hidden


def export_contract(obs, hidden, destination, dt_ids=None):
    """Partitioned Parquet. Labels live in a separate hidden directory."""
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    d, t = obs.dt_kwh.shape
    ts = pd.date_range('2026-06-01', periods=t, freq='15min', tz='Asia/Kolkata')
    selected = list(range(d)) if dt_ids is None else list(dt_ids)
    keep = obs.meters.dt_id.isin(selected)
    obs.meters.loc[keep].to_parquet(dest/'meters.parquet', index=False)
    hidden_path = dest/'hidden'
    hidden_path.mkdir(exist_ok=True)
    hidden.labels_meter.loc[keep].to_parquet(hidden_path/'labels_meter.parquet', index=False)
    rows, edges, labels = [], [], []
    for g in selected:
        ix = np.flatnonzero(obs.meters.dt_id.to_numpy() == g)
        records = pd.DataFrame({'meter_id': np.repeat(obs.meters.meter_id.to_numpy()[ix], t),
                               'ts': np.tile(ts, len(ix)), 'kwh': obs.kwh[ix].ravel(),
                               'kvarh': obs.kvarh[ix].ravel(), 'voltage_v': obs.voltage[ix].ravel(),
                               'event_flags': obs.events[ix].ravel(), 'is_missing': obs.missing[ix].ravel()})
        part = dest/'readings'/f'dt_id={g}'
        part.mkdir(parents=True, exist_ok=True)
        records.to_parquet(part/'part.parquet', index=False, compression='zstd')
        rows.append(pd.DataFrame({'dt_id':g, 'ts':ts, 'kwh_in':obs.dt_kwh[g],
                                  'kvarh_in':obs.dt_kvarh[g], 'v_lv':obs.dt_voltage[g],
                                  'i_a':obs.dt_currents[g,:,0], 'i_b':obs.dt_currents[g,:,1],
                                  'i_c':obs.dt_currents[g,:,2]}))
        labels.append(pd.DataFrame({'dt_id':g, 'ts':ts, 'ntl_kwh':hidden.ntl[g],
                                    'tech_loss_kwh':hidden.technical[g],
                                    'meter_visible_ntl_kwh':hidden.visible[g],
                                    'invisible_ntl_kwh':hidden.invisible[g]}))
        topo = obs.topologies[g]
        public_names = [f'DT{g:03d}'] + [f'DT{g:03d}-P{j:02d}' for j in range(1,topo.pole_count+1)]
        public_names += obs.meters.meter_id.to_numpy()[ix].tolist()
        for node in range(1, topo.size):
            edges.append({'dt_id':g, 'from_node':public_names[topo.parent[node]],
                          'to_node':public_names[node], 'r_ohm':topo.r[node],
                          'x_ohm':topo.x[node], 'length_m':topo.length_m[node]})
    pd.concat(rows, ignore_index=True).to_parquet(dest/'dt_readings.parquet', index=False)
    pd.DataFrame(edges).to_parquet(dest/'topology_edges.parquet', index=False)
    pd.concat(labels, ignore_index=True).to_parquet(hidden_path/'labels_dt.parquet', index=False)
    (dest/'metadata.json').write_text(json.dumps({'profile_source':obs.cfg['profile_source'],
        'seed':obs.seed, 'schema':'energy in kWh per interval; signed net import; synthetic coordinates',
        'selected_dts':selected, 'label_access':'evaluation only'}, indent=2), encoding='utf-8')
