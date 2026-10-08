"""WP-13 synthetic generator v2: India-relevant theft families, honest
confounders, typed event log, data problems, hold-out hooks.

Extends the inherited generator WITHOUT modifying it: reuses the physics
layer primitives and the Observations/HiddenLabels interfaces so the
existing feature/evaluation stack runs unchanged. Every parameter is an
ASSUMPTION recorded in docs/ASSUMPTIONS.md. Nothing is calibrated to
Indian measurements.

Honesty rules implemented (WP-13 Section 1):
- ground truth (labels, families, hooking, true consumption) lives only in
  HiddenLabels / ground-truth exports, never in Observations;
- T3 hooking never appears in meter-level labels or theft_kwh;
- deterministic given (cfg, seed): single rng stream, fixed draw order.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .physics import make_topology, sweep, transformer_loss
from .simulator import Observations, HiddenLabels, dt_splits

FAMILIES = ['t1_partial_bypass', 't2_night_bypass', 't3_hooking', 't4_magnetic',
            't5_neutral_phase', 't6_slow_ramp', 't7_zero_flat', 't8_cluster']
HONEST_ARCHETYPES = ['vacant', 'seasonal_occupancy', 'meter_fault_stuck',
                     'meter_fault_intermittent', 'meter_fault_drift',
                     'meter_fault_spike', 'behaviour_up', 'behaviour_down',
                     'solar', 'disconnect_reconnect']
EVENT_TYPES = ['magnetic_influence', 'cover_open', 'neutral_disturbance',
               'power_failure', 'voltage_sag_swell', 'reverse_current',
               'communication_failure']
CLUSTER_FAMILIES = ['t3_hooking', 't1_partial_bypass']


@dataclass
class ObservationsV2(Observations):
    typed_events: pd.DataFrame = None
    pack: str = ''


def _load_model(cfg, rng, n, steps, days, hour, day, category):
    """True consumption kW per interval. Same structure as the inherited
    generator plus an explicit annual temperature envelope for seasonal
    packs. All shapes are assumptions (docs/ASSUMPTIONS.md)."""
    level = np.clip(rng.lognormal(-.55, .55, n), .08, 2.)
    morning = np.exp(-((hour - 8) / 2.5) ** 2)
    evening = np.exp(-((hour - 20) / 3.) ** 2)
    domestic = .3 + .3 * morning + .7 * evening
    shapes = np.broadcast_to(domestic, (n, steps)).copy()
    shapes[category == 'small_commercial'] = .12 + .85 * ((hour >= 9) & (hour < 21))
    shapes[category == 'agricultural'] = .2 + .6 * ((hour >= 5) & (hour < 12))
    shapes *= (1 + .12 * (day % 7 >= 5))
    temp_annual = (cfg['temp_mean_c'] +
                   cfg['temp_seasonal_amplitude_c'] * np.sin(2 * np.pi * (day + cfg['temp_seasonal_phase_days']) / 365.) +
                   cfg['temp_daily_amplitude_c'] * np.sin(2 * np.pi * (hour - 9) / 24))
    temperature = (temp_annual + rng.normal(0, .8, steps)).astype('float32')
    cooling = rng.uniform(.0, .055, n)[:, None] * np.maximum(temperature - cfg['cooling_threshold_c'], 0)
    daily_jitter = rng.lognormal(-.5 * .18 ** 2, .18, (n, days))
    p = ((shapes + cooling) * level[:, None] * np.repeat(daily_jitter, 96, axis=1)).astype('float32')
    p *= rng.lognormal(-.5 * .10 ** 2, .10, (n, steps)).astype('float32')
    # Unlabelled gradual changes prevent a perfectly stable synthetic baseline.
    p *= 1 + rng.uniform(-.15, .20, n)[:, None] * np.linspace(0, 1, steps)
    return p, level, temperature, evening


def _assign(cfg, rng, d, m, split, true_dt, n, steps, base, hour, three_phase):
    """Assign theft families and honest confounders.

    Per-DT prevalence drawn from an assumption range (2-15%) with a
    configurable share of zero-theft DTs. Family weights are per customer;
    t5 restricted to three-phase meters; t8 converts to a same-DT cluster of
    neighbours receiving t1/t3 together (recorded via params['cluster']).
    """
    family = np.full(n, '', dtype='<U24')
    honest = np.full(n, '', dtype='<U24')
    start = np.full(n, -1, dtype=int)
    params = [{} for _ in range(n)]
    weights = np.array(cfg['family_weights'], dtype=float)
    weights = weights / weights.sum()
    zero_frac = cfg['zero_theft_dt_fraction']
    prev_lo, prev_hi = cfg['per_dt_prevalence_range']
    onset_span = max(base + 1, min(steps - 7 * 96, base + 15 * 96))

    def new_onset():
        return int(rng.integers(base, onset_span))

    for s in ['train', 'val', 'test']:
        ids = np.flatnonzero(split == s)
        for g in ids:
            members = np.flatnonzero(true_dt == g)
            prev = 0. if rng.random() < zero_frac else rng.uniform(prev_lo, prev_hi)
            count = int(round(prev * len(members)))
            if count == 0:
                continue
            chosen = list(rng.choice(members, count, replace=False))
            unassigned = list(chosen)
            while unassigned:
                i = unassigned.pop(0)
                eligible = weights.copy()
                if not three_phase[i]:
                    eligible[FAMILIES.index('t5_neutral_phase')] = 0.
                fam = FAMILIES[int(rng.choice(len(FAMILIES), p=eligible / eligible.sum()))]
                if fam == 't8_cluster':
                    avail = len(unassigned) + 1
                    k = int(rng.integers(cfg['cluster_min_customers'], cfg['cluster_max_customers'] + 1))
                    k = max(1, min(k, avail))
                    cluster = [i] + unassigned[:k - 1]
                    unassigned = unassigned[k - 1:]
                    cluster_fam = CLUSTER_FAMILIES[int(rng.random() < .5)]
                    for j in cluster:
                        family[j] = cluster_fam
                        start[j] = new_onset()
                        params[j] = ({'cluster': True, 'cluster_size': len(cluster)}
                                     if len(cluster) > 1 else {})
                else:
                    family[i] = fam
                    start[i] = new_onset()
    free = np.flatnonzero(family == '')
    rng.shuffle(free)
    fr = cfg['confounder_fractions']
    mf = fr['meter_fault']
    split_mf = cfg['meter_fault_split']
    blocks = [('vacant', fr['vacancy']), ('seasonal_occupancy', fr['seasonal_occupancy']),
              ('meter_fault_stuck', mf * split_mf[0]),
              ('meter_fault_intermittent', mf * split_mf[1]),
              ('meter_fault_drift', mf * split_mf[2]),
              ('meter_fault_spike', mf * split_mf[3]),
              ('behaviour_up', fr['behaviour_up']), ('behaviour_down', fr['behaviour_down']),
              ('solar', fr['solar']), ('disconnect_reconnect', fr['disconnect_reconnect'])]
    pos = 0
    for name, fraction in blocks:
        count = int(round(n * fraction))
        for i in free[pos:pos + count]:
            honest[i] = name
            start[i] = new_onset()
        pos += count
    return family, honest, start, params


def _apply_true_changes(cfg, rng, p, honest, start, steps, hour):
    """Behavioural honest confounders change TRUE consumption. Meter faults
    are reporting faults and are applied to metered values only."""
    for i in np.flatnonzero(honest != ''):
        a, onset = honest[i], start[i]
        if a == 'vacant':
            p[i, onset:] *= rng.uniform(.02, .22)
        elif a == 'seasonal_occupancy':
            fraction = rng.uniform(.08, .42)
            p[i, onset:] *= fraction
            for j in range(onset, steps, 7 * 96):
                if rng.random() < .4:
                    p[i, j:j + 48] /= max(fraction, 1e-6)
        elif a == 'solar':
            solar = np.maximum(np.sin(np.pi * (hour - 6) / 12), 0)
            p[i, onset:] -= rng.uniform(.4, 2.) * solar[onset:]
        elif a == 'behaviour_up':
            p[i, onset:] += rng.uniform(1.2, 3.2) * ((hour[onset:] >= 22) | (hour[onset:] < 1))
        elif a == 'behaviour_down':
            p[i, onset:] *= rng.uniform(.4, .7)
        elif a == 'disconnect_reconnect':
            for _ in range(int(rng.integers(1, 3))):
                at = int(rng.integers(onset, steps - 96))
                p[i, at:at + int(rng.integers(2, 10)) * 96] = 0
    return np.maximum(p, 0)


def _apply_meter_distortion(cfg, rng, p_true, metered, family, start, steps,
                            hour, phase_weight):
    """Theft families distort the METERED series only. True consumption is
    untouched, so every family except hooking produces a meter-level deficit
    signal and all produce a DT-level deficit."""
    n = p_true.shape[0]
    for i in range(n):
        fam, onset = family[i], start[i]
        if fam == 't1_partial_bypass':
            alpha = rng.uniform(*cfg['t1_alpha_range'])
            metered[i, onset:] *= 1 - alpha
        elif fam == 't2_night_bypass':
            alpha = rng.uniform(*cfg['t2_alpha_range'])
            window = ((hour >= 11) & (hour < 18)) if cfg['t2_daytime_variant'] else ((hour >= 18) | (hour < 2))
            metered[i, window & (np.arange(steps) >= onset)] *= 1 - alpha
        elif fam == 't4_magnetic':
            metered[i, onset:] *= rng.uniform(*cfg['t4_scale_range'])
        elif fam == 't5_neutral_phase':
            factor = rng.uniform(*cfg['t5_phase_scale_range'])
            w = phase_weight[i]
            tampered_share = w.max()
            metered[i, onset:] *= 1 - (1 - factor) * tampered_share
        elif fam == 't6_slow_ramp':
            alpha_end = rng.uniform(*cfg['t6_alpha_end_range'])
            ramp_days = rng.uniform(*cfg['t6_ramp_days_range'])
            frac = 1 - alpha_end * np.clip((np.arange(steps) - onset) / (ramp_days * 96), 0, 1)
            metered[i, onset:] *= frac[onset:]
        elif fam == 't7_zero_flat':
            const = 0 if rng.random() < .5 else float(np.quantile(p_true[i], .1) * .2)
            metered[i, onset:] = np.minimum(p_true[i, onset:], const)
        # t3_hooking: meter untouched; the deficit exists only at the DT.


def _reporting_faults(cfg, rng, metered, honest, start, steps):
    """H3 faulty-meter sub-types: stuck, intermittent zeros, drift, spikes.
    The fault is NOT intentional; consumption continues, the reading lies."""
    for i in np.flatnonzero(np.char.startswith(honest, 'meter_fault')):
        a, onset = honest[i], start[i]
        if a == 'meter_fault_stuck':
            hist = metered[i, max(0, onset - 96 * 7):onset]
            const = float(np.median(hist)) if hist.size and np.isfinite(np.median(hist)) and np.median(hist) > 0 \
                else float(np.quantile(metered[i], .25))
            metered[i, onset:] = np.minimum(metered[i, onset:], max(const, 0))
        elif a == 'meter_fault_intermittent':
            for _ in range(int(rng.integers(2, 6))):
                at = int(rng.integers(onset, steps - 96))
                metered[i, at:at + int(rng.integers(4, 24))] = 0
        elif a == 'meter_fault_drift':
            rate = rng.uniform(*cfg['meter_fault_drift_rate_range']) * (1 if rng.random() < .5 else -1)
            metered[i, onset:] *= np.clip(1 + rate * (np.arange(steps)[onset:] - onset) / (96 * 30.), .05, 1.5)
        elif a == 'meter_fault_spike':
            idx = np.flatnonzero((rng.random(steps) < cfg['meter_fault_spike_rate']) & (np.arange(steps) >= onset))
            metered[i, idx] *= rng.uniform(3., 15., len(idx))


def _hook_draws(cfg, rng, family, start, steps, evening, d, true_dt, outage):
    """T3/T8 hooking: energy taken directly from the line, upstream of the
    meter. Never visible in any meter; only the DT balance sees it."""
    hook_power = np.zeros((d, steps), np.float32)
    active_all = np.arange(steps)[None, :] >= np.maximum(start, 0)[:, None]
    for i in np.flatnonzero(np.isin(family, ['t3_hooking', 't8_cluster'])):
        g = true_dt[i]
        kw = rng.uniform(*cfg['hook_power_kw_range']) * (.4 + evening)
        kw = kw * active_all[i]
        kw[outage[g]] = 0
        hook_power[g] += kw.astype('float32')
    return hook_power


def _place_at_nodes(topo, p_true_meters, phase, phase_late, three_phase,
                    phase_weight, q_ratio, shift_at=None):
    """Like physics.meter_power_to_nodes but three-phase customers are split
    across phases with fixed per-customer weights (assumption). TRUE power is
    placed; the physics layer never sees metered values."""
    n, t = p_true_meters.shape
    p = np.zeros((topo.size, t, 3), np.float64)
    for i, node in enumerate(topo.meter_nodes):
        if three_phase[i]:
            for ph in range(3):
                p[node, :, ph] += p_true_meters[i] * phase_weight[i, ph]
        elif shift_at is not None and phase_late[i] != phase[i]:
            p[node, :shift_at, phase[i]] += p_true_meters[i, :shift_at]
            p[node, shift_at:, phase_late[i]] += p_true_meters[i, shift_at:]
        else:
            p[node, :, phase[i]] += p_true_meters[i]
    q = np.maximum(p, 0) * q_ratio
    return p, q


def _typed_events(cfg, rng, family, honest, start, is_theft, steps, dt_voltage,
                  outage, true_dt, missing, three_phase, tamper_phase):
    """Typed event log (WP-13 Section 6). Event names are ASSUMPTIONS
    pending confirmation against the Indian smart-meter standard and the
    DISCOM meter specification. Some theft produces NO events (stealth);
    honest meters produce frequent noise events (false-alarm source).
    Returns (long-format DataFrame, legacy uint8 flag array)."""
    n = len(family)
    cols = {'meter_idx': [], 'ts': [], 'event_type': [], 'duration_s': [],
            'phase': [], 'severity': []}

    def add(i, t0, etype, duration, phase, severity):
        cols['meter_idx'].append(int(i))
        cols['ts'].append(int(t0))
        cols['event_type'].append(etype)
        cols['duration_s'].append(int(duration))
        cols['phase'].append(int(phase))
        cols['severity'].append(round(float(severity), 3))

    def sprinkle(i, etype, per_day, dur_range, phase=-1, sev=(.3, 1.)):
        hits = (rng.random(steps) < per_day / 96.) & active[i]
        for t0 in np.flatnonzero(hits):
            add(i, t0, etype, rng.uniform(*dur_range), phase, rng.uniform(*sev))

    active = (np.arange(steps)[None, :] >= np.maximum(start, 0)[:, None]) & ~missing
    for i in np.flatnonzero(is_theft):
        fam = family[i]
        ph = tamper_phase[i] if three_phase[i] else -1
        if fam in ('t4_magnetic',) and rng.random() < cfg['event_link_probability']:
            sprinkle(i, 'magnetic_influence', cfg['magnetic_events_per_day'], (60, 3600), ph)
        if fam == 't5_neutral_phase' and rng.random() < cfg['event_link_probability']:
            sprinkle(i, 'neutral_disturbance', cfg['neutral_events_per_day'], (30, 1800), ph)
        if fam in ('t1_partial_bypass', 't2_night_bypass', 't7_zero_flat') \
                and rng.random() < cfg['cover_open_link_probability']:
            sprinkle(i, 'cover_open', cfg['cover_open_events_per_day'], (60, 1800))
    for g in range(outage.shape[0]):
        on = outage[g].astype(int)
        starts = np.flatnonzero(np.diff(np.r_[0, on]) == 1)
        lengths = np.diff(np.r_[np.flatnonzero(np.diff(np.r_[0, on]) == 1), on.size])
        members = np.flatnonzero(true_dt == g)
        for t0, ln in zip(starts, lengths):
            for i in members:
                add(i, t0, 'power_failure', int(ln) * 900, -1, 1.0)
    sag = (dt_voltage < cfg['nominal_voltage_v'] * (1 - cfg['voltage_event_threshold'])) & (dt_voltage > 1)
    swell = dt_voltage > cfg['nominal_voltage_v'] * (1 + cfg['voltage_event_threshold'])
    for g in range(dt_voltage.shape[0]):
        members = np.flatnonzero(true_dt == g)
        for mask in (sag[g], swell[g]):
            edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int)]) == 1)
            for t0 in edges[::max(1, len(edges) // 40)]:
                for i in rng.choice(members, min(3, len(members)), replace=False):
                    add(int(i), int(t0), 'voltage_sag_swell', 0, -1, .8)
    for i in range(n):
        if honest[i] == 'solar' and rng.random() < cfg['reverse_current_link_probability']:
            sprinkle(i, 'reverse_current', cfg['reverse_current_events_per_day'], (600, 7200))
        if rng.random() < cfg['honest_event_meter_fraction']:
            for etype, rate, dur in [('cover_open', cfg['honest_cover_open_per_day'], (300, 1800)),
                                     ('communication_failure', cfg['comm_failure_per_day'], (300, 14400)),
                                     ('magnetic_influence', cfg['honest_magnetic_per_day'], (60, 600))]:
                sprinkle(i, etype, rate, dur)
    for i in np.flatnonzero(missing.any(1)):
        drops = np.flatnonzero(np.diff(np.r_[0, missing[i].astype(int)]) == 1)
        for t0 in drops[:20]:
            if rng.random() < cfg['comm_failure_on_dropout_probability']:
                add(int(i), int(t0), 'communication_failure', 900, -1, .5)
    events = pd.DataFrame(cols)
    legacy = np.zeros((n, steps), np.uint8)
    theft_idx = np.flatnonzero(is_theft)
    honest_idx = np.flatnonzero(~is_theft)
    for idx, rate in ((theft_idx, cfg['event_sensitivity_per_day']),
                      (honest_idx, cfg['event_false_positive_per_day'])):
        if len(idx):
            draw = (rng.random((len(idx), steps)) < rate / 96.) & active[idx]
            legacy[idx] |= (1 * draw).astype(np.uint8)
    return events, legacy


def simulate_v2(cfg, seed, clean=False):
    """Returns (ObservationsV2, HiddenLabels) compatible with the inherited
    feature/evaluation stack."""
    rng = np.random.default_rng(seed)
    d, m, days = cfg['n_dts'], cfg['meters_per_dt'], cfg['days']
    n, steps = d * m, days * 96
    base = cfg['baseline_days'] * 96
    dt_hours = cfg['interval_minutes'] / 60.
    if cfg['interval_minutes'] != 15:
        raise ValueError('This generator implements quarter-hour intervals only.')
    split = dt_splits(d, rng)
    true_dt = np.repeat(np.arange(d), m)
    # --- classes, phases ---
    class_mix = np.array(cfg['class_mix'], dtype=float)
    category = rng.choice(['domestic', 'small_commercial', 'agricultural'],
                          n, p=class_mix / class_mix.sum())
    tp_prob = np.array([cfg['three_phase_prob_by_class'][c] for c in category])
    three_phase = rng.random(n) < tp_prob
    phase = rng.integers(0, 3, n)
    late_phase = phase.copy()
    phase_changes = rng.random(n) < cfg['phase_shift_fraction']
    late_phase[phase_changes] = (phase[phase_changes] + 1) % 3
    phase_weight = np.ones((n, 3))
    if three_phase.any():
        phase_weight[three_phase] = rng.dirichlet([8, 8, 8], int(three_phase.sum()))
    declared_phase = np.where(three_phase, np.argmax(phase_weight, axis=1), phase)
    hour = (np.arange(steps) % 96) / 4
    day = np.arange(steps) // 96
    # --- true load, assignments, honest true-side changes ---
    p, level, temperature, evening = _load_model(cfg, rng, n, steps, days, hour, day, category)
    family = np.full(n, '', dtype='<U24')
    honest = np.full(n, '', dtype='<U24')
    start = np.full(n, -1, dtype=int)
    params = [{} for _ in range(n)]
    if not clean:
        family, honest, start, params = _assign(cfg, rng, d, m, split, true_dt, n,
                                                steps, base, hour, three_phase)
        p = _apply_true_changes(cfg, rng, p, honest, start, steps, hour)
    is_theft = family != ''
    # --- outages zero TRUE load before any meter copy (matches inherited
    # semantics: nothing flows, nothing is reported, no phantom deficit) ---
    outage = np.zeros((d, steps), bool)
    if not clean:
        for g in range(d):
            for day_id in range(days):
                if day_id % cfg['planned_outage_days'] == g % cfg['planned_outage_days']:
                    outage[g, day_id * 96 + 44:day_id * 96 + 48] = True
                if rng.random() < cfg['unplanned_outage_probability_per_day']:
                    at = day_id * 96 + int(rng.integers(0, 90))
                    outage[g, at:at + int(rng.integers(2, 10))] = True
        p[outage[true_dt]] = 0
    # --- reported (meter) series ---
    metered = p.copy()
    if not clean:
        _apply_meter_distortion(cfg, rng, p, metered, family, start, steps, hour, phase_weight)
        _reporting_faults(cfg, rng, metered, honest, start, steps)
        for i in np.flatnonzero(is_theft):
            if rng.random() < .15:
                start[i] = 0  # Persistent theft defeats own-history-only expectations.
    visible_meter = np.maximum(p - metered, 0)
    theft_kwh = visible_meter * is_theft[:, None] * dt_hours
    visible = visible_meter.reshape(d, m, steps).sum(1) * dt_hours
    hook_power = np.zeros((d, steps), np.float32)
    if not clean:
        hook_power = _hook_draws(cfg, rng, family, start, steps, evening, d, true_dt, outage)
    invisible = hook_power * dt_hours
    # --- physics on TRUE load ---
    technical = np.zeros((d, steps), np.float32)
    dt_input = np.zeros_like(technical)
    dt_q = np.zeros_like(technical)
    v = np.zeros((n, steps), np.float32)
    dt_v = np.zeros((d, steps), np.float32)
    currents = np.zeros((d, steps, 3), np.float32)
    topologies = []
    q_ratio = np.tan(np.arccos(cfg['power_factor']))
    lr_count = int(round(d * cfg['long_radial_dt_fraction']))
    lr_dts = set(rng.choice(d, lr_count, replace=False).tolist()) if lr_count else set()
    shift_at = base + (steps - base) // 2
    for g in range(d):
        topo = make_topology(m, rng, 'long_radial' if g in lr_dts else cfg['topology_source'])
        topologies.append(topo)
        ids = slice(g * m, (g + 1) * m)
        pp, qq = _place_at_nodes(topo, p[ids], phase[ids], late_phase[ids], three_phase[ids],
                                 phase_weight[ids], q_ratio, shift_at)
        # Hooking is a pole load, never assigned a consumer meter label.
        hook_node = max(1, topo.pole_count // 2)
        pp[hook_node, :, g % 3] += hook_power[g]
        qq[hook_node, :, g % 3] += hook_power[g] * q_ratio
        root_v = np.full(steps, cfg['nominal_voltage_v'], np.float64)
        if not clean:
            root_v += 1.5 * np.sin(np.arange(steps) / 100) + rng.normal(0, .5, steps)
        root_v[outage[g]] = 0
        vv, line_loss, ii = sweep(topo, pp, qq, root_v, cfg['solver_iterations'])
        load_p = pp.sum((0, 2))
        load_q = qq.sum((0, 2))
        dt_loss = transformer_loss(load_p, load_q, cfg) * ~outage[g]
        technical[g] = (line_loss + dt_loss) * dt_hours
        dt_input[g] = load_p * dt_hours + technical[g]
        dt_q[g] = load_q * dt_hours
        currents[g] = np.abs(ii)
        dt_v[g] = root_v
        for j in range(m):
            gi = g * m + j
            node = topo.meter_nodes[j]
            if three_phase[gi]:
                v[gi] = vv[node, :, declared_phase[gi]]
            else:
                v[gi, :shift_at] = vv[node, :shift_at, phase[gi]]
                v[gi, shift_at:] = vv[node, shift_at:, late_phase[gi]]
    # Physics conservation (WP-13 test 1): DT input = sum(true consumption)
    # + technical loss + hooking theft, exactly, before reporting noise.
    balance_error = float(np.max(np.abs(
        dt_input - (p.reshape(d, m, steps).sum(1) * dt_hours + technical + invisible))))
    # --- reporting side: noise, dt errors, events, data problems ---
    kwh = metered * dt_hours
    kvarh = np.maximum(metered, 0) * q_ratio * dt_hours
    missing = np.zeros((n, steps), bool)
    events = np.zeros((n, steps), np.uint8)
    events_typed = pd.DataFrame({c: [] for c in
                                 ['meter_idx', 'ts', 'event_type', 'duration_s', 'phase', 'severity']})
    if not clean:
        kwh *= (1 + rng.normal(0, cfg['energy_noise_fraction'], (n, 1))).astype('float32')
        kvarh *= (1 + rng.normal(0, cfg['energy_noise_fraction'], (n, 1))).astype('float32')
        v += rng.normal(0, cfg['voltage_noise_v'], v.shape).astype('float32')
        resolution = cfg['meter_voltage_resolution_v']
        v = np.round(v / resolution) * resolution
        v[outage[true_dt]] = 0
        for g in range(d):
            offset = rng.uniform(-cfg['dt_skew_minutes'], cfg['dt_skew_minutes']) / cfg['interval_minutes']
            dt_input[g] = np.interp(np.arange(steps) + offset, np.arange(steps), dt_input[g])
            dt_input[g] *= 1 + rng.uniform(-cfg['dt_accuracy_fraction'], cfg['dt_accuracy_fraction'])
        dt_v += rng.normal(0, cfg['dt_voltage_noise_v'], dt_v.shape)
        dt_v[outage] = 0
        missing = rng.random((n, steps)) < cfg['dropout']
        if cfg['dropout'] > 0:
            for i in np.flatnonzero(rng.random(n) < cfg['burst_dropout_fraction']):
                at = int(rng.integers(base, steps - 96))
                missing[i, at:at + int(rng.integers(24, 3 * 96))] = True
        # Negative / spike glitches at a low rate, left in the reported data
        # to test cleaning (WP-13 Section 7).
        glitch = rng.random((n, steps)) < cfg['glitch_rate']
        kwh[glitch] *= rng.choice([-1., 10.], int(glitch.sum()))
        kvarh[glitch] *= rng.choice([-1., 10.], int(glitch.sum()))
        if cfg['events']:
            events_typed, events = _typed_events(cfg, rng, family, honest, start, is_theft,
                                                 steps, dt_v, outage, true_dt, missing,
                                                 three_phase, declared_phase)
    # --- per-meter clock drift (reported side, sub-interval interpolation) ---
    if not clean and cfg['per_meter_clock_drift_minutes'] > 0:
        for i in range(n):
            off = rng.uniform(-cfg['per_meter_clock_drift_minutes'],
                              cfg['per_meter_clock_drift_minutes']) / cfg['interval_minutes']
            if abs(off) > 1e-6:
                kwh[i] = np.interp(np.arange(steps) + off, np.arange(steps), kwh[i])
                v[i] = np.interp(np.arange(steps) + off, np.arange(steps), v[i])
    kwh[missing] = np.nan
    kvarh[missing] = np.nan
    v[missing] = np.nan
    events[missing] = 0
    # --- mapping errors: whole-record cross-DT swaps within a partition ---
    observed_order = np.arange(n)
    if not clean:
        for s in ['train', 'val', 'test']:
            ids = np.flatnonzero(np.isin(true_dt, np.flatnonzero(split == s)))
            chosen = rng.choice(ids, int(round(cfg['mapping_error'] * len(ids))), replace=False)
            if len(chosen) > 1:
                observed_order[chosen] = np.roll(chosen, 1)
    orig = observed_order
    mapped_dt = np.repeat(np.arange(d), m)
    area_dt = rng.choice(['urban', 'semi_urban', 'rural'], d, p=np.array(cfg['area_mix']) / np.sum(cfg['area_mix']))
    meters = pd.DataFrame({
        'meter_id': [f'M{i:05d}' for i in orig],
        'dt_id': mapped_dt, 'mapped_dt_id': mapped_dt,
        'feeder_id': [f'DT{g:03d}-LV' for g in mapped_dt],
        'phase': declared_phase[orig],
        'three_phase': three_phase[orig],
        'category': category[orig],
        'sanctioned_load_kw': np.ceil(np.maximum(level[orig] * 3, 1)),
        'lat': 17.38 + mapped_dt * .0003 + rng.normal(0, .0001, n),
        'lon': 78.48 + np.tile(np.arange(m), d) * .00006,
        'has_event_log': bool(cfg['events']),
        'area_type': area_dt[true_dt[orig]],
    })
    true_daily_kwh = (p.reshape(n, days, 96).sum(2) * dt_hours)
    labels_meter = pd.DataFrame({
        'meter_id': meters.meter_id,
        'family': family[orig],
        'honest_archetype': honest[orig],
        'is_theft': is_theft[orig],
        # WP-13 T3 rule: hooking customers are thieves in ground truth but
        # are NOT detectable meter-level positives; their signal is DT-level.
        'meter_level_positive': is_theft[orig] & (family[orig] != 't3_hooking'),
        'honest': ~is_theft[orig],
        'event_start': start[orig],
        'theft_start': start[orig],
        'theft_params': [json.dumps(params[i]) for i in orig],
        'three_phase': three_phase[orig],
        'tamper_phase': declared_phase[orig],
        'true_daily_kwh': [true_daily_kwh[i].round(4) for i in orig],
        'physical_dt_id': true_dt[orig],
    })
    obs = ObservationsV2(
        meters=meters, kwh=kwh[orig], kvarh=kvarh[orig], voltage=v[orig],
        events=events[orig], missing=missing[orig], dt_kwh=dt_input, dt_kvarh=dt_q,
        dt_voltage=dt_v, dt_currents=currents, temperature=temperature,
        topologies=topologies, split=split, cfg=dict(cfg), seed=seed,
        typed_events=events_typed, pack=str(cfg.get('pack', '')))
    hidden = HiddenLabels(labels_meter=labels_meter, ntl=visible + invisible,
                          technical=technical, visible=visible, invisible=invisible,
                          theft_kwh=theft_kwh[orig], true_dt=true_dt[orig],
                          clean_balance_error=balance_error)
    return obs, hidden


def export_pack(obs, hidden, destination):
    """WP-13 Section 12 outputs: meters/dt_meter/events/topology.json plus a
    SEPARATE ground truth (labels, families, true consumption). Ground truth
    is evaluation-only; the feature pipeline never reads it."""
    from src.common import config_hash, jsonable
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    d, t = obs.dt_kwh.shape
    n = obs.kwh.shape[0]
    m = n // d
    ts = pd.date_range('2026-06-01', periods=t, freq='15min', tz='Asia/Kolkata')
    obs.meters.to_parquet(dest / 'meters.parquet', index=False)
    pd.DataFrame({'dt_id': np.repeat(np.arange(d), t), 'ts': np.tile(np.arange(t), d),
                  'kwh_in': obs.dt_kwh.ravel(), 'kvarh_in': obs.dt_kvarh.ravel(),
                  'v_lv': obs.dt_voltage.ravel(),
                  'i_a': obs.dt_currents[:, :, 0].ravel(),
                  'i_b': obs.dt_currents[:, :, 1].ravel(),
                  'i_c': obs.dt_currents[:, :, 2].ravel()}
                 ).to_parquet(dest / 'dt_meter.parquet', index=False, compression='zstd')
    ev = obs.typed_events
    if ev is not None and len(ev):
        out = ev.copy()
        out['meter_id'] = obs.meters.meter_id.to_numpy()[out['meter_idx'].to_numpy()]
        out = out[['meter_id', 'ts', 'event_type', 'duration_s', 'phase', 'severity']]
    else:
        out = pd.DataFrame(columns=['meter_id', 'ts', 'event_type', 'duration_s', 'phase', 'severity'])
    out.to_parquet(dest / 'events.parquet', index=False, compression='zstd')
    edges = []
    for g, topo in enumerate(obs.topologies):
        for node in range(1, topo.size):
            edges.append({'dt_id': g, 'from_node': f'DT{g:03d}-N{topo.parent[node]:03d}',
                          'to_node': f'DT{g:03d}-N{node:03d}', 'r_ohm': float(topo.r[node]),
                          'x_ohm': float(topo.x[node]), 'length_m': float(topo.length_m[node])})
    (dest / 'topology.json').write_text(json.dumps(
        {'dts': d, 'meters_per_dt': m, 'topology_sources':
            sorted({t_.source for t_ in obs.topologies}), 'edges': edges}, indent=1), encoding='utf-8')
    hidden.labels_meter.to_parquet(dest / 'ground_truth.parquet', index=False)
    pd.DataFrame({'dt_id': np.repeat(np.arange(d), t), 'ts': np.tile(np.arange(t), d),
                  'ntl_kwh': hidden.ntl.ravel(), 'tech_loss_kwh': hidden.technical.ravel(),
                  'meter_visible_ntl_kwh': hidden.visible.ravel(),
                  'invisible_ntl_kwh': hidden.invisible.ravel()}
                 ).to_parquet(dest / 'ground_truth_dt.parquet', index=False, compression='zstd')
    meta = {'profile_source': 'SYNTHETIC', 'generator': 'synth_v2 (WP-13)',
            'pack': obs.pack, 'seed': obs.seed, 'config_hash': config_hash(obs.cfg),
            'n_meters': n, 'n_dts': d, 'days': t // 96,
            'achieved_meter_prevalence': float(hidden.labels_meter.is_theft.mean()),
            'schema': 'kWh per 15-min interval; dt_meter ts is an interval index into metadata timestamps',
            'label_access': 'evaluation only; never visible to the feature pipeline',
            'note': 'SYNTHETIC demonstration data; every parameter is an assumption, '
                    'not an Indian measurement; not calibrated to any real dataset'}
    (dest / 'metadata.json').write_text(json.dumps(jsonable(meta), indent=2), encoding='utf-8')
    return meta
