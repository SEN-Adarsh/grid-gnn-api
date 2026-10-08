"""Phase-decoupled radial backward/forward sweep, in volts, kW and ohms.

Vectorised over timesteps. The neutral is ideal and mutual impedances are
omitted. Balanced line-only equivalence is validated against runpp; this is
NOT a validation of a coupled four-wire unbalanced Indian feeder.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np


@dataclass
class Topology:
    parent: np.ndarray
    r: np.ndarray
    x: np.ndarray
    length_m: np.ndarray
    pole_count: int
    meter_nodes: np.ndarray
    source: str

    @property
    def size(self):
        return len(self.parent)


@lru_cache(maxsize=1)
def cigre_template():
    import pandapower.networks as pn
    return pn.create_cigre_network_lv()


def _long_radial_topology(n_meters, rng):
    """Assumption-driven long radial LT feeder: one trunk of many poles with
    long spans, meters clustered toward the far end, higher technical loss.
    Reflects the general structure of low-tension feeders, not a measured
    Indian feeder. No pandapower reference exists for this variant."""
    n_poles = int(rng.integers(24, 34))
    parents, r, x, lengths = [-1], [0.], [0.], [0.]
    for i in range(1, n_poles + 1):
        parents.append(i - 1)
        span = rng.uniform(0.025, 0.045)  # km, trunk span (assumption)
        r.append(rng.uniform(0.45, 0.90) * span)   # Ω/km LV conductor (assumption)
        x.append(rng.uniform(0.07, 0.11) * span)
        lengths.append(span * 1000)
    meter_nodes = np.arange(n_poles + 1, n_poles + 1 + n_meters)
    drop_r, drop_x = 1.83, 0.083  # Ω/km, NAYY 4x50 SE service drop
    for _ in range(n_meters):
        # Bias service connections toward the far half of the trunk.
        pole = int(rng.integers(max(1, n_poles // 2), n_poles + 1))
        parents.append(pole)
        l = rng.uniform(0.008, 0.030)
        r.append(drop_r * l)
        x.append(drop_x * l)
        lengths.append(l * 1000)
    return Topology(np.array(parents), np.array(r), np.array(x),
                    np.array(lengths), n_poles, meter_nodes, 'long_radial')


def make_topology(n_meters, rng, source='cigre_residential'):
    if source == 'long_radial':
        return _long_radial_topology(n_meters, rng)
    net = cigre_template()
    root = 24 if source == 'cigre_commercial' else 2
    links = {}
    for row in net.line.itertuples():
        if net.bus.at[row.from_bus, 'vn_kv'] > 1:
            continue
        for a, b in [(row.from_bus, row.to_bus), (row.to_bus, row.from_bus)]:
            links.setdefault(a, []).append((b, row))
    order, parents, rows = [root], [-1], [None]
    seen = {root}
    for bus in order:
        for other, row in links.get(bus, []):
            if other not in seen:
                seen.add(other)
                parents.append(order.index(bus))
                rows.append(row)
                order.append(other)
    n_poles = len(order) - 1
    r, x, lengths = [0.], [0.], [0.]
    stretch = rng.uniform(0.8, 1.2)
    for row in rows[1:]:
        l = row.length_km * stretch
        r.append(row.r_ohm_per_km * l)
        x.append(row.x_ohm_per_km * l)
        lengths.append(l * 1000)
    meter_nodes = np.arange(len(order), len(order) + n_meters)
    # Service impedance from pandapower's NAYY 4x50 SE standard type.
    drop_type = net.std_types['line']['NAYY 4x50 SE']
    for _ in range(n_meters):
        parents.append(int(rng.integers(1, n_poles + 1)))
        l = rng.uniform(0.008, 0.025)
        r.append(drop_type['r_ohm_per_km'] * l)
        x.append(drop_type['x_ohm_per_km'] * l)
        lengths.append(l * 1000)
    return Topology(np.array(parents), np.array(r), np.array(x),
                    np.array(lengths), n_poles, meter_nodes, source)


def sweep(topo, p_kw, q_kvar, slack_v=230., iterations=6):
    """p,q have shape [nodes, times, phases]; slack_v scalar or [times]."""
    p = np.asarray(p_kw, dtype=np.float64)
    q = np.asarray(q_kvar, dtype=np.float64)
    s = (p + 1j * q) * 1000.
    slack = np.broadcast_to(np.asarray(slack_v), (p.shape[1],))[:, None]
    v = np.broadcast_to(slack, p.shape).astype(np.complex128).copy()
    z = topo.r + 1j * topo.x
    energized = slack > 1
    for _ in range(iterations):
        denom = np.where(np.abs(v) > 50, v, 230.+0j)
        branch = np.conj(s / denom) * energized[None]
        for node in range(topo.size - 1, 0, -1):
            branch[topo.parent[node]] += branch[node]
        v[0] = slack
        for node in range(1, topo.size):
            v[node] = v[topo.parent[node]] - z[node] * branch[node]
    loss_kw = (np.abs(branch[1:]) ** 2 * topo.r[1:, None, None]).sum(axis=(0, 2)) / 1000.
    return np.abs(v), loss_kw, branch[0]


def meter_power_to_nodes(topo, p_kw, phase, q_ratio, phase_late=None, shift_at=None):
    n, t = p_kw.shape
    p = np.zeros((topo.size, t, 3), np.float64)
    for i, node in enumerate(topo.meter_nodes):
        if phase_late is not None and shift_at is not None:
            p[node, :shift_at, phase[i]] = p_kw[i, :shift_at]
            p[node, shift_at:, phase_late[i]] = p_kw[i, shift_at:]
        else:
            p[node, :, phase[i]] = p_kw[i]
    # Solar reactive output is zero; positive load has the configured PF.
    q = np.maximum(p, 0) * q_ratio
    return p, q


def transformer_loss(p_kw, q_kvar, cfg):
    apparent = np.sqrt(p_kw ** 2 + q_kvar ** 2)
    return cfg['dt_no_load_kw'] + cfg['dt_full_load_loss_kw'] * (apparent / cfg['dt_rating_kva']) ** 2


def validate_against_pandapower(cfg):
    import pandapower as pp
    rng = np.random.default_rng(9173)
    meter_count = cfg['meters_per_dt']
    topo = make_topology(meter_count, rng)
    n = cfg['loadflow_samples']
    # Balanced loads span normal loading, peak loading, and reverse flow.
    p = np.zeros((topo.size, n, 3))
    draws = rng.uniform(-0.1, 1.5, (meter_count, n))
    p[topo.meter_nodes] = draws[..., None] / 3
    q = np.maximum(p, 0) * np.tan(np.arccos(cfg['power_factor']))
    v, losses, _ = sweep(topo, p, q, cfg['nominal_voltage_v'], cfg['solver_iterations'])
    net = pp.create_empty_network()
    nominal_kv = cfg['nominal_voltage_v'] * np.sqrt(3) / 1000
    buses = [pp.create_bus(net, nominal_kv) for _ in range(topo.size)]
    pp.create_ext_grid(net, buses[0], vm_pu=1.)
    for i in range(1, topo.size):
        pp.create_line_from_parameters(net, buses[topo.parent[i]], buses[i], 1.,
                                      topo.r[i], topo.x[i], 0., 1.)
    loads = [pp.create_load(net, buses[node], 0.) for node in topo.meter_nodes]
    errs, loss_errors, abs_loss_errors = [], [], []
    for t in range(n):
        net.load.loc[loads, 'p_mw'] = p[topo.meter_nodes, t].sum(1) / 1000
        net.load.loc[loads, 'q_mvar'] = q[topo.meter_nodes, t].sum(1) / 1000
        pp.runpp(net, numba=False, tolerance_mva=1e-10)
        expected = net.res_bus.vm_pu.to_numpy() * cfg['nominal_voltage_v']
        errs.extend(np.abs(expected - v[:, t, 0]).tolist())
        ref = float(net.res_line.pl_mw.sum() * 1000)
        abs_loss_errors.append(abs(ref - losses[t]))
        loss_errors.append(abs(ref - losses[t]) / max(abs(ref), 1e-9))
    out = dict(status='passed', samples=n, reference='pandapower.runpp',
               pandapower_version=pp.__version__, voltage_mean_error_v=np.mean(errs),
               voltage_max_error_v=np.max(errs), loss_mean_relative_error=np.mean(loss_errors),
               loss_max_relative_error=np.max(loss_errors), loss_max_absolute_error_kw=max(abs_loss_errors),
               voltage_tolerance_v=cfg['voltage_tolerance_v'], loss_relative_tolerance=cfg['loss_relative_tolerance'],
               scope='balanced, line-only, zero shunt, fixed LV slack; no neutral or transformer validation')
    if out['voltage_max_error_v'] > cfg['voltage_tolerance_v'] or out['loss_max_relative_error'] > cfg['loss_relative_tolerance']:
        out['status'] = 'failed'
    return out
