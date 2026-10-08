"""WP-13 Section 12 tests. All must pass BEFORE any model evaluation runs
on synth_v2 data (generator freeze discipline)."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.common import load_config, hash_file
from src.sim.physics import make_topology
from src.sim.simulator import simulate
from src.sim.synth_v2 import simulate_v2, export_pack, FAMILIES, HONEST_ARCHETYPES, EVENT_TYPES
from src.features.build import build_features, feature_contract

ROOT = Path(__file__).parents[1]
CFG_PATH = ROOT / 'configs' / 'synth_v2.yaml'


def small_cfg(**over):
    cfg = load_config(CFG_PATH)
    cfg.update(n_dts=6, meters_per_dt=10, days=40)
    cfg.update(over)
    return cfg


@pytest.fixture(scope='module')
def sim():
    return small_cfg(), simulate_v2(small_cfg(), 410001)


def test_energy_conservation(sim):
    cfg, (obs, hidden) = sim
    # Physics identity (WP-13 test 1): per DT and interval, DT input =
    # sum(true consumption) + technical loss + hooking theft. Closes exactly
    # before reporting noise, hooking included.
    assert hidden.clean_balance_error < 1e-5
    d = cfg['n_dts']
    # Reporting side, DATA PROBLEMS OFF (dropout/glitches corrupt the balance
    # by design -- that is WP-13 Section 7 working as intended). Clock
    # misalignment between meters and the DT meter remains, so the
    # per-interval reported identity holds only to skew tolerance; the daily
    # identity closes tightly.
    cfg_rep = small_cfg(dropout=0.0, glitch_rate=0.0)
    obs_r, hid_r = simulate_v2(cfg_rep, 410002)
    metered_sum = np.nan_to_num(obs_r.kwh, nan=0.).reshape(d, cfg['meters_per_dt'], -1).sum(1)
    total = metered_sum + hid_r.visible + hid_r.invisible + hid_r.technical
    frac = np.abs(obs_r.dt_kwh - total) / np.maximum(obs_r.dt_kwh, 1.)
    live = obs_r.dt_voltage > 50
    live[:, 1:-1] &= live[:, :-2] & live[:, 2:]
    frac_live = frac[live]
    assert frac_live.mean() < 0.15, 'gross per-interval imbalance'
    daily = np.abs((obs_r.dt_kwh - total).sum(1)) / np.maximum(obs_r.dt_kwh.sum(1), 1.)
    assert daily.max() < 0.015
    # With data problems ON the reported balance genuinely degrades (the
    # false-evidence source the pipeline must handle causally).
    metered_bad = np.nan_to_num(obs.kwh, nan=0.).reshape(d, cfg['meters_per_dt'], -1).sum(1)
    total_bad = metered_bad + hidden.visible + hidden.invisible + hidden.technical
    daily_bad = np.abs((obs.dt_kwh - total_bad).sum(1)) / np.maximum(obs.dt_kwh.sum(1), 1.)
    assert daily_bad.max() > daily.max()


def test_ground_truth_never_reaches_features(sim):
    from copy import deepcopy
    cfg, (obs, hidden) = sim
    before = build_features(obs).full
    changed = deepcopy(hidden)
    changed.labels_meter['is_theft'] = ~changed.labels_meter.is_theft
    changed.labels_meter['meter_level_positive'] = ~changed.labels_meter.meter_level_positive
    changed.labels_meter['true_daily_kwh'] = [np.zeros(40) for _ in range(len(changed.labels_meter))]
    changed.technical[:] = 1e9
    changed.ntl[:] = 1e9
    changed.true_dt[:] = 0
    after = build_features(obs).full
    np.testing.assert_array_equal(before, after)
    assert not hasattr(obs, 'labels_meter') and not hasattr(obs, 'hidden')
    with pytest.raises(ValueError):
        feature_contract(['safe', 'true_daily_kwh'])
    with pytest.raises(ValueError):
        feature_contract(['family_weight_rank'])


def test_t3_hooking_rule():
    cfg = small_cfg(family_weights=[0., 0., 1., 0., 0., 0., 0., 0.],
                    per_dt_prevalence_range=[0.15, 0.15], zero_theft_dt_fraction=0.2)
    obs, hidden = simulate_v2(cfg, 410003)
    lm = hidden.labels_meter
    t3 = lm.family == 't3_hooking'
    assert t3.any(), 'fixture must contain hooking customers'
    # Hooking never shows in any meter: no theft kWh, not a meter-level positive.
    assert not lm.loc[t3, 'meter_level_positive'].any()
    np.testing.assert_array_equal(hidden.theft_kwh[lm.family.to_numpy() == 't3_hooking'], 0)
    # The DT-level signal exists: DTs with hooked customers carry invisible NTL.
    hooked_dts = lm.loc[t3, 'physical_dt_id'].unique()
    assert all(hidden.invisible[g].sum() > 0 for g in hooked_dts)


def test_determinism(tmp_path):
    cfg = small_cfg()
    a, ha = simulate_v2(cfg, 14)
    b, hb = simulate_v2(cfg, 14)
    np.testing.assert_array_equal(a.kwh, b.kwh)
    np.testing.assert_array_equal(ha.ntl, hb.ntl)
    pd.testing.assert_frame_equal(ha.labels_meter, hb.labels_meter)
    hashes = []
    for run in (1, 2):
        dest = tmp_path / f'run{run}'
        meta = export_pack(a, ha, dest)
        assert meta['profile_source'] == 'SYNTHETIC'
        hashes.append({f.name: hash_file(f) for f in sorted(dest.iterdir())})
    assert hashes[0] == hashes[1], 'export must be byte-identical for the same seed'
    assert json.loads((tmp_path / 'run1' / 'metadata.json').read_text())['config_hash']


def test_honest_labels(sim):
    cfg, (obs, hidden) = sim
    lm = hidden.labels_meter
    honest_rows = lm.honest_archetype != ''
    assert honest_rows.any()
    assert lm.loc[honest_rows, 'honest'].all()
    assert not lm.loc[honest_rows, 'is_theft'].any()
    assert not lm.loc[honest_rows, 'meter_level_positive'].any()
    assert set(lm.honest_archetype[honest_rows]) <= set(HONEST_ARCHETYPES)
    assert set(lm.family[lm.is_theft]) <= set(FAMILIES)
    assert (lm.family[~lm.is_theft] == '').all()
    # t5 exists only on three-phase meters.
    t5 = lm.family == 't5_neutral_phase'
    if t5.any():
        assert lm.loc[t5, 'three_phase'].all()
    # Typed events use only declared event names.
    ev = obs.typed_events
    assert len(ev) > 0
    assert set(ev.event_type) <= set(EVENT_TYPES)
    assert {'meter_idx', 'ts', 'event_type', 'duration_s', 'phase', 'severity'} <= set(ev.columns)


def test_backward_compat():
    # Pin the inherited generator and entrypoint against drift. These are
    # the WP-0-patched working copies (Windows portability fixes recorded in
    # docs/HANDOFF_LOG.md) -- the exact code that produced the reproduced
    # results in results/reproduction_check.md. The original-zip manifest
    # hashes differ only by those patches; pristine originals live in
    # inherited_pristine/ and are verified separately.
    pinned = {
        'src/sim/simulator.py': '51e81ab713712ce65738aeb41b901b0962b9dbf390a74ccda306d653b9f6b0ac',
        'src/common.py': 'e04f188853ebf4521ce8db76ce46b025bdf54464263c783c396dd7095c04f470',
        # Deploy-layout api.py: inherited file + CORS middleware + /health
        # models list (both deltas documented in docs/HANDOFF_LOG.md).
        'app/api.py': 'f99f22598b4b92af4cfbea3627a131b91eb9c6bf2363e97423a1a1d674613181',
        'app/scoring.py': 'bf4a5f86f9d30c1da3e1f622f24025e120d34bf485cee81df8e8e76036ef3d42',
    }
    for rel, want in pinned.items():
        assert hash_file(ROOT / rel) == want, f'{rel} drifted from the WP-0-pinned version'
    # physics.py (additive long_radial variant) and features/build.py
    # (additive FORBIDDEN ground-truth names, WP-13 Section 1.3) carry two
    # documented additive deviations; both are covered by the tests here.
    # Pin the inherited topology path (cigre) against silent drift.
    topo = make_topology(12, np.random.default_rng(123))
    pin = (hashlib.sha256(topo.parent.tobytes()).hexdigest()[:16] + '|' +
           hashlib.sha256(np.round(topo.r, 6).tobytes()).hexdigest()[:16])
    assert pin == 'd6c8c62287e3b828|da9b1bdd2aa26f2e'
    # The inherited generator still runs and deterministically reproduces.
    cfg = load_config(ROOT / 'configs' / 'fast.yaml')
    cfg.update(n_dts=12, meters_per_dt=8)
    o1, h1 = simulate(cfg, 31)
    o2, h2 = simulate(cfg, 31)
    np.testing.assert_array_equal(o1.kwh, o2.kwh)
    np.testing.assert_array_equal(h1.ntl, h2.ntl)
