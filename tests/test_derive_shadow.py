import numpy as np
import pandas as pd
import pytest

from src.data.derive_shadow import (ASSESS_END, ASSESS_START, HOOK_FACTOR,
                                    flip_truth_frame, hook_dt_frame,
                                    inject_readings_frame, select_eligible)


def _reads_frame(meter_id: str, days: int = 14) -> pd.DataFrame:
    ts = pd.date_range('2025-10-01', periods=days * 96, freq='15min', tz='Asia/Kolkata')
    return pd.DataFrame({'meter_id': [meter_id] * len(ts), 'ts': ts,
                         'kwh': np.linspace(0.5, 1.5, len(ts)),
                         'kvarh': 0.0, 'voltage_v': 230.0, 'is_missing': False})


def test_inject_only_assessment_days_and_selected_meter():
    clean = _reads_frame('M000001')
    dirty = _reads_frame('M000002')
    reads = pd.concat([clean, dirty], ignore_index=True)
    out = inject_readings_frame(reads, {'M000002': 5})          # T5 = flat
    a, b = ASSESS_START, ASSESS_END
    sel = (out.meter_id == 'M000002')
    assess = sel & (out.ts >= a) & (out.ts < b)
    base = sel & (out.ts < a)
    vals = out.loc[assess, 'kwh'].to_numpy().reshape(7, 96)
    assert np.allclose(vals.std(axis=1), 0.0)                   # flat per day
    assert np.allclose(out.loc[base, 'kwh'].to_numpy(),
                       reads.loc[base, 'kwh'].to_numpy())       # baseline untouched
    assert np.allclose(out.loc[~sel, 'kwh'].to_numpy(),
                       reads.loc[~sel, 'kwh'].to_numpy())       # other meters untouched
    assert len(out.loc[assess]) == 7 * 96


def test_inject_preserves_nan_slots():
    reads = _reads_frame('M000003')
    reads.loc[reads.ts == ASSESS_START + pd.Timedelta(minutes=15), 'kwh'] = np.nan
    out = inject_readings_frame(reads, {'M000003': 1})
    row = out.loc[out.kwh.isna()]
    assert len(row) == 1 and (row.ts == ASSESS_START + pd.Timedelta(minutes=15)).all()


def test_flip_truth_marks_only_selected():
    truth = pd.DataFrame({'meter_id': ['M000001', 'M000002', 'M000003'],
                          'is_theft': [False, False, True],
                          'archetype': ['none', 'none', 'tamper_assisted'],
                          'start_date': pd.to_datetime(
                              ['NaT', 'NaT', '2025-10-06 06:45'], utc=True),
                          'intensity': [0.0, 0.0, 0.49]})
    out = flip_truth_frame(truth, {'M000001': 3})
    r = out.set_index('meter_id')
    assert bool(r.loc['M000001', 'is_theft']) is True
    assert r.loc['M000001', 'archetype'] == 'zidi_t3_injected'
    assert r.loc['M000001', 'start_date'] == ASSESS_START
    assert r.loc['M000002', 'archetype'] == 'none' and not bool(r.loc['M000002', 'is_theft'])
    assert r.loc['M000003', 'archetype'] == 'tamper_assisted'


def test_hook_dt_frame_scales_assessment_only():
    ts = pd.date_range('2025-10-01', periods=14 * 96, freq='15min', tz='Asia/Kolkata')
    dtr = pd.DataFrame({'dt_id': ['DT0000', 'DT0001'], 'ts': [ts[0], ts[0]],
                        'kwh_in': [10.0, 10.0], 'kvarh_in': [4.0, 4.0]})
    dtr = pd.concat([dtr, pd.DataFrame({'dt_id': ['DT0000'], 'ts': [ASSESS_START],
                                        'kwh_in': [10.0], 'kvarh_in': [4.0]})],
                    ignore_index=True)
    out = hook_dt_frame(dtr, [0], factor=HOOK_FACTOR)           # local dt idx 0
    hooked = out.loc[out.ts == ASSESS_START].iloc[0]
    assert hooked.kwh_in == pytest.approx(10.0 * HOOK_FACTOR)
    assert hooked.kvarh_in == pytest.approx(4.0 * HOOK_FACTOR)
    assert out.loc[out.dt_id == 'DT0001', 'kwh_in'].iloc[0] == 10.0
    assert out.loc[(out.dt_id == 'DT0000') & (out.ts < ASSESS_START), 'kwh_in'].iloc[0] == 10.0


def test_select_eligible_skips_existing_theft_and_train_dts_only():
    kept = [f'M{i:06d}' for i in range(10)]
    meters = pd.DataFrame({'meter_id': kept,
                           'gis_dt_id': [f'DT{i:04d}' for i in range(10)]},
                          ).set_index('meter_id')          # 10 DTs, 1 meter each
    truth = pd.DataFrame({'meter_id': kept,
                          'is_theft': [False, True] + [False] * 8})
    eligible, types, hooked = select_eligible(kept, meters, truth)
    # train DTs are local 0-6 -> DT0000..DT0006; M000001 (DT0001) is existing theft
    assert set(eligible) == {'M000000', 'M000002', 'M000003',
                             'M000004', 'M000005', 'M000006'}
    assert len(hooked) == 2 and all(0 <= h < 7 for h in hooked)
