"""Track C adapter tests. See docs/superpowers/specs/2026-10-27-track-c-ceew-injection-design.md."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.data.ceew_adapter import resample_15min, discover_windows


def _tiny_csv_rows(n_meters=3, days=14):
    rows = []
    for m in range(n_meters):
        mid = f"BR{m:02d}"
        for d in range(days):
            for s in range(480):  # 3-min slots
                ts = pd.Timestamp("2020-01-01", tz="Asia/Kolkata") + pd.Timedelta(days=d, minutes=3 * s)
                rows.append((ts.tz_localize(None), 0.01, 240.0, 1.5, 50.0, mid))
    return pd.DataFrame(rows, columns=["x_Timestamp", "t_kWh", "z_Avg Voltage (Volt)",
                                       "z_Avg Current (Amp)", "y_Freq (Hz)", "meter"])


def test_resample_3min_to_15min():
    df = _tiny_csv_rows(n_meters=1, days=2)
    out = resample_15min(df)
    # 2 days * 96 fifteen-min slots
    assert len(out) == 192
    assert list(out.columns) == ["meter_id", "ts", "kwh", "voltage_v", "current_a", "freq_hz"]
    # each 15-min kwh = 5 x 0.01
    assert np.allclose(out.kwh.to_numpy(), 0.05)
    assert np.allclose(out.voltage_v.to_numpy(), 240.0)
    assert str(out.ts.dt.tz) == "Asia/Kolkata"


def test_resample_keeps_zero_voltage_outage_as_zero():
    df = _tiny_csv_rows(n_meters=1, days=1)
    # 160 three-min rows = 8 hours = 32 fifteen-min slots
    df.loc[df.index[:160], "z_Avg Voltage (Volt)"] = 0.0
    df.loc[df.index[:160], "t_kWh"] = 0.0
    out = resample_15min(df)
    assert (out.voltage_v.to_numpy()[:32] == 0.0).all()   # 32 fifteen-min slots
    assert (out.kwh.to_numpy()[:32] == 0.0).all()


def test_discover_windows_uniform_grid_and_coverage():
    # 26 meters, 14 complete days -> one window, uniform 2 DTs x 13
    frames = []
    for m in range(26):
        mid = f"BR{m:02d}"
        ts = pd.date_range("2020-01-01", periods=14 * 96, freq="15min", tz="Asia/Kolkata")
        frames.append(pd.DataFrame({"meter_id": [f"CEEW-{mid}"] * len(ts), "ts": ts,
                                    "kwh": 0.5, "voltage_v": 240.0,
                                    "current_a": 1.0, "freq_hz": 50.0}))
    base = pd.concat(frames, ignore_index=True)
    wins, skipped = discover_windows("BR", base, min_coverage=0.95, target_m=11)
    assert len(wins) == 1
    w = wins[0]
    assert w["n_dts"] * w["meters_per_dt"] == len(w["kept_meters"])
    assert w["n_dts"] == 2 and w["meters_per_dt"] == 13
    assert len(w["dropped_meters"]) == 0
    assert skipped == []


def test_discover_windows_drops_low_coverage_meter():
    frames = []
    for m in range(12):
        mid = f"BR{m:02d}"
        ts = pd.date_range("2020-01-01", periods=14 * 96, freq="15min", tz="Asia/Kolkata")
        frames.append(pd.DataFrame({"meter_id": [f"CEEW-{mid}"] * len(ts), "ts": ts,
                                    "kwh": 0.5, "voltage_v": 240.0,
                                    "current_a": 1.0, "freq_hz": 50.0}))
    # meter BR00 loses half its slots -> below coverage, excluded from valid set
    f0 = frames[0]
    frames[0] = f0.iloc[: len(f0) // 2]
    base = pd.concat(frames, ignore_index=True)
    wins, skipped = discover_windows("BR", base, min_coverage=0.95, target_m=11)
    assert len(wins) == 1
    # 11 valid meters -> n_dts=1 x m=11, all kept; BR00 filtered by coverage
    assert "CEEW-BR00" not in wins[0]["kept_meters"]
    assert len(wins[0]["kept_meters"]) == 11
    assert wins[0]["n_dts"] == 1 and wins[0]["meters_per_dt"] == 11


# ---- Task 2: virtual DTs + build_window ------------------------------------
from pathlib import Path

from src.data.ceew_adapter import (build_window, star_topology, WINDOW_STEPS,
                                   discover_windows, load_district)
from src.common import load_config


def _frame_14d(meter_ids, start="2020-01-01"):
    frames = []
    for mid in meter_ids:
        ts = pd.date_range(start, periods=WINDOW_STEPS, freq="15min", tz="Asia/Kolkata")
        # gentle daily profile, V=245, I varies
        kwh = 0.4 + 0.3 * np.sin(np.arange(WINDOW_STEPS) * 2 * np.pi / 96)
        frames.append(pd.DataFrame({"meter_id": mid, "ts": ts, "kwh": kwh,
                                    "voltage_v": 245.0, "current_a": 1.2,
                                    "freq_hz": 50.0}))
    return pd.concat(frames, ignore_index=True)


def _win_cfg(district="BR"):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "final.yaml")
    cfg.update(n_dts=2, meters_per_dt=6, days=14, interval_minutes=15, baseline_days=7)
    return cfg


def test_star_topology_shape():
    topo = star_topology(6)
    assert topo.size == 7                    # root + 6 meters
    assert list(topo.meter_nodes) == [1, 2, 3, 4, 5, 6]
    assert list(topo.parent) == [-1, 0, 0, 0, 0, 0, 0]
    assert topo.source == "ceew_star"


def test_build_window_uniform_and_conservation(monkeypatch):
    import src.data.ceew_adapter as A
    meter_ids = [f"CEEW-BR{i:02d}" for i in range(12)]
    frame = _frame_14d(meter_ids)
    monkeypatch.setattr(A, "load_district", lambda d: frame)
    win = {"window_id": "BR-2020-01", "district": "BR", "start": "2020-01-01",
           "end": "2020-01-15", "n_dts": 2, "meters_per_dt": 6,
           "kept_meters": sorted(meter_ids), "dropped_meters": [],
           "n_valid_before_drop": 12}
    cfg = _win_cfg()
    obs, info = build_window("BR", win, cfg)
    n, d = 12, 2
    assert obs.kwh.shape == (n, WINDOW_STEPS)
    assert obs.dt_kwh.shape == (d, WINDOW_STEPS)
    assert obs.missing.sum() == 0
    assert len(obs.topologies) == d
    # energy conservation: dt_in >= sum of true loads; loss in a sane band
    for g in range(d):
        members = obs.kwh[g * 6:(g + 1) * 6].sum(axis=0)
        assert np.isfinite(obs.dt_kwh[g]).all()
        assert (obs.dt_kwh[g] >= members - 1e-6).all()
        excess = (obs.dt_kwh[g] - members) / np.maximum(members, 1e-9)
        assert np.nanmax(excess) < 0.35      # loss model sanity (<35%)
    # row order: sorted by (dt_id, meter_id)
    assert list(obs.meters.dt_id) == [0] * 6 + [1] * 6
    assert list(obs.meters.meter_id) == sorted(meter_ids)
    # neutral temperature
    assert (obs.temperature == 25.0).all()
    assert info["meters"] == sorted(meter_ids)


def test_build_window_missing_slot_becomes_missing_mask(monkeypatch):
    import src.data.ceew_adapter as A
    meter_ids = [f"CEEW-BR{i:02d}" for i in range(12)]
    frame = _frame_14d(meter_ids)
    frame = frame[~((frame.meter_id == meter_ids[0]) &
                    (frame.ts == pd.Timestamp("2020-01-02 03:15", tz="Asia/Kolkata")))]
    monkeypatch.setattr(A, "load_district", lambda d: frame)
    win = {"window_id": "BR-2020-01", "district": "BR", "start": "2020-01-01",
           "end": "2020-01-15", "n_dts": 2, "meters_per_dt": 6,
           "kept_meters": sorted(meter_ids), "dropped_meters": [],
           "n_valid_before_drop": 12}
    obs, info = build_window("BR", win, _win_cfg())
    assert obs.missing[0].sum() == 1
    assert np.isnan(obs.kwh[0, :]).sum() == 1
