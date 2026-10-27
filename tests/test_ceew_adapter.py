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
