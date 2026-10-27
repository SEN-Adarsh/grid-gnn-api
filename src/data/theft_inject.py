"""Shared pure theft-injection functions (Zidi et al. 2023, Sec. 3.2 + the
exact-k type selector). Extracted from ceew_adapter for Track C/D reuse;
behavior pinned by tests/test_theft_inject.py.

Assessment-only injection contract: build_features uses days 0-6 as the clean
own-history baseline and days 7-13 as the assessment window; injecting the
baseline too would self-cancel every ratio feature (both terms scaled).
"""
from __future__ import annotations

import zlib

import numpy as np

# Assessment-only injection: see module docstring.
INJECT_START_DAY = 7
PREVALENCE = 0.09


def meter_rng(meter_id: str, salt: int = 0) -> np.random.Generator:
    """Order-independent deterministic RNG per meter (seed 42 + crc32 + salt)."""
    return np.random.default_rng([42, zlib.crc32(meter_id.encode()), salt])


def assign_types(meter_ids: list[str], prevalence: float = PREVALENCE) -> dict:
    """Type fixed per meter across all windows (seed 42, order-independent).

    Exact-k selection: the round(prevalence*n) meters with the lowest per-meter
    hash draws are thieves, so realized prevalence matches the spec (a Bernoulli
    draw on 84 meters landed at 3.6% vs the specified 9%)."""
    ids = sorted(meter_ids)
    n_thieves = min(len(ids), max(0, int(round(prevalence * len(ids)))))
    ranked = sorted(ids, key=lambda mid: (meter_rng(mid, salt=1).random(), mid))
    thieves = set(ranked[:n_thieves])
    types = {}
    for mid in ids:
        if mid in thieves:
            rng = meter_rng(mid, salt=2)
            types[mid] = int(rng.integers(1, 7))
        else:
            types[mid] = None
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
