import numpy as np

from src.data.theft_inject import INJECT_START_DAY, apply_theft, assign_types, meter_rng


def test_inject_start_day_is_7():
    assert INJECT_START_DAY == 7


def test_meter_rng_deterministic_and_salted():
    a = meter_rng('M000001', salt=4101).random()
    b = meter_rng('M000001', salt=4101).random()
    c = meter_rng('M000001', salt=7717).random()
    assert a == b
    assert a != c


def test_assign_types_exact_k_and_types_in_1_to_6():
    ids = [f'M{i:06d}' for i in range(100)]
    types = assign_types(ids, prevalence=0.10)
    thieves = [m for m, t in types.items() if t is not None]
    assert len(thieves) == 10
    assert all(1 <= t <= 6 for t in types.values() if t is not None)
    assert all(t is None for m, t in types.items() if m not in thieves)
    # deterministic given the same ids
    assert types == assign_types(ids, prevalence=0.10)


def test_apply_theft_six_types_shapes_and_t6_reversal():
    day = np.linspace(0.2, 1.5, 96)
    rng = meter_rng('probe', salt=1)
    t1 = apply_theft(day, 1, rng)
    assert t1.shape == (96,) and (t1 <= day * 0.8 + 1e-12).all()
    t2 = apply_theft(day, 2, meter_rng('probe', salt=2))
    assert (t2 == 0).sum() >= 36                      # >= 4h of 15-min slots
    t3 = apply_theft(day, 3, meter_rng('probe', salt=3))
    assert t3.shape == (96,)
    t4 = apply_theft(day, 4, meter_rng('probe', salt=4))
    assert 0.1 * day.mean() <= t4.mean() <= 0.8 * day.mean() + 1e-9
    t5 = apply_theft(day, 5, meter_rng('probe', salt=5))
    assert np.allclose(t5, day.mean())
    t6 = apply_theft(day, 6, meter_rng('probe', salt=6))
    assert np.allclose(t6, day[::-1])
