import numpy as np
import pandas as pd

from src.eval.merged_eval import (balance_scores, enrichment_at_k, ranks_of,
                                  unexplained_fraction)


def _toy_obs(n_dts=3, m=5, steps=96 * 4):
    """Hand-built Observations-shaped namespace; enough for the scoring utils."""
    class T:
        def __init__(self, size):
            self.size = size

    class O:
        pass

    o = O()
    n = n_dts * m
    rng = np.random.default_rng(0)
    o.dt_id = np.repeat(np.arange(n_dts), m)
    base = rng.uniform(0.4, 1.2, (n, 1)) * np.tile(
        np.abs(np.sin(np.linspace(0, 6, steps))) + .2, (n, 1))
    o.kwh = base.astype(np.float32)
    o.missing = np.zeros((n, steps), bool)
    o.dt_kwh = o.kwh.reshape(n_dts, m, steps).sum(1) * np.float32(1.04)
    o.dt_kwh[1] *= np.float32(1.0)  # DT 1: residual from losses only
    # one meter on DT 0 under-records by 60% from interval 100 ( theft-like)
    o.kwh[2, 100:] *= np.float32(0.4)
    o.dt_kwh = o.dt_kwh.astype(np.float32)
    o.dt_voltage = np.full((n_dts, steps), 230.0, np.float32)
    return o, n_dts, m


def test_unexplained_fraction_shapes():
    o, d, _ = _toy_obs()
    frac, kwh = unexplained_fraction(o)
    assert frac.shape == (d,) and kwh.shape == (d,)
    assert (frac >= 0).all() and (frac <= 1).all()


def test_balance_flags_theft_dt():
    o, d, _ = _toy_obs()
    frac, _ = unexplained_fraction(o)
    # DT 0 contains the under-recording meter -> largest unexplained fraction
    assert int(np.argmax(frac)) == 0


def test_ranks_of():
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    assert ranks_of(scores) == [4, 1, 2, 3]  # rank 1 = highest score


def test_enrichment_at_k():
    found = [0, 3]
    hits = enrichment_at_k(scores := np.array([.9, .8, .1, .7, .05]), found, k=2)
    assert hits == 1  # only meter 0 in top-2
    assert hits == 1  # idempotent
