import numpy as np

from src.eval.track_d_eval import _metrics_block, _verdict


def test_metrics_block_bounded_and_complete():
    rng = np.random.default_rng(0)
    y = np.array([0, 0, 1, 1] * 25)
    s = np.concatenate([rng.uniform(0, .4, 50), rng.uniform(.6, 1, 50)])
    m = _metrics_block(y, s)
    assert 0.0 <= m['roc_auc'] <= 1.0 and 0.0 <= m['pr_auc'] <= 1.0
    assert m['n'] == 100 and m['prevalence'] == 0.5


def test_gates_verdicts():
    assert _verdict(g1={'max_pr_drop': 0.01, 'max_roc_drop': 0.01})['G1'] is True
    assert _verdict(g1={'max_pr_drop': 0.05, 'max_roc_drop': 0.01})['G1'] is False
    assert _verdict(g2={'best_t56_recall': 0.55})['G2'] is True
    assert _verdict(g2={'best_t56_recall': 0.11})['G2'] is False
    assert _verdict(g3={'m3_delta_pr': 0.05, 'm4_delta_pr': -0.01})['G3'] is False
    assert _verdict(g3={'m3_delta_pr': 0.05, 'm4_delta_pr': 0.02})['G3'] is True
