import numpy as np

from adsrank.calibration import (Isotonic, Platt, correct_downsampling, downsample_negatives, ece,
                                 segment_calibration, segment_error)


def _miscalibrated(n=50000, seed=0):
    rng = np.random.default_rng(seed)
    true_p = rng.beta(2, 8, n)
    y = (rng.random(n) < true_p).astype(int)
    return y, np.clip(true_p * 1.6, 0, 0.99)           # model over-predicts by 60%


def test_ece_zero_for_perfect_and_positive_for_biased():
    y, biased = _miscalibrated()
    rng = np.random.default_rng(1)
    p = rng.random(200000)
    assert ece((rng.random(200000) < p).astype(int), p) < 0.01
    assert ece(y, biased) > 0.05


def test_platt_and_isotonic_fix_bias():
    y, p = _miscalibrated()
    for cal in (Platt(), Isotonic()):
        q = cal.fit(p[:25000], y[:25000]).predict(p[25000:])
        assert ece(y[25000:], q) < ece(y[25000:], p[25000:]) / 3


def test_downsampling_correction_inverts_sampling():
    p = np.array([0.01, 0.1, 0.5])
    rate = 0.2
    sampled = p / (p + (1 - p) * rate)                 # what a model learns on sampled data
    assert np.allclose(correct_downsampling(sampled, rate), p)


def test_downsample_keeps_all_positives():
    y = np.array([1] * 100 + [0] * 10000)
    keep = downsample_negatives(y, 0.1)
    assert keep[:100].all() and 0.07 < keep[100:].mean() < 0.13


def test_segment_error():
    seg = np.repeat(np.arange(5), 2000)
    y = np.tile(np.r_[np.ones(200), np.zeros(1800)], 5)
    t = segment_calibration(y, np.full(len(y), 0.1), seg)
    assert np.allclose(t.ratio, 1.0) and segment_error(t) < 1e-9
