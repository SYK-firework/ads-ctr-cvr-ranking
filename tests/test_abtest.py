import numpy as np

from adsrank.abtest import assign, mde, mean_diff, ratio_diff, sample_size, srm_check


def test_assign_is_deterministic_balanced_and_salted():
    u = np.arange(200000)
    a1, a2, b = assign(u, "x"), assign(u, "x"), assign(u, "y")
    assert (a1 == a2).all()                                   # same unit + salt -> same arm
    assert abs(a1.mean() - 0.5) < 0.005                       # balanced
    assert abs((a1 == b).mean() - 0.5) < 0.01                 # different salts are independent
    rep = assign(np.repeat(u[:1000], 3), "x")
    assert (rep.reshape(-1, 3) == rep.reshape(-1, 3)[:, :1]).all()   # a user's rows share one arm


def test_srm_flags_imbalance():
    assert not srm_check(50100, 49900)["srm_detected"]
    assert srm_check(51500, 48500)["srm_detected"]


def test_mean_diff_ci_coverage_and_power():
    rng = np.random.default_rng(0)
    covered = 0
    for _ in range(400):
        r = mean_diff(rng.normal(1.1, 1, 2000), rng.normal(1.0, 1, 2000))
        covered += r["ci_low"] <= 0.1 <= r["ci_high"]
    assert 0.92 < covered / 400 < 0.98                        # ~95% coverage


def test_aa_false_positive_rate_near_alpha():
    rng = np.random.default_rng(1)
    x = rng.exponential(1, 20000)
    fp = np.mean([mean_diff(x[g == 1], x[g == 0])["p_value"] < 0.05
                  for g in (assign(np.arange(20000), f"aa{i}") for i in range(300))])
    assert 0.02 < fp < 0.09


def test_ratio_delta_method_matches_bootstrap():
    rng = np.random.default_rng(2)
    den_t, den_c = rng.exponential(1, 5000), rng.exponential(1, 5000)
    num_t, num_c = den_t * rng.normal(2.2, 0.5, 5000), den_c * rng.normal(2.0, 0.5, 5000)
    r = ratio_diff(num_t, den_t, num_c, den_c)
    boots = []
    for _ in range(300):
        i, j = rng.integers(0, 5000, 5000), rng.integers(0, 5000, 5000)
        boots.append(num_t[i].sum() / den_t[i].sum() - num_c[j].sum() / den_c[j].sum())
    assert abs(r["se"] / np.std(boots) - 1) < 0.15


def test_mde_and_sample_size_are_consistent():
    n = sample_size(sigma=2.0, delta=0.1)
    assert abs(mde(2.0, n) - 0.1) < 1e-3
