import numpy as np

from adsrank.bidding import alpha_for_roi, alpha_for_spend, pid_pacing, replay


def _log(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 30, n)                              # true conversion probability
    attr = (rng.random(n) < p).astype(int)
    cpo = rng.uniform(0.5, 1.5, n)
    cost = rng.uniform(0.001, 0.05, n)
    return p, attr, cpo, cost


def test_replay_win_rule_and_totals():
    r = replay(np.array([1.0, 0.1]), np.array([0.5, 0.5]), np.array([1, 1]), np.array([2.0, 2.0]))
    assert r["win_rate"] == 0.5 and r["spend"] == 0.5 and r["value"] == 2.0 and r["roi"] == 4.0


def test_alpha_for_spend_hits_budget():
    p, attr, cpo, cost = _log()
    ev = cpo * p
    budget = 0.3 * cost.sum()
    a = alpha_for_spend(ev, cost, budget)
    assert abs(replay(ev * a, cost, attr, cpo)["spend"] / budget - 1) < 0.01


def test_value_based_bids_beat_random_at_same_budget():
    p, attr, cpo, cost = _log()
    budget = 0.2 * cost.sum()
    good = cpo * p
    rnd = np.random.default_rng(1).random(len(p))
    v_good = replay(good * alpha_for_spend(good, cost, budget), cost, attr, cpo)["value"]
    v_rnd = replay(rnd * alpha_for_spend(rnd, cost, budget), cost, attr, cpo)["value"]
    assert v_good > 1.5 * v_rnd


def test_alpha_for_roi_meets_target():
    p, attr, cpo, cost = _log()
    ev = cpo * p
    target = 1.5 * replay(np.full(len(p), np.inf), cost, attr, cpo)["roi"]
    a = alpha_for_roi(ev, cost, attr, cpo, target)
    assert replay(ev * a, cost, attr, cpo)["roi"] >= target


def test_pacing_spends_close_to_budget():
    p, attr, cpo, cost = _log(40000)
    ts = np.arange(len(p))
    budget = 0.2 * cost.sum()
    pace = pid_pacing(cpo * p, cost, attr, cpo, ts, budget, n_steps=40, alpha0=0.01)
    assert 0.8 < pace.spend.sum() / budget < 1.2


def _bursty_log(n=60000, seed=3):
    """Traffic whose volume varies 4x over the day, like real ad traffic."""
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.random(n) ** 1.6)                     # denser early, sparser late
    p = rng.beta(1, 30, n)
    attr = (rng.random(n) < p).astype(int)
    cpo = rng.uniform(0.5, 1.5, n)
    cost = rng.uniform(0.001, 0.05, n)
    return p, attr, cpo, cost, ts


def test_adaptive_pacing_hits_budget_without_overdraft_or_idle():
    from adsrank.bidding import adaptive_pacing
    p, attr, cpo, cost, ts = _bursty_log()
    ev, budget = cpo * p, 0.2 * cost.sum()
    alpha0 = alpha_for_spend(ev[:5000], cost[:5000], 0.2 * cost[:5000].sum())   # "previous period"
    pace = adaptive_pacing(ev, cost, attr, cpo, ts, budget, alpha0, n_steps=40)
    assert 0.9 < pace.spend.sum() / budget <= 1.0 + 1e-9        # spends the budget, never more
    assert (pace.spend > 0).all()                               # no idle steps
    assert (pace.spend <= 1.5 * pace.planned + 1e-9).all()      # throttled: never above 1.5x the step target


def test_adaptive_beats_naive_pi_on_bursty_traffic():
    from adsrank.bidding import adaptive_pacing
    p, attr, cpo, cost, ts = _bursty_log()
    ev, budget = cpo * p, 0.2 * cost.sum()
    a_static = alpha_for_spend(ev, cost, budget)
    naive = pid_pacing(ev, cost, attr, cpo, ts, budget, n_steps=40, alpha0=a_static * 0.1)
    alpha0 = alpha_for_spend(ev[:5000], cost[:5000], 0.2 * cost[:5000].sum())
    adaptive = adaptive_pacing(ev, cost, attr, cpo, ts, budget, alpha0, n_steps=40)
    assert adaptive.value.sum() > naive.value.sum()


def test_spend_plan_follows_time_of_day_and_sums_to_one():
    from adsrank.bidding import spend_plan
    day = 86400.0
    ts = np.arange(0, 3 * day, 60.0)                          # 3 days of history, one row a minute
    spend = np.where((ts % day) >= day / 2, 3.0, 1.0)         # second half of each day spends 3x
    edges = np.linspace(3 * day, 4 * day, 5)                  # next day in 4 steps
    w = spend_plan(ts, spend, edges)
    assert abs(w.sum() - 1) < 1e-9
    assert w[2] > 2.5 * w[0] and abs(w[0] - w[1]) < 1e-6      # afternoon steps get ~3x the morning


def test_adaptive_pacing_with_plan_tracks_plan_and_even_plan_is_default():
    from adsrank.bidding import adaptive_pacing
    rng = np.random.default_rng(0)
    n = 40000
    ts = np.sort(rng.uniform(0, 86400, n))
    cost = rng.exponential(1.0, n)
    ev = cost * rng.lognormal(0, 0.5, n)
    attr, cpo = (rng.random(n) < 0.05).astype(float), np.full(n, 10.0)
    budget = 0.2 * cost.sum()
    even = adaptive_pacing(ev, cost, attr, cpo, ts, budget, alpha0=0.5, n_steps=8)
    same = adaptive_pacing(ev, cost, attr, cpo, ts, budget, alpha0=0.5, n_steps=8, plan=np.full(8, 1 / 8))
    assert np.allclose(even.spend, same.spend)
    plan = np.array([1, 1, 1, 1, 3, 3, 3, 3], float) / 16
    shaped = adaptive_pacing(ev, cost, attr, cpo, ts, budget, alpha0=0.5, n_steps=8, plan=plan)
    assert shaped.spend.sum() <= budget * (1 + 1e-9)
    assert shaped.spend[4:].sum() > 2 * shaped.spend[:4].sum()   # spends more where the plan says
