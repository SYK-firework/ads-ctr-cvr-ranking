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
