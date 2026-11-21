"""Drift metrics, the canary gate, and the step-wise rollout (with fake model servers)."""
import asyncio

import numpy as np
import pandas as pd

from adsrank.canary import rollout, to_canary
from adsrank.monitoring import (ArmStats, GateConfig, canary_gate, categorical_psi, daily_drift,
                                drift_level, psi)


def test_psi_near_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = rng.beta(2, 30, 50000)
    assert psi(ref, rng.beta(2, 30, 50000)) < 0.01
    assert psi(ref, ref * 1.5) > 0.25                      # every score scaled up by 50%
    assert drift_level(0.05) == "stable" and drift_level(0.2) == "watch" and drift_level(0.3) == "act"


def test_categorical_psi_catches_new_campaigns():
    rng = np.random.default_rng(1)
    ref = rng.integers(0, 20, 20000)
    same = rng.integers(0, 20, 20000)
    new_mix = np.where(rng.random(20000) < 0.4, 999, rng.integers(0, 20, 20000))   # 40% unseen
    assert categorical_psi(ref, same) < 0.01
    assert categorical_psi(ref, new_mix) > 0.25


def test_daily_drift_flags_the_drifted_day_and_calibration():
    rng = np.random.default_rng(2)
    n = 20000

    def window(t0, scale):
        p = rng.beta(2, 30, n)
        click = (rng.random(n) < p).astype(int)           # labels follow the unscaled score
        return pd.DataFrame({"timestamp": t0 + np.sort(rng.integers(0, 86400, n)), "p_ctr": p * scale,
                             "p_cvr": rng.beta(2, 10, n), "click": click,
                             "attribution": click * (rng.random(n) < 0.2), "campaign": rng.integers(0, 10, n)})

    ref = window(0, 1.0)
    cur = pd.concat([window(86400, 1.0), window(2 * 86400, 1.4)], ignore_index=True)
    t = daily_drift(ref, cur, label_cols={"p_ctr": "click", "p_cvr": "attribution"})
    assert list(t.day) == [0, 1]
    assert t.psi_p_ctr[0] < 0.02 and t.psi_p_ctr[1] > 0.1
    assert abs(t.calib_p_ctr[0] - 1) < 0.05 and t.calib_p_ctr[1] > 1.3


def _arm(scores, requests=500, errors=0, p99_ms=50.0):
    return ArmStats(requests, errors, p99_ms, np.asarray(scores))


def test_gate_promotes_equivalent_model_and_rejects_each_failure():
    rng = np.random.default_rng(3)
    s = rng.beta(2, 30, 20000)
    stable = _arm(s)
    assert canary_gate(stable, _arm(rng.beta(2, 30, 5000))).decision == "promote"
    assert canary_gate(stable, _arm(s[:5000] * 1.2)).decision == "rollback"            # scale shift
    assert canary_gate(stable, _arm(s[:5000], errors=20)).decision == "rollback"       # 4% errors
    assert canary_gate(stable, _arm(s[:5000], p99_ms=200)).decision == "rollback"      # latency
    assert canary_gate(stable, _arm(s[:5000], p99_ms=60)).decision == "promote"        # within slack
    assert canary_gate(stable, _arm(s[:10], requests=10)).decision == "wait"           # too little traffic


def test_hash_routing_matches_share():
    share = np.mean([to_canary(f"req-{i}", 0.1) for i in range(20000)])
    assert 0.09 < share < 0.11


def _fake_servers(canary_scale=1.0):
    rng = np.random.default_rng(4)

    async def send(arm, body):
        k = len(body["instances"])
        p = rng.beta(2, 30, k) * (canary_scale if arm == "canary" else 1.0)
        return {"predictions": [{"p_ctr": float(x)} for x in p]}
    return send


def _run(send):
    rows = [{"x": i} for i in range(100)]
    return asyncio.run(rollout(send, rows, steps=(0.1, 0.25, 0.5), rps=2000, seconds=0.5, batch=20,
                               cfg=GateConfig(min_requests=50), log=lambda *_: None))


def test_rollout_promotes_good_canary_through_all_steps():
    res = _run(_fake_servers(1.0))
    assert res["decision"] == "promote" and [h["share"] for h in res["history"]] == [0.1, 0.25, 0.5]


def test_rollout_stops_bad_canary_at_first_step():
    res = _run(_fake_servers(1.3))                       # canary over-predicts CTR by 30%
    assert res["decision"] == "rollback" and len(res["history"]) == 1


def test_canary_path_imports_without_pandas():
    """The canary gate runs inside the serving image, which has numpy but not pandas."""
    import subprocess
    import sys
    code = ("import builtins; real = builtins.__import__\n"
            "def imp(n, *a, **k):\n"
            "    if n.split('.')[0] == 'pandas': raise ImportError(n)\n"
            "    return real(n, *a, **k)\n"
            "builtins.__import__ = imp\n"
            "import adsrank.canary\n")
    subprocess.run([sys.executable, "-c", code], check=True)
