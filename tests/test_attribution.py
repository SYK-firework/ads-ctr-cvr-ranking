"""Smoke tests for the Criteo Attribution pipeline on synthetic data with the same schema."""
import numpy as np
import pandas as pd

from adsrank.attribution import FEATURES, LEAKY, build_features, time_split
from adsrank.train_attribution import run


def fake_attribution(n: int = 30000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    campaign = rng.integers(0, 50, n)
    cat1 = rng.integers(0, 9, n)
    p_click = 0.2 + 0.3 * (campaign % 5 == 0) + 0.1 * (cat1 == 3)
    click = (rng.random(n) < p_click).astype(int)
    attribution = (click & (rng.random(n) < 0.05 + 0.2 * (campaign % 7 == 0))).astype(int)
    df = pd.DataFrame({
        "timestamp": np.sort(rng.integers(0, 4 * 86400, n)),
        "uid": rng.integers(0, 10**7, n),
        "campaign": campaign,
        "conversion": attribution, "conversion_timestamp": -1, "conversion_id": -1,
        "attribution": attribution, "click": click, "click_pos": -1, "click_nb": -1,
        "cost": rng.random(n) * 1e-3, "cpo": rng.random(n),
        "time_since_last_click": np.where(rng.random(n) < 0.3, -1, rng.integers(0, 10**6, n)),
        "cat1": cat1,
    })
    for i in range(2, 10):
        df[f"cat{i}"] = rng.integers(0, 30, n)
    return df


def test_no_leaky_features():
    assert not set(LEAKY) & set(FEATURES)
    assert "cost" not in FEATURES and "uid" not in FEATURES


def test_time_split_is_ordered():
    df = fake_attribution(5000)
    s = time_split(df)
    assert df.timestamp[s == "train"].max() <= df.timestamp[s == "valid"].min()
    assert df.timestamp[s == "valid"].max() <= df.timestamp[s == "test"].min()


def test_vocab_built_on_train_only():
    df = fake_attribution(5000)
    df.loc[df.index[-100:], "campaign"] = 999          # only appears in the test window
    feats = build_features(df, time_split(df), min_count=1)
    assert (feats["campaign"].iloc[-100:].astype(int) == 0).all()   # mapped to OOV


def test_run_learns_ctr_and_cvr():
    _, _, r = run(fake_attribution(), n_jobs=2, verbose=False)
    assert r["ctr_auc"] > 0.6 and r["cvr_auc"] > 0.55
    assert 0.8 < r["ctr_calibration"] < 1.2
