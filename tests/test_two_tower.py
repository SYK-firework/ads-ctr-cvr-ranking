"""Two-tower retrieval: metrics, ad-side feature detection, and a small training run."""
import numpy as np
import pytest

from adsrank.attribution import build_features, time_split
from adsrank.two_tower import ad_side_features, recall_at_k
from test_attribution import fake_attribution


def test_recall_at_k():
    scores = np.array([[0.9, 0.1, 0.5], [0.2, 0.3, 0.1]])
    r = recall_at_k(scores, np.array([2, 2]), ks=(1, 2))          # true item ranked 2nd and 3rd
    assert r["recall@1"] == 0.0 and r["recall@2"] == 0.5
    assert np.isclose(r["mrr"], (1 / 2 + 1 / 3) / 2)


def test_ad_side_feature_detected():
    df = fake_attribution(20000)
    df["cat9"] = df["campaign"] % 7                                   # a function of the campaign
    p = ad_side_features(df, ["cat1", "cat9"])
    assert p["cat9"] == 1.0 and p["cat1"] < 0.9


def test_two_tower_learns_context():
    pytest.importorskip("torch")
    from adsrank.two_tower import train_eval
    rng = np.random.default_rng(0)
    df = fake_attribution(60000)
    df["click"] = 1
    df["cat2"] = rng.integers(0, 30, len(df))
    df["campaign"] = (df["cat2"] * 7 + rng.integers(0, 2, len(df))) % 50   # campaign depends on context
    df["attribution"] = 0
    feats = build_features(df, time_split(df))
    r = train_eval(feats, df, epochs=3, batch=512, threads=2, log=lambda *_: None)
    assert r["two_tower"]["recall@10"] > r["popularity"]["recall@10"] + 0.2
    assert r["n_campaigns"] == df.campaign.nunique()          # every campaign seen in train is retrievable
    r0 = train_eval(feats, df, epochs=1, batch=512, threads=2, query_cats=[], log=lambda *_: None)
    assert "context_popularity" not in r0 and r0["identifiability"] < r["identifiability"]


def test_identifiability():
    df = fake_attribution(20000)
    df["cat9"] = df["campaign"] % 7
    feats = build_features(df, time_split(df))
    from adsrank.two_tower import identifiability
    full = identifiability(feats.assign(campaign=feats.campaign), ["campaign"])
    assert full == 1.0                                              # the campaign names itself
    assert identifiability(feats, ["cat9"]) < identifiability(feats, ["cat9", "cat2", "cat3", "cat4"])
    assert identifiability(feats, []) < 0.1                         # global top-1 share of 50 campaigns
