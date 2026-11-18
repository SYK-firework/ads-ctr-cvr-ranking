"""Serving path: exported features + Scorer reproduce the training-pipeline predictions."""
import json

import numpy as np
import pytest

from adsrank.attribution import CATS, FEATURES, build_features, time_split
from adsrank.serving.scorer import Scorer
from adsrank.train_attribution import fit
from test_attribution import fake_attribution


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    df = fake_attribution(20000)
    split = time_split(df)
    feats = build_features(df, split)
    tr, va, te = (feats[feats.split == s] for s in ("train", "valid", "test"))
    ctr = fit(tr, va, "click", 2, verbose=False)
    cvr = fit(tr[tr.click == 1], va[va.click == 1], "attribution", 2, verbose=False)
    ctr.get_booster().save_model(d / "ctr.json")
    cvr.get_booster().save_model(d / "cvr.json")
    vocab = {}
    for c in CATS:
        counts = df.loc[split == "train", c].value_counts()
        vocab[c] = [int(v) for v in counts[counts >= 20].index]
    (d / "features.json").write_text(json.dumps({"features": FEATURES, "vocab": vocab}))
    return d, df, te, ctr, cvr


def test_no_training_serving_skew(exported):
    d, df, te, ctr, cvr = exported
    rows = df.loc[te.index[:500], CATS + ["cpo", "time_since_last_click", "timestamp"]].to_dict("records")
    s = Scorer(d).score(rows)
    assert np.allclose(s["p_ctr"], ctr.predict_proba(te[FEATURES].iloc[:500])[:, 1], atol=1e-6)
    assert np.allclose(s["p_cvr"], cvr.predict_proba(te[FEATURES].iloc[:500])[:, 1], atol=1e-6)
    assert np.allclose(s["expected_value"], np.asarray([r["cpo"] for r in rows]) * s["p_ctr"] * s["p_cvr"], rtol=1e-5)


def test_unknown_campaign_maps_to_oov(exported):
    d, df, te, *_ = exported
    row = df.loc[te.index[0], CATS + ["cpo", "time_since_last_click", "timestamp"]].to_dict()
    row["campaign"] = 10**12                               # never seen in training
    s = Scorer(d).score([row])
    assert 0 < s["p_ctr"][0] < 1


def test_http_endpoint(exported, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    d, df, te, *_ = exported
    monkeypatch.setenv("MODEL_DIR", str(d))
    import importlib
    import adsrank.serving.app as app_mod
    importlib.reload(app_mod)
    c = TestClient(app_mod.app)
    assert c.get("/health").json()["status"] == "ok"
    rows = df.loc[te.index[:3], CATS + ["cpo", "time_since_last_click", "timestamp"]].to_dict("records")
    r = c.post("/predict", json={"instances": [{k: (float(v) if k == "cpo" else int(v)) for k, v in x.items()} for x in rows]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 3
