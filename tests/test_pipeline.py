"""Smoke tests: run preprocess -> train -> metrics end to end on small synthetic data in the Criteo format."""
import numpy as np
import pandas as pd

from adsrank import CAT, NUM
from adsrank.metrics import ads_metrics
from adsrank.prep import COLS, preprocess
from adsrank.train_gbdt import train


def fake_criteo(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"label": np.zeros(n, dtype=int)}
    for c in NUM:
        v = rng.exponential(50, n).astype(int).astype(float)
        v[rng.random(n) < 0.2] = np.nan
        data[c] = v
    for c in CAT:
        v = pd.array([format(x % 500, "08x") for x in rng.zipf(1.5, n)], dtype="string")
        v[rng.random(n) < 0.1] = pd.NA
        data[c] = v
    df = pd.DataFrame(data)[COLS]
    signal = (df["C1"] == format(1, "08x")).fillna(False).to_numpy()
    df["label"] = (rng.random(n) < 0.15 + 0.3 * signal).astype(int)
    return df


def test_preprocess_split_and_vocab():
    out, vocab = preprocess(fake_criteo(), min_count=5)
    assert set(out.split.unique()) == {"train", "valid", "test"}
    assert (out.split == "train").mean() == 0.8
    assert all(out[c].min() >= 0 for c in CAT)                  # 0 = OOV
    assert all(out[c].max() < vocab[c] for c in CAT)             # indices fit the vocabulary


def test_metrics_perfect_and_constant():
    y = np.array([0, 0, 1, 1])
    m = ads_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["test_auc"] == 1.0 and m["test_ne"] < 1.0
    m = ads_metrics(y, np.full(4, 0.5))
    assert abs(m["test_ne"] - 1.0) < 1e-9 and abs(m["test_calibration"] - 1.0) < 1e-9


def test_train_gbdt_learns_signal():
    out, _ = preprocess(fake_criteo(20000), min_count=5)
    _, _, r = train(out, params={"n_estimators": 50}, n_jobs=2, verbose=False)
    assert r["test_auc"] > 0.55


def test_read_raw_accepts_tsv_and_csv_mirror(tmp_path):
    from adsrank.prep import read_raw
    df = fake_criteo(50)
    tsv, csv = tmp_path / "train.txt", tmp_path / "train.csv"
    df.to_csv(tsv, sep="\t", header=False, index=False)
    df.rename(columns={"label": "Label"}).to_csv(csv, index=False)
    a, b = read_raw(str(tsv)), read_raw(str(csv))
    assert list(a.columns) == list(b.columns) == COLS
    assert (a.label.to_numpy() == b.label.to_numpy()).all()
    assert a["C1"].fillna("").tolist() == b["C1"].fillna("").tolist()


def test_dac_deep_models_learn():
    import pytest
    pytest.importorskip("torch")
    from adsrank.train_dac_deep import run
    out, vocab = preprocess(fake_criteo(30000), min_count=5)
    for name in ("deepfm", "dcnv2"):
        r = run(out, vocab, name, epochs=2, batch=1024, lr=5e-3, threads=2, log=lambda *_: None)
        assert r["test_auc"] > 0.55
