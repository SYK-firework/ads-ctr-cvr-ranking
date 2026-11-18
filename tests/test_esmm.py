"""Smoke tests for ESMM on synthetic data with the Criteo Attribution schema."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adsrank.attribution import build_features, time_split   # noqa: E402
from adsrank.esmm import ESMM, NN_CATS, NN_NUMS, loss_fn, train_esmm, vocab_sizes   # noqa: E402
from test_attribution import fake_attribution   # noqa: E402


def test_forward_shapes_and_ctcvr_bound():
    feats = build_features(df := fake_attribution(2000), time_split(df))
    m = ESMM(vocab_sizes(feats), len(NN_NUMS), emb_dim=4)
    x_cat = torch.zeros(8, len(NN_CATS), dtype=torch.long)
    p_ctr, p_cvr = m(x_cat, torch.zeros(8, len(NN_NUMS)))
    assert p_ctr.shape == p_cvr.shape == (8,)
    assert torch.all(p_ctr * p_cvr <= p_ctr)                  # CTCVR never exceeds CTR


def test_clicked_loss_ignores_unclicked_rows():
    p_ctr = torch.tensor([0.5, 0.5])
    p_cvr = torch.tensor([0.9, 0.1])
    y_click = torch.tensor([0.0, 1.0])
    a = loss_fn(p_ctr, p_cvr, y_click, torch.tensor([0.0, 0.0]), "clicked")
    b = loss_fn(p_ctr, torch.tensor([0.2, 0.1]), y_click, torch.tensor([0.0, 0.0]), "clicked")
    assert torch.isclose(a, b)                                # changing pCVR on an unclicked row has no effect


@pytest.mark.parametrize("cvr_loss", ["esmm", "clicked"])
def test_train_esmm_learns(cvr_loss):
    df = fake_attribution(30000)
    feats = build_features(df, time_split(df))
    _, r = train_esmm(feats, cvr_loss=cvr_loss, epochs=3, batch=1024, threads=2, log=lambda *_: None)
    assert r["ctr_auc"] > 0.6 and r["cvr_auc"] > 0.55
