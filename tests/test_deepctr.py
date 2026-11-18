import pytest

torch = pytest.importorskip("torch")

from adsrank.attribution import build_features, time_split   # noqa: E402
from adsrank.deepctr import DCNv2, DeepFM   # noqa: E402
from adsrank.train_deepctr import run   # noqa: E402
from test_attribution import fake_attribution   # noqa: E402


@pytest.mark.parametrize("cls", [DeepFM, DCNv2])
def test_forward_shape(cls):
    m = cls([10, 20, 5], n_num=3, emb_dim=4)
    out = m(torch.zeros(7, 3, dtype=torch.long), torch.zeros(7, 3))
    assert out.shape == (7,)


def test_fm_term_equals_pairwise_inner_products():
    e = torch.randn(2, 4, 3)
    fast = 0.5 * ((e.sum(1) ** 2) - (e ** 2).sum(1)).sum(1)
    slow = sum((e[:, i] * e[:, j]).sum(1) for i in range(4) for j in range(i + 1, 4))
    assert torch.allclose(fast, slow, atol=1e-5)


@pytest.mark.parametrize("name", ["deepfm", "dcnv2"])
def test_models_learn(name):
    df = fake_attribution(60000)         # CVR trains on clicks only: small data, so a higher lr
    r = run(build_features(df, time_split(df)), name, epochs=3, batch=1024, lr=5e-3, threads=2,
            log=lambda *_: None)
    assert r["ctr_auc"] > 0.6 and r["cvr_auc"] > 0.55
