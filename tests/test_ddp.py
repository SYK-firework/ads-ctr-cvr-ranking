"""DDP on CPU (gloo): W ranks x batch/W must equal one process on the full batch, replicas identical."""
import pytest

torch = pytest.importorskip("torch")

from adsrank.ddp import train_ddp, train_single  # noqa: E402


def _data(n=4096, seed=0):
    g = torch.Generator().manual_seed(seed)
    vs = [20, 15, 24]
    x_cat = torch.stack([torch.randint(0, v, (n,), generator=g) for v in vs], 1)
    x_num = torch.randn(n, 3, generator=g)
    logit = (x_cat[:, 0] % 3 == 0).float() * 1.5 - 1 + 0.5 * x_num[:, 0]
    y = (torch.rand(n, generator=g) < torch.sigmoid(logit)).float()
    return vs, (x_cat, x_num, y)


def test_ddp_matches_single_process_and_replicas_agree():
    vs, train = _data()
    single = train_single(vs, 3, train, epochs=2, batch=256, dropout=0.0, threads=1)
    ddp, replica_diff, _ = train_ddp(vs, 3, train, world=2, epochs=2, batch=256, dropout=0.0, threads=1)
    assert replica_diff == 0.0                                    # every rank holds the same weights
    for (n1, p1), (n2, p2) in zip(single.state_dict().items(), ddp.state_dict().items()):
        assert n1 == n2
        assert torch.allclose(p1, p2, atol=1e-5), n1               # same as one process, full batch
