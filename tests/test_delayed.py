import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adsrank.delayed import dfm_loss, observed_labels   # noqa: E402


def test_observed_labels_hide_late_conversions():
    click = np.array([0, 0, 50, 90])
    conv = np.array([10, 200, 60, -1])
    y, e, d = observed_labels(click, conv, np.array([1, 1, 1, 0]), cutoff=100)
    assert y.tolist() == [1, 0, 1, 0]                 # the conversion at t=200 is not visible yet
    assert e.tolist() == [100, 100, 50, 10]
    assert d[0] == 10 and np.isnan(d[3])


def test_dfm_recovers_true_rate_and_delay():
    rng = np.random.default_rng(0)
    n, p, mean_delay = 40000, 0.3, 50.0
    click = rng.uniform(0, 200, n)
    conv_flag = rng.random(n) < p
    conv = np.where(conv_flag, click + rng.exponential(mean_delay, n), -1)
    y, e, d = observed_labels(click, conv, conv_flag, cutoff=200)
    assert y.mean() < 0.8 * p                          # naive rate is biased low
    logit = torch.nn.Parameter(torch.zeros(1))
    log_lam = torch.nn.Parameter(torch.tensor(np.log(1 / 10.0), dtype=torch.float32))
    opt = torch.optim.Adam([logit, log_lam], lr=0.05)
    Y, E, D = (torch.as_tensor(x) for x in (y, e, d))
    for _ in range(600):
        loss = dfm_loss(logit.expand(n), log_lam, Y, E, D)
        opt.zero_grad(); loss.backward(); opt.step()
    assert abs(torch.sigmoid(logit).item() - p) < 0.03
    assert abs(1 / log_lam.exp().item() - mean_delay) < 8
