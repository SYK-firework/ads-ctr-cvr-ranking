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


def _mixed_delays(n=30000, p=0.3, seed=1):
    rng = np.random.default_rng(seed)
    click = rng.uniform(0, 200, n)
    conv_flag = rng.random(n) < p
    fast = rng.random(n) < 0.4
    d = np.where(fast, rng.exponential(2, n), rng.exponential(150, n))     # fast + slow buyers
    conv = np.where(conv_flag, click + d, -1)
    return click, conv, conv_flag, d


def test_mixture_dfm_beats_single_exponential_on_mixed_delays():
    from adsrank.delayed import ExpMixtureDelay, mixture_dfm_loss
    p = 0.3
    click, conv, flag, _ = _mixed_delays(p=p)
    y, e, d = (torch.as_tensor(x) for x in observed_labels(click, conv, flag, cutoff=200))
    est = {}
    for k in (1, 2):
        dm = ExpMixtureDelay(k, init_hours=(1 / 3600, 100 / 3600))   # units here are "seconds" = hours / 3600
        logit = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([logit, *dm.parameters()], lr=0.05)
        for _ in range(500):
            loss = mixture_dfm_loss(logit.expand(len(y)), dm, y, e, d)
            opt.zero_grad(); loss.backward(); opt.step()
        est[k] = torch.sigmoid(logit).item()
    assert abs(est[2] - p) < abs(est[1] - p) + 0.01
    assert abs(est[2] - p) < 0.05


def test_known_delay_loss_recovers_rate():
    from adsrank.delayed import empirical_survival, known_delay_loss
    p = 0.3
    click, conv, flag, d_all = _mixed_delays(p=p)
    y, e, _ = observed_labels(click, conv, flag, cutoff=200)
    log_s = torch.as_tensor(np.log(np.maximum(empirical_survival(d_all[flag], e), 1e-6)))
    logit = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([logit], lr=0.05)
    Y = torch.as_tensor(y)
    for _ in range(500):
        loss = known_delay_loss(logit.expand(len(Y)), Y, log_s)
        opt.zero_grad(); loss.backward(); opt.step()
    assert abs(torch.sigmoid(logit).item() - p) < 0.02
