"""Delayed-feedback CVR (Chapelle, 2014).

When a model is trained at time T, a click from time t whose conversion arrives at t + d with
t + d > T still looks like a negative. Recent clicks are the most affected, so a naive model
learns a CVR that is too low.

Simulation on Criteo Attribution: labels in the dataset are complete, so the training cutoff T is
replayed. For each training click,
    observed label  y_obs = 1 if it converted and conversion_timestamp <= T
    elapsed time    e     = T - click time
and models are evaluated on the test window against the complete labels.

Methods (same network for all, so only the label handling differs):
  oracle : complete labels (upper bound, not available in practice)
  naive  : observed labels, all clicks
  wait   : observed labels, only clicks older than a waiting window W (fewer, cleaner samples)
  dfm    : delayed feedback model. Conversion probability p(x) and an exponential delay with rate
           lambda are fitted jointly by maximum likelihood:
             converted (observed, delay d) : log p + log lambda - lambda d
             not converted yet (elapsed e) : log(1 - p + p exp(-lambda e))
"""
import numpy as np
import torch
import torch.nn.functional as F


def observed_labels(click_ts, conv_ts, converted, cutoff: float):
    """Observed label, elapsed time and delay at training cutoff `cutoff` (all in seconds)."""
    click_ts, conv_ts = np.asarray(click_ts, np.float64), np.asarray(conv_ts, np.float64)
    converted = np.asarray(converted).astype(bool)
    y_obs = converted & (conv_ts <= cutoff)
    elapsed = np.maximum(cutoff - click_ts, 1.0)
    delay = np.where(converted, np.maximum(conv_ts - click_ts, 1.0), np.nan)
    return y_obs.astype(np.float32), elapsed.astype(np.float32), delay.astype(np.float32)


def dfm_loss(logit, log_lambda, y_obs, elapsed, delay):
    """Negative log-likelihood of the exponential delayed feedback model (mean over the batch)."""
    lam = log_lambda.exp()
    log_p = F.logsigmoid(logit)
    log_1mp = F.logsigmoid(-logit)
    pos = log_p + log_lambda - lam * torch.nan_to_num(delay, nan=0.0)
    # log(1 - p + p e^{-lam e}) = logaddexp(log(1-p), log p - lam e)
    neg = torch.logaddexp(log_1mp, log_p - lam * elapsed)
    return -torch.where(y_obs > 0, pos, neg).mean()


class ExpMixtureDelay(torch.nn.Module):
    """Delay distribution as a mixture of K exponentials (e.g. "buys right away" + "comes back later").

    K = 1 is the original DFM. Rates are initialized on a log grid between init_hours."""

    def __init__(self, k: int = 2, init_hours: tuple[float, float] = (1.0, 200.0)):
        super().__init__()
        hours = np.geomspace(init_hours[0], init_hours[1], k) if k > 1 else np.array([init_hours[1]])
        self.log_rate = torch.nn.Parameter(torch.tensor(np.log(1 / (hours * 3600)), dtype=torch.float32))
        self.weight_logit = torch.nn.Parameter(torch.zeros(k))

    def _log_w(self):
        return torch.log_softmax(self.weight_logit, 0)

    def log_pdf(self, d):                        # log sum_k w_k lam_k exp(-lam_k d)
        d = torch.nan_to_num(d, nan=0.0)[:, None]
        return torch.logsumexp(self._log_w() + self.log_rate - self.log_rate.exp() * d, dim=1)

    def log_surv(self, e):                       # log P(delay > e) = log sum_k w_k exp(-lam_k e)
        return torch.logsumexp(self._log_w() - self.log_rate.exp() * e[:, None], dim=1)

    def mean_hours(self) -> float:
        w = torch.softmax(self.weight_logit.detach(), 0)
        return float((w / self.log_rate.exp()).sum().detach() / 3600)


def mixture_dfm_loss(logit, delay_model: ExpMixtureDelay, y_obs, elapsed, delay):
    """DFM likelihood with a parametric delay model learned jointly with p(x)."""
    log_p, log_1mp = F.logsigmoid(logit), F.logsigmoid(-logit)
    pos = log_p + delay_model.log_pdf(delay)
    neg = torch.logaddexp(log_1mp, log_p + delay_model.log_surv(elapsed))
    return -torch.where(y_obs > 0, pos, neg).mean()


def empirical_survival(delays_hist: np.ndarray, elapsed: np.ndarray) -> np.ndarray:
    """P(delay > e) from a matured historical sample of conversion delays."""
    d = np.sort(np.asarray(delays_hist, np.float64))
    return (1 - np.searchsorted(d, elapsed, side="right") / len(d)).astype(np.float32)


def known_delay_loss(logit, y_obs, log_surv_e):
    """DFM with a fixed, known delay distribution: only p(x) is learned.
    Converted: log p (the delay density does not depend on p). Not yet: log(1 - p + p S(e))."""
    log_p, log_1mp = F.logsigmoid(logit), F.logsigmoid(-logit)
    neg = torch.logaddexp(log_1mp, log_p + log_surv_e)
    return -torch.where(y_obs > 0, log_p, neg).mean()
