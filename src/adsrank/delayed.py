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
