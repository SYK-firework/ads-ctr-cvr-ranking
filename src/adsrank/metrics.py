"""Ads quality metrics.

AUC         : ranking quality - how well clicked impressions are ordered above non-clicked ones
LogLoss     : accuracy of the predicted probabilities
NE          : Normalized Entropy = LogLoss / LogLoss of a model that always predicts the average CTR.
              Lower is better (< 1.0 means the model beats the base rate). Common in ads CTR work.
Calibration : mean(predicted) / mean(actual). 1.0 is ideal.
              Ads are ranked and priced by eCPM = bid x pCTR, so the probability values themselves
              must be accurate, not just their order.
"""
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def ads_metrics(y, p, prefix: str = "test") -> dict[str, float]:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    ll = log_loss(y, p)
    base = log_loss(y, np.full_like(p, y.mean()))
    return {
        f"{prefix}_auc": roc_auc_score(y, p),
        f"{prefix}_logloss": ll,
        f"{prefix}_ne": ll / base,
        f"{prefix}_calibration": p.mean() / y.mean(),
    }
