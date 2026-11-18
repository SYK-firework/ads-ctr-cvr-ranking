"""Probability calibration for ads models.

Ads are ranked and priced on expected value (bid x pCTR x pCVR), so the probability values must be
right, not only their order. Three tools:

1. Post-hoc calibration, fitted on the validation window and applied to test:
     Platt    : logistic regression on the model's logit (2 parameters, smooth, monotone)
     Isotonic : monotone step function (flexible, needs more data, can create ties)
2. Reliability metrics: Expected Calibration Error (ECE) and a per-bin table / diagram.
3. Negative-downsampling correction. If non-click impressions are kept with rate w during
   training, the model learns p' = p / (p + (1 - p) * w). Inverting:
     p = p' * w / (1 - p' + p' * w)
   This is how production CTR models trained on sampled data are mapped back to true rates.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Platt:
    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class Isotonic:
    def fit(self, p, y):
        self.iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(p, y)
        return self

    def predict(self, p):
        return self.iso.predict(p)


CALIBRATORS = {"platt": Platt, "isotonic": Isotonic}


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of predicted probability: mean prediction vs observed rate per bin."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df.p.rank(method="first"), n_bins, labels=False)
    return df.groupby("bin").agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index()


def ece(y, p, n_bins: int = 10) -> float:
    """Expected Calibration Error: count-weighted mean |pred - actual| over equal-count bins."""
    t = reliability_table(y, p, n_bins)
    return float(np.average((t.pred - t.actual).abs(), weights=t.n))


def segment_calibration(y, p, segment, min_n: int = 200, top: int = 30) -> pd.DataFrame:
    """Calibration ratio (mean pred / actual) for the largest segments, e.g. campaigns."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "seg": np.asarray(segment)})
    g = df.groupby("seg").agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    g = g[(g.n >= min_n) & (g.actual > 0)].sort_values("n", ascending=False).head(top)
    g["ratio"] = g.pred / g.actual
    return g


def segment_error(seg_table: pd.DataFrame) -> float:
    """Count-weighted mean |log(pred / actual)| across segments (0 = every segment calibrated).
    NaN when no segment is large enough to measure."""
    if seg_table.empty:
        return float("nan")
    return float(np.average(np.abs(np.log(seg_table.ratio)), weights=seg_table.n))


def downsample_negatives(y, rate: float, seed: int = 0) -> np.ndarray:
    """Boolean mask that keeps all positives and a `rate` fraction of negatives."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    return (y == 1) | (rng.random(len(y)) < rate)


def correct_downsampling(p, rate: float) -> np.ndarray:
    """Map probabilities learned on negative-downsampled data back to the original scale."""
    p = np.asarray(p, dtype=np.float64)
    return p * rate / (1 - p + p * rate)
