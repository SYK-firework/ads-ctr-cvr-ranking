"""Criteo Attribution dataset: features and time-based splits for CTR / CVR / CTCVR.

Each row is an impression. The funnel is impression -> click -> attributed conversion:
  click        : clicked or not                                   -> CTR label (all impressions)
  attribution  : conversion attributed to the ad (post-click)     -> CVR label (clicked impressions)
                 every attributed conversion has click == 1, so attribution is also the
                 CTCVR label on all impressions: P(click & conversion) = pCTR x pCVR

Columns that only exist after the click / conversion happened are excluded (label leakage):
  conversion, conversion_timestamp, conversion_id, click_pos, click_nb
`cost` (price paid) is the auction outcome, kept for ranking / bidding stages, not as a feature.
`uid` is ~1.3M unique values in 2M rows (almost no repeats), so it is not used directly;
user recency comes from time_since_last_click.
"""
import numpy as np
import pandas as pd

CATS = ["campaign"] + [f"cat{i}" for i in range(1, 10)]
NUMS = ["cpo", "log_tslc", "no_prev_click", "hour"]
FEATURES = CATS + NUMS
LEAKY = ["conversion", "conversion_timestamp", "conversion_id", "click_pos", "click_nb"]


def time_split(df: pd.DataFrame, valid_frac: float = 0.1, test_frac: float = 0.2) -> pd.Series:
    """Split by timestamp quantiles: earliest -> train, then valid, latest -> test."""
    t = df["timestamp"]
    v0, t0 = t.quantile(1 - valid_frac - test_frac), t.quantile(1 - test_frac)
    return pd.Series(np.where(t < v0, "train", np.where(t < t0, "valid", "test")), index=df.index)


def build_features(df: pd.DataFrame, split: pd.Series, min_count: int = 20) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    train = split == "train"
    for c in CATS:
        counts = df.loc[train, c].value_counts()
        keep = counts[counts >= min_count].index           # vocabulary from the train window only
        codes = df[c].map(pd.Series(np.arange(1, len(keep) + 1), index=keep)).fillna(0).astype(np.int32)
        out[c] = pd.Categorical(codes, categories=np.arange(len(keep) + 1))   # 0 = OOV
    tslc = df["time_since_last_click"]
    out["no_prev_click"] = (tslc < 0).astype(np.int8)
    out["log_tslc"] = np.log1p(tslc.clip(lower=0)).astype(np.float32)
    out["cpo"] = df["cpo"].astype(np.float32)               # campaign's target cost per order
    out["hour"] = ((df["timestamp"] // 3600) % 24).astype(np.int8)
    out["click"] = df["click"].astype(np.int8)
    out["attribution"] = df["attribution"].astype(np.int8)
    out["split"] = split
    return out
