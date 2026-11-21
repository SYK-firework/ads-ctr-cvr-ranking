"""How much user history does Criteo Attribution have? (Decides whether user-history retrieval is possible.)

For growing prefixes of the time-ordered data, with the usual 70/10/20 time split:
  unique_uid_share          unique uids / rows
  rows_with_earlier_row     share of rows whose uid already appeared earlier
  test_clicks_with_train_click  share of test-window clicks whose uid clicked in the train window:
                            the clicks a user-history feature ("campaigns this user clicked before")
                            could help retrieve
  test_clicks_same_campaign_before  ... and clicked the same campaign in train (upper bound for a
                            "show again what they clicked" signal)

usage: python scripts/uid_repeats.py data/attribution_full.parquet
"""
import json
import sys

import numpy as np
import pandas as pd

df = pd.read_parquet(sys.argv[1], columns=["timestamp", "uid", "campaign", "click"])
df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
out = {"rows_total": len(df), "days_total": float((df.timestamp.max() - df.timestamp.min()) / 86400)}
for n in [2_000_000, 5_000_000, 10_000_000, len(df)]:
    if n > len(df):
        continue
    d = df.iloc[:n]
    t = d.timestamp
    v0, t0 = t.quantile(0.7), t.quantile(0.8)
    train, test = d[t < v0], d[t >= t0]
    tr_clicks = train[train.click == 1]
    te_clicks = test[test.click == 1]
    clicked_users = set(tr_clicks.uid.to_numpy())
    pairs = set(zip(tr_clicks.uid.to_numpy(), tr_clicks.campaign.to_numpy()))
    out[f"first_{n}"] = {
        "days": float((t.max() - t.min()) / 86400),
        "unique_uid_share": float(d.uid.nunique() / n),
        "rows_with_earlier_row": float(d.uid.duplicated().mean()),
        "test_clicks": int(len(te_clicks)),
        "test_clicks_with_train_click": float(np.mean([u in clicked_users for u in te_clicks.uid.to_numpy()])),
        "test_clicks_same_campaign_before": float(np.mean([(u, c) in pairs for u, c in
                                                           zip(te_clicks.uid.to_numpy(), te_clicks.campaign.to_numpy())])),
    }
print(json.dumps(out, indent=1))
