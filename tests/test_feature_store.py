"""Campaign feature table for the online store (pandas part; the Feast round trip runs in CI)."""
import numpy as np

from adsrank.attribution import time_split
from adsrank.feature_store import build_campaign_features
from test_attribution import fake_attribution


def test_campaign_features_use_train_window_only():
    df = fake_attribution(20000)
    split = time_split(df)
    f = build_campaign_features(df, split)
    tr = df[split == "train"]
    assert set(f.campaign) == set(tr.campaign)
    c = int(f.campaign.iloc[0])
    last = tr[tr.campaign == c].sort_values("timestamp").cpo.iloc[-1]
    assert np.isclose(f.set_index("campaign").loc[c, "cpo"], last)
    assert f.set_index("campaign").loc[c, "campaign_impressions"] == (tr.campaign == c).sum()
    assert str(f.event_timestamp.dt.tz) == "UTC"
