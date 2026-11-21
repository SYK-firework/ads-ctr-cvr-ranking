"""Online feature store (Feast + Redis) for ad-side features.

Until now every request carried all features, including the ad's own attributes. In production
the ad server sends ids and context; ad-side features come from a low-latency online store that a
batch job keeps up to date (Coupang's platform uses Feast with offline and online stores).

Here the ad-side feature is the campaign's `cpo` (target cost per order, which also enters the
expected value cpo x pCTR x pCVR), plus campaign impression / click counts:

  build  : campaign features from the train window -> feature_repo/data/campaign_features.parquet
           (offline source; event time = the campaign's last train impression)
  feast apply / feast materialize : offline parquet -> Redis online store
  bench  : for test-window sample requests with cpo removed, look up cpo by campaign in Redis and
           measure lookup latency per request, coverage (campaigns found), and training-serving
           skew: how often the stored value differs from the value logged with the impression, and
           what that does to the served scores.

usage: python -m adsrank.feature_store build --data data/attribution_2000000.parquet
       (cd feature_repo && feast apply && feast materialize 2025-12-01T00:00:00 2026-03-01T00:00:00)
       python -m adsrank.feature_store bench --model serving/model
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2] / "feature_repo"
SOURCE = REPO / "data" / "campaign_features.parquet"
BASE = pd.Timestamp("2026-01-01", tz="UTC")          # Criteo timestamps are seconds from the start
FEATURES = ["campaign_features:cpo", "campaign_features:campaign_impressions",
            "campaign_features:campaign_clicks"]


def build_campaign_features(raw: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """Latest cpo and train-window counts per campaign (train window only: no leakage)."""
    tr = raw.loc[split == "train", ["campaign", "timestamp", "cpo", "click"]].sort_values("timestamp")
    g = tr.groupby("campaign")
    out = pd.DataFrame({
        "cpo": g["cpo"].last().astype("float32"),
        "campaign_impressions": g.size().astype("int64"),
        "campaign_clicks": g["click"].sum().astype("int64"),
        "last_ts": g["timestamp"].last(),
        "cpo_distinct_values": g["cpo"].nunique().astype("int64"),
    }).reset_index()
    out["campaign"] = out["campaign"].astype("int64")
    out["event_timestamp"] = BASE + pd.to_timedelta(out.pop("last_ts"), unit="s")
    out["created"] = out["event_timestamp"]
    return out


class OnlineCampaignFeatures:
    """Fills `cpo` in request rows from the online store, keyed by campaign."""

    def __init__(self, repo: str | Path = REPO):
        from feast import FeatureStore
        self.store = FeatureStore(repo_path=str(repo))

    def lookup(self, campaigns) -> dict:
        uniq = sorted({int(c) for c in campaigns})
        d = self.store.get_online_features(features=FEATURES[:1],
                                           entity_rows=[{"campaign": c} for c in uniq]).to_dict()
        return {c: v for c, v in zip(d["campaign"], d["cpo"])}

    def enrich(self, rows: list[dict], default_cpo: float = 0.0) -> tuple[list[dict], int]:
        found = self.lookup(r["campaign"] for r in rows)
        out, missing = [], 0
        for r in rows:
            v = found.get(int(r["campaign"]))
            if v is None:
                missing += 1
                v = default_cpo
            out.append({**{k: x for k, x in r.items() if k != "cpo"}, "cpo": float(v)})
        return out, missing


def cmd_build(a):
    from adsrank.attribution import load, time_split
    raw = load(a.data)
    f = build_campaign_features(raw, time_split(raw))
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    f.drop(columns=["cpo_distinct_values"]).to_parquet(SOURCE, index=False)
    print(json.dumps({"campaigns": len(f), "source": str(SOURCE),
                      "campaigns_with_more_than_one_cpo_in_train": int((f.cpo_distinct_values > 1).sum()),
                      "max_cpo_values_per_campaign": int(f.cpo_distinct_values.max())}))


def cmd_bench(a):
    from adsrank.serving.scorer import Scorer
    rows = json.loads((Path(a.model) / "sample_requests.json").read_text())
    scorer = Scorer(a.model)
    fs = OnlineCampaignFeatures()
    rng = np.random.default_rng(0)
    lat, missing_total, n_total = [], 0, 0
    for _ in range(a.requests):                                     # latency: one lookup per request
        req = [rows[i] for i in rng.integers(0, len(rows), a.batch)]
        t = time.perf_counter()
        _, missing = fs.enrich(req)
        lat.append((time.perf_counter() - t) * 1000)
        missing_total += missing
        n_total += len(req)
    # skew: stored cpo vs the cpo logged with each test impression, and its effect on the scores
    enriched, missing = fs.enrich(rows)
    logged = np.array([r["cpo"] for r in rows])
    stored = np.array([r["cpo"] for r in enriched])
    s0, s1 = scorer.score(rows), scorer.score(enriched)
    differ = ~np.isclose(stored, logged, rtol=1e-5, atol=1e-7)
    report = {
        "requests": a.requests, "candidates_per_request": a.batch,
        "lookup_ms_p50": float(np.percentile(lat, 50)), "lookup_ms_p99": float(np.percentile(lat, 99)),
        "campaign_coverage": 1 - missing_total / n_total,
        "skew_rows": len(rows), "skew_missing_campaigns": int(missing),
        "share_cpo_differs_from_logged": float(differ.mean()),
        "max_abs_pctr_change": float(np.abs(s1["p_ctr"] - s0["p_ctr"]).max()),
        "mean_pctr_ratio_store_vs_logged": float(s1["p_ctr"].mean() / s0["p_ctr"].mean()),
        "mean_expected_value_ratio_store_vs_logged": float(s1["expected_value"].mean() / s0["expected_value"].mean()),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/feature_store.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", required=True)
    c = sub.add_parser("bench")
    c.add_argument("--model", default="serving/model")
    c.add_argument("--requests", type=int, default=500)
    c.add_argument("--batch", type=int, default=50)
    a = p.parse_args()
    {"build": cmd_build, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
