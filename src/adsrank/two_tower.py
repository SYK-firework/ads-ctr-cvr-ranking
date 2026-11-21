"""Two-tower candidate retrieval on Criteo Attribution.

The ranking models score a short list of candidates (50 per request in the serving benchmark).
Retrieval produces that list: from every eligible ad, pick the few hundred worth ranking, fast
enough to run on every request. A two-tower model makes that a nearest-neighbour search:

    query tower : request context (cat features of the impression, hour, user recency) -> q in R^d
    ad tower    : campaign id (+ ad-side attributes)                                   -> a in R^d
    score       : q . a    -> top-K by dot product (exact here; an ANN index in production)

Training data: clicked impressions of the train window, one positive (query, clicked campaign) each.
Loss: in-batch sampled softmax (every other row's campaign in the batch is a negative) with
logQ correction: popular campaigns appear as in-batch negatives more often, so log(sampling
probability) is subtracted from their logits; without it the model learns to bury popular ads.

Which cat features describe the ad, not the request? Criteo does not say, and a feature that is
(almost) a function of the campaign would leak the answer into the query tower. `ad_side_features`
measures it: the share of rows where the feature equals its campaign's most common value.
Features above `purity` go to the ad tower (as the campaign's modal value), the rest to the query.

Evaluation on test-window clicks: recall@K = share of clicks whose campaign is in the top K,
against two baselines: the K most clicked campaigns of the train window (popularity), and a
lookup of the most clicked campaigns among train clicks with exactly the same query features
(context popularity; falls back to global popularity for contexts never seen in training).

usage: python -m adsrank.two_tower --data data/attribution_2000000.parquet
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from adsrank.attribution import CATS, build_features, load, time_split

CTX_NUMS = ["log_tslc", "no_prev_click"]
KS = (10, 50, 100)


def ad_side_features(df: pd.DataFrame, cats, purity: float = 0.95) -> dict:
    """Share of rows equal to the campaign's modal value, per cat feature."""
    out = {}
    for c in cats:
        modal = df.groupby("campaign")[c].agg(lambda s: s.value_counts().index[0])
        out[c] = float((df[c].to_numpy() == df["campaign"].map(modal).to_numpy()).mean())
    return out


def recall_at_k(scores: np.ndarray, truth: np.ndarray, ks=KS) -> dict:
    """scores: (n_queries, n_items); truth: item index per query. Ties broken by item index."""
    # rank of the true item = number of items scored strictly higher
    true_s = scores[np.arange(len(truth)), truth][:, None]
    rank = (scores > true_s).sum(axis=1)
    return {f"recall@{k}": float((rank < k).mean()) for k in ks} | {"mrr": float((1 / (rank + 1)).mean())}


def popularity_scores(train_items: np.ndarray, n_items: int, n_queries: int) -> np.ndarray:
    pop = np.bincount(train_items, minlength=n_items).astype(np.float64)
    return np.broadcast_to(pop, (n_queries, n_items))


def build(feats: pd.DataFrame, raw: pd.DataFrame, purity: float = 0.95):
    """Clicked rows only; campaign code 0 (OOV in train) cannot be retrieved and is dropped."""
    clicked = feats[(feats.click == 1) & (feats.campaign.cat.codes > 0)]
    others = [c for c in CATS if c != "campaign"]
    tr_raw = raw.loc[clicked.index[clicked.split == "train"]]
    purity_by_feature = ad_side_features(tr_raw, others)
    ad_side = [c for c in others if purity_by_feature[c] >= purity]
    query_side = [c for c in others if c not in ad_side]
    return clicked, query_side, ad_side, purity_by_feature


def train_eval(feats, raw, epochs: int = 3, batch: int = 2048, dim: int = 32, lr: float = 2e-3,
               temperature: float = 0.05, threads: int = 4, max_eval: int = 50_000, log=print) -> dict:
    import torch
    from torch import nn

    torch.manual_seed(0)
    torch.set_num_threads(threads)
    clicked, q_cats, a_cats, purity_by_feature = build(feats, raw)
    log(f"ad-side features (purity >= 0.95): {a_cats or 'none'}; query features: {q_cats}")
    tr, te = clicked[clicked.split == "train"], clicked[clicked.split == "test"]
    n_items = len(feats.campaign.cat.categories)
    item_tr = tr.campaign.cat.codes.to_numpy().astype(np.int64)
    item_te = te.campaign.cat.codes.to_numpy().astype(np.int64)

    q_sizes = [len(feats[c].cat.categories) for c in q_cats] + [24]
    def query_x(d):
        cat = np.stack([d[c].cat.codes.to_numpy() for c in q_cats] + [d["hour"].to_numpy()], axis=1)
        num = d[CTX_NUMS].to_numpy(np.float32)
        return torch.as_tensor(cat.astype(np.int64)), torch.as_tensor(num)
    # ad-side attributes: each campaign's modal code in the train window (0 if unseen)
    a_sizes = [len(feats[c].cat.categories) for c in a_cats]
    ad_attr = np.zeros((n_items, len(a_cats)), dtype=np.int64)
    for j, c in enumerate(a_cats):
        m = tr.groupby(tr.campaign.cat.codes)[c].agg(lambda s: s.cat.codes.value_counts().index[0])
        ad_attr[m.index.to_numpy(), j] = m.to_numpy()
    ad_attr = torch.as_tensor(ad_attr)

    num_mean = tr[CTX_NUMS].to_numpy(np.float32).mean(0)
    num_std = tr[CTX_NUMS].to_numpy(np.float32).std(0) + 1e-6

    class Tower(nn.Module):
        def __init__(self, sizes, n_num, emb=16):
            super().__init__()
            self.embs = nn.ModuleList(nn.Embedding(s, emb) for s in sizes)
            d_in = emb * len(sizes) + n_num
            self.mlp = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, dim))

        def forward(self, cat, num=None):
            x = [e(cat[:, i]) for i, e in enumerate(self.embs)]
            if num is not None:
                x.append(num)
            return nn.functional.normalize(self.mlp(torch.cat(x, 1)), dim=1)

    query = Tower(q_sizes, len(CTX_NUMS))
    ad = Tower([n_items] + a_sizes, 0)
    params = list(query.parameters()) + list(ad.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    def ad_vectors(items):
        return ad(torch.cat([items[:, None], ad_attr[items]], 1))

    qx_cat, qx_num = query_x(tr)
    qx_num = (qx_num - torch.as_tensor(num_mean)) / torch.as_tensor(num_std)
    items = torch.as_tensor(item_tr)
    log_q = torch.as_tensor(np.log(np.bincount(item_tr, minlength=n_items) / len(item_tr) + 1e-12),
                            dtype=torch.float32)
    t0 = time.time()
    n = len(items)
    for ep in range(epochs):
        perm, total = torch.randperm(n), 0.0
        for i in range(0, n - batch + 1, batch):
            idx = perm[i:i + batch]
            q = query(qx_cat[idx], qx_num[idx])
            it = items[idx]
            a = ad_vectors(it)
            logits = q @ a.T / temperature - log_q[it][None, :]          # logQ correction
            same = it[:, None] == it[None, :]                               # duplicates are not negatives
            logits = logits.masked_fill(same & ~torch.eye(len(it), dtype=torch.bool), -1e9)
            loss = nn.functional.cross_entropy(logits, torch.arange(len(it)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        log(f"two-tower epoch {ep + 1}: train loss {total / max(1, n // batch):.4f} ({time.time() - t0:.0f}s)")
    train_s = time.time() - t0

    # evaluation: test-window clicks, scored against every campaign in the vocabulary
    te_s = te.iloc[:max_eval] if len(te) > max_eval else te
    truth = item_te[:len(te_s)]
    with torch.no_grad():
        all_items = torch.arange(1, n_items)                                # code 0 = OOV, not retrievable
        A = ad_vectors(all_items).numpy()
        c, nm = query_x(te_s)
        Q = query(c, (nm - torch.as_tensor(num_mean)) / torch.as_tensor(num_std)).numpy()
        t1 = time.perf_counter()
        S = Q @ A.T
        score_ms = (time.perf_counter() - t1) * 1000 / len(Q)
    truth_i = truth - 1
    res = {"two_tower": recall_at_k(S, truth_i),
           "popularity": recall_at_k(popularity_scores(item_tr - 1, n_items - 1, len(Q)), truth_i)}
    # context-popularity reference: most clicked campaigns among train clicks with the same cat1..cat9
    key_cols = [c for c in q_cats]
    tr_key = pd.MultiIndex.from_frame(tr[key_cols].apply(lambda s: s.cat.codes)) if key_cols else None
    if key_cols:
        ctx = pd.DataFrame({"key": list(tr_key), "item": item_tr - 1})
        top = ctx.groupby(["key", "item"]).size()
        te_key = list(pd.MultiIndex.from_frame(te_s[key_cols].apply(lambda s: s.cat.codes)))
        pop = np.bincount(item_tr - 1, minlength=n_items - 1).astype(np.float64)
        Sc = np.tile(pop / pop.sum() * 1e-3, (len(te_key), 1))             # popularity breaks ties
        lookup = {}
        for (k, i), v in top.items():
            lookup.setdefault(k, []).append((i, v))
        for r, k in enumerate(te_key):
            for i, v in lookup.get(k, []):
                Sc[r, i] += v
        res["context_popularity"] = recall_at_k(Sc, truth_i)
        res["context_seen_share"] = float(np.mean([k in lookup for k in te_key]))
    return res | {
        "n_campaigns": int(n_items - 1), "train_clicks": int(n), "test_clicks_evaluated": int(len(Q)),
        "query_features": q_cats + ["hour"] + CTX_NUMS, "ad_features": ["campaign"] + a_cats,
        "purity_by_feature": purity_by_feature, "dim": dim, "epochs": epochs,
        "train_seconds": round(train_s, 1), "exact_scoring_ms_per_query": round(score_ms, 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="results/two_tower.json")
    a = p.parse_args()
    raw = load(a.data)
    feats = build_features(raw, time_split(raw))
    r = train_eval(feats, raw, epochs=a.epochs, threads=a.threads)
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(f"\n{'method':20s} " + " ".join(f"{k:>10s}" for k in [f'recall@{k}' for k in KS] + ['mrr']))
    for m in ("popularity", "context_popularity", "two_tower"):
        if m in r:
            print(f"{m:20s} " + " ".join(f"{r[m][k]:>10.4f}" for k in [f'recall@{k}' for k in KS] + ['mrr']))
    print(json.dumps({k: v for k, v in r.items() if k not in ("two_tower", "popularity", "context_popularity")}))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
