"""Delayed-feedback CVR study on Criteo Attribution.

The training cutoff is the end of the training window: conversions that happened after it are
hidden from training. All models share the DCN-v2 network and are evaluated on test clicks
against complete labels.

usage: python -m adsrank.run_delayed --data data/attribution_2000000.parquet
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adsrank.attribution import build_features, load, time_split
from adsrank.deepctr import DCNv2, predict_logits, train_binary
from adsrank.delayed import dfm_loss, observed_labels
from adsrank.esmm import NN_NUMS, to_tensors, vocab_sizes
from adsrank.metrics import ads_metrics


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def train_dfm(model, x_cat, x_num, y_obs, elapsed, delay, init_rate: float, epochs: int = 5,
              batch: int = 4096, lr: float = 1e-3, log=print):
    log_lambda = torch.nn.Parameter(torch.tensor(float(np.log(init_rate))))
    opt = torch.optim.Adam(list(model.parameters()) + [log_lambda], lr=lr)
    n, t0 = len(x_cat), time.time()
    for ep in range(epochs):
        model.train()
        perm, total = torch.randperm(n), 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = dfm_loss(model(x_cat[idx], x_num[idx]), log_lambda, y_obs[idx], elapsed[idx], delay[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        log(f"dfm epoch {ep + 1}: nll {total / n:.5f}, mean delay {1 / log_lambda.exp().item() / 3600:.1f}h "
            f"({time.time() - t0:.0f}s)")
    return model.eval(), float(log_lambda.exp().item())


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--data", required=True)
    a.add_argument("--wait-hours", type=float, default=24.0)
    a.add_argument("--epochs", type=int, default=5)
    a.add_argument("--threads", type=int, default=4)
    a = a.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)

    df = load(a.data, extra=["conversion_timestamp"])
    feats = build_features(df, time_split(df))
    tr, va, te = (feats[(feats.split == s) & (feats.click == 1)] for s in ("train", "valid", "test"))
    cutoff = float(df.loc[feats.split == "train", "timestamp"].max())
    y_obs, elapsed, delay = observed_labels(df.loc[tr.index, "timestamp"], df.loc[tr.index, "conversion_timestamp"],
                                            tr.attribution, cutoff)
    y_true = tr.attribution.to_numpy(np.float32)
    d_hours = delay[y_true == 1] / 3600
    stats = {"train_clicks": len(tr), "cutoff_hours": cutoff / 3600,
             "true_cvr": float(y_true.mean()), "observed_cvr": float(y_obs.mean()),
             "share_of_conversions_observed": float(y_obs.sum() / y_true.sum()),
             "delay_hours_median": float(np.median(d_hours)), "delay_hours_p90": float(np.percentile(d_hours, 90)),
             "share_within_1h": float((d_hours <= 1).mean()), "share_within_24h": float((d_hours <= 24).mean())}

    mean, std = tr[NN_NUMS].mean(), tr[NN_NUMS].std().replace(0, 1)
    Xtr, Xva, Xte = (to_tensors(d, mean, std) for d in (tr, va, te))
    vs = vocab_sizes(feats)
    y_te = Xte[3].numpy()
    report = {"stats": stats}

    def fit_eval(name, mask, labels):
        m = train_binary(DCNv2(vs, len(NN_NUMS)), (Xtr[0][mask], Xtr[1][mask], labels[mask]),
                         (Xva[0], Xva[1], Xva[3]), a.epochs, name=name)
        report[name] = ads_metrics(y_te, sigmoid(predict_logits(m, Xte[0], Xte[1])), "cvr") | {"train_clicks": int(mask.sum())}

    all_rows = torch.ones(len(tr), dtype=torch.bool)
    fit_eval("oracle", all_rows, torch.as_tensor(y_true))
    fit_eval("naive", all_rows, torch.as_tensor(y_obs))
    wait = torch.as_tensor(elapsed >= a.wait_hours * 3600)
    fit_eval(f"wait_{a.wait_hours:g}h", wait, torch.as_tensor(y_obs))

    model, lam = train_dfm(DCNv2(vs, len(NN_NUMS)), Xtr[0], Xtr[1], torch.as_tensor(y_obs),
                           torch.as_tensor(elapsed), torch.as_tensor(delay),
                           init_rate=1 / (np.nanmean(delay) + 1), epochs=a.epochs)
    report["dfm"] = ads_metrics(y_te, sigmoid(predict_logits(model, Xte[0], Xte[1])), "cvr") | {
        "train_clicks": len(tr), "fitted_mean_delay_hours": 1 / lam / 3600}

    Path("results").mkdir(exist_ok=True)
    Path("results/delayed_feedback.json").write_text(json.dumps(report, indent=1))
    s = report["stats"]
    print(f"\n== Delays (training clicks, cutoff at {s['cutoff_hours']:.1f}h) ==")
    print(f"true CVR {s['true_cvr']:.4f} | observed at cutoff {s['observed_cvr']:.4f} "
          f"({s['share_of_conversions_observed']:.1%} of conversions visible)")
    print(f"delay median {s['delay_hours_median']:.1f}h, p90 {s['delay_hours_p90']:.1f}h, "
          f"within 1h {s['share_within_1h']:.1%}, within 24h {s['share_within_24h']:.1%}")
    print("\n== CVR on test clicks (complete labels) ==")
    for k, v in report.items():
        if k != "stats":
            print(f"{k:10s} AUC {v['cvr_auc']:.4f}  LogLoss {v['cvr_logloss']:.4f}  NE {v['cvr_ne']:.4f}  "
                  f"calibration {v['cvr_calibration']:.3f}  train clicks {v['train_clicks']:,}")
    print("\nsaved results/delayed_feedback.json")


if __name__ == "__main__":
    main()
