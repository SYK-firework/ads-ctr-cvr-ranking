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
from adsrank.delayed import (ExpMixtureDelay, empirical_survival, known_delay_loss, mixture_dfm_loss,
                              observed_labels)
from adsrank.esmm import NN_NUMS, to_tensors, vocab_sizes
from adsrank.metrics import ads_metrics
from adsrank.optim import adam


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def train_custom(model, loss_fn, extra_params, x_cat, x_num, epochs: int = 5, batch: int = 4096,
                 lr: float = 1e-3, name: str = "", log=print, describe=lambda: ""):
    """Generic minibatch loop: loss_fn(logits, idx) -> scalar."""
    opt = adam(list(model.parameters()) + list(extra_params), lr=lr)
    n, t0 = len(x_cat), time.time()
    for ep in range(epochs):
        model.train()
        perm, total = torch.randperm(n), 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = loss_fn(model(x_cat[idx], x_num[idx]), idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        log(f"{name} epoch {ep + 1}: nll {total / n:.5f}{describe()} ({time.time() - t0:.0f}s)")
    return model.eval()


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
        p = sigmoid(predict_logits(m, Xte[0], Xte[1]))
        report[name] = ads_metrics(y_te, p, "cvr") | {"train_clicks": int(mask.sum())}
        return p

    all_rows = torch.ones(len(tr), dtype=torch.bool)
    fit_eval("oracle", all_rows, torch.as_tensor(y_true))
    p_naive = fit_eval("naive", all_rows, torch.as_tensor(y_obs))
    wait = torch.as_tensor(elapsed >= a.wait_hours * 3600)
    fit_eval(f"wait_{a.wait_hours:g}h", wait, torch.as_tensor(y_obs))

    Y, E, D = torch.as_tensor(y_obs), torch.as_tensor(elapsed), torch.as_tensor(delay)

    def eval_p(model):
        return ads_metrics(y_te, sigmoid(predict_logits(model, Xte[0], Xte[1])), "cvr")

    # DFM, delay learned jointly: 1 exponential (Chapelle, 2014) and a 2-component mixture
    for k, name in ((1, "dfm_exp"), (2, "dfm_mix2")):
        dm = ExpMixtureDelay(k)
        m = train_custom(DCNv2(vs, len(NN_NUMS)),
                         lambda z, idx, dm=dm: mixture_dfm_loss(z, dm, Y[idx], E[idx], D[idx]),
                         dm.parameters(), Xtr[0], Xtr[1], a.epochs, name=name,
                         describe=lambda dm=dm: f", mean delay {dm.mean_hours():.1f}h")
        report[name] = eval_p(m) | {"train_clicks": len(tr), "fitted_mean_delay_hours": dm.mean_hours()}

    # DFM with the delay distribution fixed from matured history. This extract has no separate
    # earlier period, so the complete delays of the training clicks stand in for that history
    # (only the delay distribution is used, not any click's label).
    log_s = torch.as_tensor(np.log(np.maximum(empirical_survival(delay[y_true == 1], elapsed), 1e-6)))
    m = train_custom(DCNv2(vs, len(NN_NUMS)), lambda z, idx: known_delay_loss(z, Y[idx], log_s[idx]),
                     [], Xtr[0], Xtr[1], a.epochs, name="dfm_hist")
    report["dfm_hist"] = eval_p(m) | {"train_clicks": len(tr)}

    # Simplest practical fix: keep the naive model's ranking and rescale it by the share of
    # conversions expected to be visible at training time, E[F(elapsed)], from the same history.
    visible = float(np.mean(1 - np.exp(log_s.numpy())))
    report["naive_rescaled"] = ads_metrics(y_te, np.clip(p_naive / visible, 1e-6, 1 - 1e-6), "cvr") | {
        "train_clicks": len(tr), "expected_visible_share": visible}

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
            print(f"{k:15s} AUC {v['cvr_auc']:.4f}  LogLoss {v['cvr_logloss']:.4f}  NE {v['cvr_ne']:.4f}  "
                  f"calibration {v['cvr_calibration']:.3f}  train clicks {v['train_clicks']:,}")
    print("\nsaved results/delayed_feedback.json")


if __name__ == "__main__":
    main()
