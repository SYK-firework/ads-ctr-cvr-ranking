"""Train DeepFM or DCN-v2 CTR + CVR models on Criteo Attribution (same split and metrics as XGBoost).

usage: python -m adsrank.train_deepctr --data data/attribution_2000000.parquet --model dcnv2
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adsrank.attribution import build_features, load, time_split
from adsrank.deepctr import MODELS, predict_logits, train_binary
from adsrank.esmm import NN_NUMS, to_tensors, vocab_sizes
from adsrank.metrics import ads_metrics


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def run(feats, model_name: str, epochs: int = 5, batch: int = 4096, lr: float = 1e-3,
        emb_dim: int = 16, threads: int = 4, log=print) -> dict:
    torch.manual_seed(0)
    torch.set_num_threads(threads)
    tr, va, te = (feats[feats.split == s] for s in ("train", "valid", "test"))
    mean, std = tr[NN_NUMS].mean(), tr[NN_NUMS].std().replace(0, 1)
    Xtr, Xva, Xte = (to_tensors(d, mean, std) for d in (tr, va, te))
    vs, cls = vocab_sizes(feats), MODELS[model_name]
    t0 = time.time()

    ctr = train_binary(cls(vs, len(NN_NUMS), emb_dim=emb_dim), (Xtr[0], Xtr[1], Xtr[2]),
                       (Xva[0], Xva[1], Xva[2]), epochs, batch, lr, log=log, name=f"{model_name} CTR")
    ctr_tr, ctr_va = Xtr[2] == 1, Xva[2] == 1          # CVR: clicked rows only
    cvr = train_binary(cls(vs, len(NN_NUMS), emb_dim=emb_dim),
                       (Xtr[0][ctr_tr], Xtr[1][ctr_tr], Xtr[3][ctr_tr]),
                       (Xva[0][ctr_va], Xva[1][ctr_va], Xva[3][ctr_va]), epochs, batch, lr,
                       log=log, name=f"{model_name} CVR")

    p_ctr = sigmoid(predict_logits(ctr, Xte[0], Xte[1]))
    p_cvr = sigmoid(predict_logits(cvr, Xte[0], Xte[1]))
    y_click, y_conv = Xte[2].numpy(), Xte[3].numpy()
    clicked = y_click == 1
    return (ads_metrics(y_click, p_ctr, "ctr") | ads_metrics(y_conv[clicked], p_cvr[clicked], "cvr")
            | ads_metrics(y_conv, p_ctr * p_cvr, "ctcvr") | {"train_seconds": time.time() - t0})


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--data", required=True)
    a.add_argument("--model", choices=list(MODELS), required=True)
    a.add_argument("--epochs", type=int, default=5)
    a.add_argument("--threads", type=int, default=4)
    a.add_argument("--mlflow", default=None)
    a = a.parse_args()
    df = load(a.data)
    feats = build_features(df, time_split(df))
    del df
    r = run(feats, a.model, epochs=a.epochs, threads=a.threads)
    print(f"\n== Criteo Attribution, {a.model} (test) ==")
    for k, v in r.items():
        print(f"{k:22s} {v:.4f}")
    out = Path(f"results/attribution_{a.model}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": a.model} | r, indent=1))
    print(f"saved {out}")
    if a.mlflow:
        import mlflow
        mlflow.set_tracking_uri(a.mlflow)
        mlflow.set_experiment("criteo-attribution")
        with mlflow.start_run(run_name=a.model):
            mlflow.log_params({"model": a.model, "epochs": a.epochs, "rows": len(feats)})
            mlflow.log_metrics({k: float(v) for k, v in r.items()})
        print("logged to MLflow experiment 'criteo-attribution'")


if __name__ == "__main__":
    main()
