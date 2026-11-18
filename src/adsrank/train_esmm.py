"""Train ESMM (or the same network with a click-only CVR loss) on Criteo Attribution.

usage:
  python -m adsrank.train_esmm --data data/attribution_2000000.parquet --cvr-loss esmm
  python -m adsrank.train_esmm --data data/attribution_2000000.parquet --cvr-loss clicked
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from adsrank.attribution import build_features, load, time_split
from adsrank.esmm import train_esmm


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--data", required=True)
    a.add_argument("--cvr-loss", choices=["esmm", "clicked"], default="esmm")
    a.add_argument("--epochs", type=int, default=5)
    a.add_argument("--batch", type=int, default=4096)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--emb-dim", type=int, default=16)
    a.add_argument("--threads", type=int, default=4)
    a.add_argument("--mlflow", default=None)
    a.add_argument("--out", default=None)
    a = a.parse_args()

    df = load(a.data)
    feats = build_features(df, time_split(df))
    del df
    model, r = train_esmm(feats, cvr_loss=a.cvr_loss, epochs=a.epochs, batch=a.batch, lr=a.lr,
                          emb_dim=a.emb_dim, threads=a.threads)
    name = "esmm" if a.cvr_loss == "esmm" else "nn_clicked_cvr"
    print(f"\n== Criteo Attribution, {name} (test) ==")
    for k, v in r.items():
        print(f"{k:22s} {v:.4f}")
    out = Path(a.out or f"results/attribution_{name}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"data": a.data, "cvr_loss": a.cvr_loss} | r, indent=1))
    print(f"\nsaved {out}")

    if a.mlflow:
        import mlflow
        mlflow.set_tracking_uri(a.mlflow)
        mlflow.set_experiment("criteo-attribution")
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "epochs": a.epochs, "batch": a.batch, "lr": a.lr,
                               "emb_dim": a.emb_dim, "rows": len(feats)})
            mlflow.log_metrics({k: float(v) for k, v in r.items()})
            mlflow.pytorch.log_model(model, name="model")
        print("logged to MLflow experiment 'criteo-attribution'")


if __name__ == "__main__":
    main()
