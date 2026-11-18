"""DeepFM / DCN-v2 CTR models on Criteo DAC (output of adsrank.prep), same split as train_gbdt.

Inputs: 26 categorical fields as vocabulary indices (0 = OOV) and 13 log-transformed integers,
standardized on the training window.

usage: python -m adsrank.train_dac_deep --data data/criteo_2000000.parquet --model dcnv2
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adsrank import CAT, NUM
from adsrank.deepctr import MODELS, predict_logits, train_binary
from adsrank.metrics import ads_metrics


def tensors(df: pd.DataFrame, mean, std):
    return (torch.as_tensor(df[CAT].to_numpy(np.int64)),
            torch.as_tensor(((df[NUM] - mean) / std).to_numpy(np.float32)),
            torch.as_tensor(df["label"].to_numpy(np.float32)))


def run(df: pd.DataFrame, vocab: dict, model_name: str, epochs: int = 3, batch: int = 4096,
        lr: float = 1e-3, emb_dim: int = 16, threads: int = 4, log=print) -> dict:
    torch.manual_seed(0)
    torch.set_num_threads(threads)
    tr, va, te = (df[df.split == s] for s in ("train", "valid", "test"))
    mean, std = tr[NUM].mean(), tr[NUM].std().replace(0, 1)
    Xtr, Xva, Xte = (tensors(d, mean, std) for d in (tr, va, te))
    t0 = time.time()
    model = MODELS[model_name]([vocab[c] for c in CAT], len(NUM), emb_dim=emb_dim)
    model = train_binary(model, Xtr, Xva, epochs, batch, lr, log=log, name=f"DAC {model_name}")
    z = np.clip(predict_logits(model, Xte[0], Xte[1]), -30, 30)
    return ads_metrics(Xte[2].numpy(), 1 / (1 + np.exp(-z)), "test") | {"train_seconds": time.time() - t0}


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--data", required=True)
    a.add_argument("--model", choices=list(MODELS), required=True)
    a.add_argument("--epochs", type=int, default=3)
    a.add_argument("--threads", type=int, default=4)
    a = a.parse_args()
    df = pd.read_parquet(a.data, columns=["label", "split"] + NUM + CAT)
    vocab = json.loads(Path(a.data.replace(".parquet", "_vocab.json")).read_text())
    r = run(df, vocab, a.model, epochs=a.epochs, threads=a.threads)
    print(f"\n== Criteo DAC, {a.model} (test) ==")
    for k, v in r.items():
        print(f"{k:20s} {v:.4f}")
    out = Path(f"results/dac_{a.model}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": a.model, "data": a.data} | r, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
