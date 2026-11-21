"""Data-parallel training (torch.distributed DDP) for the DeepFM CTR model.

Coupang's platform trains on multiple A100s with DDP / FSDP. This repository has no GPUs, so the
same code path runs on CPU processes with the gloo backend: what is verified here is the
distributed logic (sharding, gradient all-reduce, identical replicas, evaluation on rank 0),
not GPU speed-up. Switching to GPUs is `backend="nccl"` and `model.to(rank)`.

How the batches are split: every rank draws the same random permutation (same seed), cuts the
same global batch of `batch` rows, and takes its own `batch / world` slice. DDP averages the
gradients across ranks, so one step on W ranks equals one step of a single process on the whole
global batch (exactly, up to float summation order, when dropout is off). `tests/test_ddp.py`
checks that equivalence and that all replicas hold identical weights after training.

usage: python -m adsrank.ddp --data data/attribution_2000000.parquet --world 2
"""
import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from adsrank.deepctr import DeepFM, predict_logits
from adsrank.optim import adam


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _batches(n: int, batch: int, epoch: int, seed: int):
    g = torch.Generator().manual_seed(seed + epoch)
    perm = torch.randperm(n, generator=g)
    for i in range(0, n - batch + 1, batch):                 # drop the last partial batch
        yield perm[i:i + batch]


def train_single(vs, n_num, train, epochs=1, batch=4096, lr=1e-3, dropout=0.1, seed=0, threads=4):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    model = DeepFM(vs, n_num, dropout=dropout)
    opt, bce = adam(model.parameters(), lr), nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        for idx in _batches(len(train[0]), batch, ep, seed):
            loss = bce(model(train[0][idx], train[1][idx]), train[2][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def _worker(rank, world, port, vs, n_num, train, epochs, batch, lr, dropout, seed, threads, out):
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    torch.set_num_threads(threads)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    torch.manual_seed(seed)                                   # same initial weights everywhere
    model = DDP(DeepFM(vs, n_num, dropout=dropout))           # DDP also broadcasts rank 0's weights
    opt, bce = adam(model.parameters(), lr), nn.BCEWithLogitsLoss()
    per = batch // world
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for idx in _batches(len(train[0]), batch, ep, seed):
            mine = idx[rank * per:(rank + 1) * per]
            loss = bce(model(train[0][mine], train[1][mine]), train[2][mine])
            opt.zero_grad()
            loss.backward()                                    # gradients all-reduced (averaged) here
            opt.step()
    elapsed = time.time() - t0
    flat = torch.cat([p.detach().flatten() for p in model.module.parameters()])
    gathered = [torch.zeros_like(flat) for _ in range(world)]
    dist.all_gather(gathered, flat)
    if rank == 0:
        diff = max(float((g - gathered[0]).abs().max()) for g in gathered)
        torch.save({"state": model.module.state_dict(), "replica_max_abs_diff": diff,
                    "train_seconds": elapsed}, out)
    dist.barrier()
    dist.destroy_process_group()


def train_ddp(vs, n_num, train, world=2, epochs=1, batch=4096, lr=1e-3, dropout=0.1, seed=0,
              threads=1, out=None):
    """Spawns `world` processes; returns (rank-0 model, max weight difference across replicas, seconds)."""
    assert batch % world == 0, "global batch must split evenly across ranks"
    out = out or f"/tmp/adsrank_ddp_{os.getpid()}.pt"
    mp.spawn(_worker, args=(world, free_port(), vs, n_num, train, epochs, batch, lr, dropout, seed,
                            threads, out), nprocs=world, join=True)
    ck = torch.load(out)
    model = DeepFM(vs, n_num, dropout=dropout)
    model.load_state_dict(ck["state"])
    return model.eval(), ck["replica_max_abs_diff"], ck["train_seconds"]


def main():
    from adsrank.attribution import build_features, load, time_split
    from adsrank.esmm import NN_NUMS, to_tensors, vocab_sizes
    from adsrank.metrics import ads_metrics

    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--world", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--threads", type=int, default=4, help="total CPU threads, split across ranks")
    p.add_argument("--out", default="results/ddp.json")
    a = p.parse_args()

    df = load(a.data)
    feats = build_features(df, time_split(df))
    tr, te = feats[feats.split == "train"], feats[feats.split == "test"]
    mean, std = tr[NN_NUMS].mean(), tr[NN_NUMS].std().replace(0, 1)
    Xtr, Xte = to_tensors(tr, mean, std), to_tensors(te, mean, std)
    vs, n_num = vocab_sizes(feats), len(NN_NUMS)
    train = (Xtr[0], Xtr[1], Xtr[2])
    y = Xte[2].numpy()

    def evaluate(m):
        z = predict_logits(m, Xte[0], Xte[1])
        return ads_metrics(y, 1 / (1 + np.exp(-np.clip(z, -30, 30))), "ctr")

    t0 = time.time()
    single = train_single(vs, n_num, train, a.epochs, a.batch, threads=a.threads)
    r = {"single": evaluate(single) | {"train_seconds": time.time() - t0, "processes": 1}}
    m, diff, secs = train_ddp(vs, n_num, train, a.world, a.epochs, a.batch,
                              threads=max(1, a.threads // a.world))
    r["ddp"] = evaluate(m) | {"train_seconds": secs, "processes": a.world,
                              "replica_max_abs_diff": diff}
    r |= {"rows_train": len(tr), "rows_test": len(te), "epochs": a.epochs, "global_batch": a.batch,
          "backend": "gloo (CPU)"}
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
