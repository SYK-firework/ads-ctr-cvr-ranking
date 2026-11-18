"""ESMM: Entire Space Multi-task Model (Ma et al., SIGIR 2018) for CTR + CVR.

Problem with separate models:
  the CVR model is trained on clicked impressions only, but at ranking time it is applied to
  every impression (sample selection bias). Clicks are also sparse, so CVR data is small.

ESMM:
  shared embeddings -> CTR tower -> pCTR
                    -> CVR tower -> pCVR
  pCTCVR = pCTR x pCVR
  loss = BCE(pCTR, click) + BCE(pCTCVR, conversion)     both over ALL impressions
The CVR tower never gets a direct, click-only loss; it is learned through pCTCVR on the
entire impression space.

cvr_loss="clicked" trains the same network the conventional way (CVR loss on clicked rows
only) so the two objectives can be compared with everything else held fixed.
"""
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from adsrank.attribution import CATS
from adsrank.metrics import ads_metrics

NN_CATS = CATS + ["hour"]
NN_NUMS = ["cpo", "log_tslc", "no_prev_click"]


class ESMM(nn.Module):
    def __init__(self, vocab_sizes: list[int], n_num: int, emb_dim: int = 16,
                 hidden: tuple[int, ...] = (128, 64), dropout: float = 0.1):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(v, emb_dim) for v in vocab_sizes)   # shared
        d_in = emb_dim * len(vocab_sizes) + n_num

        def tower() -> nn.Sequential:
            layers, d = [], d_in
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
                d = h
            return nn.Sequential(*layers, nn.Linear(d, 1))

        self.ctr_tower, self.cvr_tower = tower(), tower()

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor):
        z = torch.cat([e(x_cat[:, i]) for i, e in enumerate(self.embs)] + [x_num], dim=1)
        p_ctr = torch.sigmoid(self.ctr_tower(z).squeeze(1))
        p_cvr = torch.sigmoid(self.cvr_tower(z).squeeze(1))
        return p_ctr, p_cvr


def to_tensors(feats: pd.DataFrame, mean: pd.Series, std: pd.Series):
    x_cat = torch.as_tensor(np.stack([feats[c].cat.codes.to_numpy() if hasattr(feats[c], "cat")
                                      else feats[c].to_numpy() for c in NN_CATS], 1).astype(np.int64))
    x_num = torch.as_tensor(((feats[NN_NUMS] - mean) / std).to_numpy(np.float32))
    y_click = torch.as_tensor(feats["click"].to_numpy(np.float32))
    y_conv = torch.as_tensor(feats["attribution"].to_numpy(np.float32))
    return x_cat, x_num, y_click, y_conv


def vocab_sizes(feats: pd.DataFrame) -> list[int]:
    return [len(feats[c].cat.categories) if hasattr(feats[c], "cat") else 24 for c in NN_CATS]


def predict(model: ESMM, x_cat, x_num, batch: int = 65536):
    model.eval()
    out_ctr, out_cvr = [], []
    with torch.no_grad():
        for i in range(0, len(x_cat), batch):
            a, b = model(x_cat[i:i + batch], x_num[i:i + batch])
            out_ctr.append(a), out_cvr.append(b)
    return torch.cat(out_ctr).numpy(), torch.cat(out_cvr).numpy()


def loss_fn(p_ctr, p_cvr, y_click, y_conv, cvr_loss: str):
    bce = nn.functional.binary_cross_entropy
    eps = 1e-7
    l_ctr = bce(p_ctr.clamp(eps, 1 - eps), y_click)
    if cvr_loss == "esmm":
        return l_ctr + bce((p_ctr * p_cvr).clamp(eps, 1 - eps), y_conv)
    # conventional: CVR loss on clicked rows only. A 0/1 weight instead of boolean indexing keeps
    # tensor shapes fixed (no data-dependent nonzero/gather kernels) and gives the same value.
    w = y_click
    n_clicked = w.sum()
    if n_clicked == 0:
        return l_ctr
    per_row = bce(p_cvr.clamp(eps, 1 - eps), y_conv, reduction="none")
    return l_ctr + (per_row * w).sum() / n_clicked


def save(model: ESMM, directory: str) -> None:
    """Save weights (state_dict) + config. Reload with load(); no pickled code involved."""
    import json
    from pathlib import Path
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "esmm_state_dict.pt")
    (d / "esmm_config.json").write_text(json.dumps(model.config, indent=1))


def load(directory: str) -> ESMM:
    import json
    from pathlib import Path
    cfg = json.loads((Path(directory) / "esmm_config.json").read_text())
    model = ESMM(cfg["vocab_sizes"], cfg["n_num"], emb_dim=cfg["emb_dim"])
    model.load_state_dict(torch.load(Path(directory) / "esmm_state_dict.pt", weights_only=True))
    model.config = cfg
    return model.eval()


def evaluate(p_ctr, p_cvr, y_click, y_conv) -> dict[str, float]:
    clicked = y_click == 1
    return (ads_metrics(y_click, p_ctr, "ctr")
            | ads_metrics(y_conv[clicked], p_cvr[clicked], "cvr")
            | ads_metrics(y_conv, p_ctr * p_cvr, "ctcvr"))


def train_esmm(feats: pd.DataFrame, cvr_loss: str = "esmm", epochs: int = 5, batch: int = 4096,
               lr: float = 1e-3, emb_dim: int = 16, patience: int = 1, seed: int = 0,
               threads: int = 4, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    tr, va, te = (feats[feats.split == s] for s in ("train", "valid", "test"))
    mean, std = tr[NN_NUMS].mean(), tr[NN_NUMS].std().replace(0, 1)
    Xtr, Xva, Xte = (to_tensors(d, mean, std) for d in (tr, va, te))

    model = ESMM(vocab_sizes(feats), len(NN_NUMS), emb_dim=emb_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad, t0 = np.inf, None, 0, time.time()
    n = len(Xtr[0])
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            p_ctr, p_cvr = model(Xtr[0][idx], Xtr[1][idx])
            loss = loss_fn(p_ctr, p_cvr, Xtr[2][idx], Xtr[3][idx], cvr_loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        vc, vv = predict(model, Xva[0], Xva[1])
        vm = evaluate(vc, vv, Xva[2].numpy(), Xva[3].numpy())
        score = vm["ctr_logloss"] + vm["ctcvr_logloss"]     # model selection on the entire space
        log(f"epoch {ep + 1}: train_loss {total / n:.4f} | valid ctr_auc {vm['ctr_auc']:.4f} "
            f"cvr_auc {vm['cvr_auc']:.4f} ctcvr_auc {vm['ctcvr_auc']:.4f} ({time.time() - t0:.0f}s)")
        if score < best - 1e-5:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad > patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    model.config = {"vocab_sizes": vocab_sizes(feats), "n_num": len(NN_NUMS), "emb_dim": emb_dim,
                    "cats": NN_CATS, "nums": NN_NUMS,
                    "num_mean": mean.tolist(), "num_std": std.tolist()}
    tc, tv = predict(model, Xte[0], Xte[1])
    r = evaluate(tc, tv, Xte[2].numpy(), Xte[3].numpy()) | {"train_seconds": time.time() - t0,
                                                             "epochs_run": ep + 1}
    return model, r
