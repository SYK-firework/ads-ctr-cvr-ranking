"""Deep CTR / CVR models: DeepFM (Guo et al., 2017) and DCN-v2 (Wang et al., 2021).

Both take the same inputs as ESMM: one embedding per categorical field plus standardized numeric
features, and output a single logit. They are trained the same way as the XGBoost baseline
(CTR on all impressions, CVR on clicked impressions) so the results compare directly.

DeepFM : first-order term + factorization machine (pairwise inner products of field embeddings)
         + an MLP on the concatenated embeddings; the three logits are summed.
DCN-v2 : a cross network x_{l+1} = x_0 * (W_l x_l + b_l) + x_l that builds explicit bounded-degree
         feature crosses, run in parallel with an MLP; their outputs are concatenated.
"""
import time

import numpy as np
import torch
from torch import nn

from adsrank.optim import adam


def mlp(d_in: int, hidden: tuple[int, ...], dropout: float) -> tuple[nn.Sequential, int]:
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers), d


def _init_embeddings(model: nn.Module, std: float = 0.01) -> nn.Module:
    """Small embedding init (PyTorch's default N(0, 1) makes FM pairwise sums explode)."""
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=std)
    return model


class DeepFM(nn.Module):
    def __init__(self, vocab_sizes: list[int], n_num: int, emb_dim: int = 16,
                 hidden: tuple[int, ...] = (128, 64), dropout: float = 0.1):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(v, emb_dim) for v in vocab_sizes)
        self.first = nn.ModuleList(nn.Embedding(v, 1) for v in vocab_sizes)
        self.num_first = nn.Linear(n_num, 1)
        self.num_emb = nn.Linear(n_num, emb_dim)           # numeric block as one extra FM field
        self.deep, d = mlp(emb_dim * (len(vocab_sizes) + 1), hidden, dropout)
        self.out = nn.Linear(d, 1)
        _init_embeddings(self)

    def forward(self, x_cat, x_num):
        e = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
                        + [self.num_emb(x_num)], dim=1)                       # (B, F, k)
        first = sum(f(x_cat[:, i]) for i, f in enumerate(self.first)).squeeze(1) + self.num_first(x_num).squeeze(1)
        fm = 0.5 * ((e.sum(1) ** 2) - (e ** 2).sum(1)).sum(1)                 # sum of pairwise <e_i, e_j>
        deep = self.out(self.deep(e.flatten(1))).squeeze(1)
        return first + fm + deep


class DCNv2(nn.Module):
    def __init__(self, vocab_sizes: list[int], n_num: int, emb_dim: int = 16, n_cross: int = 3,
                 hidden: tuple[int, ...] = (128, 64), dropout: float = 0.1):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(v, emb_dim) for v in vocab_sizes)
        d0 = emb_dim * len(vocab_sizes) + n_num
        self.cross = nn.ModuleList(nn.Linear(d0, d0) for _ in range(n_cross))
        self.deep, d = mlp(d0, hidden, dropout)
        self.out = nn.Linear(d0 + d, 1)
        _init_embeddings(self)

    def forward(self, x_cat, x_num):
        x0 = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)] + [x_num], dim=1)
        x = x0
        for layer in self.cross:
            x = x0 * layer(x) + x
        return self.out(torch.cat([x, self.deep(x0)], dim=1)).squeeze(1)


MODELS = {"deepfm": DeepFM, "dcnv2": DCNv2}


def predict_logits(model, x_cat, x_num, batch: int = 65536) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = [model(x_cat[i:i + batch], x_num[i:i + batch]) for i in range(0, len(x_cat), batch)]
    return torch.cat(out).numpy()


def train_binary(model, train, valid, epochs: int = 5, batch: int = 4096, lr: float = 1e-3,
                 patience: int = 1, seed: int = 0, log=print, name: str = ""):
    """train / valid = (x_cat, x_num, y). Early stopping on valid logloss; returns the best model."""
    torch.manual_seed(seed)
    opt = adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    best, best_state, bad, t0 = np.inf, None, 0, time.time()
    n = len(train[0])
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = bce(model(train[0][idx], train[1][idx]), train[2][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        z = torch.as_tensor(predict_logits(model, valid[0], valid[1]))
        v = bce(z, valid[2]).item()
        log(f"{name} epoch {ep + 1}: valid logloss {v:.5f} ({time.time() - t0:.0f}s)")
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad > patience:
                break
    model.load_state_dict(best_state)
    return model.eval()
