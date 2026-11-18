"""Preprocessing for the Criteo Display Advertising Challenge (Kaggle) data.

Raw format (train.txt, tab-separated, no header):
  label | I1..I13 (integer, anonymized) | C1..C26 (categorical, 32-bit hashed strings, anonymized)
  Rows are in time order (7 days) -> time-based split: first 80% train / 10% valid / 10% test.

Processing (the standard recipe used in DCN / DeepFM-style papers):
  - integer features: missing -> 0, and floor(log(x)^2) when x > 2 (very wide value range)
  - categorical features: vocabulary built on the train window only (no leakage);
    values seen fewer than min_count times map to OOV (index 0)
  - count encoding (C*_cnt) for tree models: the hash value itself carries no meaning,
    so "how common is this value" is used as a feature instead
"""
import argparse
import json

import numpy as np
import pandas as pd

from adsrank import CAT, NUM

COLS = ["label"] + NUM + CAT


def log_transform(x: pd.Series) -> np.ndarray:
    x = x.fillna(0).astype(np.float64)
    return np.where(x > 2, np.floor(np.log(np.maximum(x, 1)) ** 2), x).astype(np.float32)


def preprocess(df: pd.DataFrame, min_count: int = 10) -> tuple[pd.DataFrame, dict[str, int]]:
    n = len(df)
    n_tr, n_va = int(n * 0.8), int(n * 0.1)
    split = np.array(["train"] * n_tr + ["valid"] * n_va + ["test"] * (n - n_tr - n_va))

    out = pd.DataFrame({"label": df.label.astype(np.int8), "split": split})
    for c in NUM:
        out[c] = log_transform(df[c])
    vocab_sizes = {}
    for c in CAT:
        s = df[c].fillna("__missing__")
        counts = s.iloc[:n_tr].value_counts()           # vocabulary from the train window only
        keep = counts[counts >= min_count].index
        mapping = pd.Series(np.arange(1, len(keep) + 1, dtype=np.int32), index=keep)
        out[c] = s.map(mapping).fillna(0).astype(np.int32)
        vocab_sizes[c] = len(keep) + 1                  # +1 for OOV
        out[f"{c}_cnt"] = np.log1p(s.map(counts).fillna(0)).astype(np.float32)
    return out, vocab_sizes


def read_raw(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Read Criteo DAC in either layout:
      - Kaggle original train.txt: tab-separated, no header
      - CSV mirrors (e.g. reczoo/Criteo_x4): comma-separated with a header (Label, I1..I13, C1..C26)
    Rows must stay in time order (the original file is ordered by time)."""
    opener = __import__("gzip").open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        first = f.readline()
    if "," in first and first.lower().startswith("label"):
        df = pd.read_csv(path, nrows=nrows, dtype={c: "string" for c in CAT})
        df.columns = [c.lower() if c.lower() == "label" else c for c in df.columns]
        return df[COLS]
    return pd.read_csv(path, sep="\t", header=None, names=COLS, nrows=nrows,
                       dtype={c: "string" for c in CAT})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nrows", type=int, default=None)
    p.add_argument("--min-count", type=int, default=10)
    a = p.parse_args()

    df = read_raw(a.input, a.nrows)
    out, vocab = preprocess(df, a.min_count)
    counts = out.split.value_counts()
    print(f"rows {len(out):,}  CTR {out.label.mean():.4f}  "
          f"(train {counts['train']:,} / valid {counts['valid']:,} / test {counts['test']:,})")
    out.to_parquet(a.out, index=False)
    with open(a.out.replace(".parquet", "_vocab.json"), "w") as f:
        json.dump(vocab, f, indent=1)
    print(f"saved {a.out}  (total vocab {sum(vocab.values()):,})")


if __name__ == "__main__":
    main()
