"""Model-only latency: Scorer.score on batches of N candidates, no network or HTTP.

usage: python -m adsrank.serving.bench --model serving/model
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from adsrank.serving.scorer import Scorer


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="serving/model")
    a.add_argument("--batches", default="1,10,50,200,1000")
    a.add_argument("--iters", type=int, default=300)
    a.add_argument("--out", default="results/serving_bench.json")
    a = a.parse_args()
    scorer = Scorer(a.model)
    rows = json.loads((Path(a.model) / "sample_requests.json").read_text())
    rng = random.Random(0)
    report = {}
    print(f"{'candidates':>10s} {'p50 ms':>8s} {'p99 ms':>8s} {'us / candidate':>15s}")
    for b in map(int, a.batches.split(",")):
        for _ in range(20):                                  # warm-up
            scorer.score(rng.sample(rows, b))
        t = []
        for _ in range(a.iters):
            batch = rng.sample(rows, b)
            s = time.perf_counter()
            scorer.score(batch)
            t.append((time.perf_counter() - s) * 1000)
        t = np.array(t)
        report[b] = {"p50_ms": float(np.percentile(t, 50)), "p99_ms": float(np.percentile(t, 99)),
                     "us_per_candidate": float(np.median(t) * 1000 / b)}
        r = report[b]
        print(f"{b:>10d} {r['p50_ms']:>8.3f} {r['p99_ms']:>8.3f} {r['us_per_candidate']:>15.1f}")
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
