"""Open-loop HTTP load test for the scoring server.

Requests are sent on a fixed schedule (target requests per second) regardless of how fast earlier
responses come back. Latency is measured from each request's scheduled send time, so queueing
delay is included. A closed loop (send the next request when the last one returns) slows down
when the server slows down and under-reports tail latency (coordinated omission).

usage: python scripts/loadtest.py --url http://localhost:8080/predict --rps 50,100,200 --batch 50
"""
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx
import numpy as np


async def stage(client, url, rows, rps: float, seconds: float, batch: int, rng) -> dict:
    n = int(rps * seconds)
    lat, errors = [], 0
    start = time.perf_counter() + 0.2

    async def one(i):
        nonlocal errors
        due = start + i / rps
        await asyncio.sleep(max(0.0, due - time.perf_counter()))
        body = {"instances": rng.sample(rows, batch)}
        try:
            r = await client.post(url, json=body)
            r.raise_for_status()
            lat.append((time.perf_counter() - due) * 1000)
        except Exception:
            errors += 1

    await asyncio.gather(*(one(i) for i in range(n)))
    elapsed = time.perf_counter() - start
    t = np.array(lat) if lat else np.array([np.nan])
    return {"target_rps": rps, "achieved_rps": len(lat) / elapsed, "requests": n, "errors": errors,
            "candidates_per_request": batch,
            **{f"p{q}_ms": float(np.percentile(t, q)) for q in (50, 90, 99)}, "max_ms": float(t.max())}


async def main_async(a):
    rows = json.loads(Path(a.samples).read_text())
    rng = random.Random(0)
    limits = httpx.Limits(max_connections=a.connections, max_keepalive_connections=a.connections)
    report = []
    async with httpx.AsyncClient(timeout=a.timeout, limits=limits) as client:
        await client.post(a.url, json={"instances": rows[:a.batch]})          # warm-up
        print(f"{'target rps':>10s} {'achieved':>9s} {'p50 ms':>8s} {'p90 ms':>8s} {'p99 ms':>8s} {'errors':>7s}")
        for rps in map(float, a.rps.split(",")):
            r = await stage(client, a.url, rows, rps, a.seconds, a.batch, rng)
            report.append(r)
            print(f"{rps:>10.0f} {r['achieved_rps']:>9.1f} {r['p50_ms']:>8.1f} {r['p90_ms']:>8.1f} "
                  f"{r['p99_ms']:>8.1f} {r['errors']:>7d}")
            if r["p99_ms"] > a.stop_p99_ms or r["errors"] or r["achieved_rps"] < 0.9 * rps:
                print(f"saturated at {rps:.0f} rps (p99 {r['p99_ms']:.0f} ms); skipping higher rates")
                break
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps({"label": a.label, "url": a.url, "batch": a.batch, "stages": report}, indent=1))
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        x = [r["achieved_rps"] for r in report]
        for q in (50, 90, 99):
            ax.plot(x, [r[f"p{q}_ms"] for r in report], "o-", label=f"p{q}")
        ax.set(xlabel="achieved requests / s", ylabel="latency (ms, log scale)", yscale="log",
               title=f"Scoring latency, {a.batch} candidates per request")
        ax.legend()
        fig.tight_layout()
        fig.savefig(a.plot, dpi=120)
        print(f"saved {a.plot}")
    print(f"saved {a.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080/predict")
    p.add_argument("--samples", default="serving/model/sample_requests.json")
    p.add_argument("--label", default=None, help="name stored in the output, e.g. local-1-worker")
    p.add_argument("--rps", default="25,50,100,200")
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--batch", type=int, default=50, help="ad candidates scored per request")
    p.add_argument("--connections", type=int, default=100)
    p.add_argument("--timeout", type=float, default=10)
    p.add_argument("--out", default="results/serving_loadtest.json")
    p.add_argument("--plot", default=None, help="optional PNG path for latency vs throughput")
    p.add_argument("--stop-p99-ms", type=float, default=1000, help="stop once p99 exceeds this")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
