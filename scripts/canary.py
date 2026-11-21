"""Canary rollout against two running scoring servers; exit code 0 = promote, 1 = rollback / wait.

usage (kind cluster, from a pod or with port-forwards):
  python scripts/canary.py --stable http://localhost:8080/predict --canary http://localhost:8081/predict \
      --steps 0.1,0.25,0.5 --rps 50 --seconds 30

`make canary` deploys k8s/inferenceservice-canary.yaml, runs this script, and then either moves the
stable service to the canary image (promote) or deletes the canary (rollback).
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

from adsrank.canary import rollout
from adsrank.monitoring import GateConfig


async def main_async(a) -> int:
    rows = json.loads(Path(a.samples).read_text())
    urls = {"stable": a.stable, "canary": a.canary}
    async with httpx.AsyncClient(timeout=a.timeout) as client:
        async def send(arm, body):
            r = await client.post(urls[arm], json=body)
            r.raise_for_status()
            return r.json()

        for arm in urls:                                            # warm-up, and fail fast if down
            await send(arm, {"instances": rows[:a.batch]})
        cfg = GateConfig(max_p99_ratio=a.max_p99_ratio, max_score_psi=a.max_psi,
                         score_ratio_band=(1 - a.score_band, 1 + a.score_band))
        res = await rollout(send, rows, steps=tuple(map(float, a.steps.split(","))), rps=a.rps,
                            seconds=a.seconds, batch=a.batch, cfg=cfg)
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"decision: {res['decision']}  (saved {a.out})")
    return 0 if res["decision"] == "promote" else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stable", default="http://adsrank-predictor/predict")
    p.add_argument("--canary", default="http://adsrank-canary-predictor/predict")
    p.add_argument("--samples", default="serving/model/sample_requests.json")
    p.add_argument("--steps", default="0.1,0.25,0.5")
    p.add_argument("--rps", type=float, default=50)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--timeout", type=float, default=10)
    p.add_argument("--max-p99-ratio", type=float, default=1.5)
    p.add_argument("--max-psi", type=float, default=0.1)
    p.add_argument("--score-band", type=float, default=0.05, help="allowed |mean pCTR ratio - 1|")
    p.add_argument("--out", default="results/canary.json")
    sys.exit(asyncio.run(main_async(p.parse_args())))


if __name__ == "__main__":
    main()
