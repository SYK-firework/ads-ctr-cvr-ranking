"""Step-wise canary rollout: send a growing share of traffic to the new model, gate each step.

  stable  = the InferenceService serving today (`adsrank`)
  canary  = the new model version (`adsrank-canary`, k8s/inferenceservice-canary.yaml)

Each step sends `share` of the requests to the canary and the rest to stable, at the same time and
from the same request pool, then runs `monitoring.canary_gate` on the two arms. Any failed check
stops the rollout (rollback); passing every step ends in promote.

Why the split is done here and not by KServe: KServe's `canaryTrafficPercent` needs Serverless
mode (Knative). This cluster runs KServe in raw-deployment mode with HPA (k8s/inferenceservice.yaml),
which has no weighted routing, so the rollout script routes requests itself. In production this
role belongs to the service mesh or the ads server's router.

Routing is by request key (hash), so the same key always hits the same arm within a step, as a
user-level split would.
"""
import asyncio
import hashlib
import time

import numpy as np

from adsrank.monitoring import ArmStats, GateConfig, canary_gate, p99


def to_canary(key: str, share: float) -> bool:
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < share


async def run_step(send, rows, share: float, rps: float, seconds: float, batch: int, rng) -> dict:
    """Open-loop step. `send(arm, body)` is an async callable returning the response JSON or
    raising; arm is "stable" or "canary". Returns ArmStats for both arms."""
    n = int(rps * seconds)
    acc = {arm: {"lat": [], "errors": 0, "requests": 0, "scores": []} for arm in ("stable", "canary")}
    start = time.perf_counter() + 0.1

    async def one(i):
        due = start + i / rps
        await asyncio.sleep(max(0.0, due - time.perf_counter()))
        arm = "canary" if to_canary(f"req-{i}", share) else "stable"
        a = acc[arm]
        a["requests"] += 1
        idx = rng.integers(0, len(rows), batch)
        try:
            r = await send(arm, {"instances": [rows[j] for j in idx]})
            a["lat"].append((time.perf_counter() - due) * 1000)
            a["scores"].extend(p["p_ctr"] for p in r["predictions"])
        except Exception:
            a["errors"] += 1

    await asyncio.gather(*(one(i) for i in range(n)))
    return {arm: ArmStats(a["requests"], a["errors"], p99(a["lat"]), np.asarray(a["scores"]))
            for arm, a in acc.items()}


async def rollout(send, rows, steps=(0.1, 0.25, 0.5), rps: float = 50, seconds: float = 30,
                  batch: int = 50, cfg: GateConfig = GateConfig(), seed: int = 0, log=print) -> dict:
    """Run the steps in order; stop at the first failed gate. Returns the decision and history."""
    rng = np.random.default_rng(seed)
    history = []
    for share in steps:
        arms = await run_step(send, rows, share, rps, seconds, batch, rng)
        g = canary_gate(arms["stable"], arms["canary"], cfg)
        history.append({"share": share, "decision": g.decision, "checks": g.checks, "reasons": g.reasons,
                        "requests": {k: v.requests for k, v in arms.items()}})
        c = g.checks
        log(f"canary {share:>4.0%}: {g.decision:8s} "
            + (f"p99 {c['p99_ms_stable']:.0f}/{c['p99_ms_canary']:.0f} ms, "
               f"pCTR ratio {c['score_mean_ratio']:.3f}, PSI {c['score_psi']:.3f}" if "score_psi" in c else "")
            + (f"  <- {'; '.join(g.reasons)}" if g.reasons else ""))
        if g.decision != "promote":
            return {"decision": "rollback" if g.decision == "rollback" else "wait", "history": history}
    return {"decision": "promote", "history": history}
