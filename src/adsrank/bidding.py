"""Value-based bidding and budget pacing, evaluated by offline replay on Criteo Attribution.

Setting (real-time bidding, second-price auction):
  every logged impression is an auction. The logged `cost` is the price that was paid, used as
  the market price: if our bid >= cost we win, pay cost, and receive the logged outcome
  (click / attributed conversion). This is the standard offline replay used for RTB research.

Bid = alpha x expected value of the impression
  expected value = cpo x pCTR x pCVR   (cpo: what the advertiser pays per attributed order)
In a second-price auction bidding the true value is optimal; alpha scales bids to meet a
budget or ROI target (auto-bidding).

Replay assumptions to keep in mind:
  - only impressions that were actually won are in the log, so losses to other bidders and the
    full bid landscape are not observed;
  - the outcome of a won impression does not depend on how much we bid.
Results compare bidding policies against each other; they are not production revenue estimates.
"""
import numpy as np
import pandas as pd


def replay(bid, cost, attribution, cpo) -> dict:
    win = np.asarray(bid) >= np.asarray(cost)
    spend = float(np.sum(np.asarray(cost)[win]))
    conv = float(np.sum(np.asarray(attribution)[win]))
    value = float(np.sum((np.asarray(cpo) * np.asarray(attribution))[win]))
    return {"win_rate": float(win.mean()), "spend": spend, "conversions": conv, "value": value,
            "roi": value / spend if spend > 0 else 0.0,
            "cpa": spend / conv if conv > 0 else float("inf")}


def alpha_for_spend(ev, cost, budget: float, lo: float = 1e-9, hi: float = 1e6, iters: int = 80) -> float:
    """Smallest bid multiplier whose replay spend reaches `budget` (spend is monotone in alpha)."""
    ev, cost = np.asarray(ev), np.asarray(cost)
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if cost[ev * mid >= cost].sum() < budget:
            lo = mid
        else:
            hi = mid
    return hi


def alpha_for_roi(ev, cost, attribution, cpo, target_roi: float,
                  lo: float = 1e-9, hi: float = 1e6, iters: int = 80) -> float:
    """Largest multiplier that still meets ROI >= target (auto-bidding with an ROI / CPA goal).

    ROI is not strictly monotone in alpha on finite data, so scan a log grid and keep the
    largest alpha (most volume) whose ROI meets the target."""
    best = float("nan")
    for a in np.logspace(np.log10(lo), np.log10(hi), 400):
        r = replay(ev * a, cost, attribution, cpo)
        if r["spend"] > 0 and r["roi"] >= target_roi:
            best = float(a)
    return best


def pid_pacing(ev, cost, attribution, cpo, timestamp, budget: float, n_steps: int = 48,
               alpha0: float = 1.0, kp: float = 0.8, ki: float = 0.2) -> pd.DataFrame:
    """Spend `budget` evenly over time by adjusting alpha after each time step (PI controller).

    error_t = (planned spend - actual spend) / planned spend for the step;
    alpha_{t+1} = alpha_t * exp(kp * error_t + ki * sum(errors))."""
    order = np.argsort(np.asarray(timestamp), kind="stable")
    ev, cost, attribution, cpo, ts = (np.asarray(x)[order] for x in (ev, cost, attribution, cpo, timestamp))
    edges = np.linspace(ts.min(), ts.max() + 1, n_steps + 1)
    plan = budget / n_steps
    alpha, integral, rows = alpha0, 0.0, []
    for t in range(n_steps):
        m = (ts >= edges[t]) & (ts < edges[t + 1])
        r = replay(ev[m] * alpha, cost[m], attribution[m], cpo[m])
        err = (plan - r["spend"]) / plan
        integral += err
        rows.append({"step": t, "alpha": alpha, "planned": plan, "spend": r["spend"],
                     "value": r["value"], "conversions": r["conversions"]})
        alpha = float(alpha * np.exp(np.clip(kp * err + ki * integral, -2, 2)))
    return pd.DataFrame(rows)
