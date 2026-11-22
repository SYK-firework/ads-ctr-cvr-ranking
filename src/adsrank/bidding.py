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


def _time_steps(ts: np.ndarray, n_steps: int) -> np.ndarray:
    """Assign each (sorted) timestamp to one of n_steps equal-width time steps, last one inclusive."""
    span = max(float(ts.max() - ts.min()), 1e-12)
    return np.minimum(((ts - ts.min()) / span * n_steps).astype(int), n_steps - 1)


def pid_pacing(ev, cost, attribution, cpo, timestamp, budget: float, n_steps: int = 48,
               alpha0: float = 1.0, kp: float = 0.8, ki: float = 0.2) -> pd.DataFrame:
    """Spend `budget` evenly over time by adjusting alpha after each time step (PI controller).

    error_t = (planned spend - actual spend) / planned spend for the step;
    alpha_{t+1} = alpha_t * exp(kp * error_t + ki * sum(errors))."""
    order = np.argsort(np.asarray(timestamp), kind="stable")
    ev, cost, attribution, cpo, ts = (np.asarray(x)[order] for x in (ev, cost, attribution, cpo, timestamp))
    step_of = _time_steps(ts, n_steps)
    plan = budget / n_steps
    alpha, integral, rows = alpha0, 0.0, []
    for t in range(n_steps):
        m = step_of == t
        r = replay(ev[m] * alpha, cost[m], attribution[m], cpo[m])
        err = (plan - r["spend"]) / plan
        integral += err
        rows.append({"step": t, "alpha": alpha, "planned": plan, "spend": r["spend"],
                     "value": r["value"], "conversions": r["conversions"]})
        alpha = float(alpha * np.exp(np.clip(kp * err + ki * integral, -2, 2)))
    return pd.DataFrame(rows)


def spend_plan(hist_ts, hist_spend, step_edges, period: float = 86400.0, n_bins: int = 24) -> np.ndarray:
    """Share of the budget to plan for each time step, shaped by the time of day.

    The spend rate per time-of-day bin is estimated from a past window (spend divided by the
    time that window spent in each bin), then integrated over each future step. Steps that
    fall in high-value hours get a larger share. Only past data is used, so the plan can be
    set before the day starts. Returns weights that sum to 1."""
    hist_ts, hist_spend = np.asarray(hist_ts, np.float64), np.asarray(hist_spend, np.float64)
    width = period / n_bins
    bin_of = ((hist_ts % period) // width).astype(int)
    spend_per_bin = np.bincount(bin_of, weights=hist_spend, minlength=n_bins)
    grid = np.linspace(hist_ts.min(), hist_ts.max(), 20000)               # time covered per bin
    seconds_per_bin = np.bincount(((grid % period) // width).astype(int), minlength=n_bins) * (
        (hist_ts.max() - hist_ts.min()) / len(grid))
    rate = np.where(seconds_per_bin > 0, spend_per_bin / np.maximum(seconds_per_bin, 1e-12), 0.0)
    if rate.sum() <= 0:
        rate = np.ones(n_bins)
    w = []
    for lo, hi in zip(step_edges[:-1], step_edges[1:]):
        g = np.linspace(lo, hi, 200, endpoint=False)
        w.append(rate[((g % period) // width).astype(int)].mean() * (hi - lo))
    w = np.asarray(w)
    return w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))


def step_edges(timestamp, n_steps: int) -> np.ndarray:
    """Boundaries of the n_steps equal-width time steps used by the pacers."""
    ts = np.asarray(timestamp, np.float64)
    return np.linspace(ts.min(), ts.max(), n_steps + 1)


def adaptive_pacing(ev, cost, attribution, cpo, timestamp, budget: float, alpha0: float,
                    n_steps: int = 48, kp: float = 0.5, max_change: float = 0.5,
                    step_cap: float = 1.5, plan=None) -> pd.DataFrame:
    """Budget pacing without integral wind-up.

    - warm start: alpha0 comes from the previous period (validation window), not a guess;
    - each step's target is recomputed as remaining budget / remaining steps, so a shortfall or
      overspend is spread over the rest of the day instead of being chased in one step
      (this plays the role of the integral term but cannot wind up);
    - proportional update on log(alpha), with the per-step change capped at +-max_change;
    - spend in a step is capped at step_cap x target and at the remaining budget: once the cap
      is reached the bidder stops buying until the next step (throttling), so a traffic burst
      cannot drain the budget early;
    - plan: optional per-step budget shares (e.g. from spend_plan); default is an even plan.
      The target is the remaining budget times this step's share of the remaining plan.
    """
    order = np.argsort(np.asarray(timestamp), kind="stable")
    ev, cost, attribution, cpo, ts = (np.asarray(x)[order] for x in (ev, cost, attribution, cpo, timestamp))
    step_of = _time_steps(ts, n_steps)
    w = np.full(n_steps, 1.0 / n_steps) if plan is None else np.asarray(plan, np.float64)
    if len(w) != n_steps:
        raise ValueError("plan must have one weight per step")
    alpha, remaining, rows = alpha0, budget, []
    for t in range(n_steps):
        rest = w[t:].sum()
        target = remaining * w[t] / rest if rest > 0 else 0.0
        m = np.flatnonzero(step_of == t)
        win = m[ev[m] * alpha >= cost[m]]
        # throttle: stop buying once the step cap (or the remaining budget) is reached
        cap = min(remaining, step_cap * target)
        win = win[np.cumsum(cost[win]) <= cap] if len(win) else win
        spend = float(cost[win].sum())
        rows.append({"step": t, "alpha": alpha, "planned": target, "spend": spend,
                     "value": float((cpo[win] * attribution[win]).sum()),
                     "conversions": float(attribution[win].sum())})
        remaining -= spend
        err = (target - spend) / target if target > 0 else 0.0
        alpha = float(alpha * np.exp(np.clip(kp * err, -max_change, max_change)))
    return pd.DataFrame(rows)
