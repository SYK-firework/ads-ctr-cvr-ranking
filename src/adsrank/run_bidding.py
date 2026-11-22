"""Compare bidding policies by offline replay on the Criteo Attribution test window.

Input: test-set predictions written by run_calibration (data/preds_test.parquet).

Policies (bid = alpha x expected value):
  random       : random scores                       (no model)
  ctr_only     : cpo x pCTR x mean CVR               (click model only, CVR ignored)
  ctcvr_raw    : cpo x pCTR x pCVR, uncalibrated
  ctcvr_cal    : cpo x pCTR x pCVR, calibrated

Experiments:
  1. Fixed budget: each policy gets the same budget; compare conversions, value, ROI.
  2. Value-vs-spend curves over alpha (efficient frontier per policy).
  3. ROI-target auto-bidding: largest alpha that keeps ROI >= target.
  4. Budget pacing over the test window, compared with a hindsight-tuned constant multiplier:
       naive PI     : cold start (alpha far too low), integral term -> wind-up
       adaptive     : warm start from the validation window, target = remaining budget /
                      remaining steps, capped proportional updates (no wind-up)
       adaptive_traffic_plan : same, but each step's share of the budget follows the
                      time-of-day spend profile of the validation window
       adaptive_hindsight_plan : same, with the plan taken from the test window itself
                      (a ceiling, not a deployable policy)

usage: python -m adsrank.run_bidding --preds data/preds_test.parquet
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from adsrank.bidding import (  # noqa: E402
    _time_steps, adaptive_pacing, alpha_for_roi, alpha_for_spend, pid_pacing, replay, spend_plan, step_edges,
)


def policies(d: pd.DataFrame, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    mean_cvr = d.attribution[d.click == 1].mean()
    return {
        "random": rng.random(len(d)) * d.cpo.mean(),
        "ctr_only": d.cpo * d.p_ctr * mean_cvr,
        "ctcvr_raw": d.cpo * d.p_ctr_raw * d.p_cvr_raw,
        "ctcvr_cal": d.cpo * d.p_ctr * d.p_cvr,
    }


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--preds", default="data/preds_test.parquet")
    a.add_argument("--budget-frac", type=float, default=0.2, help="budget as a fraction of total logged spend")
    a.add_argument("--target-roi", type=float, default=None, help="default: 1.5x the ROI of buying every impression")
    a.add_argument("--history", default=None,
                   help="raw Attribution parquet; rows before the test window shape the time-of-day spend plan")
    a.add_argument("--results", default="results")
    a = a.parse_args()
    out = Path(a.results)
    out.mkdir(exist_ok=True)

    d = pd.read_parquet(a.preds)
    cost, attr, cpo = d.cost.to_numpy(), d.attribution.to_numpy(), d.cpo.to_numpy()
    total = replay(np.full(len(d), np.inf), cost, attr, cpo)          # buy everything
    budget = a.budget_frac * total["spend"]
    target_roi = a.target_roi or 1.5 * total["roi"]
    evs = {k: np.asarray(v, dtype=np.float64) for k, v in policies(d).items()}

    report = {"buy_everything": total, "budget": budget, "target_roi": target_roi,
              "fixed_budget": {}, "roi_target": {}}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, ev in evs.items():
        alpha = alpha_for_spend(ev, cost, budget)
        report["fixed_budget"][name] = replay(ev * alpha, cost, attr, cpo) | {"alpha": alpha}
        a_roi = alpha_for_roi(ev, cost, attr, cpo, target_roi)
        report["roi_target"][name] = (replay(ev * a_roi, cost, attr, cpo) | {"alpha": a_roi}
                                      if np.isfinite(a_roi) else {"alpha": None})
        curve = [replay(ev * x, cost, attr, cpo) for x in np.logspace(np.log10(alpha) - 3, np.log10(alpha) + 3, 60)]
        ax.plot([c["spend"] / total["spend"] for c in curve], [c["value"] / total["value"] for c in curve], label=name)
    ax.axvline(a.budget_frac, color="grey", ls="--", lw=1)
    ax.set(xlabel="share of total spend", ylabel="share of total conversion value",
           title="Value vs spend by bidding policy (test replay)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "bidding_value_vs_spend.png", dpi=120)
    plt.close(fig)

    n_steps = 48
    ev_cal = evs["ctcvr_cal"]
    static = report["fixed_budget"]["ctcvr_cal"]                    # hindsight: knows the whole window
    naive = pid_pacing(ev_cal, cost, attr, cpo, d.timestamp.to_numpy(), budget, n_steps=n_steps,
                       alpha0=static["alpha"] * 0.1)
    valid_path = Path(a.preds.replace("_test", "_valid"))
    if valid_path.exists():                                         # warm start from the previous period
        v = pd.read_parquet(valid_path)
        ev_v = (v.cpo * v.p_ctr * v.p_cvr).to_numpy(dtype=np.float64)
        alpha0 = alpha_for_spend(ev_v, v.cost.to_numpy(), a.budget_frac * v.cost.sum())
        warm = "validation window"
    else:
        alpha0, warm = static["alpha"] * 0.1, "none (preds_valid.parquet missing)"
    ts = d.timestamp.to_numpy()
    # traffic-shaped plan: time-of-day profile of logged traffic value (market cost) in all data
    # before the test window. It needs at least a full day of history, which the validation
    # window alone (a few hours on the 2M-row sample) does not have.
    plan, plan_source = None, None
    if a.history and Path(a.history).exists():
        h = pd.read_parquet(a.history, columns=["timestamp", "cost"])
        h = h[h.timestamp < ts.min()]
        if len(h) and h.timestamp.max() - h.timestamp.min() >= 86400:
            plan = spend_plan(h.timestamp.to_numpy(), h.cost.to_numpy(), step_edges(ts, n_steps))
            plan_source = f"logged cost by hour of day, {(h.timestamp.max() - h.timestamp.min()) / 86400:.1f} days before test"
    adaptive = adaptive_pacing(ev_cal, cost, attr, cpo, ts, budget, alpha0, n_steps=n_steps)
    paced = {"naive_pi": naive, "adaptive": adaptive}
    if plan is not None:
        paced["adaptive_traffic_plan"] = adaptive_pacing(ev_cal, cost, attr, cpo, ts, budget, alpha0,
                                                         n_steps=n_steps, plan=plan)
    # ceiling: the plan a hindsight constant multiplier would have followed on the test window itself
    srt = np.sort(ts)
    hind_spend = np.where(ev_cal * static["alpha"] >= cost, cost, 0.0)[np.argsort(ts, kind="stable")]
    hind_plan = np.bincount(_time_steps(srt, n_steps), weights=hind_spend, minlength=n_steps)
    paced["adaptive_hindsight_plan"] = adaptive_pacing(ev_cal, cost, attr, cpo, ts, budget, alpha0,
                                                       n_steps=n_steps, plan=hind_plan / hind_plan.sum())

    even = budget / n_steps
    def pace_summary(p):
        return {"spend": float(p.spend.sum()), "value": float(p.value.sum()),
                "conversions": float(p.conversions.sum()), "budget_used": float(p.spend.sum() / budget),
                "max_step_vs_even_plan": float(p.spend.max() / even),
                "idle_steps": int((p.spend < 0.1 * even).sum()),
                "value_vs_hindsight_static": float(p.value.sum() / static["value"])}
    report["pacing"] = {"warm_start": warm, "alpha0": alpha0, "hindsight_static_alpha": static["alpha"],
                        **{k: pace_summary(p) for k, p in paced.items()}}
    report["pacing"]["traffic_plan_source"] = plan_source
    if plan is not None:
        report["pacing"]["traffic_plan_vs_even_max_ratio"] = float(plan.max() * n_steps)
        report["pacing"]["traffic_plan_vs_even_min_ratio"] = float(plan.min() * n_steps)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(n_steps), np.arange(1, n_steps + 1) / n_steps, "--", color="grey", label="even plan")
    ax.plot(naive.step, naive.spend.cumsum() / budget, label="naive PI (cold start, wind-up)")
    ax.plot(adaptive.step, adaptive.spend.cumsum() / budget, label="adaptive, even plan")
    if "adaptive_traffic_plan" in paced:
        tp = paced["adaptive_traffic_plan"]
        ax.plot(np.arange(n_steps), np.cumsum(plan), ":", color="black", lw=1, label="traffic-shaped plan")
        ax.plot(tp.step, tp.spend.cumsum() / budget, label="adaptive, traffic-shaped plan")
    ax.set(xlabel="time step", ylabel="cumulative share of budget", title="Budget pacing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "bidding_pacing.png", dpi=120)
    plt.close(fig)
    (out / "bidding.json").write_text(json.dumps(report, indent=1))

    cols = ["spend", "conversions", "value", "roi", "cpa", "win_rate"]
    print(f"\n== Fixed budget ({a.budget_frac:.0%} of total spend), test replay ==")
    print(pd.DataFrame(report["fixed_budget"]).T[cols].round(4).to_string())
    print(f"\n== ROI-target auto-bidding (target ROI {target_roi:.3f}; buy-everything ROI {total['roi']:.3f}) ==")
    print(pd.DataFrame({k: v for k, v in report["roi_target"].items() if v.get("alpha")}).T
          .reindex(columns=cols).round(4).to_string())
    p = report["pacing"]
    print(f"\n== Budget pacing (ctcvr_cal, {n_steps} steps; warm start: {p['warm_start']}) ==")
    print(pd.DataFrame({k: p[k] for k in paced}).T.round(3).to_string())
    print(f"\nsaved {out}/bidding.json, bidding_value_vs_spend.png, bidding_pacing.png")


if __name__ == "__main__":
    main()
