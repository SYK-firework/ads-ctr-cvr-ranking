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
  4. Budget pacing: a PI controller adjusts alpha over time to spend the budget evenly.

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

from adsrank.bidding import alpha_for_roi, alpha_for_spend, pid_pacing, replay  # noqa: E402


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

    pace = pid_pacing(evs["ctcvr_cal"], cost, attr, cpo, d.timestamp.to_numpy(), budget,
                      alpha0=report["fixed_budget"]["ctcvr_cal"]["alpha"] * 0.1)
    report["pacing"] = {"spend": float(pace.spend.sum()), "value": float(pace.value.sum()),
                        "conversions": float(pace.conversions.sum()),
                        "budget_used": float(pace.spend.sum() / budget),
                        "max_step_overspend": float((pace.spend / pace.planned).max())}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pace.step, pace.planned.cumsum() / budget, "--", color="grey", label="plan (even pacing)")
    ax.plot(pace.step, pace.spend.cumsum() / budget, label="actual (PI controller)")
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
    print(f"\n== Budget pacing (PI controller, ctcvr_cal) ==\nbudget used {p['budget_used']:.1%}, "
          f"conversions {p['conversions']:.0f}, value {p['value']:.3f}, worst step {p['max_step_overspend']:.2f}x plan")
    print(f"\nsaved {out}/bidding.json, bidding_value_vs_spend.png, bidding_pacing.png")


if __name__ == "__main__":
    main()
