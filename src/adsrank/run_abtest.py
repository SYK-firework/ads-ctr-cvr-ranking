"""Simulated online A/B test of bidding policies on the Criteo Attribution test window.

Control   : CTR-only bids       (cpo x pCTR x mean CVR)
Treatment : value-based bids    (cpo x pCTR x pCVR, calibrated)

Users are hash-assigned 50/50. Each arm bids only on its own users' impressions with a
multiplier tuned on the validation window to spend the same budget share (no peeking at the
test window). Outcomes come from second-price replay (see adsrank.bidding).

Reported:
  - SRM check on user counts
  - primary metric: conversion value per user (Welch CI, p-value)
  - secondary: conversions per user, spend per user (guardrail), ROI via delta method
  - power: MDE at this sample size and users needed for the observed lift
  - A/A: false-positive rate over many random splits of the control policy
  - peeking: false-positive rate when an A/A test is checked at many interim looks and
    stopped at the first p < 0.05

usage: python -m adsrank.run_abtest --preds data/preds_test.parquet
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from adsrank.abtest import assign, mde, mean_diff, ratio_diff, sample_size, srm_check  # noqa: E402
from adsrank.bidding import alpha_for_spend  # noqa: E402


def expected_value(d: pd.DataFrame, policy: str, mean_cvr: float) -> np.ndarray:
    if policy == "ctr_only":
        return (d.cpo * d.p_ctr * mean_cvr).to_numpy(np.float64)
    return (d.cpo * d.p_ctr * d.p_cvr).to_numpy(np.float64)


def outcomes(d: pd.DataFrame, ev: np.ndarray, alpha: float) -> pd.DataFrame:
    """Per-impression replay outcome, then summed per user."""
    win = ev * alpha >= d.cost.to_numpy()
    o = pd.DataFrame({"uid": d.uid.to_numpy(), "ts": d.timestamp.to_numpy(),
                      "spend": np.where(win, d.cost, 0.0),
                      "conv": np.where(win, d.attribution, 0),
                      "value": np.where(win, d.cpo * d.attribution, 0.0)})
    return o


def per_user(o: pd.DataFrame) -> pd.DataFrame:
    return o.groupby("uid")[["spend", "conv", "value"]].sum()


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--preds", default="data/preds_test.parquet")
    a.add_argument("--budget-frac", type=float, default=0.2)
    a.add_argument("--aa-runs", type=int, default=200)
    a.add_argument("--looks", type=int, default=20)
    a.add_argument("--results", default="results")
    a = a.parse_args()
    out = Path(a.results)
    out.mkdir(exist_ok=True)

    d = pd.read_parquet(a.preds)
    v = pd.read_parquet(a.preds.replace("_test", "_valid"))
    if "uid" not in d.columns:
        raise SystemExit("preds have no uid column: rerun `make calibrate` with the current code")
    mean_cvr = v.attribution[v.click == 1].mean()

    # multipliers tuned on the validation window only
    alphas = {p: alpha_for_spend(expected_value(v, p, mean_cvr), v.cost.to_numpy(), a.budget_frac * v.cost.sum())
              for p in ("ctr_only", "ctcvr_cal")}

    arm = assign(d.uid.to_numpy(), salt="bidding-v1")
    t, c = d[arm == 1], d[arm == 0]
    u_t = per_user(outcomes(t, expected_value(t, "ctcvr_cal", mean_cvr), alphas["ctcvr_cal"]))
    u_c = per_user(outcomes(c, expected_value(c, "ctr_only", mean_cvr), alphas["ctr_only"]))

    report = {"alphas_from_validation": alphas, "srm": srm_check(len(u_t), len(u_c)),
              "value_per_user": mean_diff(u_t.value, u_c.value),
              "conversions_per_user": mean_diff(u_t.conv, u_c.conv),
              "spend_per_user": mean_diff(u_t.spend, u_c.spend),
              "roi": ratio_diff(u_t.value, u_t.spend, u_c.value, u_c.spend)}

    sigma = float(np.sqrt((u_t.value.var() + u_c.value.var()) / 2))
    n = min(len(u_t), len(u_c))
    obs = report["value_per_user"]["diff"]
    report["power"] = {"users_per_arm": n, "sigma_value_per_user": sigma,
                       "mde_abs": mde(sigma, n), "mde_rel": mde(sigma, n) / report["value_per_user"]["mean_control"],
                       "users_per_arm_for_observed_lift": sample_size(sigma, abs(obs)) if obs else None}

    # ---- A/A: control policy on both arms, many random salts
    oc = outcomes(d, expected_value(d, "ctr_only", mean_cvr), alphas["ctr_only"])
    users = per_user(oc)
    uids = users.index.to_numpy()
    pvals = []
    for i in range(a.aa_runs):
        g = assign(uids, salt=f"aa-{i}")
        pvals.append(mean_diff(users.value[g == 1], users.value[g == 0])["p_value"])
    pvals = np.array(pvals)
    report["aa"] = {"runs": a.aa_runs, "false_positive_rate": float((pvals < 0.05).mean())}

    # ---- peeking: A/A checked at `looks` interim points, stop at first p < 0.05
    ts_edges = np.quantile(oc.ts, np.linspace(0, 1, a.looks + 1)[1:])
    looks = [per_user(oc[oc.ts <= e]) for e in ts_edges]          # per-user totals at each look
    stopped = 0
    for i in range(a.aa_runs):
        salt = f"peek-{i}"
        for lk in looks:
            g = assign(lk.index.to_numpy(), salt=salt)              # same salt -> same arm at every look
            if mean_diff(lk.value[g == 1], lk.value[g == 0])["p_value"] < 0.05:
                stopped += 1
                break
    report["peeking"] = {"runs": a.aa_runs, "looks": a.looks, "false_positive_rate": stopped / a.aa_runs}

    # ---- lift over time for the real A/B test (cumulative CI)
    ot = outcomes(t, expected_value(t, "ctcvr_cal", mean_cvr), alphas["ctcvr_cal"])
    ocon = outcomes(c, expected_value(c, "ctr_only", mean_cvr), alphas["ctr_only"])
    rows = []
    for edge in ts_edges:
        r = mean_diff(per_user(ot[ot.ts <= edge]).value, per_user(ocon[ocon.ts <= edge]).value)
        rows.append({"t": edge, **r})
    curve = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = (curve.t - d.timestamp.min()) / 3600
    ax.plot(x, curve.rel_lift * 100, label="lift in value per user")
    ax.fill_between(x, curve.ci_low / curve.mean_control * 100, curve.ci_high / curve.mean_control * 100,
                    alpha=0.25, label="95% CI")
    ax.axhline(0, color="grey", lw=1)
    ax.set(xlabel="hours into the experiment", ylabel="lift (%)",
           title="A/B: value-based vs CTR-only bidding (cumulative)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "abtest_lift.png", dpi=120)
    plt.close(fig)
    (out / "abtest.json").write_text(json.dumps(report, indent=1, default=float))

    s = report["srm"]
    print(f"\n== SRM ==\nusers treatment {s['n_treat']:,} / control {s['n_control']:,}, "
          f"p = {s['p_value']:.3f} -> {'SRM DETECTED' if s['srm_detected'] else 'ok'}")
    print("\n== A/B result (treatment = value-based bids, control = CTR-only bids) ==")
    for k in ("value_per_user", "conversions_per_user", "spend_per_user", "roi"):
        r = report[k]
        base = r.get("mean_control", r.get("ratio_control"))
        print(f"{k:22s} lift {r['rel_lift']:+6.1%}   95% CI [{r['ci_low'] / base:+6.1%}, "
              f"{r['ci_high'] / base:+6.1%}]   p = {r['p_value']:.2e}")
    pw = report["power"]
    print(f"\n== Power ==\nusers per arm {pw['users_per_arm']:,} | MDE {pw['mde_rel']:.1%} of control "
          f"| users per arm needed for the observed lift: {pw['users_per_arm_for_observed_lift']:,}")
    print(f"\n== A/A ({a.aa_runs} splits) ==\nfalse positive rate at p < 0.05: {report['aa']['false_positive_rate']:.1%}")
    print(f"\n== Peeking ({a.looks} interim looks, stop at first p < 0.05) ==\n"
          f"false positive rate: {report['peeking']['false_positive_rate']:.1%}")
    print(f"\nsaved {out}/abtest.json, abtest_lift.png")


if __name__ == "__main__":
    main()
