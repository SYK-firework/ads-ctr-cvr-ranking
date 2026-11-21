"""Prediction-drift report by time window (daily, or equal slices) on the test window, with the validation window as reference.

Inputs are the prediction files written by `make calibrate`:
  data/preds_valid.parquet  reference (the window the calibrators were fitted on)
  data/preds_test.parquet   monitored window, one report row per day

For each day: PSI of pCTR and pCVR, PSI of the campaign mix, and the calibration ratio
(mean prediction / observed rate). PSI needs no labels, so in production it is the early alarm;
the calibration ratio confirms the damage once (delayed) labels arrive.

usage: python -m adsrank.run_drift --ref data/preds_valid.parquet --cur data/preds_test.parquet
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from adsrank.monitoring import drift_level, daily_drift, psi


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--ref", default="data/preds_valid.parquet")
    a.add_argument("--cur", default="data/preds_test.parquet")
    a.add_argument("--out", default="results")
    a.add_argument("--watch", type=float, default=0.1)
    a.add_argument("--act", type=float, default=0.25)
    a.add_argument("--slices", type=int, default=8,
                   help="equal-time windows of the monitored period (0 = one per day); a 2M-row "
                        "subset covers only a few days, so its test window is shorter than a day")
    a = a.parse_args()

    ref, cur = pd.read_parquet(a.ref), pd.read_parquet(a.cur)
    table = daily_drift(ref, cur, label_cols={"p_ctr": "click", "p_cvr": "attribution"},
                        slices=a.slices or None)
    worst = max(table.psi_p_ctr.max(), table.psi_p_cvr.max())
    report = {
        "reference_rows": len(ref), "current_rows": len(cur),
        "overall_psi_p_ctr": psi(ref.p_ctr, cur.p_ctr), "overall_psi_p_cvr": psi(ref.p_cvr, cur.p_cvr),
        "worst_daily_score_psi": float(worst), "status": drift_level(worst, a.watch, a.act),
        "thresholds": {"watch": a.watch, "act": a.act}, "slices": a.slices,
        "window_hours": float((cur.timestamp.max() - cur.timestamp.min()) / 3600),
        "daily": table.to_dict(orient="records"),
    }
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / "drift.json").write_text(json.dumps(report, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for c in ("psi_p_ctr", "psi_p_cvr", "psi_campaign"):
        ax1.plot(table.hours, table[c], "o-", label=c.replace("psi_", ""))
    ax1.axhline(a.watch, ls="--", c="gray", lw=1)
    ax1.axhline(a.act, ls="--", c="red", lw=1)
    ax1.set(xlabel="hours into test window", ylabel="PSI vs validation window", title="Input / score drift")
    ax1.legend()
    for c in ("calib_p_ctr", "calib_p_cvr"):
        ax2.plot(table.hours, table[c], "o-", label=c.replace("calib_", ""))
    ax2.axhline(1.0, c="gray", lw=1)
    ax2.set(xlabel="hours into test window", ylabel="mean prediction / observed rate", title="Calibration by day")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out / "drift.png", dpi=120)

    print(table.round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in report.items() if k != "daily"}))
    print(f"\nworst daily score PSI {worst:.4f} -> {report['status']}  (saved {out / 'drift.json'})")


if __name__ == "__main__":
    main()
