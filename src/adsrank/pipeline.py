"""Retraining decision for the scheduled pipeline (dags/adsrank_retrain.py).

The DAG runs the batch drift report every day; this decides whether the day ends there or goes on
to retrain -> skew-checked export -> canary. Retraining is not free (a canary on live traffic,
an image, a rollout), so it runs only when the report shows a reason:

  - score PSI of any window at "act" level (> 0.25)
  - campaign-mix PSI at "act" level in a window with enough rows to trust it
  - calibration off by more than `max_calib_error` in a window with enough rows
    (pricing uses pCTR x pCVR directly, so a calibration error is a pricing error)
  - `force` (manual trigger, or a fixed retrain cadence)

usage: python -m adsrank.pipeline results/drift.json   -> prints the decision as JSON, exit 0
"""
import json
import math
import sys


def retrain_decision(report: dict, act: float = 0.25, max_calib_error: float = 0.05,
                     min_rows: int = 20_000, force: bool = False) -> dict:
    reasons = []
    if force:
        reasons.append("forced")
    for w in report.get("daily", []):
        where = f"window {w.get('day')} ({w.get('hours', '?')} h, {w['n']:,} rows)"
        for k in ("psi_p_ctr", "psi_p_cvr"):
            if w.get(k, 0) > act:
                reasons.append(f"{k} {w[k]:.3f} > {act} in {where}")
        if w["n"] < min_rows:
            continue                                          # too few rows for mix / calibration
        if w.get("psi_campaign", 0) > act:
            reasons.append(f"psi_campaign {w['psi_campaign']:.3f} > {act} in {where}")
        for k in ("calib_p_ctr", "calib_p_cvr"):
            v = w.get(k)
            if v is not None and not math.isnan(v) and abs(v - 1) > max_calib_error:
                reasons.append(f"{k} {v:.3f} off by more than {max_calib_error:.0%} in {where}")
    return {"retrain": bool(reasons), "reasons": reasons,
            "thresholds": {"act": act, "max_calib_error": max_calib_error, "min_rows": min_rows}}


def main():
    report = json.loads(open(sys.argv[1]).read())
    force = len(sys.argv) > 2 and sys.argv[2].lower() in ("1", "true", "force")
    d = retrain_decision(report, force=force)
    print(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
