"""Calibration study on Criteo Attribution with the XGBoost CTR / CVR models.

  1. Train CTR and CVR models (same setup as train_attribution).
  2. Fit Platt and Isotonic calibrators on the validation window, evaluate on test:
     LogLoss, calibration ratio, ECE, AUC, and per-campaign calibration error.
  3. Negative-downsampling experiment: train CTR on data with non-clicks kept at --neg-rate,
     then compare raw vs corrected probabilities on the full test set.
  4. Save reliability diagrams and test-set predictions (input to the bidding simulation).

usage: python -m adsrank.run_calibration --data data/attribution_2000000.parquet
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from adsrank.attribution import FEATURES, build_features, load, time_split  # noqa: E402
from adsrank.calibration import (CALIBRATORS, correct_downsampling, downsample_negatives,  # noqa: E402
                                 ece, reliability_table, segment_calibration, segment_error)
from adsrank.metrics import ads_metrics  # noqa: E402
from adsrank.train_attribution import fit  # noqa: E402


def summarize(name, y, p, seg=None) -> dict:
    r = ads_metrics(y, p, name) | {f"{name}_ece": ece(y, p)}
    if seg is not None:
        r[f"{name}_campaign_error"] = segment_error(segment_calibration(y, p, seg))
    return r


def reliability_plot(curves: dict, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    hi = max(max(t.pred.max(), t.actual.max()) for t in curves.values()) * 1.05
    ax.plot([0, hi], [0, hi], "--", color="grey", lw=1, label="perfect")
    for name, t in curves.items():
        ax.plot(t.pred, t.actual, "o-", ms=4, label=name)
    ax.set(xlabel="mean predicted probability", ylabel="observed rate", title=title,
           xlim=(0, hi), ylim=(0, hi))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--data", required=True)
    a.add_argument("--neg-rate", type=float, default=0.25, help="non-click keep rate for the downsampling experiment")
    a.add_argument("--jobs", type=int, default=4)
    a.add_argument("--results", default="results")
    a.add_argument("--preds-out", default="data/preds_test.parquet")
    a = a.parse_args()
    out = Path(a.results)
    out.mkdir(exist_ok=True)

    df = load(a.data, extra=["cost"])
    feats = build_features(df, time_split(df))
    tr, va, te = (feats[feats.split == s] for s in ("train", "valid", "test"))
    campaign_te = df.loc[te.index, "campaign"].to_numpy()

    ctr = fit(tr, va, "click", a.jobs, verbose=False)
    cvr = fit(tr[tr.click == 1], va[va.click == 1], "attribution", a.jobs, verbose=False)
    pred = {s: (ctr.predict_proba(d[FEATURES])[:, 1], cvr.predict_proba(d[FEATURES])[:, 1])
            for s, d in (("valid", va), ("test", te))}
    va_clicked, te_clicked = (va.click == 1).to_numpy(), (te.click == 1).to_numpy()

    report, curves_ctr, curves_cvr, calibrated = {}, {}, {}, {}
    for method in ["raw"] + list(CALIBRATORS):
        if method == "raw":
            p_ctr, p_cvr = pred["test"]
        else:
            c_ctr = CALIBRATORS[method]().fit(pred["valid"][0], va.click.to_numpy())
            c_cvr = CALIBRATORS[method]().fit(pred["valid"][1][va_clicked], va.attribution.to_numpy()[va_clicked])
            p_ctr, p_cvr = c_ctr.predict(pred["test"][0]), c_cvr.predict(pred["test"][1])
        calibrated[method] = (p_ctr, p_cvr)
        report[method] = (summarize("ctr", te.click.to_numpy(), p_ctr, campaign_te)
                          | summarize("cvr", te.attribution.to_numpy()[te_clicked], p_cvr[te_clicked],
                                      campaign_te[te_clicked])
                          | summarize("ctcvr", te.attribution.to_numpy(), p_ctr * p_cvr, campaign_te))
        curves_ctr[method] = reliability_table(te.click, p_ctr)
        curves_cvr[method] = reliability_table(te.attribution[te_clicked], p_cvr[te_clicked])

    # ---- negative downsampling: train CTR on sampled data, correct back to the full-data scale
    keep_tr = downsample_negatives(tr.click, a.neg_rate, seed=1)
    keep_va = downsample_negatives(va.click, a.neg_rate, seed=2)
    ctr_ds = fit(tr[keep_tr], va[keep_va], "click", a.jobs, verbose=False)
    p_ds = ctr_ds.predict_proba(te[FEATURES])[:, 1]
    p_fixed = correct_downsampling(p_ds, a.neg_rate)
    y = te.click.to_numpy()
    report["downsampling"] = {"neg_rate": a.neg_rate,
                              "train_ctr_after_sampling": float(tr.click[keep_tr].mean()),
                              "test_ctr": float(y.mean())}
    report["downsampling"] |= summarize("uncorrected", y, p_ds) | summarize("corrected", y, p_fixed)
    curves_ds = {"trained on downsampled, uncorrected": reliability_table(y, p_ds),
                 "corrected": reliability_table(y, p_fixed),
                 "full-data model": curves_ctr["raw"]}

    # ---- outputs
    (out / "calibration.json").write_text(json.dumps(report, indent=1))
    reliability_plot(curves_ctr, "CTR reliability (test)", out / "reliability_ctr.png")
    reliability_plot(curves_cvr, "CVR reliability (test, clicked)", out / "reliability_cvr.png")
    reliability_plot(curves_ds, f"CTR with negatives kept at {a.neg_rate:.0%}", out / "reliability_downsampling.png")
    best = min(CALIBRATORS, key=lambda m: report[m]["ctcvr_logloss"])
    preds = pd.DataFrame({
        "timestamp": df.loc[te.index, "timestamp"].to_numpy(), "campaign": campaign_te,
        "cost": df.loc[te.index, "cost"].to_numpy(), "cpo": df.loc[te.index, "cpo"].to_numpy(),
        "click": te.click.to_numpy(), "attribution": te.attribution.to_numpy(),
        "p_ctr_raw": pred["test"][0], "p_cvr_raw": pred["test"][1],
        "p_ctr": calibrated[best][0], "p_cvr": calibrated[best][1],
    })
    Path(a.preds_out).parent.mkdir(exist_ok=True)
    preds.to_parquet(a.preds_out, index=False)

    rows = ["ctr_logloss", "ctr_calibration", "ctr_ece", "ctr_campaign_error", "ctr_auc",
            "cvr_logloss", "cvr_calibration", "cvr_ece", "cvr_campaign_error", "cvr_auc",
            "ctcvr_logloss", "ctcvr_calibration", "ctcvr_ece", "ctcvr_auc"]
    table = pd.DataFrame({m: [report[m][r] for r in rows] for m in ["raw"] + list(CALIBRATORS)}, index=rows)
    print("\n== Calibration on test (fitted on valid) ==")
    print(table.round(4).to_string())
    d = report["downsampling"]
    print(f"\n== Negative downsampling (non-clicks kept at {d['neg_rate']:.0%}) ==")
    print(f"train CTR after sampling {d['train_ctr_after_sampling']:.3f} | test CTR {d['test_ctr']:.3f}")
    for k in ["calibration", "ece", "logloss", "auc"]:
        print(f"{k:12s} uncorrected {d['uncorrected_' + k]:.4f}   corrected {d['corrected_' + k]:.4f}")
    print(f"\nbest calibrator by CTCVR logloss: {best}")
    print(f"saved {out}/calibration.json, reliability_*.png, {a.preds_out}")


if __name__ == "__main__":
    main()
