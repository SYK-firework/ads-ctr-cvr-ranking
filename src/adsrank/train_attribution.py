"""CTR + CVR baseline on Criteo Attribution with XGBoost.

  CTR model   : all impressions,     label = click
  CVR model   : clicked impressions, label = attribution (post-click attributed conversion)
  CTCVR       : pCTR x pCVR evaluated on all impressions against attribution

The CVR model is trained on clicks only but applied to every impression at ranking time.
That mismatch (sample selection bias) is what ESMM addresses in the next stage.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb

from adsrank.attribution import FEATURES, build_features, load, time_split
from adsrank.metrics import ads_metrics

PARAMS = dict(n_estimators=1000, max_depth=7, learning_rate=0.08, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=20, max_cat_to_onehot=1,
              tree_method="hist", enable_categorical=True, eval_metric="logloss",
              early_stopping_rounds=30)


def fit(tr: pd.DataFrame, va: pd.DataFrame, label: str, n_jobs: int, verbose):
    m = xgb.XGBClassifier(**PARAMS, n_jobs=n_jobs)
    m.fit(tr[FEATURES], tr[label], eval_set=[(va[FEATURES], va[label])], verbose=verbose)
    return m


def run(df: pd.DataFrame, n_jobs: int = 4, verbose=100):
    feats = build_features(df, time_split(df))
    tr, va, te = (feats[feats.split == s] for s in ("train", "valid", "test"))
    t = time.time()
    ctr = fit(tr, va, "click", n_jobs, verbose)
    cvr = fit(tr[tr.click == 1], va[va.click == 1], "attribution", n_jobs, verbose)
    p_ctr = ctr.predict_proba(te[FEATURES])[:, 1]
    p_cvr = cvr.predict_proba(te[FEATURES])[:, 1]            # applied to every impression
    clicked = (te.click == 1).to_numpy()
    r = {}
    r |= ads_metrics(te.click, p_ctr, "ctr")
    r |= ads_metrics(te.attribution[clicked], p_cvr[clicked], "cvr")
    r |= ads_metrics(te.attribution, p_ctr * p_cvr, "ctcvr")
    r |= {"train_seconds": time.time() - t,
          "rows_train": len(tr), "rows_valid": len(va), "rows_test": len(te),
          "base_ctr": float(te.click.mean()), "base_cvr": float(te.attribution[clicked].mean())}
    return ctr, cvr, r


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--data", required=True)
    a.add_argument("--mlflow", default=None)
    a.add_argument("--jobs", type=int, default=4)
    a.add_argument("--out", default="results/attribution_xgb.json")
    a = a.parse_args()

    df = load(a.data)
    ctr, cvr, r = run(df, a.jobs)
    print("\n== Criteo Attribution, XGBoost (test) ==")
    for k, v in r.items():
        print(f"{k:22s} {v:.4f}" if isinstance(v, float) else f"{k:22s} {v}")
    imp = pd.Series(ctr.feature_importances_, FEATURES).sort_values(ascending=False)
    print("\nCTR feature importance:\n", imp.round(3).to_string())
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps({"data": a.data, "rows": len(df)} | r, indent=1))
    print(f"\nsaved {a.out}")

    if a.mlflow:
        import mlflow
        mlflow.set_tracking_uri(a.mlflow)
        mlflow.set_experiment("criteo-attribution")
        with mlflow.start_run(run_name="xgb-ctr-cvr"):
            mlflow.log_params({k: v for k, v in PARAMS.items() if k != "eval_metric"} | {"rows": len(df)})
            mlflow.log_metrics({k: float(v) for k, v in r.items()})
            mlflow.xgboost.log_model(ctr, name="ctr_model")
            mlflow.xgboost.log_model(cvr, name="cvr_model")
        print("logged to MLflow experiment 'criteo-attribution'")


if __name__ == "__main__":
    main()
