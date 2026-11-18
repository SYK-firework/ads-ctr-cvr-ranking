"""Criteo CTR baseline: XGBoost on integer features + count-encoded categorical features."""
import argparse
import time

import pandas as pd
import xgboost as xgb

from adsrank import CAT, NUM
from adsrank.metrics import ads_metrics

FEATURES = NUM + [f"{c}_cnt" for c in CAT]
DEFAULT_PARAMS = dict(n_estimators=1000, max_depth=8, learning_rate=0.1, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
                      eval_metric="logloss", early_stopping_rounds=30)


def train(df: pd.DataFrame, params: dict | None = None, n_jobs: int = 4, verbose=100):
    params = {**DEFAULT_PARAMS, **(params or {}), "n_jobs": n_jobs}
    tr, va, te = (df[df.split == s] for s in ("train", "valid", "test"))
    t = time.time()
    model = xgb.XGBClassifier(**params)
    model.fit(tr[FEATURES], tr.label, eval_set=[(va[FEATURES], va.label)], verbose=verbose)
    p = model.predict_proba(te[FEATURES])[:, 1]
    result = ads_metrics(te.label, p, "test") | {"train_seconds": time.time() - t,
                                                 "best_iteration": float(model.best_iteration)}
    return model, params, result


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--data", required=True)
    a.add_argument("--mlflow", default=None, help="MLflow tracking URI, e.g. http://localhost:5000")
    a.add_argument("--jobs", type=int, default=4)
    a = a.parse_args()

    df = pd.read_parquet(a.data)
    model, params, r = train(df, n_jobs=a.jobs)
    print("\n== XGBoost (test) ==")
    for k, v in r.items():
        print(f"{k:20s} {v:.4f}")
    imp = pd.Series(model.feature_importances_, FEATURES).sort_values(ascending=False)
    print("\nTop 10 features:\n", imp.head(10).round(3).to_string())

    if a.mlflow:
        import mlflow
        mlflow.set_tracking_uri(a.mlflow)
        mlflow.set_experiment("criteo-ctr")
        with mlflow.start_run(run_name="xgboost"):
            mlflow.log_params({k: v for k, v in params.items() if k != "eval_metric"} | {"rows": len(df)})
            mlflow.log_metrics(r)
            mlflow.xgboost.log_model(model, name="model")
        print("logged to MLflow experiment 'criteo-ctr'")


if __name__ == "__main__":
    main()
