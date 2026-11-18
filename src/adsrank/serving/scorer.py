"""Request-time scoring: raw ad candidates -> pCTR, pCVR, expected value.

Feature logic mirrors adsrank.attribution.build_features exactly (same vocabularies, same
transforms), but works on plain dicts / numpy so a request does not go through pandas.
export.py checks that both paths give identical predictions (training-serving skew test).
"""
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

CATS = ["campaign"] + [f"cat{i}" for i in range(1, 10)]


class Scorer:
    def __init__(self, model_dir: str):
        d = Path(model_dir)
        meta = json.loads((d / "features.json").read_text())
        self.features = meta["features"]
        self.version = meta.get("version", "unknown")
        # raw value -> code; code 0 = out of vocabulary
        self.vocab = {c: {int(v): i + 1 for i, v in enumerate(vals)} for c, vals in meta["vocab"].items()}
        self.ctr, self.cvr = xgb.Booster(), xgb.Booster()
        self.ctr.load_model(d / "ctr.json")
        self.cvr.load_model(d / "cvr.json")
        for b in (self.ctr, self.cvr):
            b.set_param({"nthread": 1})          # one request = one core; scale with replicas
        # early stopping: training-time predict_proba stops at best_iteration, so must serving
        self.ranges = [(0, int(b.attributes().get("best_iteration", b.num_boosted_rounds() - 1)) + 1)
                       for b in (self.ctr, self.cvr)]

    def matrix(self, rows: list[dict]) -> np.ndarray:
        x = np.empty((len(rows), len(self.features)), dtype=np.float32)
        for j, c in enumerate(CATS):
            v = self.vocab[c]
            x[:, j] = [v.get(int(r[c]), 0) for r in rows]
        tslc = np.array([float(r["time_since_last_click"]) for r in rows])
        num = {"cpo": np.array([float(r["cpo"]) for r in rows]),
               "log_tslc": np.log1p(np.clip(tslc, 0, None)),
               "no_prev_click": (tslc < 0).astype(np.float32),
               "hour": (np.array([int(r["timestamp"]) for r in rows]) // 3600) % 24}
        for j, f in enumerate(self.features[len(CATS):], start=len(CATS)):
            x[:, j] = num[f]
        return x

    def score(self, rows: list[dict]) -> dict:
        x = self.matrix(rows)
        p_ctr = self.ctr.inplace_predict(x, iteration_range=self.ranges[0])
        p_cvr = self.cvr.inplace_predict(x, iteration_range=self.ranges[1])
        cpo = x[:, self.features.index("cpo")]
        return {"p_ctr": p_ctr, "p_cvr": p_cvr, "expected_value": cpo * p_ctr * p_cvr}
