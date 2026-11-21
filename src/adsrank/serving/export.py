"""Train the XGBoost CTR / CVR models and export everything the server needs.

Writes to --out (default serving/model):
  ctr.json, cvr.json        XGBoost boosters
  features.json             feature order and categorical vocabularies (train window only)
  sample_requests.json      raw test-window candidates for the load test
  reference_scores.json     validation-window pCTR deciles, the reference for live drift (/metrics)
Then runs a training-serving skew check: the same test rows scored through the training feature
pipeline (pandas) and through the serving path (Scorer) must give identical predictions.

usage: python -m adsrank.serving.export --data data/attribution_2000000.parquet
"""
import argparse
import json
from pathlib import Path

import numpy as np

from adsrank.attribution import CATS, FEATURES, build_features, load, time_split
from adsrank.serving.live import REFERENCE_FILE, reference_from_scores
from adsrank.serving.scorer import Scorer
from adsrank.train_attribution import fit

RAW = CATS + ["cpo", "time_since_last_click", "timestamp"]


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--data", required=True)
    a.add_argument("--out", default="serving/model")
    a.add_argument("--samples", type=int, default=20000)
    a.add_argument("--jobs", type=int, default=4)
    a = a.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load(a.data)
    split = time_split(df)
    feats = build_features(df, split)
    tr, va, te = (feats[feats.split == s] for s in ("train", "valid", "test"))
    ctr = fit(tr, va, "click", a.jobs, verbose=False)
    cvr = fit(tr[tr.click == 1], va[va.click == 1], "attribution", a.jobs, verbose=False)
    ctr.get_booster().save_model(out / "ctr.json")
    cvr.get_booster().save_model(out / "cvr.json")

    # vocabularies exactly as build_features made them: code i <-> i-th kept raw value (0 = OOV)
    vocab = {}
    for c in CATS:
        counts = df.loc[split == "train", c].value_counts()
        vocab[c] = [int(v) for v in counts[counts >= 20].index]
    (out / "features.json").write_text(json.dumps(
        {"features": FEATURES, "vocab": vocab, "version": Path(a.data).stem}, indent=0))

    # reference distribution for live drift: validation-window pCTR, as served (same booster)
    (out / REFERENCE_FILE).write_text(json.dumps(reference_from_scores(ctr.predict_proba(va[FEATURES])[:, 1])))

    raw = df.loc[te.index, RAW].sample(n=min(a.samples, len(te)), random_state=0)
    rows = [{k: (float(v) if k == "cpo" else int(v)) for k, v in r.items()} for r in raw.to_dict("records")]
    (out / "sample_requests.json").write_text(json.dumps(rows))

    # training-serving skew check
    scorer = Scorer(out)
    s = scorer.score(rows)
    ref_ctr = ctr.predict_proba(te.loc[raw.index, FEATURES])[:, 1]
    ref_cvr = cvr.predict_proba(te.loc[raw.index, FEATURES])[:, 1]
    diff = max(float(np.abs(s["p_ctr"] - ref_ctr).max()), float(np.abs(s["p_cvr"] - ref_cvr).max()))
    print(f"exported to {out}: ctr.json, cvr.json, features.json, reference_scores.json, sample_requests.json ({len(rows):,} rows)")
    print(f"training-serving skew check: max |diff| = {diff:.2e} -> {'OK' if diff < 1e-5 else 'MISMATCH'}")
    if diff >= 1e-5:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
