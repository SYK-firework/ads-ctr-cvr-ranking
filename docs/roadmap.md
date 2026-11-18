# Roadmap

Each stage maps to a responsibility in an ads quality & ranking team.

| # | Stage | Data | What | Status |
|---|-------|------|------|--------|
| 1 | CTR / CVR baseline | Criteo Attribution, Criteo DAC | Time-based split, XGBoost CTR + CVR, AUC / LogLoss / NE / calibration, MLflow tracking | done (Attribution) |
| 2 | Deep CTR models | Criteo Attribution (DAC planned) | DeepFM and DCN-v2 in PyTorch on the XGBoost split | done |
| 3 | Calibration | Criteo Attribution | Isotonic / Platt scaling, ECE, reliability diagrams, per-campaign calibration, negative-downsampling correction | done |
| 4 | CVR and delayed feedback | Criteo Attribution | Post-click conversion, ESMM (entire-space CVR), delayed conversions | ESMM and delayed feedback done |
| 5 | Ranking, auction, bidding | Criteo Attribution | Second-price replay with logged cost, value-based bids (cpo x pCTR x pCVR), ROI-target auto-bidding, adaptive budget pacing | done |
| 5b | A/B testing | Criteo Attribution | User-level A/B of bidding policies, SRM check, delta-method ROI, power / MDE, A/A and peeking false-positive rates | done |
| 6 | Serving | XGBoost models | Skew-checked export, FastAPI scorer, model + HTTP latency, open-loop load test, KServe with limits | implemented |

## Datasets

- **Criteo Display Advertising Challenge (Kaggle)**: ~45M impressions over 7 days, click label,
  13 integer + 26 hashed categorical features. Standard CTR benchmark (DeepFM, DCN, DCN-v2 papers).
- **Criteo Attribution Modeling for Bidding**: ~16M impressions over 30 days with click, conversion,
  attribution, cost and cost-per-order. Used for CVR, ranking and bidding stages.

Data is not committed to this repository. Use the download scripts and follow Criteo's terms of use.
