# Roadmap

Each stage maps to a responsibility in an ads quality & ranking team.

| # | Stage | Data | What | Status |
|---|-------|------|------|--------|
| 1 | CTR / CVR baseline | Criteo Attribution, Criteo DAC | Time-based split, XGBoost CTR + CVR, AUC / LogLoss / NE / calibration, MLflow tracking | done (Attribution) |
| 2 | Deep CTR models | Criteo DAC | DCN-v2 and DeepFM in PyTorch; embedding size, cross depth ablations | planned |
| 3 | Calibration | Criteo DAC | Isotonic / Platt scaling, reliability diagrams, calibration by segment | planned |
| 4 | CVR and delayed feedback | Criteo Attribution | Post-click conversion, ESMM (entire-space CVR), delayed conversions | in progress (ESMM) |
| 5 | Ranking, auction, bidding | Criteo Attribution | eCPM = bid x pCTR x pCVR, GSP auction simulation, ROI-target auto-bidding, budget pacing | planned |
| 6 | Serving | models above | KServe deployment, p50/p99 latency, autoscaling under load | planned |

## Datasets

- **Criteo Display Advertising Challenge (Kaggle)**: ~45M impressions over 7 days, click label,
  13 integer + 26 hashed categorical features. Standard CTR benchmark (DeepFM, DCN, DCN-v2 papers).
- **Criteo Attribution Modeling for Bidding**: ~16M impressions over 30 days with click, conversion,
  attribution, cost and cost-per-order. Used for CVR, ranking and bidding stages.

Data is not committed to this repository. Use the download scripts and follow Criteo's terms of use.
