# Roadmap

Each stage maps to a responsibility in an ads quality & ranking team.

| # | Stage | Data | What | Status |
|---|-------|------|------|--------|
| 1 | CTR / CVR baseline | Criteo Attribution, Criteo DAC | Time-based split, XGBoost CTR + CVR, AUC / LogLoss / NE / calibration, MLflow tracking | done |
| 2 | Deep CTR models | Criteo Attribution, Criteo DAC | DeepFM and DCN-v2 in PyTorch on the XGBoost split | done |
| 3 | Calibration | Criteo Attribution | Isotonic / Platt scaling, ECE, reliability diagrams, per-campaign calibration, negative-downsampling correction | done |
| 4 | CVR and delayed feedback | Criteo Attribution | Post-click conversion, ESMM (entire-space CVR), delayed conversions | done |
| 5 | Ranking, auction, bidding | Criteo Attribution | Second-price replay with logged cost, value-based bids (cpo x pCTR x pCVR), ROI-target auto-bidding, adaptive budget pacing | done |
| 5b | A/B testing | Criteo Attribution | User-level A/B of bidding policies, SRM check, delta-method ROI, power / MDE, A/A and peeking false-positive rates | done |
| 6 | Serving | XGBoost models | Skew-checked export, FastAPI scorer, model + HTTP latency, open-loop load test, KServe with limits | done |
| 7 | Rollout and monitoring | Criteo Attribution predictions, serving image | Canary InferenceService with gated 10/25/50% rollout, promote / rollback; daily PSI and calibration drift report; live `/metrics` PSI with Prometheus alert rules | done |
| 8 | Orchestration | same | Airflow DAG: drift check -> retrain decision -> skew-checked export -> canary | done (full run on kind + KServe; daily schedule in `daily.yml`) |
| 9 | Candidate retrieval | Criteo Attribution clicks | Two-tower with logQ-corrected in-batch softmax, recall@K vs popularity, feature leakage check, real-time user history | done |
| 10 | Platform | same | Feast + Redis online store with skew check; DDP training (CPU, gloo); daily scheduled DAG run with the stable image in GHCR | done |

## Datasets

- **Criteo Display Advertising Challenge (Kaggle)**: ~45M impressions over 7 days, click label,
  13 integer + 26 hashed categorical features. Standard CTR benchmark (DeepFM, DCN, DCN-v2 papers).
- **Criteo Attribution Modeling for Bidding**: ~16M impressions over 30 days with click, conversion,
  attribution, cost and cost-per-order. Used for CVR, ranking and bidding stages.

Data is not committed to this repository. Use the download scripts and follow Criteo's terms of use.
