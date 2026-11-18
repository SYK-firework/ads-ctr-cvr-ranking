# ads-ctr-cvr-ranking

CTR / CVR prediction, calibration and value-based bidding on Criteo's public
[Attribution dataset](https://huggingface.co/datasets/criteo/criteo-attribution-dataset),
built as an end-to-end **ads quality & ranking** pipeline.

## Key findings

On 2M impressions with a held-out, later-in-time test window:

- **XGBoost is the strongest CVR / CTCVR model** (CVR AUC 0.850 vs 0.845 for ESMM) and trains about 4x faster.
- **ESMM gives no gain over click-only CVR training** on this data. The public sample has a ~36%
  click rate, so the clicked subset is large and close to the impression distribution, leaving
  little sample selection bias to correct.
- **All models under-predict CTR by 3-4%** on the test window (temporal drift in base CTR),
  which is why calibration is treated as its own stage.
- **Calibrating pCTR and pCVR separately made pCTR x pCVR less calibrated** (0.989 -> 0.962):
  their errors had been cancelling. Calibrate the quantity you bid on.
- **Value-based bidding (`cpo x pCTR x pCVR`) buys 43% more conversion value than CTR-only
  bidding at the same budget** in a second-price replay; CTR-only bids halve conversions.

## Why this project

Ads platforms rank candidates by expected value per impression:

```
eCPM = bid x pCTR            (cost-per-click ads)
eCPM = bid x pCTR x pCVR     (cost-per-conversion / ROI-driven ads)
```

That makes the quality of `pCTR` and `pCVR` - both their **ranking** (AUC) and their
**calibration** (are 2% predictions really clicked 2% of the time?) - the core of ad quality,
advertiser ROI and platform revenue. This repository builds and evaluates each piece.

| Ads quality & ranking problem | In this repo | Status |
|---|---|---|
| CTR / CVR prediction | XGBoost CTR + CVR, time-based split, leakage-safe features | done |
| CVR sample selection bias | ESMM vs click-only CVR loss (controlled comparison) | done |
| Probability accuracy for pricing | Platt / Isotonic, ECE, per-campaign calibration, downsampling correction | done |
| Automated bidding, advertiser ROI | Second-price replay, value-based bids, ROI-target auto-bidding, budget pacing | done (pacing needs tuning) |
| Deep CTR models | DCN-v2 / DeepFM on Criteo DAC | planned |
| Delayed conversions | Delayed-feedback CVR | planned |
| Low-latency serving | KServe deployment, p99 latency | planned |

See [docs/roadmap.md](docs/roadmap.md) for the full plan.

## Results

### Criteo Attribution: CTR + CVR (impression -> click -> attributed conversion)

First 2M impressions (~3.75 days), time-based split (70/10/20). Test set.
CVR is evaluated on clicked impressions; CTCVR = pCTR x pCVR on all impressions.

| Model | CTR AUC | CTR NE | CTR Calib. | CVR AUC | CVR NE | CVR Calib. | CTCVR AUC |
|---|---|---|---|---|---|---|---|
| XGBoost, separate CTR / CVR | 0.6766 | 0.9273 | 0.966 | 0.8502 | 0.7406 | 1.023 | 0.8502 |
| NN, CVR loss on clicks only | 0.6764 | 0.9276 | 0.966 | 0.8457 | 0.7475 | 1.024 | 0.8475 |
| ESMM (CTCVR loss on all impressions) | 0.6766 | 0.9277 | 0.960 | 0.8446 | 0.7499 | 1.023 | 0.8472 |

The two neural rows share the same architecture (shared embeddings, CTR and CVR towers) and
differ only in how the CVR tower is trained, which isolates the effect of the ESMM objective.

Observations from the XGBoost baseline ([results/attribution_xgb.json](results/attribution_xgb.json)):

- The strongest CTR feature is time since the user's last click. Criteo traffic is largely
  retargeting, so recent engagement dominates the click signal.
- CTR calibration is 0.966: the test window has a higher base CTR (35.9%) than training,
  a small temporal drift that a frequently retrained or recalibrated model would absorb.
- CTCVR AUC is almost identical to CVR AUC (0.85016 vs 0.85022): impressions without a click
  are scored as low as clicked-but-not-converted ones, so ranking by pCTR x pCVR on all
  impressions is driven mostly by the CVR model.

XGBoost vs neural models, and ESMM vs click-only CVR training:

- **XGBoost is the strongest CVR / CTCVR model here** (CVR AUC 0.8502 vs 0.8446-0.8457).
  With 11 categorical fields and 3 numeric features, gradient-boosted trees fit this tabular
  signal at least as well as an embedding MLP, at a fraction of the training cost (72s vs ~300s on CPU).
- **ESMM shows no gain over the click-only objective** on this data (CVR AUC 0.8446 vs 0.8457,
  CTCVR AUC 0.8472 vs 0.8475; single seed, differences are within run-to-run noise).
  ESMM targets sample selection bias when clicks are a small, skewed subset of impressions
  (the original paper works with click rates of a few percent). In this public sample the click
  rate is ~36%, so the clicked subset (~500K rows) is large and close to the impression
  distribution, leaving little bias for ESMM to correct.
- All models under-predict CTR on the test window (calibration 0.96-0.97), consistent with the
  base-rate drift noted above; this is the target of the calibration stage.

The public sample keeps a subset of non-clicked impressions, so the base CTR (~35%) is far above
production levels. Calibration is reported against this sample; correcting for negative
downsampling is part of the calibration stage.

### Calibration (stage 3)

`make calibrate` fits Platt and Isotonic calibrators on the validation window and evaluates on
test: LogLoss, calibration ratio, Expected Calibration Error (ECE), per-campaign calibration error
and AUC, with reliability diagrams in `results/`. It also runs a negative-downsampling experiment:
train CTR with non-clicks kept at 25%, then map predictions back with
`p = p' w / (1 - p' + p' w)` and check calibration on the full test set.

| Test set | Raw | Platt | Isotonic |
|---|---|---|---|
| CTR calibration (pred / actual) | 0.966 | 0.976 | 0.976 |
| CTR ECE | 0.0122 | 0.0090 | 0.0092 |
| CTR per-campaign error | 0.0489 | 0.0434 | 0.0427 |
| CVR calibration | 1.023 | 0.984 | 0.985 |
| CVR ECE | 0.0023 | 0.0030 | **0.0017** |
| CVR per-campaign error | 0.1388 | 0.1328 | **0.1273** |
| CTCVR calibration (pCTR x pCVR) | **0.989** | 0.962 | 0.963 |
| CTR / CVR / CTCVR AUC | 0.677 / 0.850 / 0.850 | unchanged | unchanged |

Negative downsampling (non-clicks kept at 25%, train CTR 0.36 -> 0.68):

| Test set | Trained on sampled data | After correction | Full-data model |
|---|---|---|---|
| CTR calibration | 1.819 | **0.963** | 0.966 |
| CTR ECE | 0.294 | **0.013** | 0.012 |
| LogLoss | 0.802 | **0.608** | 0.605 |
| AUC | 0.672 | 0.672 | 0.677 |

![Reliability with downsampling](results/reliability_downsampling.png)

Findings:

- Calibrating on the validation window halves part of the CTR drift (0.966 -> 0.976) and lowers
  ECE by ~25% without changing AUC. The rest of the gap is drift that continues past the
  validation window, which favors frequent recalibration over a one-off fit.
- **Calibrating CTR and CVR separately made the product worse** (CTCVR 0.989 -> 0.962). Raw CTR
  was under-predicted and raw CVR over-predicted, and the errors cancelled in pCTR x pCVR.
  Fixing each factor removed the cancellation. The quantity used for bidding should be
  calibrated as a whole (or the factors calibrated jointly), not piece by piece.
- Training on negative-downsampled data inflates CTR by 82%; the closed-form correction brings
  it back to the full-data model's calibration with no AUC change, so downsampling for
  training cost is safe as long as predictions are corrected before use.

### Bidding and auctions (stage 5)

`make bid` replays the test window as a second-price auction: the logged `cost` is the market
price, a bid wins if `bid >= cost`, and a win pays `cost` and receives the logged outcome.
Bids are `alpha x cpo x pCTR x pCVR` (expected value of the impression). Compared policies:
random, CTR-only, pCTR x pCVR raw, pCTR x pCVR calibrated. Experiments: fixed budget,
value-vs-spend curves, ROI-target auto-bidding (largest alpha meeting the ROI goal) and a
PI-controller budget pacer.

Replay caveats: only impressions that were won are logged, so the full bid landscape and
competition are not observed, and outcomes are assumed independent of the bid. The results
compare policies against each other rather than estimate production revenue.

Fixed budget (20% of total logged spend), test replay:

| Policy | Conversions | Value | ROI | CPA |
|---|---|---|---|---|
| Random bids | 4,253 | 349 | 15.7 | 0.0052 |
| CTR only (`cpo x pCTR`) | 2,120 | 355 | 16.0 | 0.0105 |
| `cpo x pCTR x pCVR` | 4,276 | **498** | **22.4** | 0.0052 |
| `cpo x pCTR x pCVR`, calibrated | 4,321 | 497 | 22.3 | 0.0052 |

ROI-target auto-bidding (target ROI 13.8 = 1.5x the ROI of buying every impression):

| Policy | Spend | Conversions | Value | ROI |
|---|---|---|---|---|
| Random bids | 36.6 | 5,893 | 511 | 13.9 |
| CTR only | 42.4 | 3,787 | 592 | 13.9 |
| `cpo x pCTR x pCVR` | 61.2 | 8,043 | 859 | 14.0 |
| `cpo x pCTR x pCVR`, calibrated | 62.3 | 8,168 | **863** | 13.9 |

![Value vs spend](results/bidding_value_vs_spend.png)

Findings:

- **Value-based bids buy 43% more conversion value than random or CTR-only bids at the same
  budget** (ROI 22.4 vs 15.7-16.0).
- **CTR-only bidding optimizes the wrong objective**: it buys clicks on high-CPO campaigns and
  ends up with half the conversions of random bidding at the same spend.
- Under an ROI constraint, value-based bidding can spend **46% more than CTR-only** (and 70%
  more than random) while still meeting the target, i.e. more volume for the advertiser at the
  same efficiency.
- Calibration barely changes these results because the bid multiplier is tuned per policy, which
  absorbs a global scale error. Calibration matters when bids are used as absolute values
  (no per-campaign multiplier) or when probabilities are compared across models.
- **The PI budget pacer needs work**: it used 98.6% of the budget, but one step overspent its
  plan by 4.9x and total value was 35% below a hindsight-tuned constant multiplier (325 vs 497).
  Causes: a cold start far from the right multiplier, integral wind-up, and an even spend plan
  while traffic value varies over the day. Next: warm start from the previous period,
  anti-windup, and a spend plan shaped by expected traffic.

### Criteo DAC: CTR benchmark

Criteo DAC, first 2M rows, time-based split (80/10/10). Test set.

| Model | AUC | LogLoss | NE | Calibration |
|---|---|---|---|---|
| XGBoost (int + count-encoded cat) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| DCN-v2 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

## Quickstart

```bash
uv venv && source .venv/bin/activate
make install              # package + mlflow + datasets + pytest
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # for ESMM (CPU build)
make test                 # smoke tests on synthetic Criteo-format data
make data ROWS=2000000    # stream the first 2M rows of Criteo DAC into data/
make data-attribution ATTR_ROWS=2000000   # stream 2M rows of Criteo Attribution from Hugging Face
make train-attribution ATTR_ROWS=2000000  # CTR + CVR XGBoost baseline -> results/attribution_xgb.json
make train-esmm ATTR_ROWS=2000000         # ESMM vs click-only CVR loss -> results/attribution_*.json
make calibrate ATTR_ROWS=2000000          # calibration study + test predictions
make bid                                  # bidding / auction replay on the test window
make prep ROWS=2000000
make train-gbdt ROWS=2000000 MLFLOW=http://localhost:5000   # MLFLOW is optional
```

## Layout

```
src/adsrank/
  prep.py         Criteo preprocessing (time split, log transform, vocab with OOV, count encoding)
  metrics.py      AUC, LogLoss, Normalized Entropy, calibration
  train_gbdt.py   XGBoost CTR baseline with optional MLflow tracking
  attribution.py        Criteo Attribution features, leakage-safe columns, time split
  train_attribution.py  CTR + CVR XGBoost baseline, CTCVR evaluation
  esmm.py               ESMM in PyTorch (shared embeddings, CTR / CVR towers, pCTCVR = pCTR x pCVR)
  train_esmm.py         train ESMM or the same network with a click-only CVR loss
  calibration.py        Platt / Isotonic, ECE, reliability tables, downsampling correction
  run_calibration.py    calibration study on Criteo Attribution
  bidding.py            second-price replay, budget / ROI-target auto-bidding, PI pacing
  run_bidding.py        bidding policy comparison
scripts/
  download_criteo.sh        stream the first N rows without downloading the full 4.3GB archive
  download_attribution.py   stream N rows of Criteo Attribution from Hugging Face into Parquet
tests/            smoke tests on synthetic data
docs/roadmap.md   stages and datasets
```

## Evaluation notes

- **Time-based split.** Criteo rows are ordered by time; random splits leak future
  information and overstate AUC. The vocabulary is built on the train window only.
- **NE (Normalized Entropy)** is reported alongside LogLoss so results are comparable across
  datasets with different base CTRs.
- **Calibration** is tracked for every model because ranking by `bid x pCTR` and pricing
  depend on probability values, not just their order.

## Data

Criteo datasets are released by Criteo AI Lab for research use. They are not included in this
repository; the scripts download them from the original source.

- Criteo Attribution Modeling for Bidding: https://huggingface.co/datasets/criteo/criteo-attribution-dataset
  (Diemert et al., 2017, "Attribution Modeling Increases Efficiency of Bidding in Display Advertising")
- Criteo Display Advertising Challenge: https://www.kaggle.com/c/criteo-display-ad-challenge
