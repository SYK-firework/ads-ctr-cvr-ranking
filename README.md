# ads-ctr-cvr-ranking

CTR / CVR prediction and ad ranking on Criteo public datasets, built as an end-to-end
**ads quality & ranking** pipeline: from click prediction to calibrated probabilities,
conversion modeling, auction-based ranking, auto-bidding and low-latency serving.

## Why this project

Search and display ads platforms rank candidates by expected value per impression:

```
eCPM = bid x pCTR            (cost-per-click ads)
eCPM = bid x pCTR x pCVR     (cost-per-conversion / ROI-driven ads)
```

That makes the quality of `pCTR` and `pCVR` - both their **ranking** (AUC) and their
**calibration** (are 2% predictions really clicked 2% of the time?) - the core of ad quality,
advertiser ROI and platform revenue. This repository builds and evaluates each piece.

| Ads quality & ranking problem | Where in this repo |
|---|---|
| CTR prediction at scale | Stage 1-2: XGBoost baseline, DCN-v2 / DeepFM |
| Probability accuracy for pricing | Stage 3: calibration |
| CVR prediction, sample selection bias | Stage 4: ESMM, delayed feedback |
| Automated auctions and bidding, advertiser ROI | Stage 5: GSP simulation, ROI-target bidding |
| High-throughput, low-latency serving | Stage 6: KServe, p99 latency |

See [docs/roadmap.md](docs/roadmap.md) for the full plan.

## Results

Criteo DAC, first 2M rows, time-based split (80/10/10). Test set.

| Model | AUC | LogLoss | NE | Calibration |
|---|---|---|---|---|
| XGBoost (int + count-encoded cat) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| DCN-v2 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

## Quickstart

```bash
uv venv && source .venv/bin/activate
make install              # package + mlflow + pytest
make test                 # smoke tests on synthetic Criteo-format data
make data ROWS=2000000    # stream the first 2M rows of Criteo DAC into data/
make prep ROWS=2000000
make train-gbdt ROWS=2000000 MLFLOW=http://localhost:5000   # MLFLOW is optional
```

## Layout

```
src/adsrank/
  prep.py         Criteo preprocessing (time split, log transform, vocab with OOV, count encoding)
  metrics.py      AUC, LogLoss, Normalized Entropy, calibration
  train_gbdt.py   XGBoost CTR baseline with optional MLflow tracking
scripts/
  download_criteo.sh   stream the first N rows without downloading the full 4.3GB archive
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
