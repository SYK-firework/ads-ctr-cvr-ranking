ROWS ?= 2000000
DATA_DIR ?= data
RAW = $(DATA_DIR)/train_$(ROWS).txt
PARQUET = $(DATA_DIR)/criteo_$(ROWS).parquet
MLFLOW ?=

.PHONY: install data data-attribution prep train-gbdt train-attribution train-esmm calibrate bid abtest train-deepctr delayed test all

ATTR_ROWS ?= 2000000

install:
	uv pip install -e ".[tracking,dev,data]"
	uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Criteo Attribution (impression -> click -> conversion, with cost): stream a subset from Hugging Face
data-attribution:
	python scripts/download_attribution.py --rows $(ATTR_ROWS) --out $(DATA_DIR)/attribution_$(ATTR_ROWS).parquet

data: $(RAW)
$(RAW):
	bash scripts/download_criteo.sh $(ROWS) $(RAW)

ATTR_PARQUET = $(DATA_DIR)/attribution_$(ATTR_ROWS).parquet

train-attribution:
	python -m adsrank.train_attribution --data $(ATTR_PARQUET) $(if $(MLFLOW),--mlflow $(MLFLOW),)

# ESMM vs the same network with a click-only CVR loss
train-esmm:
	python -m adsrank.train_esmm --data $(ATTR_PARQUET) --cvr-loss esmm $(if $(MLFLOW),--mlflow $(MLFLOW),)
	python -m adsrank.train_esmm --data $(ATTR_PARQUET) --cvr-loss clicked $(if $(MLFLOW),--mlflow $(MLFLOW),)

# Calibration study (Platt / Isotonic, ECE, per-campaign, negative-downsampling correction)
# also writes test-set predictions used by the bidding simulation
calibrate:
	python -m adsrank.run_calibration --data $(ATTR_PARQUET)

# Bidding replay: fixed budget, value-vs-spend curves, ROI-target auto-bidding, budget pacing
bid:
	python -m adsrank.run_bidding --preds $(DATA_DIR)/preds_test.parquet

# Simulated A/B test of bidding policies (user-level split, SRM, delta method, power, A/A, peeking)
abtest:
	python -m adsrank.run_abtest --preds $(DATA_DIR)/preds_test.parquet

# Deep CTR / CVR models on the same split as XGBoost
train-deepctr:
	python -m adsrank.train_deepctr --data $(ATTR_PARQUET) --model deepfm $(if $(MLFLOW),--mlflow $(MLFLOW),)
	python -m adsrank.train_deepctr --data $(ATTR_PARQUET) --model dcnv2 $(if $(MLFLOW),--mlflow $(MLFLOW),)

# Delayed-feedback CVR: naive vs waiting window vs DFM, against complete labels
delayed:
	python -m adsrank.run_delayed --data $(ATTR_PARQUET)

prep: $(PARQUET)
$(PARQUET): $(RAW)
	python -m adsrank.prep --input $(RAW) --out $(PARQUET)

train-gbdt: $(PARQUET)
	python -m adsrank.train_gbdt --data $(PARQUET) $(if $(MLFLOW),--mlflow $(MLFLOW),)

test:
	pytest -q

all: data prep train-gbdt
