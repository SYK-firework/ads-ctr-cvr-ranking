ROWS ?= 2000000
DATA_DIR ?= data
RAW = $(DATA_DIR)/train_$(ROWS).txt
PARQUET = $(DATA_DIR)/criteo_$(ROWS).parquet
MLFLOW ?=

.PHONY: install data data-attribution prep train-gbdt train-attribution test all

ATTR_ROWS ?= 2000000

install:
	uv pip install -e ".[tracking,dev,data]"

# Criteo Attribution (impression -> click -> conversion, with cost): stream a subset from Hugging Face
data-attribution:
	python scripts/download_attribution.py --rows $(ATTR_ROWS) --out $(DATA_DIR)/attribution_$(ATTR_ROWS).parquet

data: $(RAW)
$(RAW):
	bash scripts/download_criteo.sh $(ROWS) $(RAW)

ATTR_PARQUET = $(DATA_DIR)/attribution_$(ATTR_ROWS).parquet

train-attribution:
	python -m adsrank.train_attribution --data $(ATTR_PARQUET) $(if $(MLFLOW),--mlflow $(MLFLOW),)

prep: $(PARQUET)
$(PARQUET): $(RAW)
	python -m adsrank.prep --input $(RAW) --out $(PARQUET)

train-gbdt: $(PARQUET)
	python -m adsrank.train_gbdt --data $(PARQUET) $(if $(MLFLOW),--mlflow $(MLFLOW),)

test:
	pytest -q

all: data prep train-gbdt
