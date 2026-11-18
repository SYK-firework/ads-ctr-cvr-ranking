ROWS ?= 2000000
DATA_DIR ?= data
RAW = $(DATA_DIR)/train_$(ROWS).txt
PARQUET = $(DATA_DIR)/criteo_$(ROWS).parquet
MLFLOW ?=

.PHONY: install data prep train-gbdt test all

install:
	uv pip install -e ".[tracking,dev]"

data: $(RAW)
$(RAW):
	bash scripts/download_criteo.sh $(ROWS) $(RAW)

prep: $(PARQUET)
$(PARQUET): $(RAW)
	python -m adsrank.prep --input $(RAW) --out $(PARQUET)

train-gbdt: $(PARQUET)
	python -m adsrank.train_gbdt --data $(PARQUET) $(if $(MLFLOW),--mlflow $(MLFLOW),)

test:
	pytest -q

all: data prep train-gbdt
