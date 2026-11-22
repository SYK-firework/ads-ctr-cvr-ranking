ROWS ?= 2000000
DATA_DIR ?= data
RAW = $(DATA_DIR)/train_$(ROWS).txt
PARQUET = $(DATA_DIR)/criteo_$(ROWS).parquet
MLFLOW ?=

.PHONY: install data data-attribution prep train-gbdt train-attribution train-esmm calibrate bid abtest train-deepctr delayed export bench serve loadtest image deploy loadtest-cluster canary-image canary drift two-tower dac-deep test all

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
	python -m adsrank.run_bidding --preds $(DATA_DIR)/preds_test.parquet --history $(ATTR_PARQUET)

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

# ---- serving ----
IMAGE ?= adsrank-serving:v1
URL ?= http://localhost:8080/predict
RPS ?= 25,50,100,200,400

# train + export models, vocabularies and sample requests; fails on training-serving skew
export:
	python -m adsrank.serving.export --data $(ATTR_PARQUET)

# model-only latency by number of candidates per request
bench:
	python -m adsrank.serving.bench --model serving/model

# local HTTP server (Ctrl+C to stop)
serve:
	MODEL_DIR=serving/model uvicorn adsrank.serving.app:app --port 8080 --workers 1

# open-loop load test against URL
loadtest:
	python scripts/loadtest.py --url $(URL) --rps $(RPS) --plot results/serving_latency.png

# container image, loaded into the kind cluster, deployed with KServe
image:
	docker build -f serving/Dockerfile -t $(IMAGE) . && kind load docker-image $(IMAGE) --name mlp

deploy:
	kubectl apply -f k8s/inferenceservice.yaml && kubectl wait --for=condition=Ready isvc/adsrank --timeout=300s

# load test from a pod inside the cluster (avoids kubectl port-forward as a bottleneck)
CLUSTER_RPS ?= 25,50,100,200
loadtest-cluster:
	kubectl delete pod adsrank-loadtest --ignore-not-found
	kubectl run adsrank-loadtest --image=$(IMAGE) --image-pull-policy=IfNotPresent --restart=Never \
	  --overrides='{"spec":{"containers":[{"name":"adsrank-loadtest","image":"$(IMAGE)","imagePullPolicy":"IfNotPresent","resources":{"limits":{"cpu":"1","memory":"512Mi"}},"command":["python","/app/loadtest.py","--url","http://adsrank-predictor/predict","--samples","/app/model/sample_requests.json","--rps","$(CLUSTER_RPS)","--out","/tmp/lt.json"]}]}}'
	kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/adsrank-loadtest --timeout=900s; kubectl logs adsrank-loadtest

# two-tower candidate retrieval (query tower: request context, ad tower: campaign), recall@K
two-tower:
	python -m adsrank.two_tower --data $(ATTR_PARQUET)

# ---- rollout and monitoring ----
CANARY_IMAGE ?= adsrank-serving:v2

# build the new model version as a separate image tag and load it into kind
canary-image:
	docker build -f serving/Dockerfile -t $(CANARY_IMAGE) . && kind load docker-image $(CANARY_IMAGE) --name mlp

# gated canary: 10% -> 25% -> 50% of traffic, promote or roll back (needs `make deploy` first)
canary:
	bash scripts/canary_rollout.sh $(CANARY_IMAGE)

# daily score / campaign-mix PSI and calibration of the test window vs the validation window
drift:
	python -m adsrank.run_drift --ref $(DATA_DIR)/preds_valid.parquet --cur $(DATA_DIR)/preds_test.parquet

prep: $(PARQUET)
$(PARQUET): $(RAW)
	python -m adsrank.prep --input $(RAW) --out $(PARQUET)

train-gbdt: $(PARQUET)
	python -m adsrank.train_gbdt --data $(PARQUET) $(if $(MLFLOW),--mlflow $(MLFLOW),)

# DeepFM and DCN-v2 on Criteo DAC (same split as train-gbdt)
dac-deep: $(PARQUET)
	python -m adsrank.train_dac_deep --data $(PARQUET) --model deepfm
	python -m adsrank.train_dac_deep --data $(PARQUET) --model dcnv2

test:
	pytest -q

all: data prep train-gbdt
