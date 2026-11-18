"""HTTP server: POST /predict scores a batch of ad candidates for one request.

  request : {"instances": [{"campaign": ..., "cat1": ..., ..., "cpo": ..., "time_since_last_click": ...,
                            "timestamp": ...}, ...]}
  response: {"model_version": ..., "predictions": [{"p_ctr": ..., "p_cvr": ..., "expected_value": ...}, ...]}

run: MODEL_DIR=serving/model uvicorn adsrank.serving.app:app --port 8080
"""
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel

from adsrank.serving.scorer import Scorer

scorer = Scorer(os.environ.get("MODEL_DIR", "serving/model"))
app = FastAPI(title="adsrank")


class Request(BaseModel):
    instances: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "model_version": scorer.version}


@app.post("/predict")
def predict(req: Request):
    t = time.perf_counter()
    s = scorer.score(req.instances)
    preds = [{"p_ctr": float(a), "p_cvr": float(b), "expected_value": float(c)}
             for a, b, c in zip(s["p_ctr"], s["p_cvr"], s["expected_value"])]
    return {"model_version": scorer.version, "predictions": preds,
            "server_ms": round((time.perf_counter() - t) * 1000, 3)}
