# Scoring server image. Build from the repository root:
#   docker build -f serving/Dockerfile -t adsrank-serving:v1 .
FROM python:3.13-slim
WORKDIR /app
COPY serving/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY src/adsrank /app/adsrank
COPY serving/model /app/model
COPY scripts/loadtest.py /app/loadtest.py
ENV MODEL_DIR=/app/model PYTHONPATH=/app
EXPOSE 8080
# one worker per pod; scale out with replicas (HPA) rather than processes inside a pod
CMD ["uvicorn", "adsrank.serving.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
