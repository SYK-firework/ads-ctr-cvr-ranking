#!/usr/bin/env bash
# Gated canary rollout on the kind cluster.
#   1. deploy CANARY_IMAGE as isvc/adsrank-canary (k8s/inferenceservice-canary.yaml)
#   2. run scripts/canary.py from a pod inside the cluster (10% -> 25% -> 50% of traffic, gate per step)
#   3. promote: point isvc/adsrank at CANARY_IMAGE and delete the canary
#      rollback: delete the canary; stable is untouched
# usage: bash scripts/canary_rollout.sh adsrank-serving:v2
set -euo pipefail
CANARY_IMAGE=${1:?usage: canary_rollout.sh IMAGE}
STEPS=${STEPS:-0.1,0.25,0.5}
RPS=${RPS:-50}
SECONDS_PER_STEP=${SECONDS_PER_STEP:-30}
POD=adsrank-canary-check

sed "s#adsrank-serving:v2#${CANARY_IMAGE}#" k8s/inferenceservice-canary.yaml | kubectl apply -f -
kubectl wait --for=condition=Ready isvc/adsrank-canary --timeout=300s

kubectl delete pod "$POD" --ignore-not-found
kubectl run "$POD" --image="$CANARY_IMAGE" --image-pull-policy=IfNotPresent --restart=Never \
  --overrides="{\"spec\":{\"containers\":[{\"name\":\"$POD\",\"image\":\"$CANARY_IMAGE\",\"imagePullPolicy\":\"IfNotPresent\",
    \"resources\":{\"limits\":{\"cpu\":\"1\",\"memory\":\"512Mi\"}},
    \"command\":[\"python\",\"/app/canary.py\",\"--samples\",\"/app/model/sample_requests.json\",
      \"--steps\",\"$STEPS\",\"--rps\",\"$RPS\",\"--seconds\",\"$SECONDS_PER_STEP\",\"--out\",\"/tmp/canary.json\"]}]}}"

phase=""
for _ in $(seq 1 180); do
  phase=$(kubectl get pod "$POD" -o jsonpath='{.status.phase}')
  [[ "$phase" == "Succeeded" || "$phase" == "Failed" ]] && break
  sleep 5
done
kubectl logs "$POD"

if [[ "$phase" == "Succeeded" ]]; then
  echo "promote: isvc/adsrank -> ${CANARY_IMAGE}"
  kubectl patch isvc adsrank --type json \
    -p "[{\"op\":\"replace\",\"path\":\"/spec/predictor/containers/0/image\",\"value\":\"${CANARY_IMAGE}\"}]"
  kubectl wait --for=condition=Ready isvc/adsrank --timeout=300s
  kubectl delete isvc adsrank-canary
else
  echo "rollback (pod phase: ${phase:-unknown}): deleting isvc/adsrank-canary, stable unchanged"
  kubectl delete isvc adsrank-canary
  exit 1
fi
