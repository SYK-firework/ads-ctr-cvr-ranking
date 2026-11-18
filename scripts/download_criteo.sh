#!/usr/bin/env bash
# Download only the first N rows of the Criteo Display Advertising Challenge (Kaggle) data.
# The 4.3GB archive is streamed and decompressed on the fly; the transfer stops once N rows are read.
# The original train.txt has ~45M rows (7 days), ordered by time.
#
# usage: bash scripts/download_criteo.sh [rows] [output file]
set -euo pipefail
ROWS="${1:-2000000}"
OUT="${2:-data/train_${ROWS}.txt}"
URL="${CRITEO_URL:-https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz}"

mkdir -p "$(dirname "$OUT")"
echo "downloading first ${ROWS} rows -> ${OUT}"
# head closes the pipe when done, which stops curl/tar (SIGPIPE is expected here)
set +o pipefail
curl -sL "$URL" | tar -xzO train.txt | head -n "$ROWS" > "$OUT"
set -o pipefail

LINES=$(wc -l < "$OUT")
if [ "$LINES" -lt "$ROWS" ]; then
  echo "error: got only ${LINES} rows. Check the URL (set CRITEO_URL to override)." >&2
  exit 1
fi
echo "done: ${LINES} rows"
