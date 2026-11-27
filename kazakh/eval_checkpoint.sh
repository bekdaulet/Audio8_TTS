#!/usr/bin/env bash
# Synthesize the fixed evaluation manifest with one model and score it.
# Usage: kazakh/eval_checkpoint.sh MODEL_OR_DIR TAG [EVAL_JSONL]
# Writes WAVs to $EVAL_ROOT/TAG/wav and metrics to $EVAL_ROOT/TAG/metrics.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${HERE}")"
PYTHON="${PYTHON:-python}"
MODEL="$1"
TAG="$2"
EVAL_JSONL="${3:-${EVAL_JSONL:?set EVAL_JSONL or pass it as the third argument}}"
EVAL_ROOT="${EVAL_ROOT:-outputs/eval}"
OUT="${EVAL_ROOT}/${TAG}"
mkdir -p "${OUT}"

"${PYTHON}" "${ROOT}/audio8_tts_infer.py" \
  --model "${MODEL}" \
  --input-jsonl "${EVAL_JSONL}" \
  --output-dir "${OUT}/wav" \
  --batch-size "${INFER_BATCH_SIZE:-8}" \
  --seed "${SEED:-42}"

"${PYTHON}" "${HERE}/eval_metrics.py" \
  --manifest "${EVAL_JSONL}" \
  --wav-dir "${OUT}/wav" \
  --output "${OUT}/metrics.json"
