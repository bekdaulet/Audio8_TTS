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

# Trainer checkpoints hold weights, code and codec but not the processor;
# copy the processor/tokenizer files from the base model snapshot.
if [[ -d "${MODEL}" && ! -f "${MODEL}/tokenizer.json" ]]; then
  "${PYTHON}" - "${MODEL}" "${BASE_MODEL:-Edge0/Audio8-TTS-Realistic-Air}" <<'EOF'
import shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download
target, base = Path(sys.argv[1]), Path(snapshot_download(sys.argv[2], local_files_only=True))
for name in ("processing_arktts.py", "processor_config.json", "special_tokens_map.json",
             "tokenizer_config.json", "tokenizer.json"):
    if not (target / name).exists():
        shutil.copy2(base / name, target / name)
EOF
fi

"${PYTHON}" "${ROOT}/audio8_tts_infer.py" \
  --model "${MODEL}" \
  --input-jsonl "${EVAL_JSONL}" \
  --output-dir "${OUT}/wav" \
  --batch-size "${INFER_BATCH_SIZE:-8}" \
  --seed "${SEED:-42}" \
  --guard-chars-per-second "${GUARD_CPS:-0}"

"${PYTHON}" "${HERE}/eval_metrics.py" \
  --manifest "${EVAL_JSONL}" \
  --wav-dir "${OUT}/wav" \
  --output "${OUT}/metrics.json"
