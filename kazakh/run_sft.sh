#!/usr/bin/env bash
# Kazakh SFT run on one GPU: assemble the manifest, then call the upstream launcher.
# Usage: RUN=name CORPORA="a.jsonl:N b.jsonl ..." EVAL_JSONL=val.jsonl kazakh/run_sft.sh [extra args]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${HERE}")"
PYTHON="${PYTHON:-python}"
RUN="${RUN:?set RUN}"
WORK="${WORK:-work}"
RUN_DIR="${WORK}/runs/${RUN}"
mkdir -p "${RUN_DIR}"

# NCCL 2.28 crashes while probing InfiniBand on this workstation.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

corpus_args=()
for corpus in ${CORPORA:?set CORPORA}; do
  corpus_args+=(--corpus "${corpus}")
done
"${PYTHON}" "${HERE}/assemble_train.py" "${corpus_args[@]}" \
  --epochs "${PAIRING_EPOCHS:-1}" --seed "${SEED:-0}" \
  --output-jsonl "${WORK}/prepared/train_${RUN}.jsonl" | tee "${RUN_DIR}/assemble.json"

MODEL="${MODEL:-Edge0/Audio8-TTS-Realistic-Air}" \
TRAIN_JSONL="${WORK}/prepared/train_${RUN}.jsonl" \
EVAL_JSONL="${EVAL_JSONL:-}" \
EVAL_STEPS="${EVAL_STEPS:-600}" \
OUTPUT_DIR="${RUN_DIR}/ckpt" \
EXPORT_DIR="${RUN_DIR}/export" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}" \
LEARNING_RATE="${LEARNING_RATE:-2e-5}" \
WARMUP_RATIO="${WARMUP_RATIO:-0.03}" \
NUM_TRAIN_EPOCHS=1 \
SAVE_STEPS="${SAVE_STEPS:-600}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}" \
LOGGING_STEPS="${LOGGING_STEPS:-20}" \
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}" \
REPORT_TO="${REPORT_TO:-tensorboard}" \
PYTHON="${PYTHON}" \
bash "${ROOT}/audio8_tts_sft.sh" --save_only_model "${SAVE_ONLY_MODEL:-true}" "$@"
