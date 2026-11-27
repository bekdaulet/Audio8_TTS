#!/bin/bash
# Training recipe of audio8-kk-v1 (was scripts/pilot_h.sh). Pilot H: pilot G with the learning rate halved again (5e-6). Data as pilot F: 6k KazakhTTS2 rows/pairing
# without reference-free rows, all filtered KSD (20 % no-ref), 2k EN + 2k RU, 2 pairings.
set -e
cd /extra/beka/audio8-kk
export HF_HUB_OFFLINE=1 HF_HOME=/extra/beka/audio8-kk/.cache/hf PYTHON=env/bin/python EVAL_ROOT=outputs/eval
RUN=pilot_h CORPORA="work/prepared/kk_pilot.jsonl:6000:0 work/prepared/ksd.jsonl work/prepared/en_train.jsonl:2000 work/prepared/ru_train.jsonl:2000" \
  PAIRING_EPOCHS=2 EVAL_JSONL=work/prepared/val_mix.jsonl SAVE_TOTAL_LIMIT=20 LEARNING_RATE=5e-6 \
  Audio8_TTS/kazakh/run_sft.sh
Audio8_TTS/kazakh/eval_checkpoint.sh work/runs/pilot_h/export pilot_h work/eval/eval_v1.jsonl
env/bin/python eval/collapse_check.py pilot_h
echo PILOT_H_OK
