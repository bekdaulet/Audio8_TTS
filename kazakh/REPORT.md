# Audio8-TTS-Realistic-Air: Kazakh fine-tune, evaluation report (2026-11-28)

**Result.** The fine-tuned model `audio8-kk-v1` (akhmed: `/extra/beka/audio8-kk/work/final/audio8-kk-v1`)
pronounces Kazakh about as well as the real recordings do, by ASR. On voices it has never heard,
Kazakh character errors fall from 24 % with the base model to 8 %. Zero-shot cloning still works
(speaker similarity 0.938 vs 0.951 for the base), and English and Russian are at the base model's
level.

## Numbers

Fixed eval set `work/eval/eval_v1.jsonl` (240 items), every number averaged over two sampling seeds
(42, 43). CER and WER come from Whisper large-v3 with the language forced. SIM is the cosine of
WavLM-base-plus-sv embeddings, output vs reference clip: the same speaker scores ≈ 0.96, different
speakers ≈ 0.73.

| group (n) | what it tests | real recordings CER | base CER | **v1 CER** | base SIM | **v1 SIM** |
|---|---|---:|---:|---:|---:|---:|
| kk/seen (100) | KazakhTTS2 voices, held-out sentences | 9.5 | 15.0 | **8.2** | 0.970 | 0.964 |
| kk/unseen (60) | FLEURS Kazakh voices never seen in training (cloning) | 6.5 | 24.0 | **8.1** | 0.951 | **0.938** |
| kk/noref (25) | no reference voice | 11.3 | 15.4 | **11.1** | – | – |
| en/clone (40) | held-out LibriTTS-R readers | 1.0 | 0.8 | 1.1 | 0.964 | 0.957 |
| ru/clone (15) | held-out RuLS readers | 4.6 | 3.1 | 2.7 | 0.946 | 0.946 |

- **"Voice collapse" check.** This is the share of unseen-voice outputs that are closer to one of
  the 5 KazakhTTS2 training voices than to their own reference. It is 6 % for the base and
  **16.5 %** for v1 (13 % and 20 % on the two seeds). The first pilot, trained on KazakhTTS2
  only, reached 100 %: every unseen Kazakh voice came out as one of the 5 training voices.
- **WER is high for every system**, real recordings included (44 % on kk/seen). This is Whisper's
  Kazakh word segmentation and spelling, not the TTS. Use CER, and compare against the real
  recordings row.
- **Seed-to-seed spread** is up to ±2 CER points on the Kazakh groups and ±1 on EN/RU, so
  differences smaller than that are noise.
- **Real-recording kk/unseen SIM.** It is 0.73 because FLEURS has no speaker ids: the real
  recording of the target sentence is usually a different person from the reference clip. That
  is why SIM is measured against the reference clip.

## What made the difference (pilots, all ≈ 1 h on one RTX 5000 Ada)

| run | change | kk/unseen CER | unseen SIM | collapse |
|---|---|---:|---:|---:|
| pilot | KazakhTTS2 only (5 voices) + EN/RU replay, LR 2e-5 | 7.0 | 0.841 | 100 % |
| B / C | pitch-formant augmented pseudo-voices / more replay | 10.6 / 6.6 | 0.809 / 0.862 | 87 / 100 % |
| D | + KSD: 196 real Kazakh speakers, 55.6 h | 7.3 | 0.898 | 85 % |
| E | KazakhTTS2 cut to 12k rows per pass; every KazakhTTS2 example has a voice reference | 6.4 | 0.915 | 38 % |
| F | KazakhTTS2 cut to 6k | 5.5 | 0.920 | 37 % |
| G | LR 1e-5 | 8.5 | 0.926 | 27 % |
| **H = v1** | **LR 5e-6** | 6.7 | **0.941** | **13 %** |
| final v1 run | H recipe, full KazakhTTS2 text pool, 3 passes | 8.5† | 0.937† | 17.5 %† |

† averaged over 2 seeds at its best checkpoint (3600). It was not better than H, so H is the release.

Lessons:
1. **The collapse lives in the slow (semantic) AR.** Splicing tests showed this: pilot slow + base
   fast still collapses, and base slow + pilot fast does not.
2. **Many real speakers fix it; synthetic voice variety does not.** Augmented pseudo-voices
   didn't help.
3. **More single-voice data hurts.** The full 268 h of KazakhTTS2 would push the model back
   toward its 5 voices, so v1 uses only about 12k KazakhTTS2 clips.
4. **A low LR keeps the base model's cloning.**

## Recipe of v1
- **Data:**
  - KazakhTTS2, 6k clips per pairing pass, always paired with a same-speaker reference.
  - KSD, all 20,445 clips that pass a Whisper filter (CER ≤ 15 %), 196 speakers, 20 % of them
    without a reference.
  - LibriTTS-R and RuLS, 2k clips each per pass.
  - 2 passes with fresh reference pairings: 60,890 examples in total.
- **Text:** normalized with `audio8_tts_kk_normalize.py`, which expands numbers, dates and
  abbreviations, and lower-cased.
- **Training:**
  - full fine-tune, LR 5e-6, warmup 3 %, batch 16, 3,806 steps;
  - 52 min on one RTX 5000 Ada (32 GB);
  - DeepSpeed ZeRO-2, `NCCL_IB_DISABLE=1`.
- **Code:** branch `kazakh` of github.com/bekdaulet/Audio8_TTS, under `kazakh/`. The run
  script is `kazakh/train_v1.sh` (run from `/extra/beka/audio8-kk`).

## Listening samples
`samples/final/` holds 19 WAVs and a README with the texts and per-item numbers. The items were
picked automatically as typical, closest to the median CER/SIM, not the best ones:
- 2 KazakhTTS2 voices: reference, base, fine-tuned;
- 3 unseen FLEURS voices (cloning): reference, base, fine-tuned;
- 1 English and 1 Russian clone: base, fine-tuned.

Older pilot comparisons are in `samples/pilot/`.

## Limits and open points
- **Collapse remains higher than base:** 16.5 % vs 6 %. On some unseen voices v1 still drifts
  toward a KazakhTTS2 timbre. The most promising fix is the rest of KSD: 93 more shards, about
  650 more speakers, about 54 GB. That needs Bek's yes.
- **Small eval set:** 60 unseen-voice items, each with its own FLEURS reference clip. No human MOS yet;
  please listen to `samples/final/`.
- **ASR-based metrics:** Whisper large-v3 Kazakh is itself imperfect; real recordings score 9.5 %
  CER.
- **KSD licence:** CC BY-SA 3.0, share-alike. Whether that binds the model weights is debated.
  It matters only if v1 is published, and nothing has been uploaded anywhere.
