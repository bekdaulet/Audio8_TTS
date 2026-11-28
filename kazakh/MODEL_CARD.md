# audio8-kk-v1: Kazakh fine-tune of Audio8-TTS-Realistic-Air (0.6B)

Kazakh text-to-speech with zero-shot voice cloning. This is a full fine-tune of
[Edge0/Audio8-TTS-Realistic-Air](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air) that keeps
English and Russian. It is not published anywhere: the checkpoint lives on akhmed at
`/extra/beka/audio8-kk/work/final/audio8-kk-v1`.

## Use
Same interface as the base model (`audio8_tts_infer.py`, manifest rows `id, text, reference_audio,
reference_text`). **Kazakh text must be normalized and lower-cased first**, because the model was
trained that way:

```bash
echo "2024 жылы 15 наурызда" | python audio8_tts_kk_normalize.py --lowercase
# -> екі мың жиырма төртінші жылы он бесінші наурызда
```

A 3-10 s reference clip of the target voice gives the best cloning. Without a reference, the model
picks a voice itself.

Recommended inference options (`audio8_tts_infer.py`):
- `--guard-chars-per-second 5` resamples runaway outputs, which are longer than the text could take.
- `--reference-rms-dbfs -23` scales the reference to a normal level. The model copies the
  reference's loudness, and quiet references also weaken the voice match.

## Evaluation
Fixed set of 240 items, averaged over two sampling seeds. CER is from Whisper large-v3 with the
language forced; SIM is the WavLM-sv cosine to the reference clip.

| | real recordings | base | **v1** |
|---|---:|---:|---:|
| Kazakh CER, seen voices (KazakhTTS2 held-out sentences) | 9.5 | 15.0 | **8.2** |
| Kazakh CER, unseen voices (FLEURS, cloning) | 6.5 | 24.0 | **8.1** |
| Kazakh CER, no reference | 11.3 | 15.4 | **11.1** |
| Kazakh SIM, unseen voices | – | 0.951 | 0.938 |
| English CER / SIM (held-out LibriTTS-R readers) | 1.0 / – | 0.8 / 0.964 | 1.1 / 0.957 |
| Russian CER / SIM (held-out RuLS readers) | 4.6 / – | 3.1 / 0.946 | 2.7 / 0.946 |

With the recommended options (2 seeds): Kazakh CER 8.8 / 9.2 / 9.0 (seen / unseen / no reference),
unseen SIM 0.928, EN 1.0, RU 3.8.

Known limitation: some unseen-voice Kazakh outputs sound closer to one of the 5 KazakhTTS2 training
voices than to the reference. That is 16.5 % with plain inference and 7.5 % with the
recommended options, against 6 % and 1.5 % for the base model. Details are in `REPORT.md`.

## Training data
- **KazakhTTS2** (ISSAI, Nazarbayev University), CC BY 4.0: 12k clips from 5 speakers.
  Mussakhojayeva, S., Khassanov, Y., Varol, H. A. *KazakhTTS2: Extending the Open-Source Kazakh
  TTS Corpus With More Data, Speakers, and Topics.* LREC 2022.
- **Kazakh Speech Dataset (KSD)**, OpenSLR 140 (Al-Farabi Kazakh National University), CC BY-SA
  3.0: 20,445 clips (55.6 h, 196 speakers) from 28 of the 121 shards of the HF copy
  `voice-biomarkers/openslr-140-hq-Kazakh`, kept when Whisper CER ≤ 15 %. Mansurova, M.,
  Kadyrbek, N. (2023), Kazakh Speech Dataset, https://www.openslr.org/140/.
  **Share-alike:** if these weights are ever distributed, check whether CC BY-SA applies to them.
- **LibriTTS-R** train-clean-100 (CC BY 4.0; Koizumi et al., 2023) and **Russian LibriSpeech**
  (OpenSLR 96, via HF `istupakov/russian_librispeech`): 4k replay clips each, against forgetting.
- **FLEURS** Kazakh test split (CC BY 4.0) was used for evaluation only.

## Training
- full fine-tune at LR 5e-6, warmup 3 %, effective batch 16, 3,806 steps over 60,890 examples;
- 2 passes, each with fresh same-speaker reference pairings;
- 52 min on one RTX 5000 Ada 32 GB, DeepSpeed ZeRO-2.

Code is in the `kazakh/` folder of branch `kazakh` at github.com/bekdaulet/Audio8_TTS.

## Licence
The code is Apache-2.0, like upstream. For the weights, the base model's licence applies (see its
model page) together with the dataset terms above.
