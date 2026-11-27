#!/usr/bin/env python3
"""Intelligibility (ASR CER/WER) and speaker similarity for synthesized speech.

Reads an evaluation manifest (rows: id, lang, text, reference_audio and
optionally gt_audio) and a directory of WAVs named ``<id>.wav``. Transcribes
every WAV with Whisper (language forced per row), scores it against ``text``,
and computes cosine similarity between speaker embeddings of the WAV and the
reference clip. Pass ``--wav-dir gt`` to score the ground-truth recordings
(``gt_audio``) instead, which gives the ASR floor for the same sentences.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio8_tts_kk_normalize import normalize_kk  # noqa: E402

WHISPER_LANG = {"kk": "kazakh", "ru": "russian", "en": "english"}


def load_16k(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.mean(axis=1))
    if rate != 16000:
        wave = AF.resample(wave, rate, 16000)
    return wave.numpy()


def score_text(text: str, lang: str) -> str:
    """Lower-case, spell numbers (Kazakh), strip punctuation for scoring."""
    if lang == "kk":
        text = normalize_kk(text)
    text = unicodedata.normalize("NFC", text).lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class Whisper:
    def __init__(self, name: str, device: str) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.processor = WhisperProcessor.from_pretrained(name)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16)
            .to(device)
            .eval()
        )
        self.device = device

    @torch.no_grad()
    def __call__(self, audios: list[np.ndarray], lang: str) -> list[str]:
        features = self.processor(audios, sampling_rate=16000, return_tensors="pt").input_features
        ids = self.model.generate(
            features.to(self.device, torch.float16),
            language=WHISPER_LANG[lang],
            task="transcribe",
            num_beams=1,
            max_new_tokens=220,
        )
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]


class SpeakerEncoder:
    def __init__(self, name: str, device: str) -> None:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.extractor = AutoFeatureExtractor.from_pretrained(name)
        self.model = WavLMForXVector.from_pretrained(name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, audio: np.ndarray) -> torch.Tensor:
        inputs = self.extractor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
        embedding = self.model(**inputs).embeddings[0]
        return torch.nn.functional.normalize(embedding, dim=-1).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wav-dir", required=True, help="Directory of <id>.wav, or 'gt'.")
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON path.")
    parser.add_argument("--asr", default="openai/whisper-large-v3")
    parser.add_argument("--sv", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    rows = [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]
    for row in rows:
        if args.wav_dir == "gt":
            row["wav"] = str((manifest.parent / row["gt_audio"]).resolve())
        else:
            row["wav"] = str(Path(args.wav_dir).resolve() / f"{row['id']}.wav")
    missing = [row["id"] for row in rows if not Path(row["wav"]).is_file()]
    rows = [row for row in rows if Path(row["wav"]).is_file()]

    asr = Whisper(args.asr, args.device)
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_lang[row["lang"]].append(row)
    for lang, items in by_lang.items():
        for start in range(0, len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            hyps = asr([load_16k(Path(r["wav"])) for r in batch], lang)
            for row, hyp in zip(batch, hyps):
                row["hyp"] = hyp
                ref, out = score_text(row["text"], lang), score_text(hyp, lang)
                row["cer"] = jiwer.cer(ref, out) if ref else None
                row["wer"] = jiwer.wer(ref, out) if ref else None
                row["ref_chars"], row["ref_words"] = len(ref), len(ref.split())
    del asr
    torch.cuda.empty_cache()

    sv = SpeakerEncoder(args.sv, args.device)
    for row in rows:
        if row.get("reference_audio"):
            ref_path = (manifest.parent / row["reference_audio"]).resolve()
            row["sim"] = float(sv(load_16k(Path(row["wav"]))) @ sv(load_16k(ref_path)))
        row["seconds"] = sf.info(row["wav"]).duration

    summary: dict[str, dict] = {}
    for group in sorted({(r["lang"], r.get("group", "all")) for r in rows}):
        items = [r for r in rows if (r["lang"], r.get("group", "all")) == group]
        chars = sum(r["ref_chars"] for r in items)
        words = sum(r["ref_words"] for r in items)
        sims = [r["sim"] for r in items if "sim" in r]
        summary["/".join(group)] = {
            "n": len(items),
            # Corpus-level rates: errors weighted by reference length.
            "cer": sum(r["cer"] * r["ref_chars"] for r in items) / max(chars, 1),
            "wer": sum(r["wer"] * r["ref_words"] for r in items) / max(words, 1),
            "sim": float(np.mean(sims)) if sims else None,
            "seconds_mean": float(np.mean([r["seconds"] for r in items])),
        }
    result = {"wav_dir": args.wav_dir, "missing": missing, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output.with_suffix(".items.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
