#!/usr/bin/env python3
"""Extract a speaker-balanced subset of the Kazakh Speech Dataset (OpenSLR 140).

Reads the Hugging Face parquet copy (``voice-biomarkers/openslr-140-hq-Kazakh``:
columns ``audio`` and ``transcription``). The audio file name is
``<participant>_<n>.wav``; clips sharing a participant id were checked to be
one voice (WavLM speaker similarity 0.965 within ids vs 0.69 between ids), so
the id is used as the speaker. Writes WAVs and a raw manifest with ids
``ksd_<participant>_<n>``.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio8_tts_kk_normalize import normalize_kk  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-speaker", type=int, default=120)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "raw.jsonl"
    done, seen_ids = set(), set()
    if manifest.exists():
        for line in manifest.open(encoding="utf-8"):
            row = json.loads(line)
            done.add(row["shard"])
            seen_ids.add(row["id"])
    rng = random.Random(args.seed)
    totals = {"rows": 0, "speakers": set(), "seconds": 0.0}
    with manifest.open("a", encoding="utf-8") as handle:
        for shard in args.parquet:
            if shard.name in done:
                continue
            table = pq.read_table(shard)
            audio, texts = table.column("audio"), table.column("transcription").to_pylist()
            by_speaker: dict[str, list[int]] = defaultdict(list)
            for index in range(len(texts)):
                name = Path(audio[index]["path"].as_py()).stem
                by_speaker[name.split("_")[0]].append(index)
            for speaker, indices in sorted(by_speaker.items()):
                rng.shuffle(indices)
                kept = 0
                for index in indices:
                    if kept >= args.per_speaker:
                        break
                    text = normalize_kk(texts[index] or "", lowercase=True)
                    if not text:
                        continue
                    payload = audio[index]["bytes"].as_py()
                    try:
                        samples, rate = sf.read(io.BytesIO(payload), dtype="float32")
                    except sf.LibsndfileError:  # a few clips in the HF copy are not decodable
                        continue
                    if samples.ndim > 1:
                        samples = samples.mean(axis=1)
                    seconds = len(samples) / rate
                    if not args.min_seconds <= seconds <= args.max_seconds:
                        continue
                    stem = Path(audio[index]["path"].as_py()).stem
                    row_id = f"ksd_{speaker}_{stem.split('_', 1)[1]}"
                    if row_id in seen_ids:  # the HF copy repeats some clips in two shards
                        continue
                    seen_ids.add(row_id)
                    path = args.out_dir / "wav" / speaker / f"{stem}.wav"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(path, samples, rate, subtype="PCM_16")
                    row = {"id": row_id, "text": text,
                           "audio": str(path.resolve()), "seconds": round(seconds, 3),
                           "shard": shard.name}
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
                    totals["rows"] += 1
                    totals["seconds"] += seconds
                totals["speakers"].add(speaker)
    print(json.dumps({"new_rows": totals["rows"], "new_speakers": len(totals["speakers"]),
                      "new_hours": round(totals["seconds"] / 3600, 2)}))


if __name__ == "__main__":
    main()
