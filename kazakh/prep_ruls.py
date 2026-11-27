#!/usr/bin/env python3
"""Extract a speaker-balanced Russian replay subset from RuLS parquet shards.

Uses the HF mirror ``istupakov/russian_librispeech`` (OpenSLR 96 audio with the
original cased, punctuated text in ``text_no_preprocessing``). The reader is
the first component of ``audio_filepath`` (``audio/<reader>/<book>/...``).
Clips with a poor alignment ``score`` are skipped. Writes WAVs and a raw
manifest whose ids are ``ru_<reader>_<utterance>``.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

COLUMNS = ["duration", "text_no_preprocessing", "score", "audio_filepath"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-speaker", type=int, default=400)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--min-score", type=float, default=-1.0)
    parser.add_argument("--manifest-name", default="raw.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    candidates: dict[str, list[tuple[Path, int, str, str, float]]] = defaultdict(list)
    for shard in args.parquet:
        meta = pq.read_table(shard, columns=COLUMNS).to_pylist()
        for index, row in enumerate(meta):
            text = (row["text_no_preprocessing"] or "").strip()
            if not text or row["score"] < args.min_score:
                continue
            if not args.min_seconds <= row["duration"] <= args.max_seconds:
                continue
            parts = Path(row["audio_filepath"]).parts
            speaker, stem = parts[1], Path(row["audio_filepath"]).stem
            candidates[speaker].append((shard, index, stem, text, row["duration"]))

    chosen: dict[Path, list[tuple[int, str, str, str, float]]] = defaultdict(list)
    for speaker, items in candidates.items():
        rng.shuffle(items)
        for shard, index, stem, text, seconds in items[: args.per_speaker]:
            chosen[shard].append((index, speaker, stem, text, seconds))

    rows = []
    for shard, wanted in chosen.items():
        audio_column = pq.read_table(shard, columns=["audio"]).column("audio")
        for index, speaker, stem, text, seconds in sorted(wanted):
            payload = audio_column[index].as_py()["bytes"]
            audio, rate = sf.read(io.BytesIO(payload), dtype="float32")
            path = args.out_dir / "wav" / speaker / f"{stem}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, audio, rate, subtype="PCM_16")
            uid = f"ru_{speaker}_{stem}".replace(" ", "_")
            rows.append(
                {"id": uid, "text": text, "audio": str(path.resolve()), "seconds": seconds}
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / args.manifest_name).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    speakers = {r["id"].split("_")[1] for r in rows}
    hours = sum(r["seconds"] for r in rows) / 3600
    print(json.dumps({"rows": len(rows), "speakers": len(speakers), "hours": round(hours, 2)}))


if __name__ == "__main__":
    main()
