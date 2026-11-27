#!/usr/bin/env python3
"""Extract a speaker-balanced English replay subset from LibriTTS-R parquet shards.

Writes WAVs to ``<out-dir>/wav/<speaker>/<id>.wav`` and a raw manifest
(id, text, audio) whose ids are ``en_<speaker>_<utterance>``.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-speaker", type=int, default=60)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    candidates: dict[str, list[tuple[Path, int, int, str, str]]] = defaultdict(list)
    shards = sorted(args.parquet_dir.glob("*.parquet"))
    for shard in shards:
        meta = pq.read_table(shard, columns=["id", "speaker_id", "text_normalized"])
        for row_index, (uid, speaker, text) in enumerate(
            zip(*(meta.column(c).to_pylist() for c in ("id", "speaker_id", "text_normalized")))
        ):
            candidates[speaker].append((shard, row_index, 0, uid, text))

    chosen: dict[Path, dict[int, tuple[str, str, str]]] = defaultdict(dict)
    for speaker, items in candidates.items():
        rng.shuffle(items)
        # Oversample; duration is checked after decoding.
        for shard, row_index, _unused, uid, text in items[: args.per_speaker * 2]:
            chosen[shard][row_index] = (speaker, uid, text)

    kept: dict[str, int] = defaultdict(int)
    rows = []
    for shard in shards:
        wanted = chosen.get(shard)
        if not wanted:
            continue
        table = pq.read_table(shard, columns=["audio"])
        audio_column = table.column("audio")
        for row_index in sorted(wanted):
            speaker, uid, text = wanted[row_index]
            if kept[speaker] >= args.per_speaker or not text.strip():
                continue
            payload = audio_column[row_index].as_py()["bytes"]
            audio, rate = sf.read(io.BytesIO(payload), dtype="float32")
            seconds = len(audio) / rate
            if not args.min_seconds <= seconds <= args.max_seconds:
                continue
            path = args.out_dir / "wav" / speaker / f"{uid}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, audio, rate, subtype="PCM_16")
            kept[speaker] += 1
            rows.append(
                {"id": f"en_{speaker}_{uid}", "text": text.strip(), "audio": str(path.resolve()),
                 "seconds": round(seconds, 3)}
            )
    with (args.out_dir / "raw.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    hours = sum(r["seconds"] for r in rows) / 3600
    print(json.dumps({"rows": len(rows), "speakers": len(kept), "hours": round(hours, 2)}))


if __name__ == "__main__":
    main()
