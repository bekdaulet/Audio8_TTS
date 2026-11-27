#!/usr/bin/env python3
"""Build raw manifests and utterance-level splits for ISSAI KazakhTTS2.

Reads ``<root>/<speaker>/[<domain>/]{Audio,Transcripts}/<name>.{wav,txt}``, normalizes
the text, drops clips that are too short or long or whose speaking rate
suggests a transcript mismatch, and writes ``train/val/test`` raw manifests
(id, text, audio, seconds) with ids ``kk_<speaker>_<name>``. Validation and
test clips are drawn per speaker; any training clip whose text also occurs
in validation or test is removed, so held-out sentences are never trained on.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio8_tts_kk_normalize import normalize_kk  # noqa: E402


def read_item(args: tuple[str, str]) -> dict | None:
    speaker, wav = args
    wav_path = Path(wav)
    txt_path = wav_path.parent.parent / "Transcripts" / f"{wav_path.stem}.txt"
    if not txt_path.is_file():
        return {"error": "no_transcript", "audio": wav}
    raw = txt_path.read_text(encoding="utf-8", errors="replace")
    text = normalize_kk(" ".join(raw.split()))
    try:
        info = sf.info(wav_path)
    except RuntimeError:
        return {"error": "unreadable", "audio": wav}
    return {
        "id": f"kk_{speaker}_{wav_path.stem}",
        "speaker": speaker,
        "text": text,
        "changed": text != " ".join(raw.split()),
        "audio": str(wav_path.resolve()),
        "seconds": round(info.frames / info.samplerate, 3),
        "rate": info.samplerate,
    }


def key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--test-per-speaker", type=int, default=100)
    parser.add_argument("--val-per-speaker", type=int, default=50)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=25.0)
    parser.add_argument("--min-cps", type=float, default=6.0, help="Characters per second.")
    parser.add_argument("--max-cps", type=float, default=25.0)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    jobs = [
        (speaker_dir.name, str(wav))
        for speaker_dir in sorted(p for p in args.root.iterdir() if p.is_dir())
        # Most speakers use <spk>/Audio/; M1 has <spk>/{Book,News,Wiki}/Audio/.
        for wav in sorted(speaker_dir.rglob("*.wav"))
        if wav.parent.name == "Audio"
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        items = list(pool.map(read_item, jobs, chunksize=256))

    stats: Counter = Counter()
    ids = Counter(item["id"] for item in items if item and "id" in item)
    if max(ids.values()) > 1:
        raise ValueError(f"duplicate ids: {[k for k, v in ids.items() if v > 1][:5]}")
    kept = []
    for item in items:
        if item is None or "error" in item:
            stats[item["error"] if item else "none"] += 1
            continue
        stats["changed_by_normalizer"] += item["changed"]
        cps = len(item["text"]) / max(item["seconds"], 1e-3)
        if not item["text"]:
            stats["empty_text"] += 1
        elif not args.min_seconds <= item["seconds"] <= args.max_seconds:
            stats["duration_out_of_range"] += 1
        elif not args.min_cps <= cps <= args.max_cps:
            stats["speaking_rate_outlier"] += 1
        else:
            kept.append(item)

    rng = random.Random(args.seed)
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for item in kept:
        by_speaker[item["speaker"]].append(item)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    held_out_keys: set[str] = set()
    for speaker, speaker_items in sorted(by_speaker.items()):
        pool_items = [i for i in speaker_items if 2.0 <= i["seconds"] <= 15.0]
        rng.shuffle(pool_items)
        test = pool_items[: args.test_per_speaker]
        val = pool_items[args.test_per_speaker : args.test_per_speaker + args.val_per_speaker]
        splits["test"] += test
        splits["val"] += val
        held_out_keys |= {key(i["text"]) for i in test + val}
    held_out_ids = {i["id"] for i in splits["test"] + splits["val"]}
    for item in kept:
        if item["id"] in held_out_ids:
            continue
        if key(item["text"]) in held_out_keys:
            stats["train_dropped_same_text_as_heldout"] += 1
            continue
        splits["train"].append(item)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"input_files": len(jobs), "filters": dict(stats), "splits": {}}
    for name, rows in splits.items():
        rows.sort(key=lambda r: r["seconds"])
        with (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                out = {k: row[k] for k in ("id", "text", "audio", "seconds")}
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
        per_speaker = Counter(r["speaker"] for r in rows)
        hours = defaultdict(float)
        for r in rows:
            hours[r["speaker"]] += r["seconds"] / 3600
        summary["splits"][name] = {
            "rows": len(rows),
            "hours": round(sum(hours.values()), 2),
            "per_speaker": {s: [per_speaker[s], round(hours[s], 2)] for s in sorted(per_speaker)},
        }
    summary["sample_rates"] = dict(Counter(i["rate"] for i in kept))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
