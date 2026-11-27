#!/usr/bin/env python3
"""Drop clips whose Whisper transcript disagrees with the given text.

Scores every row of a raw manifest (id, text, audio, ...) with Whisper
(language forced) and writes the rows with CER <= ``--max-cer`` to
``--output``. Scores are cached in ``<output>.scores.jsonl`` so re-runs only
transcribe new rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_metrics import Whisper, load_16k, score_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lang", default="kk")
    parser.add_argument("--max-cer", type=float, default=0.15)
    parser.add_argument("--asr", default="openai/whisper-large-v3")
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input_jsonl.open(encoding="utf-8") if line.strip()]
    cache_path = args.output.with_suffix(".scores.jsonl")
    cache = {}
    if cache_path.exists():
        cache = {r["id"]: r for r in map(json.loads, cache_path.open(encoding="utf-8"))}
    todo = [r for r in rows if r["id"] not in cache]
    if todo:
        asr = Whisper(args.asr, "cuda")
        with cache_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(todo), args.batch_size):
                batch = todo[start : start + args.batch_size]
                hyps = asr([load_16k(Path(r["audio"])) for r in batch], args.lang)
                for row, hyp in zip(batch, hyps):
                    ref = score_text(row["text"], args.lang)
                    cer = jiwer.cer(ref, score_text(hyp, args.lang)) if ref else 1.0
                    cache[row["id"]] = {"id": row["id"], "hyp": hyp, "cer": cer}
                    handle.write(json.dumps(cache[row["id"]], ensure_ascii=False) + "\n")
    kept = [r for r in rows if cache[r["id"]]["cer"] <= args.max_cer]
    kept.sort(key=lambda r: r["seconds"])
    with args.output.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    cers = sorted(cache[r["id"]]["cer"] for r in rows)
    print(json.dumps({"rows": len(rows), "kept": len(kept),
                      "median_cer": round(cers[len(cers) // 2], 3),
                      "kept_hours": round(sum(r["seconds"] for r in kept) / 3600, 2)}))


if __name__ == "__main__":
    main()
