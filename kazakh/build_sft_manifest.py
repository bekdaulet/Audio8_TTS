#!/usr/bin/env python3
"""Pair prepared utterances with same-speaker reference clips for SFT.

Input is one or more prepared manifests (output of audio8_tts_prepare.py,
one row per utterance, no references). Row ids must start with
``<corpus>_<speaker>_``; that prefix defines the speaker.
For each target row a different utterance of the same speaker is chosen as
the reference, from clips within the reference duration window. A fraction
of rows is kept without a reference so plain (no-prompt) synthesis is still
trained. Paths in the output are relative to the output manifest.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from audio8_tts_data import json_line, read_jsonl, relative_to_manifest  # noqa: E402

FRAMES_PER_SECOND = 44100 / 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--no-reference-ratio", type=float, default=0.2)
    parser.add_argument("--min-ref-seconds", type=float, default=2.0)
    parser.add_argument("--max-ref-seconds", type=float, default=10.0)
    parser.add_argument("--max-target-seconds", type=float, default=25.0)
    parser.add_argument("--max-rows", type=int, help="Random subset of this many target rows.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output = args.output_jsonl.resolve()
    rows = []
    for manifest in args.input_jsonl:
        manifest = manifest.resolve()
        for _line, row in read_jsonl(manifest):
            row = dict(row)
            row["target_codes"] = (manifest.parent / row["target_codes"]).resolve()
            row["speaker"] = "_".join(row["id"].split("_")[:2])
            row["frames"] = np.load(row["target_codes"], mmap_mode="r").shape[1]
            rows.append(row)

    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        seconds = float(row["frames"]) / FRAMES_PER_SECOND
        if args.min_ref_seconds <= seconds <= args.max_ref_seconds:
            by_speaker[row["speaker"]].append(row)

    targets = [r for r in rows if float(r["frames"]) / FRAMES_PER_SECOND <= args.max_target_seconds]
    rng.shuffle(targets)
    if args.max_rows:
        targets = targets[: args.max_rows]

    output.parent.mkdir(parents=True, exist_ok=True)
    written = with_reference = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in targets:
            out = {
                "id": row["id"],
                "text": row["text"],
                "target_codes": relative_to_manifest(row["target_codes"], output),
            }
            pool = by_speaker.get(row["speaker"], [])
            if rng.random() >= args.no_reference_ratio and len(pool) > 1:
                reference = row
                while reference["id"] == row["id"]:
                    reference = rng.choice(pool)
                out["reference_codes"] = relative_to_manifest(reference["target_codes"], output)
                out["reference_text"] = reference["text"]
                with_reference += 1
            handle.write(json_line(out))
            written += 1
    print(json.dumps({"rows": written, "with_reference": with_reference, "output": str(output)}))


if __name__ == "__main__":
    main()
