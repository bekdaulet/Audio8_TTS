#!/usr/bin/env python3
"""Assemble a shuffled multi-corpus SFT manifest with fresh references per epoch.

Each ``--corpus`` is ``path.jsonl[:max_rows]``. For every epoch the rows of
each corpus are re-sampled (if ``max_rows`` is set) and re-paired with new
same-speaker references, then all epochs are concatenated in epoch order,
each epoch shuffled. Training one Trainer epoch over the result therefore
sees every epoch's pairing once.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--no-reference-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = args.output_jsonl.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with output.open("w", encoding="utf-8") as handle:
        for epoch in range(args.epochs):
            rows = []
            for index, spec in enumerate(args.corpus):
                path, _, limit = spec.partition(":")
                # Same directory as the output, so relative paths stay valid.
                part = output.parent / f".{output.stem}.e{epoch}_{index}.tmp.jsonl"
                command = [
                    sys.executable, str(HERE / "build_sft_manifest.py"),
                    "--input-jsonl", path, "--output-jsonl", str(part),
                    "--no-reference-ratio", str(args.no_reference_ratio),
                    "--seed", str(args.seed + 1000 * epoch + index),
                ]
                if limit:
                    command += ["--max-rows", limit]
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
                lines = part.read_text(encoding="utf-8").splitlines()
                part.unlink()
                for line in lines:
                    row = json.loads(line)
                    row["id"] = f"{row['id']}_e{epoch}"
                    rows.append(row)
                    corpus = row["id"].split("_")[0]
                    counts[corpus] = counts.get(corpus, 0) + 1
            random.Random(args.seed + epoch).shuffle(rows)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "rows_per_corpus": counts}))


if __name__ == "__main__":
    main()
