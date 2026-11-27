#!/usr/bin/env python3
"""Pick listening samples from two eval runs (base vs fine-tuned) without cherry-picking.

Within each group, items are ranked by how close the fine-tuned model's CER and SIM are to the
group medians (typical outputs, not the best ones); seen items also use distinct speakers.
Copies reference / base / fine-tuned WAVs into ``--out-dir`` and writes a README with the texts
and per-item numbers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path

# group -> (number of items, include the reference recording)
PLAN = {("kk", "seen"): (2, True), ("kk", "unseen"): (3, True),
        ("en", "clone"): (1, False), ("ru", "clone"): (1, False)}


def load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.open(encoding="utf-8"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="eval dir of the base model")
    parser.add_argument("--tuned", type=Path, required=True, help="eval dir of the fine-tuned model")
    parser.add_argument("--tuned-name", default="finetuned")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-seconds", type=float, default=4.0)
    parser.add_argument("--max-seconds", type=float, default=14.0)
    args = parser.parse_args()

    base = load(args.base / "metrics.items.jsonl")
    tuned = load(args.tuned / "metrics.items.jsonl")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"Listening samples: base Audio8-TTS-Realistic-Air vs {args.tuned_name}.",
             "Items are typical (closest to the group median CER/SIM of the fine-tuned model), not best-case.",
             "seen = KazakhTTS2 voice used in training; unseen = FLEURS voice never seen in training",
             "(zero-shot cloning); en/ru = cloning a held-out English / Russian reader.", ""]
    index = 0
    for (lang, group), (count, with_ref) in PLAN.items():
        rows = [r for r in tuned.values() if (r["lang"], r["group"]) == (lang, group) and r["id"] in base
                and args.min_seconds <= r["seconds"] <= args.max_seconds]
        med_cer = statistics.median(r["cer"] for r in rows)
        med_sim = statistics.median(r.get("sim", 0.0) for r in rows)
        rows.sort(key=lambda r: abs(r["cer"] - med_cer) + abs(r.get("sim", 0.0) - med_sim))
        picked, speakers = [], set()
        for row in rows:
            if group == "seen" and row["speaker"] in speakers:
                continue
            picked.append(row)
            speakers.add(row.get("speaker"))
            if len(picked) == count:
                break
        for row in picked:
            index += 1
            stem = f"{index:02d}_{lang}_{group}"
            if with_ref:
                shutil.copy(row["reference_audio"], args.out_dir / f"{stem}_reference.wav")
            shutil.copy(base[row["id"]]["wav"], args.out_dir / f"{stem}_base.wav")
            shutil.copy(row["wav"], args.out_dir / f"{stem}_{args.tuned_name}.wav")
            b = base[row["id"]]
            lines += [f"{stem} (voice {row.get('speaker', row['id'])}): {row['text']}",
                      f"    base: CER {b['cer'] * 100:.1f} %, SIM {b.get('sim', 0):.3f}   "
                      f"{args.tuned_name}: CER {row['cer'] * 100:.1f} %, SIM {row.get('sim', 0):.3f}"]
    (args.out_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{index} items -> {args.out_dir}")


if __name__ == "__main__":
    main()
