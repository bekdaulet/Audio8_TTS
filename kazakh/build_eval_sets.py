#!/usr/bin/env python3
"""Build the fixed evaluation manifest used for every checkpoint.

Groups (``lang/group``):
  kk/seen    KazakhTTS2 test sentences, cloned from another held-out clip of
             the same speaker (speaker seen in training, sentence not).
  kk/unseen  FLEURS kk_kz test: reference clip from a FLEURS speaker (never in
             training), target text a different FLEURS sentence.
  kk/noref   KazakhTTS2 test sentences with no reference audio.
  en/clone   LibriTTS-R clips from speakers held out of the replay set.
  ru/clone   RuLS test readers (disjoint from the training readers).
Every row has id, lang, group, text, and (except noref) reference_audio and
reference_text; rows with a recording of the target sentence carry gt_audio.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio8_tts_data import read_jsonl  # noqa: E402
from audio8_tts_kk_normalize import normalize_kk  # noqa: E402


def speaker_of(row_id: str) -> str:
    return row_id.split("_")[1]


def clone_rows(rows, lang, group, per_speaker, rng, ref_range=(3.0, 10.0)):
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[speaker_of(row["id"])].append(row)
    out = []
    for speaker, items in sorted(by_speaker.items()):
        items = list(items)
        rng.shuffle(items)
        refs = [r for r in items if ref_range[0] <= r["seconds"] <= ref_range[1]]
        targets = [r for r in items if r["seconds"] <= 15.0]
        for target in targets[:per_speaker]:
            reference = next(r for r in refs if r["id"] != target["id"])
            out.append(
                {
                    "id": f"{group}_{target['id']}",
                    "lang": lang,
                    "group": group,
                    "speaker": speaker,
                    "text": target["text"],
                    "gt_audio": target["audio"],
                    "reference_audio": reference["audio"],
                    "reference_text": reference["text"],
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kk-test", type=Path, required=True)
    parser.add_argument("--fleurs-dir", type=Path, required=True)
    parser.add_argument("--en-heldout", type=Path, required=True)
    parser.add_argument("--ru-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kk-per-speaker", type=int, default=20)
    parser.add_argument("--kk-unseen", type=int, default=60)
    parser.add_argument("--noref-per-speaker", type=int, default=5)
    parser.add_argument("--en-per-speaker", type=int, default=5)
    parser.add_argument("--ru-per-speaker", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    kk = [row for _line, row in read_jsonl(args.kk_test)]
    rows = clone_rows(kk, "kk", "seen", args.kk_per_speaker, rng)
    used = {r["gt_audio"] for r in rows}
    by_speaker = defaultdict(list)
    for row in kk:
        if row["audio"] not in used:
            by_speaker[speaker_of(row["id"])].append(row)
    for speaker, items in sorted(by_speaker.items()):
        for target in items[: args.noref_per_speaker]:
            rows.append(
                {
                    "id": f"noref_{target['id']}",
                    "lang": "kk",
                    "group": "noref",
                    "speaker": speaker,
                    "text": target["text"],
                    "gt_audio": target["audio"],
                }
            )

    # FLEURS: one row per distinct sentence; reference from a different sentence.
    fleurs = []
    seen_text = set()
    for line in (args.fleurs_dir / "test.tsv").read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        wav = args.fleurs_dir / "test" / fields[1]
        seconds = int(fields[5]) / 16000
        if wav.is_file():
            fleurs.append({"audio": str(wav.resolve()), "text": normalize_kk(fields[2]),
                           "seconds": seconds, "first": fields[2] not in seen_text})
            seen_text.add(fields[2])
    rng.shuffle(fleurs)
    references = [f for f in fleurs if 3.0 <= f["seconds"] <= 10.0]
    targets = [f for f in fleurs if f["first"] and len(f["text"]) <= 150]
    for index, target in enumerate(targets[: args.kk_unseen]):
        reference = next(r for r in references[index:] + references if r["text"] != target["text"])
        rows.append(
            {
                "id": f"unseen_{index:03d}",
                "lang": "kk",
                "group": "unseen",
                "text": target["text"],
                "gt_audio": target["audio"],
                "reference_audio": reference["audio"],
                "reference_text": reference["text"],
            }
        )

    rows += clone_rows([r for _l, r in read_jsonl(args.en_heldout)], "en", "clone",
                       args.en_per_speaker, rng)
    rows += clone_rows([r for _l, r in read_jsonl(args.ru_test)], "ru", "clone",
                       args.ru_per_speaker, rng)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = defaultdict(int)
    for row in rows:
        counts[f"{row['lang']}/{row['group']}"] += 1
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
