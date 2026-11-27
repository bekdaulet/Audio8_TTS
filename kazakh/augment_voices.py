#!/usr/bin/env python3
"""Create pseudo-speakers from a few real speakers by pitch/formant shifting.

KazakhTTS2 has five voices, which teaches the model to ignore the reference
and fall back to one of them. Each real speaker is split into ``--pseudo``
pseudo-speakers, each with one fixed (pitch ratio, formant ratio) pair, and
each selected clip is resynthesised with Praat "Change Gender" using its
pseudo-speaker's pair. Because the pair is fixed per pseudo-speaker, a
reference and a target drawn from the same pseudo-speaker share one voice.

Output ids are ``<corpus>_<speaker>v<k>_<name>`` so build_sft_manifest.py
treats each pseudo-speaker as its own speaker.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call


def perturb(job: tuple[str, str, float, float]) -> tuple[str, str | None]:
    source, target, pitch_ratio, formant_ratio = job
    try:
        sound = parselmouth.Sound(source)
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=500)
        median = call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")
        if not median or math.isnan(median):
            return source, "no pitch"
        shifted = call(
            sound, "Change gender", 60, 500, formant_ratio, median * pitch_ratio, 1.0, 1.0
        )
        values = shifted.values[0]
        peak = float(np.max(np.abs(values)))
        if not np.isfinite(peak) or peak == 0:
            return source, "silent"
        if peak > 0.99:
            values = values * (0.99 / peak)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, values.astype(np.float32), int(shifted.sampling_frequency),
                 subtype="PCM_16")
        return source, None
    except Exception as exc:  # noqa: BLE001 - report and skip bad clips
        return source, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Raw manifest.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clips-per-speaker", type=int, default=3000)
    parser.add_argument("--pseudo", type=int, default=40, help="Pseudo-speakers per speaker.")
    parser.add_argument("--pitch-range", type=float, nargs=2, default=(0.75, 1.35))
    # Formant ratio is what moves a speaker embedding; ratios within ±7 % of 1 barely
    # change the voice, so sample |log ratio| from this range with a random sign.
    parser.add_argument("--formant-range", type=float, nargs=2, default=(1.08, 1.22))
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(line) for line in args.input_jsonl.open(encoding="utf-8") if line.strip()]
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_speaker[row["id"].split("_")[1]].append(row)

    def log_uniform(low: float, high: float) -> float:
        return math.exp(rng.uniform(math.log(low), math.log(high)))

    jobs, planned, voices = [], [], {}
    for speaker, items in sorted(by_speaker.items()):
        params = [
            (log_uniform(*args.pitch_range),
             log_uniform(*args.formant_range) ** rng.choice((-1, 1)))
            for _ in range(args.pseudo)
        ]
        rng.shuffle(items)
        for index, row in enumerate(items[: args.clips_per_speaker]):
            k = index % args.pseudo
            pitch_ratio, formant_ratio = params[k]
            voice = f"{speaker}v{k:02d}"
            voices[voice] = {"pitch_ratio": round(pitch_ratio, 4),
                             "formant_ratio": round(formant_ratio, 4)}
            name = row["id"].split("_", 2)[2]
            target = args.out_dir / "wav" / voice / f"{name}.wav"
            jobs.append((row["audio"], str(target.resolve()), pitch_ratio, formant_ratio))
            planned.append({"id": f"kk_{voice}_{name}", "text": row["text"],
                            "audio": str(target.resolve()), "seconds": row["seconds"]})

    with ProcessPoolExecutor(args.workers) as pool:
        results = dict(pool.map(perturb, jobs, chunksize=16))
    errors = {source: error for source, error in results.items() if error}
    kept = [row for row, job in zip(planned, jobs) if job[0] not in errors]
    kept.sort(key=lambda r: r["seconds"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "raw.jsonl").open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "voices.json").write_text(json.dumps(voices, indent=1), encoding="utf-8")
    print(json.dumps({"rows": len(kept), "voices": len(voices), "errors": len(errors),
                      "example_errors": list(errors.values())[:3]}))


if __name__ == "__main__":
    main()
