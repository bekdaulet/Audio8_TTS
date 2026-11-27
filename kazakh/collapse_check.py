"""For kk/unseen outputs: similarity to their reference vs to the nearest KazakhTTS2 speaker centroid."""
import json, sys, random
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "Audio8_TTS/kazakh"); sys.path.insert(0, "Audio8_TTS")
from eval_metrics import SpeakerEncoder, load_16k
sv = SpeakerEncoder("microsoft/wavlm-base-plus-sv", "cuda")
rows = [json.loads(l) for l in open("data/kk_splits/val.jsonl")]
cent = {}
for spk in ["F1", "F2", "F3", "M1", "M2"]:
    embs = [sv(load_16k(Path(r["audio"]))) for r in rows if r["id"].split("_")[1] == spk][:20]
    cent[spk] = torch.nn.functional.normalize(torch.stack(embs).mean(0), dim=-1)
ev = [json.loads(l) for l in open("work/eval/eval_v1.jsonl") if '"unseen"' in l]
for tag in sys.argv[1:]:
    to_ref, to_spk, ref_to_spk = [], [], []
    for r in ev:
        e = sv(load_16k(Path(f"outputs/eval/{tag}/wav/{r['id']}.wav")))
        ref = sv(load_16k(Path(r["reference_audio"])))
        to_ref.append(float(e @ ref)); to_spk.append(max(float(e @ c) for c in cent.values()))
        ref_to_spk.append(max(float(ref @ c) for c in cent.values()))
    print(f"{tag}: sim(out,ref)={np.mean(to_ref):.3f}  max sim(out,KK train spk)={np.mean(to_spk):.3f}  "
          f"[refs themselves vs KK spk: {np.mean(ref_to_spk):.3f}]  out closer to a KK spk than to ref: "
          f"{np.mean(np.array(to_spk) > np.array(to_ref))*100:.0f}%")
