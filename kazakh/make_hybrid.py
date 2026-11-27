"""Splice two checkpoints: slow-AR weights from one, fast-AR weights from the other."""
import shutil, sys
from pathlib import Path
from safetensors.torch import load_file, save_file
slow_src, fast_src, out = map(Path, sys.argv[1:4])
FAST = ("fast_project_in.", "fast_embeddings.", "fast_layers.", "fast_norm.", "fast_output.")
slow, fast = load_file(slow_src / "model.safetensors"), load_file(fast_src / "model.safetensors")
assert slow.keys() == fast.keys()
merged = {k: (fast[k] if k.startswith(FAST) else slow[k]) for k in slow}
print("fast tensors from", fast_src, sum(k.startswith(FAST) for k in merged), "of", len(merged))
out.mkdir(parents=True, exist_ok=True)
for f in slow_src.iterdir():
    if f.name != "model.safetensors" and f.is_file():
        shutil.copy2(f, out / f.name)
save_file(merged, out / "model.safetensors", metadata={"format": "pt"})
