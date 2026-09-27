"""Step 2: Load one MUSDB track — mixture vs vocals (requires MUSDB_PATH)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from src.data.musdb_dataset import get_musdb_path

try:
    import musdb
except ImportError:
    raise SystemExit("Run: pip install musdb")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main() -> None:
    root = get_musdb_path()
    db = musdb.DB(root=str(root), subsets="train", is_wav=False)
    track = db[0]
    print(f"Track: {track.name}")

    sr = track.rate
    vocals = track.targets["vocals"].audio
    if vocals.ndim == 2:
        vocals = vocals.mean(axis=1)
    drums = track.targets["drums"].audio.mean(axis=1)
    bass = track.targets["bass"].audio.mean(axis=1)
    other = track.targets["other"].audio.mean(axis=1)
    mixture = vocals + drums + bass + other

    # 7s sample tracks — use full clip or cap at track length
    n = len(mixture)
    sf.write(str(OUT / "step02_mixture_clip.wav"), mixture[:n].astype(np.float32), sr)
    sf.write(str(OUT / "step02_vocals_clip.wav"), vocals[:n].astype(np.float32), sr)

    print(f"MUSDB root: {root}")
    print(f"Saved clips under {OUT}")
    print("Listen: mixture vs vocals — that's what training tries to recover.")


if __name__ == "__main__":
    main()
