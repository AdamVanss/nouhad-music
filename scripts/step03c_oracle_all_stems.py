"""Oracle separation: vocals + drums + bass + other (perfect masks from MUSDB)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf
import torch

from src.audio.stft import StftConfig, istft_from_magnitude, magnitude_spectrogram, oracle_vocal_mask
from src.data.musdb_dataset import STEM_NAMES, get_musdb_path
try:
    import musdb
except ImportError:
    raise SystemExit("pip install musdb")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "oracle_stems"
CFG = StftConfig()


def main() -> None:
    root = get_musdb_path()
    db = musdb.DB(root=str(root), subsets="train", is_wav=False)
    track = db[0]
    print(f"Track: {track.name}")

    def mono(name: str) -> torch.Tensor:
        a = track.targets[name].audio
        if a.ndim == 2:
            a = a.mean(axis=1)
        return torch.from_numpy(a.astype("float32"))

    stems = {n: mono(n) for n in STEM_NAMES}
    mixture = sum(stems.values())
    n = mixture.shape[0]

    mix_mag, mix_phase = magnitude_spectrogram(mixture, CFG)
    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT / "mixture.wav"), mixture.numpy(), CFG.sample_rate)

    for name in STEM_NAMES:
        stem_mag, _ = magnitude_spectrogram(stems[name], CFG)
        mask = oracle_vocal_mask(mix_mag, stem_mag).clamp(0, 1)
        est = istft_from_magnitude(mask * mix_mag, mix_phase, CFG, length=n)
        path = OUT / f"{name}.wav"
        sf.write(str(path), est.numpy(), CFG.sample_rate)
        print(f"  saved {path.name}")

    print(f"\nAll oracle stems in {OUT}")
    print("Compare to ground truth in MUSDB — this is the quality ceiling before AI.")


if __name__ == "__main__":
    main()
