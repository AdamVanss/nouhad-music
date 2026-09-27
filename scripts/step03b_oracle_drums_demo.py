"""Step 3b: Oracle DRUMS mask (same idea as vocals — preview multi-stem)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from src.audio.stft import StftConfig, istft_from_magnitude, magnitude_spectrogram, oracle_vocal_mask
from src.data.musdb_dataset import get_musdb_path

try:
    import musdb
except ImportError:
    raise SystemExit("Run: pip install musdb")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
CFG = StftConfig()


def main() -> None:
    root = get_musdb_path()
    db = musdb.DB(root=str(root), subsets="train", is_wav=False)
    track = db[0]
    sr = track.rate

    def mono(name: str) -> torch.Tensor:
        a = track.targets[name].audio
        if a.ndim == 2:
            a = a.mean(axis=1)
        return torch.from_numpy(a.astype("float32"))

    drums = mono("drums")
    acc = mono("vocals") + mono("bass") + mono("other")
    mixture = drums + acc

    n = min(int(8 * sr), mixture.shape[0])
    mixture, drums = mixture[:n], drums[:n]

    mix_mag, mix_phase = magnitude_spectrogram(mixture, CFG)
    drum_mag, _ = magnitude_spectrogram(drums, CFG)
    mask = oracle_vocal_mask(mix_mag, drum_mag).clamp(0, 1)

    est_mag = mask * mix_mag
    est_drums = istft_from_magnitude(est_mag, mix_phase, CFG, length=n)

    sf.write(str(OUT / "step03b_oracle_drums.wav"), est_drums.numpy(), CFG.sample_rate)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    extent = [0, n / sr, 0, CFG.sample_rate / 2]
    axes[0].imshow(20 * np.log10(mix_mag.numpy() + 1e-8), aspect="auto", origin="lower", extent=extent)
    axes[0].set_title("Mixture")
    axes[1].imshow(mask.numpy(), aspect="auto", origin="lower", vmin=0, vmax=1, extent=extent, cmap="gray")
    axes[1].set_title("Oracle DRUMS mask")
    axes[2].imshow(20 * np.log10(est_mag.numpy() + 1e-8), aspect="auto", origin="lower", extent=extent)
    axes[2].set_title("Drums after mask")
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hz")
    fig.tight_layout()
    fig.savefig(OUT / "step03b_oracle_drums_panels.png", dpi=150)
    plt.close()

    print("Saved step03b_oracle_drums.wav and step03b_oracle_drums_panels.png")
    print("Drums look like vertical bursts on the spectrogram — compare to vocal horizontal bands in step03.")


if __name__ == "__main__":
    main()
