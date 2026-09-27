"""Step 3: Oracle vocal mask from ground truth (upper bound before neural net)."""

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

    vocals = mono("vocals")
    acc = mono("drums") + mono("bass") + mono("other")
    mixture = vocals + acc

    n = min(int(8 * sr), mixture.shape[0])
    mixture, vocals = mixture[:n], vocals[:n]

    mix_mag, mix_phase = magnitude_spectrogram(mixture, CFG)
    voc_mag, _ = magnitude_spectrogram(vocals, CFG)
    mask = oracle_vocal_mask(mix_mag, voc_mag).clamp(0, 1)

    est_voc_mag = mask * mix_mag
    est_voc = istft_from_magnitude(est_voc_mag, mix_phase, CFG, length=n)

    sf.write(str(OUT / "step03_oracle_vocals.wav"), est_voc.numpy(), CFG.sample_rate)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    extent = [0, n / sr, 0, CFG.sample_rate / 2]
    axes[0].imshow(20 * np.log10(mix_mag.numpy() + 1e-8), aspect="auto", origin="lower", extent=extent)
    axes[0].set_title("Mixture")
    axes[1].imshow(mask.numpy(), aspect="auto", origin="lower", vmin=0, vmax=1, extent=extent, cmap="gray")
    axes[1].set_title("Oracle vocal mask (0–1)")
    axes[2].imshow(
        20 * np.log10(est_voc_mag.numpy() + 1e-8), aspect="auto", origin="lower", extent=extent
    )
    axes[2].set_title("Vocals after mask")
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hz")
    fig.tight_layout()
    fig.savefig(OUT / "step03_oracle_panels.png", dpi=150)
    plt.close()

    print(f"Saved {OUT / 'step03_oracle_vocals.wav'} and step03_oracle_panels.png")
    print("The neural net will learn to predict a mask like panel 2 from the mixture alone.")


if __name__ == "__main__":
    main()
