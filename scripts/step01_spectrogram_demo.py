"""Step 1: Create a fake mixture and plot its spectrogram (no MUSDB needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import soundfile as sf
import torch

from src.audio.stft import StftConfig, magnitude_spectrogram

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

CFG = StftConfig(sample_rate=44100)
DURATION = 3.0
T = int(CFG.sample_rate * DURATION)


def tone(freq: float, amp: float = 0.3) -> torch.Tensor:
    t = torch.linspace(0, DURATION, T)
    return amp * torch.sin(2 * torch.pi * freq * t)


def main() -> None:
    # Two sustained "notes" + noise ≈ simple mixture
    mix = tone(220.0) + tone(330.0) + 0.05 * torch.randn(T)
    mix = mix / mix.abs().max().clamp(min=1e-8)

    sf.write(str(OUT / "step01_mixture.wav"), mix.numpy(), CFG.sample_rate)

    mag, _ = magnitude_spectrogram(mix, CFG)
    mag_db = 20 * torch.log10(mag + 1e-8).numpy()
    nyquist = CFG.sample_rate / 2

    def save_spec(
        path: Path,
        fmax: float,
        title: str,
        mark_tones: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(
            mag_db,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0, DURATION, 0, nyquist],
            vmin=mag_db.max() - 60,
            vmax=mag_db.max(),
        )
        ax.set_ylim(0, fmax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title)
        if mark_tones:
            for hz, label in ((220, "220 Hz (A3)"), (330, "330 Hz (E4)")):
                ax.axhline(hz, color="cyan", linewidth=1.2, alpha=0.9)
                ax.text(0.05, hz + fmax * 0.02, label, color="cyan", fontsize=9)
        fig.colorbar(im, ax=ax, label="Magnitude (dB)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close()

    save_spec(
        OUT / "step01_spectrogram.png",
        fmax=nyquist,
        title="Step 1 — Full range (tones sit at the very bottom)",
    )
    save_spec(
        OUT / "step01_spectrogram_zoomed.png",
        fmax=800,
        title="Step 1 — Zoomed 0–800 Hz (two separate tone lines)",
        mark_tones=True,
    )

    print(f"Saved: {OUT / 'step01_mixture.wav'}")
    print(f"Saved: {OUT / 'step01_spectrogram.png'}")
    print(f"Saved: {OUT / 'step01_spectrogram_zoomed.png'}  ← open this one first")
    print("You should see two horizontal bands at 220 Hz and 330 Hz on the zoomed plot.")


if __name__ == "__main__":
    main()
