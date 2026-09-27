"""Practice: load WAVs, plot waveforms, compare levels (normalization intuition)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MIX_PATH = OUT / "step02_mixture_clip.wav"
VOC_PATH = OUT / "step02_vocals_clip.wav"


def stats(name: str, x: np.ndarray) -> dict[str, float]:
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    print(f"{name:10s}  peak |max|: {peak:.4f}   RMS: {rms:.4f}")
    return {"peak": peak, "rms": rms}


def main() -> None:
    if not MIX_PATH.is_file() or not VOC_PATH.is_file():
        raise SystemExit(
            "Run step02 first:\n"
            "  $env:MUSDB_PATH = ...\n"
            "  python scripts/step02_listen_stems.py"
        )

    mix, sr_m = sf.read(MIX_PATH, dtype="float32")
    voc, sr_v = sf.read(VOC_PATH, dtype="float32")
    if mix.ndim > 1:
        mix = mix.mean(axis=1)
    if voc.ndim > 1:
        voc = voc.mean(axis=1)
    assert sr_m == sr_v

    print(f"Sample rate: {sr_m} Hz   length: {len(mix) / sr_m:.2f} s\n")
    print("--- Amplitude (before any normalization) ---")
    sm = stats("mixture", mix)
    sv = stats("vocals", voc)
    print(f"\nVocals peak is {100 * sv['peak'] / sm['peak']:.1f}% of mixture peak")

    # Time axis in seconds
    t = np.arange(len(mix)) / sr_m
    n = min(len(mix), len(voc))
    t, mix, voc = t[:n], mix[:n], voc[:n]

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(t, mix, color="steelblue", linewidth=0.5)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Mixture (all instruments)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, voc, color="coral", linewidth=0.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Vocals only (ground truth stem)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = OUT / "practice_waveforms.png"
    fig.savefig(plot_path, dpi=150)
    plt.close()

    # Same plot but each signal scaled to its own peak (visual compare shape)
    mix_n = mix / (sm["peak"] + 1e-8)
    voc_n = voc / (sv["peak"] + 1e-8)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, mix_n, label="mixture (peak-normalized)", alpha=0.7, linewidth=0.6)
    ax.plot(t, voc_n, label="vocals (peak-normalized)", alpha=0.8, linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Peak-normalized overlay — compare shape, not loudness")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    overlay_path = OUT / "practice_waveforms_normalized.png"
    fig.savefig(overlay_path, dpi=150)
    plt.close()

    print(f"\nSaved: {plot_path}")
    print(f"Saved: {overlay_path}")
    print(
        "\nWhy normalization matters:\n"
        "  - Neural nets train more stably when input/target scales are consistent.\n"
        "  - Step01 divides by max so the fake mix peaks at 1.0.\n"
        "  - MUSDB stems can have different peaks; training uses L1 on waveforms,\n"
        "    so very loud clips dominate the loss unless you balance or normalize.\n"
        "  - Peak-normalized plots let you SEE timing without louder stems hiding quieter ones."
    )


if __name__ == "__main__":
    main()
