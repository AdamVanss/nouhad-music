"""Export JSON + spectrogram PNGs for TouchDesigner (Python outside, TD displays)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from src.audio.stft import StftConfig, log_mag, magnitude_spectrogram
from src.pipeline import STEM_NAMES

ROOT = Path(__file__).resolve().parents[1]
CFG = StftConfig()


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return x, sr


def rms_envelope(audio: np.ndarray, sample_rate: int, fps: float) -> np.ndarray:
    """One RMS value per video frame."""
    hop = max(1, int(sample_rate / fps))
    n_frames = int(np.ceil(len(audio) / hop))
    out = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = audio[i * hop : (i + 1) * hop]
        if chunk.size:
            out[i] = float(np.sqrt(np.mean(chunk ** 2)))
    return out


def save_spec_png(wave: np.ndarray, path: Path) -> None:
    w = torch.from_numpy(wave)
    mag, _ = magnitude_spectrogram(w, CFG)
    img = log_mag(mag).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.imshow(img, aspect="auto", origin="lower", cmap="magma")
    plt.xlabel("time frames")
    plt.ylabel("frequency")
    plt.title(path.stem)
    plt.colorbar(fraction=0.02)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def find_stems(folder: Path, stem: str) -> Path | None:
    for p in folder.glob(f"*_{stem}.wav"):
        return p
    p = folder / f"{stem}.wav"
    return p if p.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TD assets from separated WAVs")
    parser.add_argument(
        "--folder",
        type=Path,
        default=ROOT / "outputs" / "separated" / "tame_28_50",
        help="Folder with *_vocals.wav etc., or stem-named wavs",
    )
    parser.add_argument(
        "--mixture",
        type=Path,
        default=None,
        help="Optional mixture WAV (else skip mixture channel)",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "td")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    folder = args.folder
    out = args.out
    spec_dir = out / "spectrograms"
    out.mkdir(parents=True, exist_ok=True)

    stems_audio: dict[str, np.ndarray] = {}
    sr_ref: int | None = None

    for name in STEM_NAMES:
        path = find_stems(folder, name)
        if path is None:
            print(f"skip missing stem: {name}")
            continue
        audio, sr = load_mono(path)
        if sr_ref is None:
            sr_ref = sr
        elif sr != sr_ref:
            raise ValueError(f"Sample rate mismatch: {path}")
        stems_audio[name] = audio
        save_spec_png(audio, spec_dir / f"{name}.png")
        print(f"spec {name} -> {spec_dir / f'{name}.png'}")

    mix_path = args.mixture
    if mix_path is None:
        for p in folder.glob("*mixture*.wav"):
            mix_path = p
            break
    mixture_audio = None
    if mix_path and mix_path.is_file():
        mixture_audio, sr = load_mono(mix_path)
        sr_ref = sr_ref or sr
        save_spec_png(mixture_audio, spec_dir / "mixture.png")
        print(f"spec mixture -> {spec_dir / 'mixture.png'}")

    if not stems_audio or sr_ref is None:
        raise SystemExit(f"No stem WAVs found in {folder}")

    n = min(len(a) for a in stems_audio.values())
    if mixture_audio is not None:
        n = min(n, len(mixture_audio))

    energy: dict[str, list[float]] = {}
    for name, audio in stems_audio.items():
        energy[name] = rms_envelope(audio[:n], sr_ref, args.fps).tolist()
    if mixture_audio is not None:
        energy["mixture"] = rms_envelope(mixture_audio[:n], sr_ref, args.fps).tolist()

    duration_s = n / sr_ref
    energy_path = out / "energy.json"
    payload = {
        "sample_rate": sr_ref,
        "duration_seconds": duration_s,
        "fps": args.fps,
        "frame_count": len(next(iter(energy.values()))),
        "stems": energy,
    }
    energy_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    manifest = {
        "project": "Music_model",
        "energy_json": str(energy_path.resolve()),
        "spectrograms": {k: str((spec_dir / f"{k}.png").resolve()) for k in stems_audio},
        "fps": args.fps,
        "duration_seconds": duration_s,
    }
    if mixture_audio is not None:
        manifest["spectrograms"]["mixture"] = str((spec_dir / "mixture.png").resolve())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote {energy_path}")
    print(f"Wrote {out / 'manifest.json'}")
    print("Open docs/TOUCHDESIGNER.md for wiring in TD.")


if __name__ == "__main__":
    main()
