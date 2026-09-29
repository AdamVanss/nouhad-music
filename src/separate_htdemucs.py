"""Separate a track with pretrained or fine-tuned Hybrid Transformer Demucs."""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from demucs.apply import apply_model
from demucs.pretrained import get_model


def load_htdemucs(checkpoint: Path | None = None, device: torch.device | None = None):
    """Single HTDemucs. Optional checkpoint is a fine-tune state dict."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bag = get_model("htdemucs")
    model = bag.models[0]
    if checkpoint is not None:
        try:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def decode_audio(path: Path, samplerate: int, channels: int) -> torch.Tensor:
    """Any ffmpeg-readable file -> float (channels, T). torchaudio.load needs torchcodec here."""
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    raw = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", str(channels),
         "-ar", str(samplerate), "-"],
        check=True,
        capture_output=True,
    ).stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, channels)
    return torch.from_numpy(audio.T.copy())


def separate_file(
    model,
    wav_path: Path,
    out_dir: Path,
    shifts: int = 1,
    overlap: float = 0.25,
    progress: bool = True,
) -> list[Path]:
    device = next(model.parameters()).device
    wav = decode_audio(Path(wav_path), model.samplerate, model.audio_channels)
    ref = wav.mean(0)
    std = ref.std().clamp(min=1e-8)
    wav = (wav - ref.mean()) / std
    with torch.no_grad():
        estimates = apply_model(
            model,
            wav[None],
            device=device,
            shifts=shifts,
            overlap=overlap,
            split=True,
            progress=progress,
        )[0]
    estimates = estimates * std + ref.mean()
    # one gain for every stem so they still sum to the mix, instead of hard-clipping peaks
    peak = estimates.abs().max().item()
    if peak > 0.99:
        estimates = estimates * (0.99 / peak)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(wav_path).stem
    paths = []
    for name, source in zip(model.sources, estimates):
        path = out_dir / f"{stem}_{name}.wav"
        sf.write(str(path), source.cpu().T.numpy(), model.samplerate, subtype="PCM_16")
        print(f"Saved {path}")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="HTDemucs separation")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/htdemucs"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Fine-tuned weights")
    parser.add_argument("--shifts", type=int, default=1)
    args = parser.parse_args()
    model = load_htdemucs(args.checkpoint)
    separate_file(model, args.input, args.out_dir, shifts=args.shifts)


if __name__ == "__main__":
    main()
