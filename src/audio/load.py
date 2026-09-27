"""Load mono audio from WAV/MP3 with optional time window."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.audio.stft import StftConfig

CFG = StftConfig()


def load_mono(
    path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    max_seconds: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load mono float32 at CFG.sample_rate. Slice [start, end) in seconds if given."""
    path = Path(path)
    try:
        audio, sr = sf.read(str(path), always_2d=True)
        audio = audio.mean(axis=1).astype(np.float32)
    except Exception:
        wave, sr = torchaudio.load(str(path))
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)
        audio = wave.squeeze(0).numpy().astype(np.float32)

    if sr != CFG.sample_rate:
        audio = (
            torchaudio.functional.resample(
                torch.from_numpy(audio).unsqueeze(0), sr, CFG.sample_rate
            )
            .squeeze(0)
            .numpy()
        )
        sr = CFG.sample_rate

    i0 = int(start_seconds * sr) if start_seconds is not None else 0
    i1 = int(end_seconds * sr) if end_seconds is not None else len(audio)
    i0 = max(0, min(i0, len(audio)))
    i1 = max(i0, min(i1, len(audio)))
    audio = audio[i0:i1]

    if max_seconds is not None:
        n = int(max_seconds * sr)
        audio = audio[:n]

    return audio, sr
