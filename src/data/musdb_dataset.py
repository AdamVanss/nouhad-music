"""MUSDB18 chunks for supervised training."""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

try:
    import musdb
except ImportError:
    musdb = None

from src.data.stems import STEM_NAMES_4 as STEM_NAMES


def get_musdb_path() -> Path:
    path = os.environ.get("MUSDB_PATH")
    if not path:
        raise EnvironmentError(
            "Set MUSDB_PATH to your musdb18 folder. See docs/MUSDB18_SETUP.md"
        )
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"MUSDB_PATH is not a directory: {p}")
    return p


class MusdbStemDataset(Dataset):
    """Random crop: mixture + stems (vocals, drums, bass, other)."""

    def __init__(
        self,
        subset: str = "train",
        sample_rate: int = 44100,
        segment_seconds: float = 6.0,
        target: str = "vocals",
    ):
        if musdb is None:
            raise ImportError("pip install musdb")

        root = get_musdb_path()
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.target = target  # used by single-stem loaders; all stems still returned

        self._db = musdb.DB(root=str(root), subsets=subset, is_wav=False)

    def __len__(self) -> int:
        return len(self._db)

    def _load_mono(self, track, name: str) -> torch.Tensor:
        audio = track.targets[name].audio
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        wav = torch.from_numpy(audio.astype("float32"))
        if track.rate != self.sample_rate:
            wav = torchaudio_resample(wav, track.rate, self.sample_rate)
        return wav

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        track = self._db[index]
        stems = {name: self._load_mono(track, name) for name in STEM_NAMES}
        mixture = sum(stems.values())

        length = mixture.shape[0]
        if length <= self.segment_samples:
            pad = self.segment_samples - length
            mixture = torch.nn.functional.pad(mixture, (0, pad))
            for name in STEM_NAMES:
                stems[name] = torch.nn.functional.pad(stems[name], (0, pad))
        else:
            start = torch.randint(0, length - self.segment_samples, (1,)).item()
            end = start + self.segment_samples
            mixture = mixture[start:end]
            for name in STEM_NAMES:
                stems[name] = stems[name][start:end]

        return {"mixture": mixture, **stems}


def torchaudio_resample(wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    import torchaudio

    if orig_sr == new_sr:
        return wav
    return torchaudio.functional.resample(wav.unsqueeze(0), orig_sr, new_sr).squeeze(0)
