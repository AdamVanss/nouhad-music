"""MoisesDB random crops — requires MOISESDB_PATH and pip install moises-db."""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.stems import moises_mix_map, stem_names_for

try:
    from moisesdb.dataset import MoisesDB
except ImportError:
    MoisesDB = None


def get_moises_path() -> Path:
    path = os.environ.get("MOISESDB_PATH")
    if not path:
        raise EnvironmentError(
            "Set MOISESDB_PATH to the folder containing provider/track UUID dirs. "
            "See docs/MODEL_UPGRADE.md"
        )
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"MOISESDB_PATH is not a directory: {p}")
    return p


class MoisesStemDataset(Dataset):
    """On-the-fly MoisesDB mixtures with 4-, 5-, or 6-stem targets."""

    def __init__(
        self,
        num_stems: int = 6,
        sample_rate: int = 44100,
        segment_seconds: float = 8.0,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        if MoisesDB is None:
            raise ImportError(
                "pip install git+https://github.com/moises-ai/moises-db.git"
            )
        if num_stems not in (4, 5, 6):
            raise ValueError("MoisesStemDataset supports num_stems 4, 5, or 6")

        self.stem_names = stem_names_for(num_stems)
        self.mix_map = moises_mix_map(num_stems)
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)

        root = get_moises_path()
        db = MoisesDB(data_path=str(root), sample_rate=sample_rate, quiet=True)
        n = len(db)
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=gen).tolist()
        n_val = max(1, int(n * val_ratio))
        val_idx = set(perm[:n_val])
        train_idx = [i for i in range(n) if i not in val_idx]

        self._indices = train_idx if split == "train" else sorted(val_idx)
        self._db = db

    def __len__(self) -> int:
        return len(self._indices)

    def _track_stems(self, track) -> dict[str, torch.Tensor]:
        mixed = track.mix_stems(self.mix_map)
        out: dict[str, torch.Tensor] = {}
        for name in self.stem_names:
            audio = mixed.get(name)
            if audio is None:
                audio = torch.zeros(1, dtype=torch.float32)
            else:
                if isinstance(audio, np.ndarray) and audio.ndim == 2:
                    audio = audio.mean(axis=0)
                audio = torch.from_numpy(np.asarray(audio, dtype=np.float32))
            out[name] = audio
        return out

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        track = self._db[self._indices[index]]
        stems = self._track_stems(track)
        mixture = sum(stems.values())

        length = mixture.shape[0]
        if length <= self.segment_samples:
            pad = self.segment_samples - length
            mixture = torch.nn.functional.pad(mixture, (0, pad))
            for name in self.stem_names:
                stems[name] = torch.nn.functional.pad(stems[name], (0, pad))
        else:
            start = torch.randint(0, length - self.segment_samples, (1,)).item()
            end = start + self.segment_samples
            mixture = mixture[start:end]
            for name in self.stem_names:
                stems[name] = stems[name][start:end]

        return {"mixture": mixture, **stems}
