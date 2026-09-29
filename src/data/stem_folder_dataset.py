"""Stereo crops from folders holding drums/bass/other/vocals wavs (e.g. data/cmt_stems)."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.data.musdb_stereo import SOURCES


class StemFolderDataset(Dataset):
    """Same (4, 2, T) layout as MusdbStereoDataset, so the two can be concatenated."""

    def __init__(self, root: Path, segment_samples: int = 343980, chunks_per_track: int = 1):
        self.root = Path(root)
        self.tracks = sorted(
            p for p in self.root.iterdir() if all((p / f"{name}.wav").is_file() for name in SOURCES)
        ) if self.root.is_dir() else []
        self.segment_samples = segment_samples
        self.chunks_per_track = chunks_per_track
        self._ram_index: int | None = None
        self._ram: torch.Tensor | None = None

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)

    def __len__(self) -> int:
        return self.n_tracks * self.chunks_per_track

    def _load_track(self, track_index: int) -> torch.Tensor:
        if self._ram_index == track_index and self._ram is not None:
            return self._ram
        stems = []
        for name in SOURCES:
            audio, _sr = sf.read(str(self.tracks[track_index] / f"{name}.wav"), dtype="float32", always_2d=True)
            stems.append(torch.from_numpy(np.ascontiguousarray(audio.T)))
        self._ram_index = track_index
        self._ram = torch.stack(stems, dim=0)
        return self._ram

    def __getitem__(self, index: int) -> torch.Tensor:
        stems = self._load_track(index // self.chunks_per_track)
        length = stems.shape[-1]
        if length <= self.segment_samples:
            return torch.nn.functional.pad(stems, (0, self.segment_samples - length))
        # prefer a crop where vocals are audible so the vocal target is not silence
        best, best_energy = 0, -1.0
        for _ in range(4):
            start = torch.randint(0, length - self.segment_samples, (1,)).item()
            energy = stems[3, :, start : start + self.segment_samples].square().mean().item()
            if energy > best_energy:
                best, best_energy = start, energy
        return stems[..., best : best + self.segment_samples].contiguous()
