"""Stereo MUSDB crops in HTDemucs source order: drums, bass, other, vocals."""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.musdb_dataset import (
    TrackGroupedSampler,
    _load_dotenv,
    _safe_name,
    get_musdb_path,
)

try:
    import musdb
except ImportError:
    musdb = None

# stem.mp4 stream index -> HTDemucs source name
SOURCES = ("drums", "bass", "other", "vocals")
_STEM_ID = {"drums": 1, "bass": 2, "other": 3, "vocals": 4}


class MusdbStereoDataset(Dataset):
    """Random stereo crop of the four stems. Mixture is their sum, as in Demucs training."""

    def __init__(
        self,
        subset: str = "train",
        segment_samples: int = 343980,
        chunks_per_track: int = 1,
        sample_rate: int = 44100,
    ):
        if musdb is None:
            raise ImportError("pip install musdb")
        if chunks_per_track < 1:
            raise ValueError("chunks_per_track must be >= 1")

        _load_dotenv()
        root = get_musdb_path()
        self.subset = subset
        self.segment_samples = segment_samples
        self.chunks_per_track = chunks_per_track
        self.sample_rate = sample_rate
        cache_env = os.environ.get("MUSDB_STEREO_CACHE")
        self.cache_dir = Path(cache_env) if cache_env else root.parent / "musdb_stereo_cache"
        self._db = musdb.DB(root=str(root), subsets=subset, is_wav=False)
        self._ram_index: int | None = None
        self._ram: torch.Tensor | None = None

    @property
    def n_tracks(self) -> int:
        return len(self._db)

    def __len__(self) -> int:
        return self.n_tracks * self.chunks_per_track

    def _stem_path(self, track_name: str, source: str) -> Path:
        return self.cache_dir / self.subset / _safe_name(track_name) / f"{source}.wav"

    def _load_track(self, track_index: int) -> torch.Tensor:
        if self._ram_index == track_index and self._ram is not None:
            return self._ram
        import soundfile as sf

        track = self._db[track_index]
        missing = [name for name in SOURCES if not self._stem_path(track.name, name).is_file()]
        stems: list[torch.Tensor] = []
        decoded = None
        if missing:
            print(f"decoding {self.subset}/{track.name}", flush=True)
            audio = track.stems  # (5, T, 2)
            decoded = {}
            for name in SOURCES:
                mono_stereo = np.asarray(audio[_STEM_ID[name]], dtype=np.float32)
                if mono_stereo.ndim == 1:
                    mono_stereo = np.stack([mono_stereo, mono_stereo], axis=1)
                path = self._stem_path(track.name, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), np.clip(mono_stereo, -1.0, 1.0), self.sample_rate, subtype="PCM_16")
                decoded[name] = torch.from_numpy(np.ascontiguousarray(mono_stereo.T))
        for name in SOURCES:
            if decoded is not None and name in decoded:
                stems.append(decoded[name])
                continue
            audio, _sr = sf.read(str(self._stem_path(track.name, name)), dtype="float32", always_2d=True)
            stems.append(torch.from_numpy(np.ascontiguousarray(audio.T)))
        stacked = torch.stack(stems, dim=0)  # (4, 2, T)
        self._ram_index = track_index
        self._ram = stacked
        return stacked

    def __getitem__(self, index: int) -> torch.Tensor:
        track_index = index // self.chunks_per_track
        stems = self._load_track(track_index)
        length = stems.shape[-1]
        if length <= self.segment_samples:
            stems = torch.nn.functional.pad(stems, (0, self.segment_samples - length))
        else:
            high = length - self.segment_samples
            if self.subset == "train":
                start = torch.randint(0, high, (1,)).item()
            else:
                generator = torch.Generator()
                generator.manual_seed(track_index + 17)
                start = torch.randint(0, high, (1,), generator=generator).item()
            stems = stems[..., start : start + self.segment_samples].contiguous()
        return stems
