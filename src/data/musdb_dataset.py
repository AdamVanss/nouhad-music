"""MUSDB18 chunks for supervised training."""

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

try:
    import musdb
except ImportError:
    musdb = None

from src.data.stems import STEM_NAMES_4 as STEM_NAMES

# Native Instruments stem order used by MUSDB18 .stem.mp4 files.
_STEM_INDEX = {"drums": 1, "bass": 2, "other": 3, "vocals": 4}


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_musdb_path() -> Path:
    _load_dotenv()
    path = os.environ.get("MUSDB_PATH")
    if not path:
        raise EnvironmentError(
            "Set MUSDB_PATH to your musdb18 folder. See docs/MUSDB18_SETUP.md"
        )
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"MUSDB_PATH is not a directory: {p}")
    return p


def _safe_name(name: str) -> str:
    cleaned = "".join(c if c not in '<>:"/\\|?*' else "_" for c in name).rstrip(" .")
    return cleaned or "track"


class TrackGroupedSampler(Sampler[int]):
    """Yield every chunk of one song before moving on, so decode cache hits."""

    def __init__(self, n_tracks: int, chunks_per_track: int, shuffle: bool = True):
        self.n_tracks = n_tracks
        self.chunks_per_track = chunks_per_track
        self.shuffle = shuffle

    def __iter__(self):
        tracks = list(range(self.n_tracks))
        if self.shuffle:
            random.shuffle(tracks)
        for track in tracks:
            base = track * self.chunks_per_track
            for chunk in range(self.chunks_per_track):
                yield base + chunk

    def __len__(self) -> int:
        return self.n_tracks * self.chunks_per_track


class MusdbStemDataset(Dataset):
    """Random crop: mixture + stems (vocals, drums, bass, other)."""

    def __init__(
        self,
        subset: str = "train",
        sample_rate: int = 44100,
        segment_seconds: float = 6.0,
        target: str = "vocals",
        chunks_per_track: int = 1,
    ):
        if musdb is None:
            raise ImportError("pip install musdb")
        if chunks_per_track < 1:
            raise ValueError("chunks_per_track must be >= 1")

        root = get_musdb_path()
        self.subset = subset
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.target = target  # used by single-stem loaders; all stems still returned
        self.chunks_per_track = chunks_per_track
        cache_env = os.environ.get("MUSDB_CACHE")
        self.cache_dir = (
            Path(cache_env) if cache_env else root.parent / "musdb_mono_cache"
        )
        self._ram_index: int | None = None
        self._ram_stems: dict[str, torch.Tensor] | None = None

        self._db = musdb.DB(root=str(root), subsets=subset, is_wav=False)

    @property
    def n_tracks(self) -> int:
        return len(self._db)

    def __len__(self) -> int:
        return len(self._db) * self.chunks_per_track

    def _cache_path(self, track_name: str, stem: str) -> Path:
        return self.cache_dir / self.subset / _safe_name(track_name) / f"{stem}.wav"

    def _load_mono(self, track, name: str) -> torch.Tensor:
        import soundfile as sf

        path = self._cache_path(track.name, name)
        if path.is_file():
            audio, _sr = sf.read(str(path), dtype="float32", always_2d=False)
            return torch.from_numpy(audio)

        audio = track.targets[name].audio
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        wav = torch.from_numpy(audio.astype("float32").copy())
        if track.rate != self.sample_rate:
            wav = torchaudio_resample(wav, track.rate, self.sample_rate)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), wav.numpy(), self.sample_rate)
        return wav

    def _load_track(self, track_index: int) -> dict[str, torch.Tensor]:
        if self._ram_index == track_index and self._ram_stems is not None:
            return self._ram_stems
        track = self._db[track_index]
        missing = [
            name for name in STEM_NAMES if not self._cache_path(track.name, name).is_file()
        ]
        decoded: dict[str, torch.Tensor] = {}
        if missing:
            # One ffmpeg pass for every stem. Per-target reads decode the mp4 four times.
            print(f"decoding {self.subset}/{track.name}", flush=True)
            audio = track.stems
            import soundfile as sf

            for name, stem_id in _STEM_INDEX.items():
                mono = audio[stem_id].mean(axis=1).astype("float32")
                wav = torch.from_numpy(mono.copy())
                if track.rate != self.sample_rate:
                    wav = torchaudio_resample(wav, int(track.rate), self.sample_rate)
                path = self._cache_path(track.name, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), wav.numpy(), self.sample_rate)
                decoded[name] = wav
        stems = {
            name: decoded[name] if name in decoded else self._load_mono(track, name)
            for name in STEM_NAMES
        }
        self._ram_index = track_index
        self._ram_stems = stems
        return stems

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        track_index = index // self.chunks_per_track
        stems_full = self._load_track(track_index)
        stems = {name: wav for name, wav in stems_full.items()}
        mixture = sum(stems.values())

        length = mixture.shape[0]
        if length <= self.segment_samples:
            pad = self.segment_samples - length
            mixture = torch.nn.functional.pad(mixture, (0, pad))
            for name in STEM_NAMES:
                stems[name] = torch.nn.functional.pad(stems[name], (0, pad))
        else:
            high = length - self.segment_samples
            if self.subset == "train":
                start = torch.randint(0, high, (1,)).item()
            else:
                generator = torch.Generator()
                generator.manual_seed(track_index + 17)
                start = torch.randint(0, high, (1,), generator=generator).item()
            end = start + self.segment_samples
            mixture = mixture[start:end].contiguous()
            for name in STEM_NAMES:
                stems[name] = stems[name][start:end].contiguous()

        return {"mixture": mixture, **stems}


def torchaudio_resample(wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    import torchaudio

    if orig_sr == new_sr:
        return wav
    return torchaudio.functional.resample(wav.unsqueeze(0), orig_sr, new_sr).squeeze(0)
