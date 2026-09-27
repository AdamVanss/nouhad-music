"""One folder = one training example: mixture.wav + vocals/drums/bass/other.wav."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.musdb_dataset import STEM_NAMES


class WavFolderStemDataset(Dataset):
    """All stems must be same length as mixture (e.g. exported from MUSDB or a DAW)."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        mix_path = self.folder / "mixture.wav"
        if not mix_path.is_file():
            raise FileNotFoundError(f"Need {mix_path}")
        for name in STEM_NAMES:
            p = self.folder / f"{name}.wav"
            if not p.is_file():
                raise FileNotFoundError(
                    f"Missing {p.name}. Training needs true stems, not just an MP3."
                )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        import soundfile as sf

        def load(name: str) -> torch.Tensor:
            x, _ = sf.read(str(self.folder / f"{name}.wav"), dtype="float32", always_2d=True)
            if x.ndim == 2:
                x = x.mean(axis=1)
            return torch.from_numpy(x)

        mixture = load("mixture")
        out = {"mixture": mixture}
        for name in STEM_NAMES:
            out[name] = load(name)
        n = mixture.shape[0]
        for name in STEM_NAMES:
            if out[name].shape[0] != n:
                raise ValueError(f"{name}.wav length != mixture.wav")
        return out
