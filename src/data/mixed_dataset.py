"""Concatenate MUSDB + MoisesDB training sources."""

import random

from torch.utils.data import ConcatDataset, Dataset

from src.data.moises_dataset import MoisesStemDataset
from src.data.musdb_dataset import MusdbStemDataset


class MixedStemDataset(Dataset):
    """Randomly sample MUSDB + Moises (both 4-stem)."""

    def __init__(
        self,
        segment_seconds: float = 8.0,
        moises_weight: float = 0.5,
    ):
        self.moises_weight = moises_weight
        self._musdb = MusdbStemDataset(
            subset="train", segment_seconds=segment_seconds
        )
        self._moises = MoisesStemDataset(
            num_stems=4, segment_seconds=segment_seconds, split="train"
        )

    def __len__(self) -> int:
        return len(self._musdb) + len(self._moises)

    def __getitem__(self, index: int) -> dict:
        use_moises = random.random() < self.moises_weight
        if use_moises:
            j = random.randint(0, len(self._moises) - 1)
            return self._moises[j]
        j = random.randint(0, len(self._musdb) - 1)
        return self._musdb[j]


def build_train_val_loaders(
    dataset: str,
    num_stems: int,
    segment_seconds: float,
    batch_size: int,
):
    """Return (train_ds, val_ds) for musdb | moises | mixed."""
    from torch.utils.data import DataLoader

    if dataset == "musdb":
        train_ds = MusdbStemDataset(subset="train", segment_seconds=segment_seconds)
        try:
            val_ds = MusdbStemDataset(subset="test", segment_seconds=segment_seconds)
        except Exception:
            val_ds = train_ds
    elif dataset == "moises":
        train_ds = MoisesStemDataset(
            num_stems=num_stems, segment_seconds=segment_seconds, split="train"
        )
        val_ds = MoisesStemDataset(
            num_stems=num_stems, segment_seconds=segment_seconds, split="val"
        )
    elif dataset == "mixed":
        if num_stems != 4:
            raise ValueError("mixed dataset only supports num_stems=4 (use moises for 5/6)")
        train_ds = MixedStemDataset(segment_seconds=segment_seconds)
        m_test = MusdbStemDataset(subset="test", segment_seconds=segment_seconds)
        moises_val = MoisesStemDataset(
            num_stems=4, segment_seconds=segment_seconds, split="val"
        )
        val_ds = ConcatDataset([m_test, moises_val])
    else:
        raise ValueError(f"Unknown dataset={dataset}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_ds, val_ds, train_loader, val_loader
