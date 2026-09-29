"""Concatenate MUSDB + MoisesDB training sources."""

import random

from torch.utils.data import ConcatDataset, Dataset

from src.data.moises_dataset import MoisesStemDataset
from src.data.musdb_dataset import MusdbStemDataset, TrackGroupedSampler


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
    chunks_per_track: int = 1,
    pin_memory: bool = False,
):
    """Return (train_ds, val_ds, train_loader, val_loader) for musdb | moises | mixed."""
    from torch.utils.data import DataLoader

    train_sampler = None
    if dataset == "musdb":
        train_ds = MusdbStemDataset(
            subset="train",
            segment_seconds=segment_seconds,
            chunks_per_track=chunks_per_track,
        )
        try:
            val_ds = MusdbStemDataset(subset="test", segment_seconds=segment_seconds)
        except Exception:
            val_ds = train_ds
        if chunks_per_track > 1:
            train_sampler = TrackGroupedSampler(train_ds.n_tracks, chunks_per_track, shuffle=True)
    elif chunks_per_track != 1:
        raise ValueError("--chunks-per-track is supported for --dataset musdb")
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
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_ds, val_ds, train_loader, val_loader
