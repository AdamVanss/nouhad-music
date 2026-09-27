"""Step 4: One batch from MusdbStemDataset (requires MUSDB_PATH)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader

from src.data.musdb_dataset import MusdbStemDataset


def main() -> None:
    ds = MusdbStemDataset(subset="train", segment_seconds=6.0)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print("mixture:", batch["mixture"].shape)
    print("vocals:", batch["vocals"].shape)
    print("OK — same idea as X (mixture) and y (vocals) in supervised learning.")


if __name__ == "__main__":
    main()
