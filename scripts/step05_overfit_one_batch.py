"""Step 5: Overfit U-Net on one batch (requires MUSDB_PATH)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.data.musdb_dataset import MusdbStemDataset
from src.models.unet_separator import UNetSeparator
from src.pipeline import vocal_l1_loss, waveform_batch_to_logspec


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MusdbStemDataset(segment_seconds=4.0)
    batch = next(iter(DataLoader(ds, batch_size=2, shuffle=True)))
    mix = batch["mixture"].to(device)
    target_voc = batch["vocals"].to(device)

    model = UNetSeparator().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training on ONE batch until loss drops (sanity check)...")
    for step in range(200):
        logspec, _ = waveform_batch_to_logspec(mix)
        mask = model(logspec.to(device))
        loss = vocal_l1_loss(mix, target_voc, mask)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 40 == 0:
            print(f"  step {step:3d}  L1 loss: {loss.item():.4f}")

    print("If loss went down a lot, your pipeline (STFT -> net -> iSTFT -> loss) works.")
    print("Next: full train.py on Colab with many tracks and epochs.")


if __name__ == "__main__":
    main()
