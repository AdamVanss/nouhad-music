"""Mini trainer — 1-stem (vocals) or 4-stem. Saves checkpoints/best.pt or best_4stem.pt."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.musdb_dataset import MusdbStemDataset
from src.data.wav_folder_dataset import WavFolderStemDataset
from src.models.unet_separator import UNetSeparator
from src.pipeline import (
    four_stem_hybrid_loss,
    four_stem_l1_loss,
    four_stem_pro_loss,
    vocal_l1_loss,
    waveform_batch_to_logspec,
)

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    num_stems: int,
    loss_name: str,
    train: bool,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            mix = batch["mixture"].to(device)
            logspec, _ = waveform_batch_to_logspec(mix)
            masks = model(logspec)
            if num_stems == 1:
                loss = vocal_l1_loss(mix, batch["vocals"].to(device), masks)
            else:
                stems = {k: batch[k].to(device) for k in ("vocals", "drums", "bass", "other")}
                if loss_name == "pro":
                    loss = four_stem_pro_loss(mix, stems, masks)
                elif loss_name == "hybrid":
                    loss = four_stem_hybrid_loss(mix, stems, masks)
                else:
                    loss = four_stem_l1_loss(mix, stems, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            n += 1
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini stem separator training")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--subset", type=str, default="train")
    parser.add_argument("--num-stems", type=int, default=1, choices=[1, 4])
    parser.add_argument(
        "--wav-folder",
        type=Path,
        default=None,
        help="Folder with mixture.wav + four stem WAVs (overfit one clip; not an MP3 alone)",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=str,
        default=None,
        help="Override checkpoint filename (default best.pt / best_4stem.pt)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Continue training from a .pt checkpoint (fine-tune)",
    )
    parser.add_argument(
        "--base",
        type=int,
        default=16,
        help="U-Net width (16=default; 32=recommended for quality)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        choices=[3, 4],
        help="3=original small U-Net; 4=deeper (use with base 32 for best results)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="l1",
        choices=["l1", "hybrid", "pro"],
        help="pro = hybrid + multi-res spec + SI-SDR (best custom U-Net; slowest)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device}  stems: {args.num_stems}  base: {args.base}  "
        f"depth: {args.depth}  loss: {args.loss}"
    )

    if args.wav_folder is not None:
        train_ds = WavFolderStemDataset(args.wav_folder)
        val_ds = train_ds
        if args.batch_size > 1:
            args.batch_size = 1
    else:
        train_ds = MusdbStemDataset(subset=args.subset, segment_seconds=args.segment_seconds)
        try:
            val_ds = MusdbStemDataset(subset="test", segment_seconds=args.segment_seconds)
        except Exception:
            val_ds = train_ds

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNetSeparator(base=args.base, num_stems=args.num_stems, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_name = args.checkpoint_out or ("best_4stem.pt" if args.num_stems == 4 else "best.pt")
    best_val = float("inf")
    start_epoch = 0

    if args.resume is not None:
        path = Path(args.resume)
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=device)
        ckpt_base = int(ckpt.get("base", 16))
        ckpt_depth = int(ckpt.get("depth", 3))
        if ckpt_base != args.base or ckpt_depth != args.depth:
            raise ValueError(
                f"Checkpoint base={ckpt_base} depth={ckpt_depth} but "
                f"--base={args.base} --depth={args.depth}. Match architecture to resume."
            )
        ckpt_stems = int(ckpt.get("num_stems", 1))
        if ckpt_stems != args.num_stems:
            raise ValueError(f"Checkpoint num_stems={ckpt_stems} but --num-stems={args.num_stems}")
        model.load_state_dict(ckpt["model"])
        best_val = float(ckpt.get("val_loss", best_val))
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"Resumed from {path}  (epoch {start_epoch}, best val L1 {best_val:.4f})")

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, opt, device, args.num_stems, args.loss, train=True)
        va = run_epoch(model, val_loader, opt, device, args.num_stems, args.loss, train=False)
        global_epoch = start_epoch + epoch
        print(f"epoch {epoch:2d} (total {global_epoch})  train loss: {tr:.4f}  val loss: {va:.4f}")
        if va < best_val:
            best_val = va
            path = CKPT_DIR / ckpt_name
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": global_epoch,
                    "val_loss": va,
                    "segment_seconds": args.segment_seconds,
                    "num_stems": args.num_stems,
                    "base": args.base,
                    "depth": args.depth,
                    "loss": args.loss,
                },
                path,
            )
            print(f"  saved {path}")

    print(f"Done. Separate: python -m src.separate --checkpoint checkpoints/{ckpt_name} --input ...")


if __name__ == "__main__":
    main()
