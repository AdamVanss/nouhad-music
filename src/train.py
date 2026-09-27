"""Train 1/4/5/6-stem U-Net on MUSDB, MoisesDB, or mixed."""

import argparse
from pathlib import Path

import torch

from src.data.mixed_dataset import build_train_val_loaders
from src.data.stems import stem_names_for
from src.data.wav_folder_dataset import WavFolderStemDataset
from src.models.checkpoint_utils import load_checkpoint, load_state_with_stem_expand
from src.models.unet_separator import UNetSeparator
from src.pipeline import (
    multi_stem_hybrid_loss,
    multi_stem_l1_loss,
    multi_stem_pro_loss,
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
    stem_names: tuple[str, ...],
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
                stems = {k: batch[k].to(device) for k in stem_names}
                if loss_name == "pro":
                    loss = multi_stem_pro_loss(mix, stems, masks, stem_names)
                elif loss_name == "hybrid":
                    loss = multi_stem_hybrid_loss(mix, stems, masks, stem_names)
                else:
                    loss = multi_stem_l1_loss(mix, stems, masks, stem_names)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            n += 1
    return total / max(n, 1)


def default_ckpt_name(num_stems: int) -> str:
    if num_stems == 1:
        return "best.pt"
    if num_stems == 4:
        return "best_4stem.pt"
    return f"best_{num_stems}stem.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stem separator training")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--segment-seconds", type=float, default=8.0)
    parser.add_argument("--subset", type=str, default="train")
    parser.add_argument("--num-stems", type=int, default=1, choices=[1, 4, 5, 6])
    parser.add_argument(
        "--dataset",
        type=str,
        default="musdb",
        choices=["musdb", "moises", "mixed"],
        help="musdb | moises (5/6 stems) | mixed (MUSDB+Moises, 4-stem only)",
    )
    parser.add_argument(
        "--wav-folder",
        type=Path,
        default=None,
        help="Folder with mixture.wav + stem WAVs (overfit one clip)",
    )
    parser.add_argument("--checkpoint-out", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--expand-stems-from",
        type=Path,
        default=None,
        help="Load 4-stem (or smaller) weights into a wider head (e.g. pro -> 6-stem)",
    )
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3, choices=[3, 4])
    parser.add_argument("--loss", type=str, default="l1", choices=["l1", "hybrid", "pro"])
    args = parser.parse_args()

    stem_names = stem_names_for(args.num_stems)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device}  dataset: {args.dataset}  stems: {args.num_stems} {stem_names}  "
        f"base: {args.base}  depth: {args.depth}  loss: {args.loss}"
    )

    if args.wav_folder is not None:
        train_ds = WavFolderStemDataset(args.wav_folder)
        val_ds = train_ds
        if args.batch_size > 1:
            args.batch_size = 1
        from torch.utils.data import DataLoader

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    else:
        if args.dataset == "musdb" and args.num_stems not in (1, 4):
            raise SystemExit("MUSDB only supports 1- or 4-stem training; use --dataset moises")
        _, _, train_loader, val_loader = build_train_val_loaders(
            args.dataset,
            args.num_stems,
            args.segment_seconds,
            args.batch_size,
        )

    model = UNetSeparator(base=args.base, num_stems=args.num_stems, depth=args.depth).to(
        device
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_name = args.checkpoint_out or default_ckpt_name(args.num_stems)
    best_val = float("inf")
    start_epoch = 0

    if args.expand_stems_from is not None:
        path = Path(args.expand_stems_from)
        ckpt = load_checkpoint(path, device)
        old = int(ckpt.get("num_stems", 4))
        ckpt_base = int(ckpt.get("base", 16))
        ckpt_depth = int(ckpt.get("depth", 3))
        if ckpt_base != args.base or ckpt_depth != args.depth:
            raise ValueError("expand-stems-from: base/depth must match checkpoint")
        load_state_with_stem_expand(model, ckpt, old, args.num_stems)
        best_val = float(ckpt.get("val_loss", best_val))
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"Expanded {old}-stem head -> {args.num_stems} from {path}")

    elif args.resume is not None:
        path = Path(args.resume)
        ckpt = load_checkpoint(path, device)
        ckpt_base = int(ckpt.get("base", 16))
        ckpt_depth = int(ckpt.get("depth", 3))
        if ckpt_base != args.base or ckpt_depth != args.depth:
            raise ValueError(
                f"Checkpoint base={ckpt_base} depth={ckpt_depth} but "
                f"--base={args.base} --depth={args.depth}"
            )
        ckpt_stems = int(ckpt.get("num_stems", 1))
        if ckpt_stems != args.num_stems:
            raise ValueError(
                f"Checkpoint num_stems={ckpt_stems} but --num-stems={args.num_stems}. "
                "Use --expand-stems-from for 4 -> 6."
            )
        model.load_state_dict(ckpt["model"])
        best_val = float(ckpt.get("val_loss", best_val))
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"Resumed from {path}  (epoch {start_epoch}, best val {best_val:.4f})")

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model,
            train_loader,
            opt,
            device,
            args.num_stems,
            stem_names,
            args.loss,
            train=True,
        )
        va = run_epoch(
            model,
            val_loader,
            opt,
            device,
            args.num_stems,
            stem_names,
            args.loss,
            train=False,
        )
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
                    "stem_names": list(stem_names),
                    "base": args.base,
                    "depth": args.depth,
                    "loss": args.loss,
                    "dataset": args.dataset,
                },
                path,
            )
            print(f"  saved {path}")

    print(f"Done. Separate: python -m src.separate --checkpoint checkpoints/{ckpt_name} --input ...")


if __name__ == "__main__":
    main()
