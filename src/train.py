"""Train 1/4/5/6-stem U-Net on MUSDB, MoisesDB, or mixed."""

import argparse
import math
from pathlib import Path

import torch

from src.data.mixed_dataset import build_train_val_loaders
from src.data.stems import stem_names_for
from src.data.wav_folder_dataset import WavFolderStemDataset
from src.models.checkpoint_utils import (
    load_checkpoint,
    load_separator_state,
    load_state_with_stem_expand,
)
from src.models.unet_separator import UNetSeparator
from src.pipeline import (
    model_outputs,
    multi_stem_hybrid_loss,
    multi_stem_l1_loss,
    multi_stem_pro_loss,
    separator_features,
    vocal_l1_loss,
)

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"


def _stem_loss(
    loss_name: str,
    mixture: torch.Tensor,
    batch: dict,
    masks: torch.Tensor,
    phase_delta: torch.Tensor | None,
    num_stems: int,
    stem_names: tuple[str, ...],
) -> torch.Tensor:
    if num_stems == 1:
        return vocal_l1_loss(mixture, batch["vocals"], masks, phase_delta)
    stems = {k: batch[k] for k in stem_names}
    if loss_name == "pro":
        return multi_stem_pro_loss(mixture, stems, masks, stem_names, phase_delta)
    if loss_name == "hybrid":
        return multi_stem_hybrid_loss(
            mixture, stems, masks, stem_names, phase_delta=phase_delta
        )
    return multi_stem_l1_loss(mixture, stems, masks, stem_names, phase_delta)


def _scale_at(step: int, total_steps: int) -> float:
    warm = max(1, total_steps // 20)
    if step < warm:
        return (step + 1) / warm
    progress = (step - warm) / max(1, total_steps - warm)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    num_stems: int,
    stem_names: tuple[str, ...],
    loss_name: str,
    train: bool,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    grad_clip: float = 1.0,
    base_lrs: list[float] | None = None,
    global_step: int = 0,
    total_steps: int = 1,
    log_every: int = 10,
) -> tuple[float, int]:
    if train:
        model.train()
    else:
        model.eval()

    total = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    amp_enabled = use_amp and device.type == "cuda"
    with ctx:
        for batch in loader:
            mix = batch["mixture"].to(device, non_blocking=True)
            moved = {
                k: batch[k].to(device, non_blocking=True)
                for k in (stem_names if num_stems > 1 else ("vocals",))
            }
            if train and base_lrs is not None and optimizer is not None:
                scale = _scale_at(global_step, total_steps)
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr * scale

            feats, _ = separator_features(mix, bool(getattr(model, "predict_phase", False)))
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                masks, phase_delta = model_outputs(model, feats)
            masks = masks.float()
            if phase_delta is not None:
                phase_delta = phase_delta.float()
            loss = _stem_loss(
                loss_name, mix, moved, masks, phase_delta, num_stems, stem_names
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                global_step += 1
                if n == 0 and device.type == "cuda":
                    mem = torch.cuda.max_memory_allocated() / (1024**3)
                    print(f"  GPU mem allocated {mem:.2f} GB", flush=True)
                if log_every > 0 and (n + 1) % log_every == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"  step {n + 1}/{len(loader)}  loss {loss.item():.4f}  lr {lr:.2e}",
                        flush=True,
                    )
            total += loss.item()
            n += 1
    return total / max(n, 1), global_step


def default_ckpt_name(num_stems: int) -> str:
    if num_stems == 1:
        return "best.pt"
    if num_stems == 4:
        return "best_4stem.pt"
    return f"best_{num_stems}stem.pt"


def _checkpoint_payload(
    model,
    epoch: int,
    val_loss: float,
    args,
    stem_names: tuple[str, ...],
) -> dict:
    return {
        "model": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "segment_seconds": args.segment_seconds,
        "num_stems": args.num_stems,
        "stem_names": list(stem_names),
        "base": args.base,
        "depth": args.depth,
        "loss": args.loss,
        "dataset": args.dataset,
        "predict_phase": bool(args.phase),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stem separator training")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="8 x 8s fits a 12GB RTX 3060 with --phase --amp",
    )
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
    parser.add_argument(
        "--phase",
        action="store_true",
        help="Predict a phase residual and condition on mixture phase",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--chunks-per-track",
        type=int,
        default=1,
        help="Random crops per song each epoch (MUSDB). Groups songs so decode hits cache.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    stem_names = stem_names_for(args.num_stems)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        props = torch.cuda.get_device_properties(0)
        print(
            f"GPU: {props.name}  VRAM: {props.total_memory / (1024**3):.1f} GB  "
            f"amp: {use_amp}"
        )
    print(
        f"Device: {device}  dataset: {args.dataset}  stems: {args.num_stems} {stem_names}  "
        f"base: {args.base}  depth: {args.depth}  loss: {args.loss}  phase: {args.phase}"
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
            chunks_per_track=args.chunks_per_track,
            pin_memory=device.type == "cuda",
        )

    model = UNetSeparator(
        base=args.base,
        num_stems=args.num_stems,
        depth=args.depth,
        predict_phase=args.phase,
    ).to(device)
    phase_params = []
    base_params = []
    for name, param in model.named_parameters():
        if name.startswith("phase_head") or name.startswith("input_proj"):
            phase_params.append(param)
        else:
            base_params.append(param)
    param_groups = [{"params": base_params, "lr": args.lr}]
    if phase_params:
        param_groups.append({"params": phase_params, "lr": args.lr * 5})
    opt = torch.optim.Adam(param_groups)
    base_lrs = [group["lr"] for group in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
        best_val = float("inf")
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
        ckpt_phase = bool(ckpt.get("predict_phase", False))
        if ckpt_phase and not args.phase:
            raise SystemExit("This checkpoint has a phase head. Re-run with --phase.")
        mode = load_separator_state(model, ckpt)
        start_epoch = int(ckpt.get("epoch", 0))
        if mode == "adapted-phase":
            best_val = float("inf")
            print(
                f"Warm-started magnitude weights from {path} "
                f"(epoch {start_epoch}); phase head starts at zero"
            )
        else:
            best_val = float(ckpt.get("val_loss", best_val))
            print(f"Resumed from {path}  (epoch {start_epoch}, best val {best_val:.4f})")

    total_steps = max(1, args.epochs * len(train_loader))
    global_step = 0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M  steps/epoch: {len(train_loader)}  total steps: {total_steps}")

    for epoch in range(1, args.epochs + 1):
        tr, global_step = run_epoch(
            model,
            train_loader,
            opt,
            device,
            args.num_stems,
            stem_names,
            args.loss,
            train=True,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
            base_lrs=base_lrs,
            global_step=global_step,
            total_steps=total_steps,
            log_every=args.log_every,
        )
        va, _ = run_epoch(
            model,
            val_loader,
            opt,
            device,
            args.num_stems,
            stem_names,
            args.loss,
            train=False,
            use_amp=False,
            grad_clip=0,
        )
        global_epoch = start_epoch + epoch
        print(
            f"epoch {epoch:2d} (total {global_epoch})  train loss: {tr:.4f}  val loss: {va:.4f}",
            flush=True,
        )
        payload = _checkpoint_payload(model, global_epoch, va, args, stem_names)
        last_path = CKPT_DIR / f"{Path(ckpt_name).stem}_last.pt"
        torch.save(payload, last_path)
        if va < best_val:
            best_val = va
            path = CKPT_DIR / ckpt_name
            torch.save(payload, path)
            print(f"  saved {path}")

    print(f"Done. Separate: python -m src.separate --checkpoint checkpoints/{ckpt_name} --input ...")


if __name__ == "__main__":
    main()
