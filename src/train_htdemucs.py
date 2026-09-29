"""Fine-tune pretrained HTDemucs on MUSDB18.

The 2M-parameter mask U-Net cannot reach current separation quality.
HTDemucs is a 42M hybrid waveform/spectrogram transformer trained by Meta on
MUSDB plus 800 extra songs. This script keeps those weights and adapts them
on the local 12GB GPU, with extra weight on the `other` stem.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from demucs.pretrained import get_model

from src.data.musdb_dataset import TrackGroupedSampler
from src.data.musdb_stereo import SOURCES, MusdbStereoDataset
from src.data.stem_folder_dataset import StemFolderDataset

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
SEGMENT_SAMPLES = 343980  # 7.8s at 44.1 kHz, HTDemucs training length
# drums, bass, other, vocals — other is upweighted because leftover bleed lands there
STEM_WEIGHTS = (1.0, 1.0, 2.0, 1.5)
SILENCE_RMS = 10 ** (-50 / 20)


def new_sdr(references: torch.Tensor, estimates: torch.Tensor) -> torch.Tensor:
    """MDX SDR. references/estimates are (B, S, C, T). Returns (B, S) in dB."""
    delta = 1e-7
    num = references.square().sum(dim=(2, 3))
    den = (references - estimates).square().sum(dim=(2, 3))
    return 10 * torch.log10((num + delta) / (den + delta))


def _augment(sources: torch.Tensor) -> torch.Tensor:
    """sources (B, S, C, T). Remix stems across the batch and apply per-stem gain."""
    batch, stems, _channels, _time = sources.shape
    if batch > 1:
        remixed = sources.clone()
        for stem in range(stems):
            if torch.rand((), device=sources.device).item() < 0.5:
                order = torch.randperm(batch, device=sources.device)
                remixed[:, stem] = sources[order, stem]
        sources = remixed
    gains = torch.empty(batch, stems, 1, 1, device=sources.device).uniform_(0.5, 1.25)
    return sources * gains


def _l1(estimate: torch.Tensor, sources: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.l1_loss(estimate, sources, reduction="none")
    loss = loss.mean(dim=(2, 3)).mean(dim=0)
    return (loss * weights).sum() / weights.sum()


def _scale_at(step: int, total_steps: int) -> float:
    warm = max(1, total_steps // 20)
    if step < warm:
        return (step + 1) / warm
    progress = (step - warm) / max(1, total_steps - warm)
    return 0.1 + 0.9 * 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())


@torch.no_grad()
def evaluate(model, loader, device, weights: torch.Tensor, base_segment) -> dict[str, float]:
    """Mean SDR per stem, skipping crops where that stem is near-silent (SDR is meaningless there)."""
    model.eval()
    model.segment = base_segment
    totals = torch.zeros(len(SOURCES))
    counts = torch.zeros(len(SOURCES))
    for sources in loader:
        sources = sources.to(device, non_blocking=True)
        mix = sources.sum(dim=1)
        estimate = model(mix).float()
        scores = new_sdr(sources.float(), estimate).cpu()
        audible = (sources.float().square().mean(dim=(2, 3)).sqrt() > SILENCE_RMS).cpu()
        totals += (scores * audible).sum(dim=0)
        counts += audible.sum(dim=0)
    totals /= counts.clamp(min=1)
    out = {name: float(totals[i]) for i, name in enumerate(SOURCES)}
    weighted = (totals * weights.cpu()).sum() / weights.sum().cpu()
    out["mean"] = float(weighted)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune HTDemucs on MUSDB18")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--extra-root",
        type=str,
        default=str(ROOT / "data" / "cmt_stems"),
        help="Folder of songs with drums/bass/other/vocals wavs, added to MUSDB train",
    )
    parser.add_argument("--chunks-per-track", type=int, default=6)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--checkpoint-out", type=str, default="htdemucs_finetune.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("HTDemucs fine-tuning needs the NVIDIA GPU.")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  VRAM: {props.total_memory / 1024**3:.1f} GB")

    bag = get_model("htdemucs")
    model = bag.models[0].to(device)
    base_segment = model.segment
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"HTDemucs {n_params:.1f}M  sources {tuple(model.sources)}  "
        f"segment {float(base_segment):.1f}s"
    )
    if tuple(model.sources) != SOURCES:
        raise SystemExit(f"Unexpected source order: {model.sources}")

    musdb_train = MusdbStereoDataset(
        "train", segment_samples=SEGMENT_SAMPLES, chunks_per_track=args.chunks_per_track
    )
    extra = StemFolderDataset(
        Path(args.extra_root), segment_samples=SEGMENT_SAMPLES, chunks_per_track=args.chunks_per_track
    )
    n_train_tracks = musdb_train.n_tracks + extra.n_tracks
    train_ds = ConcatDataset([musdb_train, extra]) if extra.n_tracks else musdb_train
    print(f"train songs: MUSDB {musdb_train.n_tracks} + extra {extra.n_tracks}")
    val_ds = MusdbStereoDataset("test", segment_samples=SEGMENT_SAMPLES, chunks_per_track=1)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=TrackGroupedSampler(n_train_tracks, args.chunks_per_track, shuffle=True),
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    weights = torch.tensor(STEM_WEIGHTS, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = max(1, args.epochs * len(train_loader))
    print(f"tracks {n_train_tracks}  steps/epoch {len(train_loader)}  total {total_steps}")

    print("Baseline (pretrained, no fine-tune yet)")
    baseline = evaluate(model, val_loader, device, weights, base_segment)
    print(
        "  "
        + "  ".join(f"{name} {baseline[name]:.2f}" for name in (*SOURCES, "mean"))
        + " dB",
        flush=True,
    )
    best = baseline["mean"]
    CKPT_DIR.mkdir(exist_ok=True)
    out_path = CKPT_DIR / args.checkpoint_out
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for sources in train_loader:
            scale = _scale_at(global_step, total_steps)
            for group in opt.param_groups:
                group["lr"] = args.lr * scale
            sources = sources.to(device, non_blocking=True)
            sources = _augment(sources)
            mix = sources.sum(dim=1)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                estimate = model(mix)
            loss = _l1(estimate.float(), sources.float(), weights)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            global_step += 1
            running += float(loss.item())
            seen += 1
            if seen == 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                print(f"  GPU mem allocated {mem:.2f} GB", flush=True)
            if args.log_every > 0 and seen % args.log_every == 0:
                print(
                    f"  step {seen}/{len(train_loader)}  loss {loss.item():.4f}  "
                    f"lr {opt.param_groups[0]['lr']:.2e}",
                    flush=True,
                )
        metrics = evaluate(model, val_loader, device, weights, base_segment)
        train_loss = running / max(seen, 1)
        print(
            f"epoch {epoch:2d}  train l1 {train_loss:.4f}  "
            + "  ".join(f"{name} {metrics[name]:.2f}" for name in SOURCES)
            + f"  mean {metrics['mean']:.2f} dB",
            flush=True,
        )
        payload = {
            "model": model.state_dict(),
            "sources": list(model.sources),
            "samplerate": model.samplerate,
            "segment": float(base_segment),
            "epoch": epoch,
            "nsdr": metrics,
            "baseline_nsdr": baseline,
            "stem_weights": STEM_WEIGHTS,
        }
        torch.save(payload, CKPT_DIR / f"{out_path.stem}_last.pt")
        if metrics["mean"] > best:
            best = metrics["mean"]
            torch.save(payload, out_path)
            print(f"  saved {out_path}")

    print(
        f"Done. Baseline mean {baseline['mean']:.2f} dB, best fine-tune {best:.2f} dB. "
        f"Checkpoint: {out_path if out_path.is_file() else '(pretrained stayed better)'}"
    )


if __name__ == "__main__":
    main()
