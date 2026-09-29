"""Run a trained checkpoint: vocals-only (1-stem) or all four MUSDB stems (4-stem)."""

import argparse
from pathlib import Path

import soundfile as sf
import torch

from src.audio.load import load_mono
from src.models.unet_separator import UNetSeparator
from src.data.stems import stem_names_for
from src.pipeline import separate_stems

# Long songs: one full STFT OOMs on CPU; process fixed-length windows instead.
_DEFAULT_CHUNK_SEC = 7.0
_AUTO_CHUNK_IF_LONGER_SEC = 45.0


def separate_chunked(
    mixture: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    chunk_samples: int,
    hop_samples: int | None = None,
    stem_names: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Overlap-add with Hann window — smoother than hard 7s concat."""
    n = mixture.shape[0]
    if hop_samples is None or hop_samples <= 0:
        hop_samples = max(1, chunk_samples // 2)
    window = torch.hann_window(chunk_samples, periodic=False)

    acc: dict[str, torch.Tensor] = {}
    weight = torch.zeros(n)

    for start in range(0, n, hop_samples):
        end = min(start + chunk_samples, n)
        seg_len = end - start
        chunk = mixture[start:end]
        if seg_len < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - seg_len))

        stems = separate_stems(chunk, model, device, stem_names=stem_names)
        w = window[:seg_len]

        for name, wav in stems.items():
            piece = wav[:seg_len]
            if name not in acc:
                acc[name] = torch.zeros(n)
            acc[name][start:end] += piece * w
        weight[start:end] += w

    weight = weight.clamp(min=1e-8)
    return {name: acc[name] / weight for name in acc}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/separated"))
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--start-seconds", type=float, default=None, help="Crop start (e.g. 28)")
    parser.add_argument("--end-seconds", type=float, default=None, help="Crop end (e.g. 50)")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help=f"Split long audio into chunks (default {_DEFAULT_CHUNK_SEC}s if input > {_AUTO_CHUNK_IF_LONGER_SEC}s)",
    )
    parser.add_argument(
        "--chunk-hop-seconds",
        type=float,
        default=None,
        help="Overlap-add hop (default: half of chunk length)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)

    num_stems = int(ckpt.get("num_stems", 1))
    if "stem_names" in ckpt:
        stem_names = tuple(ckpt["stem_names"])
    elif num_stems == 1:
        stem_names = ("vocals",)
    else:
        stem_names = stem_names_for(num_stems)
    base = int(ckpt.get("base", 16))
    depth = int(ckpt.get("depth", 3))
    predict_phase = bool(ckpt.get("predict_phase", False))
    model = UNetSeparator(
        base=base, num_stems=num_stems, depth=depth, predict_phase=predict_phase
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    audio, sr = load_mono(
        args.input,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        max_seconds=args.max_seconds,
    )
    duration = len(audio) / sr
    print(f"Segment: {duration:.2f}s at {sr} Hz")
    mixture = torch.from_numpy(audio)

    chunk_sec = args.chunk_seconds
    if chunk_sec is None and duration > _AUTO_CHUNK_IF_LONGER_SEC:
        chunk_sec = _DEFAULT_CHUNK_SEC
    if chunk_sec is not None and chunk_sec > 0:
        chunk_samples = max(1, int(chunk_sec * sr))
        if mixture.shape[0] > chunk_samples:
            n_chunks = (mixture.shape[0] + chunk_samples - 1) // chunk_samples
            hop_samples = None
            if args.chunk_hop_seconds is not None and args.chunk_hop_seconds > 0:
                hop_samples = max(1, int(args.chunk_hop_seconds * sr))
            hop_sec = (hop_samples or chunk_samples // 2) / sr
            print(
                f"Chunked separation: {chunk_sec:.1f}s windows, "
                f"{hop_sec:.1f}s hop (~{n_chunks} passes)"
            )
            stems = separate_chunked(
                mixture,
                model,
                device,
                chunk_samples,
                hop_samples=hop_samples,
                stem_names=stem_names,
            )
        else:
            stems = separate_stems(mixture, model, device, stem_names=stem_names)
    else:
        stems = separate_stems(mixture, model, device, stem_names=stem_names)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem_name = args.input.stem
    for name, wav in stems.items():
        path = args.out_dir / f"{stem_name}_{name}.wav"
        sf.write(str(path), wav.numpy(), sr)
        print(f"Saved {path}")

    if num_stems == 1:
        backing = mixture.cpu() - stems["vocals"]
        back_path = args.out_dir / f"{stem_name}_backing.wav"
        sf.write(str(back_path), backing.numpy(), sr)
        print(f"Saved {back_path}")


if __name__ == "__main__":
    main()
