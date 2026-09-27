"""Run a trained checkpoint: vocals-only (1-stem) or all four MUSDB stems (4-stem)."""

import argparse
from pathlib import Path

import soundfile as sf
import torch

from src.audio.load import load_mono
from src.models.unet_separator import UNetSeparator
from src.pipeline import separate_stems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/separated"))
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--start-seconds", type=float, default=None, help="Crop start (e.g. 28)")
    parser.add_argument("--end-seconds", type=float, default=None, help="Crop end (e.g. 50)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)

    num_stems = int(ckpt.get("num_stems", 1))
    base = int(ckpt.get("base", 16))
    depth = int(ckpt.get("depth", 3))
    model = UNetSeparator(base=base, num_stems=num_stems, depth=depth).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    audio, sr = load_mono(
        args.input,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        max_seconds=args.max_seconds,
    )
    print(f"Segment: {len(audio) / sr:.2f}s at {sr} Hz")
    mixture = torch.from_numpy(audio)
    stems = separate_stems(mixture, model, device)

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
