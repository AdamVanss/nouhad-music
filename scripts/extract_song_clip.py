"""Cut a time range from a song into outputs/custom_clips/<name>/mixture.wav."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf

from src.audio.load import load_mono

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--start", type=float, required=True, help="Start time in seconds (e.g. 28)")
    p.add_argument("--end", type=float, required=True, help="End time in seconds (e.g. 50)")
    p.add_argument("--name", type=str, default=None, help="Folder name under outputs/custom_clips/")
    args = p.parse_args()

    name = args.name or args.input.stem.replace(" ", "_")[:40]
    out_dir = ROOT / "outputs" / "custom_clips" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = load_mono(args.input, start_seconds=args.start, end_seconds=args.end)
    mix_path = out_dir / "mixture.wav"
    sf.write(str(mix_path), audio, sr)
    dur = len(audio) / sr
    print(f"Wrote {mix_path} ({dur:.2f}s)")
    print(
        "\nTo TRAIN on this clip you also need true stems in the same folder:\n"
        "  vocals.wav, drums.wav, bass.wav, other.wav\n"
        "An MP3 alone cannot provide those — use MUSDB, a DAW export, or wait for full dataset.\n"
        "Then:\n"
        f"  python -m src.train --num-stems 4 --wav-folder {out_dir} --epochs 50 "
        f"--checkpoint-out tame_clip.pt"
    )


if __name__ == "__main__":
    main()
