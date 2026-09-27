"""Production-quality 4-stem separation using pretrained Meta Demucs (htdemucs).

This is NOT your U-Net — use for demos / ceiling comparison / when you need pro audio.
Install once: pip install demucs
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate with pretrained htdemucs")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/separated_demucs"))
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Not found: {args.input}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        "htdemucs",
        "-o",
        str(args.out_dir),
        str(args.input),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # demucs writes: out_dir/htdemucs/track_name/stem.wav
    stem = args.input.stem
    src_dir = args.out_dir / "htdemucs" / stem
    if src_dir.is_dir():
        flat = args.out_dir / stem
        flat.mkdir(exist_ok=True)
        for wav in src_dir.glob("*.wav"):
            shutil.copy2(wav, flat / wav.name)
        print(f"Stems also copied to {flat}")
    print("Done.")


if __name__ == "__main__":
    main()
