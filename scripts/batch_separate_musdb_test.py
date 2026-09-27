"""Export mixture from several MUSDB test tracks and separate with a checkpoint."""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import musdb
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
STEM_NAMES = ("vocals", "drums", "bass", "other")


def safe_name(title: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", title)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best_4stem_pro.pt")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 5, 15, 20, 30])
    parser.add_argument("--subset", choices=("test", "train"), default="test")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from src.data.musdb_dataset import get_musdb_path

    root = get_musdb_path()
    db = musdb.DB(root=str(root), subsets=args.subset, is_wav=False)

    clips = ROOT / "outputs" / "musdb_clips"
    out_root = ROOT / "outputs" / "separated" / "pro_musdb_more"
    py = sys.executable

    for idx in args.indices:
        if idx >= len(db):
            print(f"Skip index {idx} (only {len(db)} tracks)")
            continue
        track = db[idx]
        title = track.name
        print(f"=== [{idx}] {title}")

        def mono(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
            return arr.astype(np.float32)

        mix = sum(mono(track.targets[n].audio) for n in STEM_NAMES)
        sr = track.rate
        clip_dir = clips / safe_name(title)
        clip_dir.mkdir(parents=True, exist_ok=True)
        mix_path = clip_dir / "mixture.wav"
        sf.write(mix_path, mix, sr)

        out_dir = out_root / safe_name(title)
        subprocess.run(
            [
                py,
                "-m",
                "src.separate",
                "--checkpoint",
                str(args.checkpoint),
                "--input",
                str(mix_path),
                "--out-dir",
                str(out_dir),
            ],
            check=True,
            cwd=str(ROOT),
        )

    print(f"Done. Separated stems under:\n  {out_root}")


if __name__ == "__main__":
    main()
