"""Check MUSDB_PATH: count tracks and print first song name."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.musdb_dataset import get_musdb_path

try:
    import musdb
except ImportError:
    raise SystemExit("pip install musdb")

def main() -> None:
    root = get_musdb_path()
    for is_wav in (False, True):
        try:
            db = musdb.DB(root=str(root), subsets="train", is_wav=is_wav)
            print(f"OK  is_wav={is_wav}  train tracks: {len(db)}  first: {db[0].name}")
            return
        except Exception as e:
            print(f"is_wav={is_wav} failed: {e}")
    raise SystemExit("Could not open dataset — check unzip layout and MUSDB_PATH")


if __name__ == "__main__":
    main()
