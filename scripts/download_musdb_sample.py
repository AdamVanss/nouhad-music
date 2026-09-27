"""Download MUSDB18 7-second sample (~quick start). Full corpus needs Zenodo access."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "MUSDB18-7"


def main() -> None:
    try:
        import musdb
    except ImportError:
        raise SystemExit("Run: pip install musdb")

    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading 7s MUSDB18 sample into:\n  {DATA_DIR}\n")
    print("(Academic full dataset: request access at https://zenodo.org/records/1117372)\n")

    db = musdb.DB(root=str(DATA_DIR), download=True)
    print(f"Tracks available: {len(db)}")
    print(f"\nSet MUSDB_PATH for this session:")
    print(f'  $env:MUSDB_PATH = "{DATA_DIR}"')
    print("\nThen run:")
    print("  python scripts/step02_listen_stems.py")

    env_path = ROOT / ".env.example"
    env_path.write_text(f"MUSDB_PATH={DATA_DIR.as_posix()}\n", encoding="utf-8")
    print(f"\nWrote {env_path} — copy to .env if you use one.")


if __name__ == "__main__":
    main()
