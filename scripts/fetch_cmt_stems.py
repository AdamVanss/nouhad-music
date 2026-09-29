"""Download Cambridge-MT multitracks and fold them into drums/bass/other/vocals.

Educational use only. Each zip is extracted, summed into four stereo 44.1 kHz
stems under data/cmt_stems/<song>/, and the zip and raw tracks are deleted.
Songs already in MUSDB18 and songs without a vocal track are skipped.
"""

import argparse
import importlib.util
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PAGE = DATA / "_cmt_page.py"
RAW = DATA / "cmt_raw"
OUT = DATA / "cmt_stems"
SR = 44100
SOURCES = ("drums", "bass", "other", "vocals")
HOST = "mtkdata.cambridgemusictechnology.co.uk"
VOCAL_SLOT = 37
SKIP_WORDS = ("orchestra", "concerto", "symphony", "quartet", "sonata", "choir", "acappella", "a capella")
AUDIO_EXT = (".wav", ".aif", ".aiff", ".flac")

DRUM_WORDS = (
    "kick", "kik", "bd", "bassdrum", "snare", "snr", "sn", "hat", "hh", "hihat", "tom", "toms",
    "oh", "ohs", "overhead", "overheads", "room", "rooms", "drum", "drums", "kit", "cymbal",
    "cymbals", "ride", "crash", "perc", "percussion", "shaker", "tamb", "tambourine", "conga",
    "congas", "bongo", "bongos", "clap", "claps", "cajon", "cowbell", "trash", "loop", "beat",
    "808kick", "sidestick", "rim", "brush", "brushes", "timbales", "djembe", "cabasa", "triangle",
)
VOCAL_WORDS = (
    "vox", "vocal", "vocals", "voc", "vocs", "bv", "bvs", "bgv", "bgvs", "harm", "harmony",
    "harmonies", "rap", "adlib", "adlibs", "sing", "singer", "voice", "voices", "leadvox",
    "backing", "backings", "dbl", "choir", "chorus", "whisper", "talk", "spoken",
)
BASS_WORDS = ("bass", "bs", "sub", "subbass", "synthbass", "bassdi", "bassamp", "db", "upright", "808")


def _tokens(name: str) -> list[str]:
    stem = Path(name).stem
    stem = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    stem = re.sub(r"([A-Za-z])(\d)", r"\1 \2", stem)
    return [t for t in re.split(r"[^a-z0-9]+", stem.lower()) if t and not t.isdigit()]


def classify(name: str) -> str:
    tokens = _tokens(name)
    joined = "".join(tokens)
    if any(t in DRUM_WORDS for t in tokens) or "bassdrum" in joined or "kick" in joined or "snare" in joined:
        return "drums"
    if any(t in VOCAL_WORDS for t in tokens) or "vox" in joined or "vocal" in joined:
        return "vocals"
    if any(t in BASS_WORDS for t in tokens) or joined.startswith("bass"):
        return "bass"
    return "other"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def candidates(max_zip_mb: int) -> list[tuple[int, str, str]]:
    spec = importlib.util.spec_from_file_location("cmt_page", PAGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    musdb = {
        _norm(p.name.replace(".stem.mp4", ""))
        for sub in ("train", "test")
        for p in (DATA / "musdb18" / sub).glob("*.stem.mp4")
    }
    rows = []
    seen = set()
    for row in mod.page_source:
        if row.get("pt") != "Full":
            continue
        match = re.search(r'href="([^"]+)"', row.get("dl") or "")
        if not match or HOST not in match.group(1):
            continue
        url = match.group(1)
        ic = row.get("ic") or []
        if len(ic) <= VOCAL_SLOT or ic[VOCAL_SLOT] == 0:
            continue
        artist, title = row.get("a") or "", row.get("p") or ""
        if any(w in (artist + " " + title).lower() for w in SKIP_WORDS):
            continue
        at, tt = _norm(artist), _norm(title)
        if any(at and tt and at in name and tt[:8] in name for name in musdb):
            continue
        size = int(row.get("dls") or 0)
        if not 50 <= size <= max_zip_mb or url in seen:
            continue
        seen.add(url)
        rows.append((size, url, f"{artist} - {title}"))
    rows.sort()
    return rows


def _load(path: Path) -> torch.Tensor:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.T.copy())
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav


def mix_song(files: list[Path], out_dir: Path) -> dict[str, int]:
    groups: dict[str, torch.Tensor | None] = {name: None for name in SOURCES}
    counts = {name: 0 for name in SOURCES}
    for path in files:
        try:
            wav = _load(path)
        except Exception as exc:  # noqa: BLE001
            print(f"    skip {path.name}: {exc}")
            continue
        name = classify(path.name)
        current = groups[name]
        if current is None:
            groups[name] = wav
        else:
            length = max(current.shape[-1], wav.shape[-1])
            current = torch.nn.functional.pad(current, (0, length - current.shape[-1]))
            groups[name] = current + torch.nn.functional.pad(wav, (0, length - wav.shape[-1]))
        counts[name] += 1
    length = max(g.shape[-1] for g in groups.values() if g is not None)
    stems = torch.stack(
        [
            torch.nn.functional.pad(g, (0, length - g.shape[-1])) if g is not None else torch.zeros(2, length)
            for g in (groups[name] for name in SOURCES)
        ]
    )
    peak = max(stems.sum(0).abs().max().item(), stems.abs().max().item())
    if peak > 0.99:
        stems = stems * (0.99 / peak)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(SOURCES):
        sf.write(str(out_dir / f"{name}.wav"), stems[i].T.numpy(), SR, subtype="PCM_16")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-songs", type=int, default=70)
    parser.add_argument("--max-zip-mb", type=int, default=700)
    parser.add_argument("--min-free-gb", type=float, default=12.0)
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = candidates(args.max_zip_mb)
    print(f"{len(rows)} candidate songs with vocals")
    done = sum(1 for p in OUT.iterdir() if (p / "vocals.wav").is_file())
    for size, url, label in rows:
        if done >= args.max_songs:
            break
        song = Path(url).stem.replace("_Full", "")
        out_dir = OUT / song
        if (out_dir / "vocals.wav").is_file() or (RAW / f"{song}.rejected").is_file():
            continue
        free = shutil.disk_usage(DATA).free / 1024**3
        if free - size / 1024 * 2.5 < args.min_free_gb:
            print(f"stopping: {free:.1f} GB free")
            break
        zip_path = RAW / Path(url).name
        print(f"[{done + 1}] {label} ({size} MB)", flush=True)
        result = subprocess.run(
            ["curl.exe", "-L", "--fail", "--retry", "3", "-s", "-o", str(zip_path), url],
            check=False,
        )
        if result.returncode != 0:
            print(f"    download failed ({result.returncode})")
            zip_path.unlink(missing_ok=True)
            continue
        extract = RAW / song
        try:
            with zipfile.ZipFile(zip_path) as zf:
                members = [
                    m for m in zf.namelist()
                    if m.lower().endswith(AUDIO_EXT) and "__macosx" not in m.lower()
                    and not Path(m).name.startswith("._")
                ]
                names = [Path(m).name for m in members]
                if not any(classify(n) == "vocals" for n in names):
                    print(f"    no vocal track, rejected: {names[:6]}")
                    (RAW / f"{song}.rejected").touch()
                    continue
                for member in members:
                    zf.extract(member, extract)
        except zipfile.BadZipFile:
            print("    bad zip")
            continue
        finally:
            zip_path.unlink(missing_ok=True)
        files = sorted(p for p in extract.rglob("*") if p.suffix.lower() in AUDIO_EXT)
        try:
            counts = mix_song(files, out_dir)
            print(f"    {counts}", flush=True)
            done += 1
        finally:
            shutil.rmtree(extract, ignore_errors=True)
    print(f"songs in {OUT}: {sum(1 for p in OUT.iterdir() if (p / 'vocals.wav').is_file())}")


if __name__ == "__main__":
    main()
