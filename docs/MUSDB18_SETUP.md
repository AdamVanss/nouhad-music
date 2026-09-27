# MUSDB18 setup

MUSDB18 is the standard dataset for music source separation research (~150 songs with **vocals, drums, bass, other** stems).

## Quick start — 7-second sample (recommended first)

From the project root with venv active:

```powershell
pip install musdb
python scripts/download_musdb_sample.py
$env:MUSDB_PATH = "c:\Users\lenovo\Documents\Music_model\data\MUSDB18-7"
python scripts\step02_listen_stems.py
```

This auto-downloads short excerpts for prototyping (enough for Steps 2–5).

## Full MUSDB18 (training for real quality)

1. Read the license: [https://sigsep.github.io/datasets/musdb.html](https://sigsep.github.io/datasets/musdb.html)
2. Request access and download **musdb18.zip** (~4.7 GB) from [Zenodo](https://zenodo.org/records/1117372).
3. Unzip so you have a folder that contains `train/` and `test/`.
4. Set `MUSDB_PATH` to that folder.

Example layout:

```text
D:\datasets\musdb18\
  train\
    A Classic Education - NightOwl\
      mixture.wav
      vocals.wav
      ...
  test\
    ...
```

## Point the code at it

**PowerShell (session):**

```powershell
$env:MUSDB_PATH = "D:\datasets\musdb18"
```

**Or** create a `.env` file in the project root (do not commit it):

```text
MUSDB_PATH=D:\datasets\musdb18
```

## Verify

```powershell
python scripts/step02_listen_stems.py
```

You should hear one short clip and see paths printed for mixture vs vocals.

## Google Colab

- Zip `musdb18` once, upload to Google Drive.
- In Colab: mount Drive, set `MUSDB_PATH` to the unzip path.
- Training notebooks should save checkpoints to Drive every few epochs.
