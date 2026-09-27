# Full MUSDB18 download (manual — ~4.7 GB)

The **7s sample** is automatic. The **full** corpus is **not** a public direct link: Zenodo requires **academic-use access** (usually approved within ~1 day).

## Steps

1. Read the license: https://sigsep.github.io/datasets/musdb.html  
2. Open: https://zenodo.org/records/1117372  
3. Click **Request access** / follow Zenodo instructions (use your **school email** if you have one).  
4. When approved, download **`musdb18.zip`** (~4.7 GB).  
5. Unzip to a folder with `train/` and `test/` inside, e.g.  
   `D:\datasets\musdb18\train\...`  
6. Point the project at it:

   ```powershell
   $env:MUSDB_PATH = "D:\datasets\musdb18"
   python scripts\verify_musdb_path.py
   ```

## Optional: MUSDB18-HQ (WAV, ~22 GB)

Higher quality, same songs: https://zenodo.org/records/3338373 — unzip and use:

```python
musdb.DB(root="path", is_wav=True)
```

Our `MusdbStemDataset` uses `is_wav=False` for standard **musdb18.zip**; switch to `is_wav=True` only for HQ layout.

## Club / Colab tip

Download once on a PC with good Wi‑Fi, upload **one zip** to Google Drive, unzip in Colab for training.
