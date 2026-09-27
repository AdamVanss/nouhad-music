# Step-by-step guide (start here)

You already know **Python**, **NumPy**, and **supervised ML**. This project is the same idea:

| Supervised ML you know | Stem separation |
|------------------------|-----------------|
| Features `X` | Mixture **spectrogram** (image of the mixed song) |
| Labels `y` | True **vocal stem** (or **mask**: % vocal per pixel) |
| `train_test_split` | Hold out some **MUSDB18 songs** entirely |
| `model.fit` | PyTorch **training loop** (many epochs) |
| `model.predict` | Forward pass → **masks** → WAV files |

Work through **one step at a time**. Do not skip to training until Steps 1–4 make sense.

---

## Step 0 — Environment (30 min)

1. Open a terminal in this folder: `c:\Users\lenovo\Documents\Music_model`
2. Create a virtual environment (on Windows, if `python` fails, use `py -3`):
   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies (PyTorch is large — first install can take 10–15 minutes):
   ```powershell
   pip install -r requirements.txt
   ```
4. Check GPU (optional):
   ```powershell
   python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
   ```
5. Run every script from the **project root** (`Music_model`), not from inside `scripts/`:
   ```powershell
   python scripts/step01_spectrogram_demo.py
   ```

**You’re done when:** no import errors for `torch`, `torchaudio`, `matplotlib`.

---

## Step 1 — Sound → spectrogram (no dataset yet)

**Goal:** See STFT with your own eyes (connects to the diagrams we discussed).

```powershell
python scripts/step01_spectrogram_demo.py
```

**Look at:** `outputs/step01_mixture.wav` and `outputs/step01_spectrogram.png`

**Understand:**
- Horizontal bands = sustained tones (we fake two sine waves + noise).
- STFT is **not** learned; it’s fixed math in `src/audio/stft.py`.

---

## Step 2 — What is a “stem”?

**Goal:** Hear mixture vs isolated parts.

- If you **don’t** have MUSDB yet: re-read Step 1 outputs (one file only).
- If you **have** MUSDB: follow [docs/MUSDB18_SETUP.md](docs/MUSDB18_SETUP.md), set `MUSDB_PATH`, then:

```powershell
python scripts/step02_listen_stems.py
```

**Understand:** `mixture = vocals + drums + bass + other` (summed waveforms).

---

## Step 3 — Oracle mask (supervised label intuition)

**Goal:** See the **upper bound** before any neural net — “if the mask were perfect, separation would sound like this.”

Requires MUSDB18:

```powershell
python scripts/step03_oracle_mask_demo.py
```

**Look at:** `outputs/step03_oracle_*.png` and listen to `outputs/step03_oracle_vocals.wav`

**Understand:** Mask = per time–frequency **percentage** of vocal energy (here from ground truth, not AI).

---

## Step 4 — PyTorch dataset (your `X` and `y`)

**Goal:** Same as `pd.read_csv` + `train_test_split`, but for audio chunks.

Read and run (after MUSDB path is set):

```powershell
python scripts/step04_dataloader_smoke_test.py
```

**Code to study:** `src/data/musdb_dataset.py` — each batch is:
- `mixture` tensor
- `vocals` tensor (target waveform for early training)

---

## Step 5 — Tiny network, one batch (sanity check)

```powershell
python scripts/step05_overfit_one_batch.py
```

**Understand:** If the model **cannot** overfit one batch, something is wrong (shapes, loss, learning rate). This is like fitting 32 rows in sklearn before using the full dataset.

---

## Step 6 — Train for real (Google Colab)

1. Upload this repo to Drive or GitHub.
2. Open `notebooks/02_train_colab.ipynb` (created in later phase).
3. GPU runtime, save `checkpoints/best.pt` to Drive.

**Metrics:** validation **SI-SDR** (higher = better). Plot loss like any ML project.

---

## Step 7 — Separate a full song

```powershell
python -m src.separate --checkpoint checkpoints/best.pt --input path\to\song.wav --out_dir outputs/separated
```

(Implemented in a later step — uses **overlap-add** because songs are longer than training crops.)

---

## Mini tools (while waiting for full MUSDB)

| Tool | Command |
|------|---------|
| Glossary | Read [docs/GLOSSARY.md](docs/GLOSSARY.md) |
| Oracle **drums** mask | `python scripts/step03b_oracle_drums_demo.py` |
| **Train** (saves checkpoint) | `python -m src.train --epochs 8` |
| **Separate** vocals | `python -m src.separate --checkpoint checkpoints/best.pt --input path\to\mix.wav` |
| **Oracle 4 stems** (no AI) | `python scripts/step03c_oracle_all_stems.py` |
| **Train 4 stems** | `python -m src.train --num-stems 4 --epochs 8` → `checkpoints/best_4stem.pt` |
| **Separate 4 stems** | `python -m src.separate --checkpoint checkpoints/best_4stem.pt --input mix.wav` |
| **Train on Colab GPU** | Open [notebooks/02_train_colab.ipynb](notebooks/02_train_colab.ipynb) → Runtime → GPU |
| **Fine-tune** (same small net) | `python -m src.train --num-stems 4 --resume checkpoints/best_4stem_colab.pt --epochs 20 --lr 3e-4` |
| **Train best custom model** | See [docs/PRO_QUALITY.md](docs/PRO_QUALITY.md) — `--base 32 --depth 4 --loss pro` |
| **Pro stems (pretrained Demucs)** | `pip install demucs` then `python scripts/separate_demucs.py --input your.wav` |

Training writes `checkpoints/best.pt`. Separation defaults to **30 s** max on CPU (`--max-seconds`).

---

## Step 8 — Gradio demo

```powershell
pip install gradio
python app/app.py
```

---

## Suggested pace

| Week | Steps |
|------|--------|
| 1 | 0–2 |
| 2 | 3–4 |
| 3 | 5 |
| 4–6 | 6 (Colab training) |
| 7 | 7–8 |

---

## When you’re stuck

1. Check tensor **shapes** (print them).
2. Listen to audio at every stage (bad data & bad phase sound obvious).
3. Compare to **oracle mask** (Step 3) — if the net is worse than oracle, that’s expected; if it’s worse than the **mixture**, debug training.

Ask in club chat with: step number, command run, error message or screenshot of spectrogram.
