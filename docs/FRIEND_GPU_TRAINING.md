# Training handoff (for your friend with a GPU)

This doc is for someone who **did not** build the project but will train the model. Owner: fine-tune for **clearer 4-stem** and/or **6-stem** separation.

**Public repo:** https://github.com/chrollo340-art/Music_model

---

## What the owner should send you

| Item | Notes |
|------|--------|
| **This repo** | `git clone` or a zip of `Music_model` |
| **`checkpoints/best_4stem_pro.pt`** | ~7.8 MB — **required** starting weights (Kaggle-trained pro model). If missing, ask owner to upload; do not train 4-stem pro from scratch unless you have days of GPU time |
| **Goals (pick one or both)** | **A)** clearer 4-stem on more genres · **B)** 6-stem (guitar + piano split out) |

Owner keeps: MUSDB/Moises zips on their machine **or** you download datasets yourself (links below).

---

## Machine setup (Linux or Windows + NVIDIA GPU)

```bash
cd Music_model
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows: .\.venv\Scripts\Activate.ps1

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install git+https://github.com/moises-ai/moises-db.git
```

Check GPU:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Put **`best_4stem_pro.pt`** in `checkpoints/` (create folder if needed).

---

## Datasets (both needed for Phase A; Moises only for Phase B)

### MUSDB18 (4-stem labels)

- Download **musdb18** from Zenodo (~4.7 GB): https://zenodo.org/records/1117372  
- Unzip so you have `train/` and `test/` folders.

```bash
export MUSDB_PATH=/path/to/musdb18
```

Windows PowerShell:

```powershell
$env:MUSDB_PATH = "D:\datasets\musdb18"
```

### MoisesDB (clearer generalization + 5/6 stems)

- Request/download: https://music.ai/research/  
- License: **CC BY-NC-SA 4.0** (non-commercial research only).  
- Unzip; folder contains provider UUIDs → track UUIDs → `data.json` + audio.

```bash
export MOISESDB_PATH=/path/to/moisesdb_v0.1
```

Verify:

```bash
python scripts/verify_musdb_path.py
# Moises: quick test
python -c "from moisesdb.dataset import MoisesDB; import os; print(len(MoisesDB(os.environ['MOISESDB_PATH'])))"
```

---

## Training commands (use GPU; adjust `--batch-size` if OOM)

Architecture must stay **`--base 32 --depth 4`** to match `best_4stem_pro.pt`.

### Phase A — clearer **4-stem** (recommended first)

Fine-tune on **MUSDB + Moises** mixed batches. Sends back: `checkpoints/best_4stem_pro_moises.pt`

```bash
python -m src.train \
  --dataset mixed \
  --num-stems 4 \
  --resume checkpoints/best_4stem_pro.pt \
  --base 32 --depth 4 --loss pro \
  --segment-seconds 8 --epochs 20 --batch-size 4 --lr 2e-4 \
  --checkpoint-out best_4stem_pro_moises.pt
```

If **out of memory**: try `--batch-size 2` or `1`.

### Phase B — **6-stem** (optional, after or instead of A)

Stems: `vocals`, `bass`, `guitar`, `piano`, `drums`, `other`.  
Sends back: `checkpoints/best_6stem_moises.pt`

```bash
python -m src.train \
  --dataset moises \
  --num-stems 6 \
  --expand-stems-from checkpoints/best_4stem_pro.pt \
  --base 32 --depth 4 --loss pro \
  --segment-seconds 8 --epochs 30 --batch-size 2 --lr 3e-4 \
  --checkpoint-out best_6stem_moises.pt
```

---

## What to send back to the owner

1. **`checkpoints/best_4stem_pro_moises.pt`** (Phase A) and/or **`best_6stem_moises.pt`** (Phase B)  
2. Last few lines of training log (final train/val loss)  
3. Note GPU used and `--batch-size`  

**Checkpoint format:** PyTorch zip; owner renames Kaggle-style zip to `.pt` if needed.

Owner tests locally:

```bash
python -m src.separate --checkpoint checkpoints/best_4stem_pro_moises.pt --input song.mp3 --start-seconds 0 --end-seconds 40 --out-dir outputs/test
```

---

## Troubleshooting

| Error | Fix |
|--------|-----|
| `Set MUSDB_PATH` | Export env var to musdb18 root |
| `Set MOISESDB_PATH` | Export env var to Moises root |
| `num_stems mismatch` | 4-stem resume needs `--num-stems 4`; 6-stem needs `--expand-stems-from`, not `--resume` |
| `base/depth mismatch` | Always use `--base 32 --depth 4` with pro checkpoint |
| CUDA OOM | Lower `--batch-size` |
| Very slow on CPU | Use a CUDA machine |

---

## Do not commit to git

- Full MUSDB / Moises folders  
- Large `.pt` checkpoints (use Drive / zip / Hugging Face file share)

Questions: open a GitHub issue or contact repo owner.
