# Model upgrade: clearer stems + more stems

## What you have now

| Checkpoint | Stems | Best for |
|------------|-------|----------|
| `best_4stem_pro.pt` | vocals, drums, bass, other | MUSDB-like rock/pop, curated demos |
| Demucs (`scripts/separate_demucs.py`) | 4 stems | Any student MP3 at showcase |

## Realistic goals

| Goal | Approach |
|------|----------|
| **Clearer on random songs** | Fine-tune on **MUSDB + MoisesDB** (`--dataset mixed`), longer segments, `--loss pro` |
| **Less junk in `other`** | Train **6 stems** on MoisesDB (guitar + piano split out) |
| **Maximum clarity** | Still capped by **mixture phase** + small U-Net; Demucs remains ceiling |

MoisesDB: [music.ai/research](https://music.ai/research/) — **CC BY-NC-SA 4.0** (non-commercial research).

## 1. Download MoisesDB

1. Request/download from the research page.
2. Unzip so you have `provider_uuid/track_uuid/data.json` + audio.
3. Set path:

```powershell
$env:MOISESDB_PATH = "D:\datasets\moisesdb\moisesdb_v0.1"
```

4. Install loader (needs Git):

```powershell
pip install git+https://github.com/moises-ai/moises-db.git
```

Verify hashes in the Moises README (MD5 / SHA256).

## 2. Recommended training path (Kaggle GPU)

### Phase A — clearer **4-stem** (keep showcase stems)

Fine-tune your pro model on **mixed** data (better generalization):

```bash
python -m src.train \
  --dataset mixed \
  --num-stems 4 \
  --resume checkpoints/best_4stem_pro.pt \
  --base 32 --depth 4 --loss pro \
  --segment-seconds 8 --epochs 20 --batch-size 4 --lr 2e-4 \
  --checkpoint-out best_4stem_pro_moises.pt
```

Requires **both** `MUSDB_PATH` and `MOISESDB_PATH` on the machine.

### Phase B — **6-stem** (guitar + piano + smaller `other`)

```bash
python -m src.train \
  --dataset moises \
  --num-stems 6 \
  --expand-stems-from checkpoints/best_4stem_pro.pt \
  --base 32 --depth 4 --loss pro \
  --segment-seconds 8 --epochs 30 --batch-size 2 --lr 3e-4 \
  --checkpoint-out best_6stem_moises.pt
```

Stems: `vocals`, `bass`, `guitar`, `piano`, `drums`, `other`.

Separate:

```bash
python -m src.separate --checkpoint checkpoints/best_6stem_moises.pt --input song.mp3 --out-dir outputs/separated/6stem
```

## 3. Club showcase strategy

1. **Hero reel** — `best_4stem_pro.pt` on MUSDB + Tame/Floyd crops.
2. **Open mic** — Demucs for arbitrary uploads.
3. **Stretch goal slide** — 6-stem Moises model + “future: waveform / phase estimation.”

## 4. What will NOT be fixed by Moises alone

- Quiet / dull drums on some masters → partly **phase** + **loss**; try `--loss pro` and 8 s segments.
- Perfect Stromae-level pop → needs **much** more data or a **Demucs-class** architecture.
- Commercial use of Moises-trained weights → check **NC** license.

## 5. Local smoke test (after Moises is installed)

```powershell
$env:MUSDB_PATH = "...\MUSDB18-7"
$env:MOISESDB_PATH = "...\moisesdb_v0.1"
python -m src.train --dataset moises --num-stems 6 --epochs 1 --batch-size 1 --base 16 --depth 3
```
