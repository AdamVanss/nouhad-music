# Closing the gap to “pro” separation

## Three levels (be honest in your presentation)

| Level | What it is | How clean |
|-------|------------|-----------|
| **Student v1** | Small U-Net, L1, `best_4stem_colab.pt` | Learns the idea; audible bleed |
| **Student v2 (best *your* model)** | `--base 32 --depth 4 --loss pro`, 50+ epochs on full MUSDB | Much better on MUSDB-style music; still mixture phase |
| **Industry** | Pretrained **Demucs** (`scripts/separate_demucs.py`) | What people mean by “pro” on arbitrary songs |

You **cannot** fully close the gap to Demucs with a mask-on-magnitude U-Net alone — pros use **time-domain** or **hybrid** models trained for **weeks on GPUs**. Your story: *we built v2 toward research metrics; Demucs shows the commercial ceiling.*

## Train the strongest custom model (Colab GPU)

```bash
python -m src.train --num-stems 4 --base 32 --depth 4 --loss pro \
  --epochs 60 --batch-size 4 --segment-seconds 8 --lr 8e-4 \
  --checkpoint-out best_4stem_pro.pt
```

`--loss pro` = waveform + spectrogram + **multi-resolution** STFT + **SI-SDR** (metrics used in papers).

Sync updated `src/` to Drive before running.

## What is still NOT fixed (even with `pro`)

- **Phase** comes from the **mixture** at inference → glassy / bleed on hard songs.
- **Capacity** is still a small U-Net, not Demucs/HTDemucs.

Next research steps (future project): predict **complex STFT**, or train a **waveform U-Net** (Demucs architecture).

## True pro output (same song, compare in demo)

```powershell
pip install demucs
python scripts/separate_demucs.py --input outputs\step02_mixture_clip.wav
```

Play `outputs/separated_demucs/step02_mixture_clip/` next to your `best_4stem_pro.pt` stems.
