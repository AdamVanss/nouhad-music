# Glossary (simple)

## Sound

| Term | Meaning |
|------|---------|
| **Waveform** | Audio as numbers over time (a `.wav` file). |
| **Sample rate** | Samples per second (we use **44100**). |
| **Stem** | One isolated part: vocals, drums, bass, or other. |
| **Mixture** | Full song = stems added together. |

## Spectrogram

| Term | Meaning |
|------|---------|
| **FFT** | Frequencies in one short slice of audio. |
| **STFT** | Many FFTs along the song → 2D image. |
| **Spectrogram** | Time × frequency picture; brightness = loudness. |
| **`n_fft`** | Window length in samples (2048). |
| **`hop_length`** | Slide step between windows (512). |
| **`mag`** | Magnitude grid (loudness per cell). |
| **Phase** | Timing detail; we often reuse the **mix** phase. |
| **iSTFT** | Spectrogram → waveform again. |

## Separation

| Term | Meaning |
|------|---------|
| **Mask** | Grid of 0–1: how much is vocals (or drums, etc.). |
| **Oracle mask** | Mask from **true** stems (teaching only). |
| **Multiply** | `est = mix × mask` per cell. |

## ML

| Term | Meaning |
|------|---------|
| **`torch`** | PyTorch — tensors + neural nets. |
| **U-Net** | Network: mix image → mask image. |
| **Loss** | How wrong the output is (lower = better). |
| **Checkpoint** | Saved weights (`.pt`). |
| **Batch** | Several clips trained together. |

## Project scripts

| Script | Purpose |
|--------|---------|
| `step01` | Fake mix → spectrogram plot |
| `step02` | Listen mix vs vocals |
| `step03` | Oracle **vocal** mask |
| `step03b` | Oracle **drums** mask |
| `step04` | DataLoader shapes |
| `step05` | Overfit one batch |
| `python -m src.train` | Mini training → checkpoint |
| `python -m src.separate` | Checkpoint → vocal WAV |
