"""Short-time Fourier transform helpers (mixture -> spectrogram -> waveform)."""

from dataclasses import dataclass

import torch
import torchaudio


@dataclass(frozen=True)
class StftConfig:
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048

    def window_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.hann_window(self.win_length, device=device)


def magnitude_spectrogram(
    waveform: torch.Tensor,
    cfg: StftConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    waveform: (channels, time) or (time,) — mono recommended for v1
    Returns:
        mag: (freq, frames) real, non-negative
        phase: (freq, frames) radians
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError(f"Expected (C, T) waveform, got shape {tuple(waveform.shape)}")

    window = cfg.window_tensor(waveform.device)
    spec = torch.stft(
        waveform,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        return_complex=True,
        center=True,
    )
    spec = spec.mean(dim=0)
    mag = spec.abs()
    phase = spec.angle()
    return mag, phase


def istft_from_magnitude(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    cfg: StftConfig,
    length: int | None = None,
) -> torch.Tensor:
    """Rebuild mono waveform from magnitude + phase."""
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)
    spec = torch.complex(real, imag)
    window = cfg.window_tensor(spec.device)
    wav = torch.istft(
        spec.unsqueeze(0),
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=True,
        length=length,
    )
    return wav.squeeze(0)


def log_mag(mag: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(mag + eps)


def oracle_vocal_mask(
    mix_mag: torch.Tensor,
    vocal_mag: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Ideal ratio mask (IRM) for teaching — uses ground-truth vocal magnitude."""
    return vocal_mag / (mix_mag + eps)
