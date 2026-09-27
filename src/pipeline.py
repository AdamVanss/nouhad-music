"""Shared STFT → mask → waveform helpers for train and separate."""

import torch
import torch.nn.functional as F

from src.audio.stft import StftConfig, istft_from_magnitude, log_mag, magnitude_spectrogram

CFG = StftConfig()
STEM_NAMES = ("vocals", "drums", "bass", "other")
# Multi-resolution STFT for --loss pro (closer to research / Demucs-style objectives)
MR_CFGS = (
    StftConfig(n_fft=1024, hop_length=256, win_length=1024),
    StftConfig(n_fft=2048, hop_length=512, win_length=2048),
    StftConfig(n_fft=4096, hop_length=1024, win_length=4096),
)


def waveform_batch_to_logspec(wave: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """wave (B, T) -> logspec (B, 1, F, Tf), list of phase tensors."""
    specs: list[torch.Tensor] = []
    phases: list[torch.Tensor] = []
    for i in range(wave.shape[0]):
        mag, phase = magnitude_spectrogram(wave[i], CFG)
        specs.append(log_mag(mag))
        phases.append(phase)
    logspec = torch.stack(specs).unsqueeze(1)
    return logspec, phases


def masked_waveform(
    mixture: torch.Tensor,
    mask: torch.Tensor,
    length: int,
) -> torch.Tensor:
    """mixture (T,), mask (F, Tf) -> estimated stem waveform."""
    mag, phase = magnitude_spectrogram(mixture, CFG)
    est_mag = mask * mag
    return istft_from_magnitude(est_mag, phase, CFG, length=length)


def vocal_l1_loss(
    mixture: torch.Tensor,
    target_vocals: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """mask: (B, 1, F, Tf) from U-Net."""
    loss = torch.tensor(0.0, device=mixture.device)
    for i in range(mixture.shape[0]):
        est = masked_waveform(mixture[i], mask[i, 0], mixture.shape[1])
        loss = loss + F.l1_loss(est, target_vocals[i])
    return loss / mixture.shape[0]


def _four_stem_estimates(
    mixture: torch.Tensor,
    masks: torch.Tensor,
) -> dict[str, list[torch.Tensor]]:
    """Per-batch-item estimated waveforms per stem."""
    out: dict[str, list[torch.Tensor]] = {n: [] for n in STEM_NAMES}
    for b in range(mixture.shape[0]):
        for i, name in enumerate(STEM_NAMES):
            out[name].append(masked_waveform(mixture[b], masks[b, i], mixture.shape[1]))
    return out


def four_stem_l1_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
) -> torch.Tensor:
    """masks: (B, 4, F, Tf) softmax over stem dim."""
    loss = torch.tensor(0.0, device=mixture.device)
    ests = _four_stem_estimates(mixture, masks)
    for name in STEM_NAMES:
        for b, est in enumerate(ests[name]):
            loss = loss + F.l1_loss(est, batch_stems[name][b])
    return loss / (mixture.shape[0] * len(STEM_NAMES))


def four_stem_spec_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
) -> torch.Tensor:
    """L1 on log-magnitude spectrograms of estimates vs targets."""
    loss = torch.tensor(0.0, device=mixture.device)
    ests = _four_stem_estimates(mixture, masks)
    for name in STEM_NAMES:
        for b, est in enumerate(ests[name]):
            em, _ = magnitude_spectrogram(est, CFG)
            tm, _ = magnitude_spectrogram(batch_stems[name][b], CFG)
            loss = loss + F.l1_loss(log_mag(em), log_mag(tm))
    return loss / (mixture.shape[0] * len(STEM_NAMES))


def four_stem_hybrid_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
    wave_weight: float = 0.5,
) -> torch.Tensor:
    w = four_stem_l1_loss(mixture, batch_stems, masks)
    s = four_stem_spec_loss(mixture, batch_stems, masks)
    return wave_weight * w + (1.0 - wave_weight) * s


def _neg_si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Minimize this to maximize SI-SDR (scale-invariant; used in pro separators)."""
    est = est - est.mean()
    ref = ref - ref.mean()
    ref_energy = (ref * ref).sum() + eps
    proj = (est * ref).sum() * ref / ref_energy
    noise = est - proj
    si_sdr = 10 * torch.log10((proj.pow(2).sum() + eps) / (noise.pow(2).sum() + eps))
    return -si_sdr


def four_stem_si_sdr_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=mixture.device)
    ests = _four_stem_estimates(mixture, masks)
    for name in STEM_NAMES:
        for b, est in enumerate(ests[name]):
            loss = loss + _neg_si_sdr(est, batch_stems[name][b])
    return loss / (mixture.shape[0] * len(STEM_NAMES))


def four_stem_multires_spec_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=mixture.device)
    ests = _four_stem_estimates(mixture, masks)
    for cfg in MR_CFGS:
        for name in STEM_NAMES:
            for b, est in enumerate(ests[name]):
                em, _ = magnitude_spectrogram(est, cfg)
                tm, _ = magnitude_spectrogram(batch_stems[name][b], cfg)
                loss = loss + F.l1_loss(log_mag(em), log_mag(tm))
    return loss / (mixture.shape[0] * len(STEM_NAMES) * len(MR_CFGS))


def four_stem_pro_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
) -> torch.Tensor:
    """
    Best training objective in this repo: hybrid + multi-res spec + SI-SDR.
    Still uses mixture phase at inference — true 'pro' also needs waveform models or Demucs.
    """
    hybrid = four_stem_hybrid_loss(mixture, batch_stems, masks)
    multires = four_stem_multires_spec_loss(mixture, batch_stems, masks)
    sisdr = four_stem_si_sdr_loss(mixture, batch_stems, masks)
    return hybrid + multires + 0.1 * sisdr


def separate_stems(
    mixture: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Return dict of stem waveforms (vocals-only or all four)."""
    mixture = mixture.to(device)
    logspec, _ = waveform_batch_to_logspec(mixture.unsqueeze(0))
    with torch.no_grad():
        masks = model(logspec.to(device))

    out: dict[str, torch.Tensor] = {}
    if masks.shape[1] == 1:
        out["vocals"] = masked_waveform(mixture, masks[0, 0], mixture.shape[0]).cpu()
        return out

    for i, name in enumerate(STEM_NAMES):
        out[name] = masked_waveform(mixture, masks[0, i], mixture.shape[0]).cpu()
    return out


def separate_mono_waveform(
    mixture: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Vocals only (1-stem checkpoints)."""
    return separate_stems(mixture, model, device)["vocals"]
