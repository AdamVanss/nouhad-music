"""Shared STFT → mask → waveform helpers for train and separate."""

import torch
import torch.nn.functional as F

from src.audio.stft import (
    StftConfig,
    complex_spectrogram,
    istft_from_magnitude,
    log_mag,
    magnitude_spectrogram,
)
from src.data.stems import STEM_NAMES_4, stem_names_for

CFG = StftConfig()
STEM_NAMES = STEM_NAMES_4  # backward compatible
# Multi-resolution STFT for --loss pro (closer to research / Demucs-style objectives)
MR_CFGS = (
    StftConfig(n_fft=1024, hop_length=256, win_length=1024),
    StftConfig(n_fft=2048, hop_length=512, win_length=2048),
    StftConfig(n_fft=4096, hop_length=1024, win_length=4096),
)


def separator_features(
    wave: torch.Tensor,
    predict_phase: bool,
    cfg: StftConfig = CFG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """wave (B, T) -> features (B, C, F, frames), complex mixture spectrogram."""
    spec = complex_spectrogram(wave, cfg)
    logspec = log_mag(spec.abs()).unsqueeze(1)
    if predict_phase:
        phase = spec.angle()
        logspec = torch.cat(
            [logspec, torch.cos(phase).unsqueeze(1), torch.sin(phase).unsqueeze(1)],
            dim=1,
        )
    return logspec, spec


def model_outputs(
    model: torch.nn.Module, feats: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    out = model(feats)
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, None


def estimate_stem_waveforms(
    mixture: torch.Tensor,
    masks: torch.Tensor,
    phase_delta: torch.Tensor | None = None,
    cfg: StftConfig = CFG,
) -> torch.Tensor:
    """mixture (B, T), masks (B, S, F, frames) -> waveforms (B, S, T)."""
    spec = complex_spectrogram(mixture, cfg)
    mag = spec.abs()
    phase = spec.angle()
    if masks.shape[-2:] != mag.shape[-2:]:
        masks = F.interpolate(masks, size=mag.shape[-2:], mode="bilinear", align_corners=False)
        if phase_delta is not None:
            phase_delta = F.interpolate(
                phase_delta, size=mag.shape[-2:], mode="bilinear", align_corners=False
            )
    est_phase = phase.unsqueeze(1).expand_as(masks)
    if phase_delta is not None:
        est_phase = est_phase + phase_delta
    est_spec = torch.polar(masks * mag.unsqueeze(1), est_phase)
    b, s, freq, frames = est_spec.shape
    window = cfg.window_tensor(est_spec.device)
    wav = torch.istft(
        est_spec.reshape(b * s, freq, frames),
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=True,
        length=mixture.shape[-1],
    )
    return wav.reshape(b, s, -1)


def _stack_targets(
    batch_stems: dict[str, torch.Tensor], stem_names: tuple[str, ...]
) -> torch.Tensor:
    return torch.stack([batch_stems[name] for name in stem_names], dim=1)


def _neg_si_sdr_batch(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """est/ref (N, T). Mean negative SI-SDR, matching the per-clip objective."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    ref_energy = ref.pow(2).sum(dim=-1, keepdim=True) + eps
    proj = (est * ref).sum(dim=-1, keepdim=True) * ref / ref_energy
    noise = est - proj
    si_sdr = 10 * torch.log10(
        (proj.pow(2).sum(dim=-1) + eps) / (noise.pow(2).sum(dim=-1) + eps)
    )
    return -si_sdr.mean()


def _logmag_loss(est: torch.Tensor, ref: torch.Tensor, cfg: StftConfig) -> torch.Tensor:
    """est/ref (B, S, T) -> L1 of log-magnitude spectrograms."""
    b, s, t = est.shape
    est_spec = complex_spectrogram(est.reshape(b * s, t), cfg).abs()
    ref_spec = complex_spectrogram(ref.reshape(b * s, t), cfg).abs()
    return F.l1_loss(log_mag(est_spec), log_mag(ref_spec))


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
    phase_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """mask: (B, 1, F, Tf) from U-Net."""
    est = estimate_stem_waveforms(mixture, mask, phase_delta)
    return F.l1_loss(est[:, 0], target_vocals)


def _multi_stem_estimates(
    mixture: torch.Tensor,
    masks: torch.Tensor,
    stem_names: tuple[str, ...],
    phase_delta: torch.Tensor | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Per-batch-item estimated waveforms per stem."""
    est = estimate_stem_waveforms(mixture, masks, phase_delta)
    out: dict[str, list[torch.Tensor]] = {n: [] for n in stem_names}
    for i, name in enumerate(stem_names):
        for b in range(est.shape[0]):
            out[name].append(est[b, i])
    return out


def multi_stem_l1_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
    stem_names: tuple[str, ...] = STEM_NAMES,
    phase_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """masks: (B, S, F, Tf) softmax over stem dim."""
    est = estimate_stem_waveforms(mixture, masks, phase_delta)
    target = _stack_targets(batch_stems, stem_names)
    return F.l1_loss(est, target)


def multi_stem_spec_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
    stem_names: tuple[str, ...] = STEM_NAMES,
    phase_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 on log-magnitude spectrograms of estimates vs targets."""
    est = estimate_stem_waveforms(mixture, masks, phase_delta)
    target = _stack_targets(batch_stems, stem_names)
    return _logmag_loss(est, target, CFG)


def multi_stem_hybrid_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
    stem_names: tuple[str, ...] = STEM_NAMES,
    wave_weight: float = 0.5,
    phase_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    est = estimate_stem_waveforms(mixture, masks, phase_delta)
    target = _stack_targets(batch_stems, stem_names)
    wave = F.l1_loss(est, target)
    spec = _logmag_loss(est, target, CFG)
    return wave_weight * wave + (1.0 - wave_weight) * spec


def four_stem_l1_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
) -> torch.Tensor:
    return multi_stem_l1_loss(mixture, batch_stems, masks, STEM_NAMES)


def four_stem_spec_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
) -> torch.Tensor:
    return multi_stem_spec_loss(mixture, batch_stems, masks, STEM_NAMES)


def four_stem_hybrid_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
    wave_weight: float = 0.5,
) -> torch.Tensor:
    return multi_stem_hybrid_loss(
        mixture, batch_stems, masks, STEM_NAMES, wave_weight=wave_weight
    )


def _neg_si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Minimize this to maximize SI-SDR (scale-invariant; used in pro separators)."""
    est = est - est.mean()
    ref = ref - ref.mean()
    ref_energy = (ref * ref).sum() + eps
    proj = (est * ref).sum() * ref / ref_energy
    noise = est - proj
    si_sdr = 10 * torch.log10((proj.pow(2).sum() + eps) / (noise.pow(2).sum() + eps))
    return -si_sdr


def multi_stem_si_sdr_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
    stem_names: tuple[str, ...] = STEM_NAMES,
    phase_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    est = estimate_stem_waveforms(mixture, masks, phase_delta)
    target = _stack_targets(batch_stems, stem_names)
    return _neg_si_sdr_batch(est.reshape(-1, est.shape[-1]), target.reshape(-1, target.shape[-1]))


def multi_stem_multires_spec_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
    stem_names: tuple[str, ...] = STEM_NAMES,
    phase_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    est = estimate_stem_waveforms(mixture, masks, phase_delta)
    target = _stack_targets(batch_stems, stem_names)
    loss = est.new_zeros(())
    for cfg in MR_CFGS:
        loss = loss + _logmag_loss(est, target, cfg)
    return loss / len(MR_CFGS)


def multi_stem_pro_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
    stem_names: tuple[str, ...] = STEM_NAMES,
    phase_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Waveform + spectrogram + multi-resolution STFT + SI-SDR.

    With a phase residual, also pull the sum of stems back toward the mixture
    so the new phase head cannot drift into cancellation.
    """
    est = estimate_stem_waveforms(mixture, masks, phase_delta)
    target = _stack_targets(batch_stems, stem_names)
    wave = F.l1_loss(est, target)
    spec = _logmag_loss(est, target, CFG)
    multires = est.new_zeros(())
    for cfg in MR_CFGS:
        multires = multires + _logmag_loss(est, target, cfg)
    multires = multires / len(MR_CFGS)
    flat_t = est.shape[-1]
    sisdr = _neg_si_sdr_batch(est.reshape(-1, flat_t), target.reshape(-1, flat_t))
    loss = 0.5 * wave + 0.5 * spec + multires + 0.1 * sisdr
    if phase_delta is not None:
        loss = loss + 0.25 * F.l1_loss(est.sum(dim=1), mixture)
    return loss


def four_stem_si_sdr_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
) -> torch.Tensor:
    return multi_stem_si_sdr_loss(mixture, batch_stems, masks, STEM_NAMES)


def four_stem_multires_spec_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
) -> torch.Tensor:
    return multi_stem_multires_spec_loss(mixture, batch_stems, masks, STEM_NAMES)


def four_stem_pro_loss(
    mixture: torch.Tensor,
    batch_stems: dict[str, torch.Tensor],
    masks: torch.Tensor,
) -> torch.Tensor:
    return multi_stem_pro_loss(mixture, batch_stems, masks, STEM_NAMES)


def separate_stems(
    mixture: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    stem_names: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Return dict of stem waveforms (1-stem vocals or multi-stem)."""
    mixture = mixture.to(device)
    predict_phase = bool(getattr(model, "predict_phase", False))
    feats, _ = separator_features(mixture.unsqueeze(0), predict_phase)
    with torch.no_grad():
        masks, phase_delta = model_outputs(model, feats)

    if stem_names is None:
        stem_names = stem_names_for(masks.shape[1])
    if len(stem_names) != masks.shape[1]:
        raise ValueError(f"Expected {len(stem_names)} stem names, masks have {masks.shape[1]}")

    waves = estimate_stem_waveforms(mixture.unsqueeze(0), masks, phase_delta)
    return {name: waves[0, i].cpu() for i, name in enumerate(stem_names)}


def separate_mono_waveform(
    mixture: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Vocals only (1-stem checkpoints)."""
    return separate_stems(mixture, model, device)["vocals"]
