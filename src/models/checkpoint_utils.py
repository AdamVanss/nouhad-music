"""Load checkpoints; expand stem head when upgrading 4 -> 6 stems."""

from pathlib import Path

import torch
from torch import nn


def load_checkpoint(
    path: Path, device: torch.device, weights_only: bool = False
) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=device)


def load_state_with_stem_expand(
    model: nn.Module,
    ckpt: dict,
    old_num_stems: int,
    new_num_stems: int,
) -> None:
    """
    Copy U-Net weights from a smaller stem head; new output channels stay random.
    Use when fine-tuning 4-stem pro -> 6-stem Moises.
    """
    if old_num_stems >= new_num_stems:
        model.load_state_dict(ckpt["model"])
        return

    current = model.state_dict()
    pretrained = ckpt["model"]
    partial_heads = {"head.weight", "head.bias", "phase_head.weight", "phase_head.bias"}
    for key, value in pretrained.items():
        if key not in current:
            continue
        if key in partial_heads and value.shape != current[key].shape:
            n = min(value.shape[0], current[key].shape[0])
            current[key][:n].copy_(value[:n])
        else:
            if value.shape != current[key].shape:
                raise ValueError(
                    f"Shape mismatch for {key}: checkpoint {tuple(value.shape)} "
                    f"vs model {tuple(current[key].shape)}"
                )
            current[key].copy_(value)
    model.load_state_dict(current)


def load_separator_state(model: nn.Module, ckpt: dict) -> str:
    """Load weights. Magnitude checkpoints warm-start a phase-aware model."""
    state = ckpt["model"]
    model_keys = model.state_dict().keys()
    phase_only = {"phase_head.weight", "phase_head.bias", "input_proj.weight"}
    missing_phase = [k for k in phase_only if k in model_keys and k not in state]
    if missing_phase:
        incompatible = model.load_state_dict(state, strict=False)
        bad_missing = [k for k in incompatible.missing_keys if k not in phase_only]
        if bad_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Cannot adapt checkpoint: "
                f"missing {bad_missing} unexpected {list(incompatible.unexpected_keys)}"
            )
        return "adapted-phase"
    model.load_state_dict(state)
    return "exact"
