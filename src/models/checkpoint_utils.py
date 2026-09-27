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
    for key, value in pretrained.items():
        if key not in current:
            continue
        if key == "head.weight":
            current[key][:old_num_stems].copy_(value)
        elif key == "head.bias":
            current[key][:old_num_stems].copy_(value)
        else:
            current[key].copy_(value)
    model.load_state_dict(current)
