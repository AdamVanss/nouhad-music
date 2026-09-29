"""Small CNN that labels a cropped face photo as Boy or Girl."""

import torch
from torch import nn

CLASSES = ("Boy", "Girl")
IMAGE_SIZE = 128
MEAN = (0.5, 0.5, 0.5)
STD = (0.25, 0.25, 0.25)


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class GenderCNN(nn.Module):
    """Five conv blocks (128 -> 4 px), global average pool, linear head over CLASSES."""

    def __init__(self, width: int = 32, dropout: float = 0.3):
        super().__init__()
        channels = [3, width, width * 2, width * 4, width * 8, width * 8]
        self.features = nn.Sequential(*(_block(a, b) for a, b in zip(channels[:-1], channels[1:])))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], len(CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def normalize(images: torch.Tensor) -> torch.Tensor:
    """uint8 or [0,1] float (B, 3, H, W) -> normalized float."""
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    mean = torch.tensor(MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std
