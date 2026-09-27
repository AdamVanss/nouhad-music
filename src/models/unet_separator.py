"""U-Net: log-magnitude spectrogram -> stem masks (softmax or sigmoid)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _crop_to_match(encoder: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
    eh, ew = encoder.shape[-2:]
    dh, dw = decoder.shape[-2:]
    if eh == dh and ew == dw:
        return encoder
    y0 = max(0, (eh - dh) // 2)
    x0 = max(0, (ew - dw) // 2)
    return encoder[:, :, y0 : y0 + dh, x0 : x0 + dw]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSeparator(nn.Module):
    """
    Input: (B, 1, F, T) log magnitude. Output masks (B, num_stems, F, T).
    depth=3: small net (default, matches early checkpoints).
    depth=4: extra encoder level — train with --base 32 for much better capacity.
    """

    def __init__(self, base: int = 16, num_stems: int = 1, depth: int = 3):
        super().__init__()
        if depth not in (3, 4):
            raise ValueError("depth must be 3 or 4")
        self.num_stems = num_stems
        self.depth = depth
        self.base = base

        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        if depth == 4:
            self.enc4 = ConvBlock(base * 4, base * 8)
            self.dec3 = ConvBlock(base * 8 + base * 4, base * 4)

        self.dec2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.dec1 = ConvBlock(base * 2 + base, base)
        self.head = nn.Conv2d(base, num_stems, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        if self.depth == 4:
            e4 = self.enc4(self.pool(e3))
            d3 = self.up(e4)
            d3 = self.dec3(torch.cat([d3, _crop_to_match(e3, d3)], dim=1))
            d2 = self.up(d3)
        else:
            d2 = self.up(e3)

        d2 = self.dec2(torch.cat([d2, _crop_to_match(e2, d2)], dim=1))
        d1 = self.up(d2)
        d1 = self.dec1(torch.cat([d1, _crop_to_match(e1, d1)], dim=1))
        out = self.head(d1)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if self.num_stems == 1:
            return torch.sigmoid(out)
        return torch.softmax(out, dim=1)
