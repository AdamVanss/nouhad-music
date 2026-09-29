"""U-Net: log-magnitude spectrogram -> stem masks (softmax or sigmoid).

Phase-aware mode also reads cos/sin of the mixture phase and predicts a
bounded per-stem phase residual. The new layers are initialized so a
magnitude checkpoint still separates exactly as before at step 0.
"""

import math

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
    Input: (B, 1, F, T) log magnitude, or (B, 3, F, T) when predict_phase.
    Output masks (B, num_stems, F, T), plus phase residual in radians if enabled.
    depth=3: small net (default, matches early checkpoints).
    depth=4: extra encoder level — train with --base 32 for much better capacity.
    """

    def __init__(
        self,
        base: int = 16,
        num_stems: int = 1,
        depth: int = 3,
        predict_phase: bool = False,
    ):
        super().__init__()
        if depth not in (3, 4):
            raise ValueError("depth must be 3 or 4")
        self.num_stems = num_stems
        self.depth = depth
        self.base = base
        self.predict_phase = predict_phase
        # 3 = log magnitude, cos(phase), sin(phase). Projected back to 1 channel
        # so the pretrained encoder weights still load unchanged.
        self.in_channels = 3 if predict_phase else 1
        if predict_phase:
            self.input_proj = nn.Conv2d(3, 1, kernel_size=1, bias=False)
            nn.init.zeros_(self.input_proj.weight)
            with torch.no_grad():
                self.input_proj.weight[:, 0, 0, 0] = 1.0

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
        if predict_phase:
            self.phase_head = nn.Conv2d(base, num_stems, kernel_size=1)
            nn.init.zeros_(self.phase_head.weight)
            nn.init.zeros_(self.phase_head.bias)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.predict_phase:
            x = self.input_proj(x)
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
            masks = torch.sigmoid(out)
        else:
            masks = torch.softmax(out, dim=1)
        if not self.predict_phase:
            return masks
        delta = self.phase_head(d1)
        if delta.shape[-2:] != x.shape[-2:]:
            delta = F.interpolate(delta, size=x.shape[-2:], mode="bilinear", align_corners=False)
        # Zero weights => zero residual, so fine-tuning starts at the magnitude model.
        return masks, torch.tanh(delta) * math.pi
