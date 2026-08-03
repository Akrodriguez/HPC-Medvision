
"""
model.py
HPC-MedVision - Phase 6: Lightweight U-Net Architecture

A scaled-down U-Net (base filters=32, 4 downsampling levels) for
binary brain tumor segmentation, designed to train efficiently
within short PBS session windows.
"""

import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    """Two 3x3 convolutions, each followed by BatchNorm + ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class LightUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()

        f = base_filters  # 32

        # --- Encoder ---
        self.enc1 = conv_block(in_channels, f)        # 128x128 -> f
        self.pool1 = nn.MaxPool2d(2)                   # 64x64

        self.enc2 = conv_block(f, f * 2)                # 64x64 -> 2f
        self.pool2 = nn.MaxPool2d(2)                    # 32x32

        self.enc3 = conv_block(f * 2, f * 4)             # 32x32 -> 4f
        self.pool3 = nn.MaxPool2d(2)                     # 16x16

        self.enc4 = conv_block(f * 4, f * 8)              # 16x16 -> 8f
        self.pool4 = nn.MaxPool2d(2)                      # 8x8

        # --- Bottleneck ---
        self.bottleneck = conv_block(f * 8, f * 16)        # 8x8 -> 16f

        # --- Decoder ---
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)  # 8x8 -> 16x16
        self.dec4 = conv_block(f * 16, f * 8)  # concat with enc4 (8f + 8f = 16f)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)   # 16x16 -> 32x32
        self.dec3 = conv_block(f * 8, f * 4)   # concat with enc3

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)   # 32x32 -> 64x64
        self.dec2 = conv_block(f * 4, f * 2)   # concat with enc2

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)        # 64x64 -> 128x128
        self.dec1 = conv_block(f * 2, f)        # concat with enc1

        # --- Final 1x1 conv to produce single-channel output ---
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return out  # raw logits; sigmoid applied in loss/inference, not here


if __name__ == "__main__":
    # Sanity check: verify shapes flow correctly end-to-end
    model = LightUNet(in_channels=1, out_channels=1, base_filters=32)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    dummy_input = torch.randn(2, 1, 128, 128)  # batch of 2, 1 channel, 128x128
    output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (2, 1, 128, 128), "Output shape mismatch!"
    print("Shape check passed: input and output spatial dimensions match.")
