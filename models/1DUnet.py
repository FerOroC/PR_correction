import torch
import torch.nn as nn
import torch.nn.functional as F


class GNSS_UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = self.conv_block(64, 128)

        # Decoder
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.final = nn.Conv1d(64, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x, mask):
        # CHANGED: Renamed F -> num_feats to avoid shadowing torch.nn.functional
        B, W, S, num_feats = x.shape

        # 1. Prepare for 1D UNet: (B*S, F, W)
        x = x.permute(0, 2, 3, 1).reshape(B * S, num_feats, W)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        # Decoder with Skip
        d1 = self.up1(e2)

        # Handle Odd Window Sizes
        diff = e1.size(2) - d1.size(2)
        if diff > 0:
            # Now F correctly refers to torch.nn.functional
            d1 = F.pad(d1, (0, diff))

        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        d1 = self.dec1(d1)

        out = self.final(d1)  # (B*S, 1, W)

        # 2. Reshape back
        out = out.permute(0, 2, 1).reshape(B, S, W, 1).permute(0, 2, 1, 3)
        return out * mask.unsqueeze(1).unsqueeze(-1).float()