import torch
import torch.nn as nn
from mamba_ssm import Mamba  # Standard Mamba implementation


class IndependentSatelliteMamba(nn.Module):
    """
    State-of-the-Art Baseline: Processes each satellite independently
    using Mamba (Selective State Space Model).

    Input shape:  (batch, window_size, num_sats, num_features)
    Output shape: (batch, num_sats, 1)
    """

    def __init__(
            self,
            input_features: int = 18,
            hidden_dim: int = 64,
            d_state: int = 16,  # SSM state expansion factor
            d_conv: int = 4,  # Local convolution width
            expand: int = 2,  # Block expansion factor
            num_layers: int = 1,
            dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Initial projection to hidden dimension
        self.input_proj = nn.Linear(input_features, hidden_dim)

        # Stacked Mamba Layers
        # Mamba inherently handles the temporal sequence
        self.layers = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Regressor Head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask):
        # B: Batch, W: Window, S: Satellites, F: Features
        B, W, S, F = x.shape

        # 1. Reshape to treat each satellite as an independent sequence
        # (B, W, S, F) -> (B, S, W, F) -> (B*S, W, F)
        x = x.permute(0, 2, 1, 3).reshape(B * S, W, F)

        # 2. Project to model dimension
        x = self.input_proj(x)

        # 3. Process through Mamba layers
        # Unlike LSTM, Mamba outputs a sequence of the same length
        for layer in self.layers:
            x = layer(x) + x  # Residual connection

        x = self.norm(x)
        x = self.dropout(x)

        # 4. Take the last timestep for error prediction
        # Shape: (B*S, hidden_dim)
        last_out = x[:, -1, :]

        # 5. Predict and Reshape back
        preds = self.regressor(last_out)  # (B*S, 1)
        preds = preds.reshape(B, S, 1)

        # 6. Apply mask (only valid satellites contribute to loss)
        return preds * mask.unsqueeze(-1).float()