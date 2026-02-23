import torch
import torch.nn as nn
import math
from typing import Optional


class IndependentSatelliteLSTM(nn.Module):
    """
    Baseline: Processes each satellite independently with no cross-satellite communication.
    Input shape:  (batch, window_size, num_sats, num_features)
    Output shape: (batch, num_sats, 1)
    """

    def __init__(self, input_features=18, hidden_dim=64, lstm_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared LSTM for all satellites
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Simple MLP Head per satellite
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask):
        B, W, S, F = x.shape

        # 1. Reshape: (B, W, S, F) -> (B*S, W, F)
        # We treat every satellite in every batch as a separate sequence
        x = x.permute(0, 2, 1, 3).reshape(B * S, W, F)

        # 2. Process through LSTM
        # lstm_out shape: (B*S, W, hidden_dim)
        lstm_out, _ = self.lstm(x)

        # 3. Take the last timestep (predicting error for the current epoch)
        last_out = lstm_out[:, -1, :]  # (B*S, hidden_dim)

        # 4. Predict
        preds = self.regressor(last_out)  # (B*S, 1)

        # 5. Reshape back and Apply mask
        preds = preds.reshape(B, S, 1)
        return preds * mask.unsqueeze(-1).float()