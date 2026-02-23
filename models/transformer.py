import torch
import torch.nn as nn


class TransformerGNSS(nn.Module):
    def __init__(self,
                 input_features: int = 19,  # Your selected features
                 d_model: int = 64,  # Match MLP hidden size
                 n_layers: int = 1,  # Single transformer layer
                 nhead: int = 4,  # Multi-head attention
                 dropout: float = 0.1,
                 ):
        """
        Parameter-matched Transformer for fair comparison with MLP.

        Architecture philosophy:
        - Same d_model as MLP hidden_size (64)
        - Minimal feedforward expansion (1x instead of 4x)
        - Single layer to match MLP parameter count
        - Focus: Does attention help vs. independent satellite processing?
        """
        super(TransformerGNSS, self).__init__()

        # Input projection: match MLP first layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_features, d_model),
            nn.ReLU(),  # Changed from GELU to match MLP
            nn.Dropout(dropout)
        )

        # MLP has: 64 → 64 (1x expansion)
        # New: 64 → 64 (1x expansion) - matches MLP
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=False
        )

        # Single layer to match MLP depth
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=None
        )

        self.output_head = nn.Linear(d_model, 1)

        # Simple initialization (match MLP default)
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.constant_(self.output_head.bias, 0)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, max_sats, input_features)
            mask: (batch, max_sats) - True for real satellites, False for padding

        Returns:
            errors: (batch, max_sats, 1) - predicted pseudorange errors
        """
        # Input projection (same as MLP first layer)
        x = self.input_projection(features)  # (batch, max_sats, d_model)

        # THE KEY DIFFERENCE: Cross-satellite attention
        # MLP processes each satellite independently
        # Transformer allows satellites to "see" each other
        padding_mask = ~mask  # Invert: True = ignore
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Output projection (same as MLP output layer)
        errors = self.output_head(x)  # (batch, max_sats, 1)

        # Zero out predictions for padded satellites
        errors = errors * mask.unsqueeze(-1).float()

        return errors