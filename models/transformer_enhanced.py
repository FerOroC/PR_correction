import torch
import torch.nn as nn
import math
from typing import Optional


class TransformerEnhancedLSTM(nn.Module):
    """
    Transformer-Enhanced LSTM Network for GNSS Pseudorange Error Prediction.

    Based on: Zhang et al., "Learning-based NLOS Detection and Uncertainty
    Prediction of GNSS Observations with Transformer-Enhanced LSTM Network"
    (IEEE ITSC 2023)

    Architecture:
        1. Per-satellite LSTM: Extract temporal features from each satellite's
           time series independently
        2. Attention Mechanism: Generate Q (query) from target satellite,
           K/V (key/value) from all satellites for spatial context
        3. Bi-LSTM: Process attention output, initialized with target satellite's
           LSTM hidden state
        4. Concatenation: Combine Bi-LSTM output with original LSTM features
        5. MLP Head: Predict pseudorange error

    Input shape:  (batch, window_size, num_sats, num_features)
    Output shape: (batch, num_sats, 1)
    """

    def __init__(
            self,
            input_features: int = 5,
            hidden_dim: int = 64,
            lstm_layers: int = 2,
            mlp_layers: int = 3,
            n_heads: int = 4,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        # ==================== TEMPORAL LSTM ====================
        # Processes each satellite's time series independently
        # Paper: "LSTM networks with two hidden layers"
        self.temporal_lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # ==================== Q, K, V PROJECTIONS ====================
        # Paper: "fully connected layer to generate contextual key K and value V"
        # "same fully-connected layer to generate the query features Q"
        self.fc_query = nn.Linear(hidden_dim, hidden_dim)
        self.fc_key = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, hidden_dim)

        # ==================== MULTI-HEAD ATTENTION ====================
        # Paper: "multi-head attention sub-network"
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ==================== BIDIRECTIONAL LSTM ====================
        # Paper: "bi-directional LSTM (Bi-LSTM) by initializing its hidden
        # states with the output features from the foregoing LSTM network"
        self.bi_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # ==================== FEATURE MLP ====================
        # Paper: "MLPfeat" - processes concatenated features
        # Bi-LSTM output (hidden_dim * 2) + LSTM features (hidden_dim)
        concat_dim = hidden_dim * 2 + hidden_dim
        self.mlp_feat = self._build_mlp(concat_dim, hidden_dim, mlp_layers, dropout)

        # ==================== ERROR PREDICTION HEAD ====================
        # Paper: "MLPerror" - predicts pseudorange error
        self.mlp_error = self._build_mlp(hidden_dim, 1, mlp_layers, dropout)

        self._init_weights()

    def _build_mlp(
            self,
            input_dim: int,
            output_dim: int,
            num_layers: int,
            dropout: float
    ) -> nn.Sequential:
        """
        Build MLP with specified number of layers.
        Paper: "MLPs with three hidden layers and use ReLU as the activation function"
        """
        layers = []
        current_dim = input_dim

        # Hidden layers
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, current_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(
            self,
            features: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: (batch, window_size, num_sats, num_features)
            mask: (batch, num_sats) - True for valid satellites

        Returns:
            errors: (batch, num_sats, 1) - predicted pseudorange errors
        """
        B, T, S, F = features.shape

        # ==================== TEMPORAL LSTM (per satellite) ====================
        # Reshape: (B, T, S, F) -> (B*S, T, F)
        x = features.permute(0, 2, 1, 3)  # (B, S, T, F)
        x = x.reshape(B * S, T, F)

        # Process all satellites through shared LSTM
        lstm_out, (h_n, c_n) = self.temporal_lstm(x)  # lstm_out: (B*S, T, hidden_dim)

        # Take last timestep output as satellite representation
        # Shape: (B*S, hidden_dim)
        lstm_features = lstm_out[:, -1, :]

        # Reshape back: (B*S, hidden_dim) -> (B, S, hidden_dim)
        lstm_features = lstm_features.reshape(B, S, self.hidden_dim)

        # Keep hidden states for Bi-LSTM initialization
        # h_n shape: (num_layers, B*S, hidden_dim)
        h_n_last = h_n[-1]  # (B*S, hidden_dim)
        c_n_last = c_n[-1]  # (B*S, hidden_dim)

        # ==================== GENERATE Q, K, V ====================
        # All satellites contribute to K, V (contextual information)
        # Paper: "LSTM output features of all satellites are forwarded to a
        # fully connected layer to generate contextual key K and value V"
        K = self.fc_key(lstm_features)  # (B, S, hidden_dim)
        V = self.fc_value(lstm_features)  # (B, S, hidden_dim)

        # ==================== PROCESS EACH TARGET SATELLITE ====================
        # Paper processes each satellite as a "target" getting context from all others
        # We vectorize this for efficiency

        # Query from each satellite
        Q = self.fc_query(lstm_features)  # (B, S, hidden_dim)

        # ==================== MULTI-HEAD ATTENTION ====================
        # Each satellite queries all satellites (including itself)
        # Create attention mask for padding
        # attn_mask: True = ignore, so we invert our mask
        attn_mask = ~mask  # (B, S)

        # Expand Q for per-satellite attention: each satellite attends to all
        # We need (B*S, 1, hidden_dim) queries attending to (B*S, S, hidden_dim) keys
        # But that's inefficient. Instead, use batch attention:

        # Attention: each position queries all positions
        # key_padding_mask: (B, S) where True = ignore
        attn_output, _ = self.attention(
            Q, K, V,
            key_padding_mask=attn_mask,
        )  # (B, S, hidden_dim)

        # ==================== BIDIRECTIONAL LSTM ====================
        # Paper: "bi-directional LSTM by initializing its hidden states with
        # the output features from the foregoing LSTM network of the target satellite"

        # Prepare attention output for Bi-LSTM: (B*S, 1, hidden_dim)
        # We treat each satellite's attention output as a length-1 sequence
        # (In the paper they may process the attention differently)
        attn_for_bilstm = attn_output.reshape(B * S, 1, self.hidden_dim)

        # Initialize Bi-LSTM with LSTM hidden states
        # Bi-LSTM needs (2, B*S, hidden_dim) for bidirectional
        h_init = torch.stack([h_n_last, h_n_last], dim=0)  # (2, B*S, hidden_dim)
        c_init = torch.stack([c_n_last, c_n_last], dim=0)  # (2, B*S, hidden_dim)

        bi_lstm_out, _ = self.bi_lstm(attn_for_bilstm, (h_init, c_init))
        # bi_lstm_out: (B*S, 1, hidden_dim*2)

        bi_lstm_features = bi_lstm_out.squeeze(1)  # (B*S, hidden_dim*2)
        bi_lstm_features = bi_lstm_features.reshape(B, S, self.hidden_dim * 2)

        # ==================== CONCATENATE FEATURES ====================
        # Paper: "concatenate the output of the Bi-LSTM network with LSTM
        # features of each target satellite"
        combined = torch.cat([bi_lstm_features, lstm_features], dim=-1)
        # combined: (B, S, hidden_dim*3)

        # ==================== MLP HEADS ====================
        feat = self.mlp_feat(combined)  # (B, S, hidden_dim)
        errors = self.mlp_error(feat)  # (B, S, 1)

        # ==================== APPLY MASK ====================
        errors = errors * mask.unsqueeze(-1).float()

        return errors