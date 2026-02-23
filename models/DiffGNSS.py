import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


# ============================================================================
# DIFFUSION UTILITIES
# ============================================================================

class DiffusionScheduler:
    """
    Manages the noise schedule for diffusion process.

    Forward process: q(ε_t | ε_0) = N(ε_t; √α_t·ε_0, (1-α_t)·I)
    """

    def __init__(
            self,
            num_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            schedule: str = "linear",
    ):
        self.num_timesteps = num_timesteps

        # Create beta schedule
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif schedule == "cosine":
            steps = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float32)
            alpha_bar = torch.cos((steps / num_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = torch.clamp(betas, max=0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)

        # Precompute values for q(x_t | x_0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

    def to(self, device, dtype=None):
        """Move all tensors to device and optionally convert dtype."""
        self.betas = self.betas.to(device=device, dtype=dtype)
        self.alphas = self.alphas.to(device=device, dtype=dtype)
        self.alpha_cumprod = self.alpha_cumprod.to(device=device, dtype=dtype)
        self.alpha_cumprod_prev = self.alpha_cumprod_prev.to(device=device, dtype=dtype)
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device=device, dtype=dtype)
        self.sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.to(device=device, dtype=dtype)
        return self

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Forward diffusion: sample x_t from q(x_t | x_0).

        x_t = √α_t · x_0 + √(1-α_t) · ε
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alpha_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t]

        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    def ddim_step(
            self,
            x_t: torch.Tensor,
            pred_x_0: torch.Tensor,
            t: torch.Tensor,
            t_prev: torch.Tensor,
            eta: float = 0.0,
    ):
        """DDIM sampling step."""
        alpha_t = self.alpha_cumprod[t]

        # FIX: Handle tensor t_prev properly
        if isinstance(t_prev, torch.Tensor):
            # All elements should be the same, so check min
            if t_prev.min().item() >= 0:
                alpha_t_prev = self.alpha_cumprod[t_prev]
            else:
                alpha_t_prev = torch.ones_like(alpha_t)
        else:
            # Scalar case
            alpha_t_prev = self.alpha_cumprod[t_prev] if t_prev >= 0 else torch.ones_like(alpha_t)

        # Reshape for broadcasting
        while alpha_t.dim() < x_t.dim():
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_t_prev = alpha_t_prev.unsqueeze(-1)

        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        pred_noise = (x_t - torch.sqrt(alpha_t) * pred_x_0) / torch.sqrt(1 - alpha_t)
        pred_dir = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * pred_noise
        x_t_prev = torch.sqrt(alpha_t_prev) * pred_x_0 + pred_dir

        if eta > 0:
            x_t_prev = x_t_prev + sigma_t * torch.randn_like(x_t)

        return x_t_prev


# ============================================================================
# COARSE ESTIMATOR
# ============================================================================

class TemporalEncoder(nn.Module):
    """Encodes temporal dynamics per satellite using LSTM."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()

        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )

        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, T, D)
        Returns:
            F_T: (B, N, hidden_dim)
        """
        B, N, T, D = x.shape
        x = x.reshape(B * N, T, D)

        x = self.input_mlp(x)
        x_conv = self.temporal_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + x_conv

        lstm_out, _ = self.lstm(x)
        F_T = lstm_out.mean(dim=1)
        F_T = F_T.reshape(B, N, self.hidden_dim)

        return F_T


class SpatialEncoder(nn.Module):
    """Encodes spatial context across satellites using Bi-LSTM."""

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.bi_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.hidden_dim = hidden_dim

    def forward(self, F_T: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F_T: (B, N, hidden_dim)
        Returns:
            F_S: (B, N, hidden_dim)
        """
        B, N, D = F_T.shape

        bi_out, _ = self.bi_lstm(F_T)
        bi_out = self.projection(bi_out)
        F_S_global = bi_out.mean(dim=1, keepdim=True)
        F_S = F_S_global.expand(-1, N, -1)

        return F_S


class CoarseEstimator(nn.Module):
    """Coarse pseudorange error estimator."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64):
        super().__init__()

        self.temporal_encoder = TemporalEncoder(input_dim, hidden_dim)
        self.spatial_encoder = SpatialEncoder(hidden_dim)

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            delta_rho_init: (B, N, 1)
            F_T: (B, N, hidden_dim)
            F_S: (B, N, hidden_dim)
        """
        F_T = self.temporal_encoder(x)
        F_S = self.spatial_encoder(F_T)

        F_combined = torch.cat([F_T, F_S], dim=-1)
        delta_rho_init = self.prediction_head(F_combined)
        delta_rho_init = delta_rho_init * mask.unsqueeze(-1).float()

        return delta_rho_init, F_T, F_S


# ============================================================================
# CONDITIONING SIGNAL GENERATOR
# ============================================================================

class ConditioningGenerator(nn.Module):
    """
    Generates conditioning signal C = F_TC ⊕ F_SC ⊕ F_IC
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64):
        super().__init__()

        self.temporal_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.temporal_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.coarse_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        self.output_dim = hidden_dim * 3
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, delta_rho_init: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            C: (B, N, hidden_dim*3)
        """
        B, N, T, D = x.shape

        # F_TC: Temporal Context
        x_flat = x.reshape(B * N, T, D)
        lstm_out, _ = self.temporal_lstm(x_flat)
        F_TC = lstm_out[:, -1, :]
        F_TC = self.temporal_mlp(F_TC)
        F_TC = F_TC.reshape(B, N, self.hidden_dim)

        # F_SC: Spatial Context
        F_SC_global = F_TC.max(dim=1, keepdim=True)[0]
        F_SC = F_SC_global.expand(-1, N, -1)

        # F_IC: Coarse Embedding
        F_IC = self.coarse_mlp(delta_rho_init)

        C = torch.cat([F_TC, F_SC, F_IC], dim=-1)
        return C


# ============================================================================
# GRU-BASED DENOISER (Simplified - no uncertainty)
# ============================================================================

class GRUDenoiser(nn.Module):
    """
    GRU-based denoising network.
    Only predicts clean residual ε̂_0 (no uncertainty).
    """

    def __init__(self, condition_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input encoder: noisy residual → hidden state init
        self.input_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Only ε_t now
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GRU denoiser
        self.gru = nn.GRU(
            input_size=condition_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output head (residual only)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(
            self,
            epsilon_t: torch.Tensor,
            t: torch.Tensor,
            C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            epsilon_t: (B, N, 1) - noisy residual at timestep t
            t: (B,) - timestep indices
            C: (B, N, condition_dim) - conditioning signal

        Returns:
            epsilon_hat_0: (B, N, 1) - predicted clean residual
        """
        B, N, _ = epsilon_t.shape

        # Encode timestep
        t_normalized = (t.float() / 1000.0).to(epsilon_t.dtype)
        t_embed = self.time_embed(t_normalized.unsqueeze(-1))
        t_embed = t_embed.unsqueeze(1).expand(-1, N, -1)

        # Encode noisy input for hidden state initialization
        h_init = self.input_encoder(epsilon_t)
        h_init = h_init.reshape(B * N, self.hidden_dim)
        h_init = h_init.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()

        # GRU input
        gru_input = torch.cat([C, t_embed], dim=-1)
        gru_input = gru_input.reshape(B * N, 1, -1)

        # GRU forward
        gru_out, _ = self.gru(gru_input, h_init)
        gru_out = gru_out.squeeze(1)
        gru_out = gru_out.reshape(B, N, self.hidden_dim)

        # Output prediction
        epsilon_hat_0 = self.residual_head(gru_out)

        return epsilon_hat_0


# ============================================================================
# FULL DIFF-GNSS MODEL (Simplified)
# ============================================================================

class DiffGNSS(nn.Module):
    """
    Diff-GNSS model (simplified without uncertainty).

    Components:
    1. Coarse Estimator → Δρ_init
    2. Diffusion Refinement → ε̂_0
    3. Final: Δρ_fine = Δρ_init + ε̂_0
    """

    def __init__(
            self,
            input_dim: int = 5,
            hidden_dim: int = 64,
            num_timesteps: int = 1000,
            beta_schedule: str = "linear",
    ):
        super().__init__()

        self.coarse_estimator = CoarseEstimator(input_dim, hidden_dim)
        self.condition_generator = ConditioningGenerator(input_dim, hidden_dim)
        condition_dim = hidden_dim * 3
        self.denoiser = GRUDenoiser(condition_dim, hidden_dim)
        self.scheduler = DiffusionScheduler(num_timesteps, schedule=beta_schedule)

        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps

    def forward(
            self,
            features: torch.Tensor,
            mask: torch.Tensor,
            labels_gt: Optional[torch.Tensor] = None,
            t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            features: (B, T, N, D) or (B, N, T, D)
            mask: (B, N)
            labels_gt: (B, N, 1) - for training
            t: (B,) - diffusion timestep (optional)

        Returns:
            Dictionary with predictions
        """
        B = features.shape[0]
        device = features.device
        self.scheduler.to(device, dtype=features.dtype)

        # Handle input shape
        if features.shape[1] != mask.shape[1]:
            features = features.permute(0, 2, 1, 3)

        N = features.shape[1]

        # Stage 1: Coarse Estimation
        delta_rho_init, F_T, F_S = self.coarse_estimator(features, mask)

        outputs = {'delta_rho_init': delta_rho_init}

        if labels_gt is None:
            return outputs

        # Stage 2: Diffusion Refinement
        epsilon_gt = labels_gt - delta_rho_init.detach()

        if t is None:
            t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # Forward diffusion
        epsilon_t, noise = self.scheduler.q_sample(epsilon_gt, t)

        # Conditioning
        C = self.condition_generator(features, delta_rho_init.detach())

        # Denoise
        epsilon_hat_0 = self.denoiser(epsilon_t, t, C)
        epsilon_hat_0 = epsilon_hat_0 * mask.unsqueeze(-1).float()

        # Refined prediction
        delta_rho_fine = delta_rho_init + epsilon_hat_0

        outputs.update({
            'epsilon_gt': epsilon_gt,
            'epsilon_hat_0': epsilon_hat_0,
            'delta_rho_fine': delta_rho_fine,
            't': t,
        })

        return outputs

    @torch.no_grad()
    def inference(
            self,
            features: torch.Tensor,
            mask: torch.Tensor,
            num_inference_steps: int = 50,
            eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Inference with iterative denoising.

        Returns:
            delta_rho_fine: (B, N, 1)
        """
        B = features.shape[0]
        device = features.device
        self.scheduler.to(device, dtype=features.dtype)

        if features.shape[1] != mask.shape[1]:
            features = features.permute(0, 2, 1, 3)

        N = features.shape[1]

        # Stage 1: Coarse estimation
        delta_rho_init, F_T, F_S = self.coarse_estimator(features, mask)

        # Conditioning
        C = self.condition_generator(features, delta_rho_init)

        # Stage 2: Iterative denoising
        epsilon_t = torch.randn(B, N, 1, device=device, dtype=features.dtype)

        step_size = self.num_timesteps // num_inference_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            epsilon_hat_0 = self.denoiser(epsilon_t, t_tensor, C)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                t_prev_tensor = torch.full((B,), t_prev, device=device, dtype=torch.long)
                epsilon_t = self.scheduler.ddim_step(epsilon_t, epsilon_hat_0, t_tensor, t_prev_tensor, eta)
            else:
                epsilon_t = epsilon_hat_0

        delta_rho_fine = delta_rho_init + epsilon_t
        delta_rho_fine = delta_rho_fine * mask.unsqueeze(-1).float()

        return delta_rho_fine