"""DDPM forward/reverse diffusion with cross-view denoising.

Implements the diffusion branch of the TD4DD architecture:
- Noise schedule with pre-computed alpha bars
- GCN encoder for auxiliary subgraph views
- Cross-view denoiser conditioned on the opposite view
- Training (forward diffusion + denoising loss) and DDIM inference
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Noise schedule
# ──────────────────────────────────────────────────────────────────────────────


class NoiseSchedule(nn.Module):
    """Linear beta schedule with pre-computed diffusion coefficients."""

    def __init__(
        self, T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02
    ) -> None:
        super().__init__()
        self.T = T

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars)
        )

    def forward_diffusion(self, h_0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Add noise at timestep *t*.

        Parameters
        ----------
        h_0 : Tensor
            Clean embeddings, shape ``(N, d)``.
        t : Tensor
            Integer timestep(s), scalar or shape ``(N,)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(h_t, noise)`` both with same shape as *h_0*.
        """
        noise = torch.randn_like(h_0)

        sqrt_ab = self.sqrt_alpha_bars[t]
        sqrt_1m_ab = self.sqrt_one_minus_alpha_bars[t]

        # Reshape for broadcasting: scalar → (1, 1), vector → (N, 1)
        if sqrt_ab.dim() == 0:
            sqrt_ab = sqrt_ab.reshape(1, 1)
            sqrt_1m_ab = sqrt_1m_ab.reshape(1, 1)
        else:
            sqrt_ab = sqrt_ab.unsqueeze(-1)
            sqrt_1m_ab = sqrt_1m_ab.unsqueeze(-1)

        h_t = sqrt_ab * h_0 + sqrt_1m_ab * noise
        return h_t, noise


# ──────────────────────────────────────────────────────────────────────────────
# GCN encoder for auxiliary subgraphs
# ──────────────────────────────────────────────────────────────────────────────


class AuxiliaryGCNEncoder(nn.Module):
    """Two-layer GCN encoder for auxiliary subgraph views."""

    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Encode node features through two GCN layers.

        Parameters
        ----------
        x : Tensor
            Node features, shape ``(N, in_dim)``.
        edge_index : Tensor
            Edge index, shape ``(2, E)``.

        Returns
        -------
        Tensor
            Encoded features, shape ``(N, hidden_dim)``.
        """
        h = self.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h


# ──────────────────────────────────────────────────────────────────────────────
# Cross-view denoiser
# ──────────────────────────────────────────────────────────────────────────────


class CrossViewDenoiser(nn.Module):
    """Denoiser conditioned on the opposite auxiliary view."""

    def __init__(self, hidden_dim: int = 128, time_emb_dim: int = 32) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, hidden_dim),
        )
        # Input: concat of h_t_self, h_t_other, time_emb → 3 * hidden_dim
        self.gcn1 = GCNConv(3 * hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(
        self, h_t_self: Tensor, h_t_other: Tensor, t: Tensor, edge_index: Tensor
    ) -> Tensor:
        """Predict noise given noisy self-view, clean other-view, and timestep.

        Parameters
        ----------
        h_t_self : Tensor
            Noisy self-view embeddings, shape ``(N, hidden)``.
        h_t_other : Tensor
            Other-view embeddings, shape ``(N, hidden)``.
        t : Tensor
            Timestep (scalar or ``(N,)``).
        edge_index : Tensor
            Edge index, shape ``(2, E)``.

        Returns
        -------
        Tensor
            Predicted noise, shape ``(N, hidden)``.
        """
        N = h_t_self.shape[0]

        # Time embedding
        if t.dim() == 0:
            t_float = t.float().expand(N, 1)
        else:
            t_float = t.float().unsqueeze(-1)  # (N, 1)
        t_emb = self.time_mlp(t_float)  # (N, hidden)

        # Concatenate self, other, time
        h = torch.cat([h_t_self, h_t_other, t_emb], dim=-1)  # (N, 3*hidden)

        # 3 GCN layers with ReLU between (no final activation)
        h = self.relu(self.gcn1(h, edge_index))
        h = self.relu(self.gcn2(h, edge_index))
        h = self.gcn3(h, edge_index)
        return h


# ──────────────────────────────────────────────────────────────────────────────
# Diffusion module
# ──────────────────────────────────────────────────────────────────────────────


class DiffusionModule(nn.Module):
    """DDPM diffusion with cross-view denoising for dual auxiliary subgraphs."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        T: int = 100,
        ddim_steps: int = 10,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.T = T
        self.ddim_steps = ddim_steps

        self.encoder = AuxiliaryGCNEncoder(in_dim, hidden_dim)
        self.denoiser = CrossViewDenoiser(hidden_dim)
        self.noise_schedule = NoiseSchedule(T)

    def training_step(
        self, target_features: Tensor, aux_edge_indices: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Forward pass for training: encode, diffuse, denoise.

        Parameters
        ----------
        target_features : Tensor
            Node features, shape ``(N, in_dim)``.
        aux_edge_indices : list[Tensor]
            Two edge-index tensors of shape ``(2, E_i)`` for the two views.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(h_diff, L_diff)`` — fused clean encoding ``(N, hidden)`` and
            scalar denoising loss.
        """
        # 1. Encode both views
        h_0_v1 = self.encoder(target_features, aux_edge_indices[0])
        h_0_v2 = self.encoder(target_features, aux_edge_indices[1])

        # 2. Sample random timestep
        t = torch.randint(0, self.T, (1,), device=target_features.device).squeeze(0)

        # 3. Forward diffusion
        h_t_v1, noise_v1 = self.noise_schedule.forward_diffusion(h_0_v1, t)
        h_t_v2, noise_v2 = self.noise_schedule.forward_diffusion(h_0_v2, t)

        # 4. Cross-view denoising
        pred_v1 = self.denoiser(h_t_v1, h_t_v2, t, aux_edge_indices[0])
        pred_v2 = self.denoiser(h_t_v2, h_t_v1, t, aux_edge_indices[1])

        # 5. Denoising loss (sum of both directions)
        L_diff = nn.functional.mse_loss(pred_v1, noise_v1) + nn.functional.mse_loss(
            pred_v2, noise_v2
        )

        # 6. Fused clean encoding
        h_diff = 0.5 * (h_0_v1 + h_0_v2)

        return h_diff, L_diff

    @torch.no_grad()
    def inference(
        self, target_features: Tensor, aux_edge_indices: list[Tensor]
    ) -> Tensor:
        """DDIM reverse diffusion for inference.

        Parameters
        ----------
        target_features : Tensor
            Node features, shape ``(N, in_dim)``.
        aux_edge_indices : list[Tensor]
            Two edge-index tensors for the two views.

        Returns
        -------
        Tensor
            Fused denoised representation, shape ``(N, hidden)``.
        """
        N = target_features.shape[0]
        device = target_features.device

        # Timestep schedule (reversed, from T-1 down to 0)
        timesteps = torch.linspace(0, self.T - 1, self.ddim_steps).long().flip(0)

        # Start from pure noise
        h_t_v1 = torch.randn(N, self.hidden_dim, device=device)
        h_t_v2 = torch.randn(N, self.hidden_dim, device=device)

        ns = self.noise_schedule

        for i, t in enumerate(timesteps):
            t_tensor = t.to(device)

            # Predict noise via cross-view denoiser
            pred_noise_v1 = self.denoiser(h_t_v1, h_t_v2, t_tensor, aux_edge_indices[0])
            pred_noise_v2 = self.denoiser(h_t_v2, h_t_v1, t_tensor, aux_edge_indices[1])

            # Current schedule values
            sqrt_ab = ns.sqrt_alpha_bars[t].reshape(1, 1)
            sqrt_1m_ab = ns.sqrt_one_minus_alpha_bars[t].reshape(1, 1)

            # Predict clean signal
            pred_h0_v1 = (h_t_v1 - sqrt_1m_ab * pred_noise_v1) / sqrt_ab
            pred_h0_v2 = (h_t_v2 - sqrt_1m_ab * pred_noise_v2) / sqrt_ab

            # DDIM step to previous timestep
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                sqrt_ab_prev = ns.sqrt_alpha_bars[t_prev].reshape(1, 1)
                sqrt_1m_ab_prev = ns.sqrt_one_minus_alpha_bars[t_prev].reshape(1, 1)
            else:
                # Last step: go to t=0 (clean)
                sqrt_ab_prev = torch.tensor(1.0, device=device).reshape(1, 1)
                sqrt_1m_ab_prev = torch.tensor(0.0, device=device).reshape(1, 1)

            h_t_v1 = sqrt_ab_prev * pred_h0_v1 + sqrt_1m_ab_prev * pred_noise_v1
            h_t_v2 = sqrt_ab_prev * pred_h0_v2 + sqrt_1m_ab_prev * pred_noise_v2

        return 0.5 * (h_t_v1 + h_t_v2)
