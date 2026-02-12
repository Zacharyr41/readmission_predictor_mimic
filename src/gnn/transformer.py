"""K-hop and hierarchical Transformer encoders.

Provides the core Transformer attention backbone for the TD4DD architecture:
- Per-hop self-attention among sampled neighbors
- Target-neighbor cross-attention to aggregate each hop
- Hierarchical cross-hop attention with interpretable weights
- Dual-track (contextual + temporal) processing with sinusoidal temporal encoding
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Temporal encoding
# ──────────────────────────────────────────────────────────────────────────────


def temporal_encoding(delta_days: Tensor, d_model: int = 128) -> Tensor:
    """Sinusoidal temporal encoding for day-gap values.

    Based on HGT (Hu et al., WWW 2020).

    Parameters
    ----------
    delta_days : Tensor
        1-D tensor of shape ``(N,)`` with day-gap values.
    d_model : int
        Embedding dimension (must be even).

    Returns
    -------
    Tensor
        Shape ``(N, d_model)`` with values in ``[-1, 1]``.
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")

    N = delta_days.shape[0]
    position = delta_days.unsqueeze(1).float()  # (N, 1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=delta_days.device)
        * -(math.log(10000.0) / d_model)
    )  # (d_model/2,)

    pe = torch.zeros(N, d_model, device=delta_days.device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ──────────────────────────────────────────────────────────────────────────────
# Hop-level Transformer
# ──────────────────────────────────────────────────────────────────────────────


class HopTransformerEncoder(nn.Module):
    """Per-hop encoder: neighbor self-attention + target-neighbor cross-attention."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.neighbor_encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        # Cross-attention projections (single-head)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self._scale = d_model**0.5

        self._last_attn_weights: Tensor | None = None

    def forward(
        self,
        target_emb: Tensor,
        neighbor_embs: Tensor,
        temporal_deltas: Tensor | None = None,
    ) -> Tensor:
        """Run per-hop encoding.

        Parameters
        ----------
        target_emb : Tensor
            Shape ``(B, d)``.
        neighbor_embs : Tensor
            Shape ``(B, N, d)``.
        temporal_deltas : Tensor or None
            Shape ``(B, N)`` day-gap values. If provided, sinusoidal temporal
            encoding is added to neighbor embeddings.

        Returns
        -------
        Tensor
            Aggregated output of shape ``(B, d)``.
        """
        if temporal_deltas is not None:
            B, N = temporal_deltas.shape
            d = neighbor_embs.shape[-1]
            flat = temporal_deltas.reshape(-1)  # (B*N,)
            te = temporal_encoding(flat, d_model=d)  # (B*N, d)
            te = te.reshape(B, N, d)
            neighbor_embs = neighbor_embs + te  # no in-place

        # Self-attention among neighbors
        encoded = self.neighbor_encoder(neighbor_embs)  # (B, N, d)

        # Cross-attention: target queries into encoded neighbors
        Q = self.W_Q(target_emb).unsqueeze(1)  # (B, 1, d)
        K = self.W_K(encoded)  # (B, N, d)
        V = self.W_V(encoded)  # (B, N, d)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self._scale  # (B, 1, N)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, 1, N)
        self._last_attn_weights = attn_weights.squeeze(1)  # (B, N)

        out = torch.bmm(attn_weights, V).squeeze(1)  # (B, d)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Meta-path Transformer
# ──────────────────────────────────────────────────────────────────────────────


class MetaPathTransformer(nn.Module):
    """Multi-hop Transformer with hierarchical cross-hop attention."""

    def __init__(
        self,
        k_hops: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        use_temporal_encoding: bool = False,
    ) -> None:
        super().__init__()
        self.k_hops = k_hops
        self.use_temporal_encoding = use_temporal_encoding

        self.hop_encoders = nn.ModuleList(
            [
                HopTransformerEncoder(d_model, nhead, dim_feedforward, dropout)
                for _ in range(k_hops)
            ]
        )
        self.cross_hop_encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.hop_attention = nn.Linear(d_model, 1)

    def forward(
        self,
        target_emb: Tensor,
        hop_neighbors: list[Tensor],
        hop_temporal_deltas: list[Tensor] | None = None,
    ) -> tuple[Tensor, dict]:
        """Run multi-hop encoding with hierarchical aggregation.

        Parameters
        ----------
        target_emb : Tensor
            Shape ``(B, d)``.
        hop_neighbors : list[Tensor]
            K tensors of shape ``(B, N_k, d)``.
        hop_temporal_deltas : list[Tensor] or None
            K tensors of shape ``(B, N_k)`` if temporal encoding is used.

        Returns
        -------
        tuple[Tensor, dict]
            ``(output, info)`` where output is ``(B, d)`` and info contains
            ``hop_attentions`` and ``hierarchical_weights``.
        """
        hop_outputs = []
        hop_attentions = []

        for k, encoder in enumerate(self.hop_encoders):
            deltas = None
            if self.use_temporal_encoding and hop_temporal_deltas is not None:
                deltas = hop_temporal_deltas[k]

            out = encoder(target_emb, hop_neighbors[k], deltas)  # (B, d)
            hop_outputs.append(out)
            hop_attentions.append(encoder._last_attn_weights)

        # Stack hop outputs → (B, K, d) and run cross-hop self-attention
        stacked = torch.stack(hop_outputs, dim=1)  # (B, K, d)
        cross_hop = self.cross_hop_encoder(stacked)  # (B, K, d)

        # Hierarchical weights via learned attention + softmax
        logits = self.hop_attention(cross_hop).squeeze(-1)  # (B, K)
        weights = torch.softmax(logits, dim=-1)  # (B, K)

        # Weighted sum across hops
        output = (weights.unsqueeze(-1) * cross_hop).sum(dim=1)  # (B, d)

        info = {
            "hop_attentions": hop_attentions,
            "hierarchical_weights": weights,
        }
        return output, info


# ──────────────────────────────────────────────────────────────────────────────
# Dual-track Transformer
# ──────────────────────────────────────────────────────────────────────────────


class DualTrackTransformer(nn.Module):
    """Dual-track (contextual + temporal) Transformer encoder."""

    def __init__(
        self,
        num_contextual_paths: int = 2,
        k_hops_contextual: int = 4,
        k_hops_temporal: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_contextual_paths = num_contextual_paths
        self.k_hops_temporal = k_hops_temporal

        self.contextual_transformers = nn.ModuleList(
            [
                MetaPathTransformer(
                    k_hops=k_hops_contextual,
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_temporal_encoding=False,
                )
                for _ in range(num_contextual_paths)
            ]
        )

        if k_hops_temporal > 0:
            self.temporal_transformer: MetaPathTransformer | None = (
                MetaPathTransformer(
                    k_hops=k_hops_temporal,
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_temporal_encoding=True,
                )
            )
            total_tracks = num_contextual_paths + 1
        else:
            self.temporal_transformer = None
            total_tracks = num_contextual_paths

        self.merge = nn.Linear(d_model * total_tracks, d_model)

    def forward(
        self,
        target_emb: Tensor,
        contextual_hop_neighbors: list[list[Tensor]],
        temporal_hop_neighbors: list[Tensor] | None = None,
        temporal_hop_deltas: list[Tensor] | None = None,
    ) -> tuple[Tensor, dict]:
        """Run dual-track encoding.

        Parameters
        ----------
        target_emb : Tensor
            Shape ``(B, d)``.
        contextual_hop_neighbors : list[list[Tensor]]
            One list per contextual path, each containing K tensors of
            shape ``(B, N_k, d)``.
        temporal_hop_neighbors : list[Tensor] or None
            K tensors of shape ``(B, N_k, d)`` for the temporal track.
        temporal_hop_deltas : list[Tensor] or None
            K tensors of shape ``(B, N_k)`` with day-gap values.

        Returns
        -------
        tuple[Tensor, dict]
            ``(output, info)`` where output is ``(B, d)`` and info contains
            ``contextual_infos`` and ``temporal_info``.
        """
        track_outputs = []
        contextual_infos = []

        for i, transformer in enumerate(self.contextual_transformers):
            out, info = transformer(target_emb, contextual_hop_neighbors[i])
            track_outputs.append(out)
            contextual_infos.append(info)

        temporal_info = None
        if (
            self.temporal_transformer is not None
            and temporal_hop_neighbors is not None
        ):
            out, temporal_info = self.temporal_transformer(
                target_emb, temporal_hop_neighbors, temporal_hop_deltas
            )
            track_outputs.append(out)

        # Concatenate all track outputs and project
        merged = torch.cat(track_outputs, dim=-1)  # (B, d * total_tracks)
        output = self.merge(merged)  # (B, d)

        info = {
            "contextual_infos": contextual_infos,
            "temporal_info": temporal_info,
        }
        return output, info
