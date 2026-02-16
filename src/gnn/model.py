"""Full TD4DD-adapted model assembly."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData

from src.gnn.diffusion import DiffusionModule
from src.gnn.hop_extraction import HopIndices
from src.gnn.transformer import DualTrackTransformer

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the TD4DD model."""

    feat_dims: dict[str, int] = field(default_factory=dict)
    d_model: int = 128
    num_contextual_paths: int = 2
    k_hops_contextual: int = 4
    k_hops_temporal: int = 2
    nhead: int = 4
    dropout: float = 0.3
    use_transformer: bool = True
    use_diffusion: bool = True
    use_temporal_encoding: bool = True
    diffusion_T: int = 100
    diffusion_ddim_steps: int = 10


class TD4DDModel(nn.Module):
    """TD4DD model: dual-track Transformer + cross-view diffusion with fusion."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Per-type linear projections
        self.proj = nn.ModuleDict(
            {
                ntype: nn.Linear(dim, config.d_model)
                for ntype, dim in config.feat_dims.items()
            }
        )

        # 2. Transformer branch (optional)
        self.transformer: DualTrackTransformer | None = None
        if config.use_transformer:
            self.transformer = DualTrackTransformer(
                num_contextual_paths=config.num_contextual_paths,
                k_hops_contextual=config.k_hops_contextual,
                k_hops_temporal=(
                    config.k_hops_temporal if config.use_temporal_encoding else 0
                ),
                d_model=config.d_model,
                nhead=config.nhead,
                dropout=config.dropout,
            )

        # 3. Diffusion branch (optional)
        self.diffusion: DiffusionModule | None = None
        if config.use_diffusion:
            self.diffusion = DiffusionModule(
                in_dim=config.d_model,
                hidden_dim=config.d_model,
                T=config.diffusion_T,
                ddim_steps=config.diffusion_ddim_steps,
            )

        # 4. Learnable fusion weight
        self.lambda_param = nn.Parameter(torch.tensor(0.5))

        # 5. Classifier head
        self.classifier = nn.Linear(config.d_model, 1)

    @staticmethod
    def _resolve_batch_size(batch_data: HeteroData, fallback: int) -> int:
        """Get batch_size from global or node stores, with fallback."""
        try:
            return batch_data.batch_size
        except AttributeError:
            pass
        for ntype in batch_data.node_types:
            store = batch_data[ntype]
            if hasattr(store, "batch_size"):
                return store.batch_size
        return fallback

    def forward(
        self,
        batch_data: HeteroData,
        aux_edge_indices: list[Tensor] | None = None,
        *,
        contextual_hop_neighbors: list[list[Tensor]] | None = None,
        temporal_hop_neighbors: list[Tensor] | None = None,
        temporal_hop_deltas: list[Tensor] | None = None,
        hop_indices: HopIndices | None = None,
    ) -> dict:
        """Forward pass through the TD4DD model.

        Parameters
        ----------
        batch_data : HeteroData
            Batched heterogeneous graph data with node features per type.
        aux_edge_indices : list[Tensor] or None
            Two auxiliary edge-index tensors for the diffusion branch.
        contextual_hop_neighbors : list[list[Tensor]] or None
            Pre-structured neighbor tensors for the transformer contextual track.
        temporal_hop_neighbors : list[Tensor] or None
            Pre-structured neighbor tensors for the transformer temporal track.
        temporal_hop_deltas : list[Tensor] or None
            Day-gap tensors for temporal encoding.
        hop_indices : HopIndices or None
            Lightweight index structures produced by ``HopExtractor``.
            When provided and ``contextual_hop_neighbors`` is None, the model
            auto-gathers projected features using these indices.

        Returns
        -------
        dict
            Keys: logits, probabilities, h_hhgat, h_diff, L_diff, attention_info
        """
        # 1. Project all node types present in batch_data
        projected = {}
        for ntype in batch_data.node_types:
            if ntype in self.proj and hasattr(batch_data[ntype], "x"):
                projected[ntype] = self.proj[ntype](batch_data[ntype].x)

        # 2. Extract target admission embeddings
        target_emb = projected["admission"]
        batch_size = self._resolve_batch_size(batch_data, target_emb.shape[0])
        target_emb = target_emb[:batch_size]

        # Auto-gather from hop_indices if explicit tensors not provided
        if contextual_hop_neighbors is None and hop_indices is not None:
            contextual_hop_neighbors, temporal_hop_neighbors, temporal_hop_deltas = (
                self._gather_hop_neighbors(projected, hop_indices)
            )

        # 3. Transformer branch
        h_hhgat = None
        attention_info = None
        if self.transformer is not None and contextual_hop_neighbors is not None:
            h_hhgat, attention_info = self.transformer(
                target_emb,
                contextual_hop_neighbors,
                temporal_hop_neighbors,
                temporal_hop_deltas,
            )

        # 4. Diffusion branch
        h_diff = None
        L_diff = torch.tensor(0.0, device=target_emb.device)
        if self.diffusion is not None and aux_edge_indices is not None:
            if self.training:
                h_diff, L_diff = self.diffusion.training_step(
                    target_emb, aux_edge_indices
                )
            else:
                h_diff = self.diffusion.inference(target_emb, aux_edge_indices)

        # 5. Fuse branches
        lam = self.lambda_param.clamp(0.0, 1.0)
        if h_hhgat is not None and h_diff is not None:
            h_fused = lam * h_diff + (1 - lam) * h_hhgat
        elif h_hhgat is not None:
            h_fused = h_hhgat
        elif h_diff is not None:
            h_fused = h_diff
        else:
            h_fused = target_emb

        # 6. Classify
        logits = self.classifier(h_fused)
        # Guard against NaN from upstream (diffusion blow-up, sparse attention)
        if torch.isnan(logits).any():
            logger.warning("NaN in logits — replacing with 0.0")
            logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        probabilities = torch.sigmoid(logits)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "h_hhgat": h_hhgat,
            "h_diff": h_diff,
            "L_diff": L_diff,
            "attention_info": attention_info,
        }

    def _gather_hop_neighbors(
        self,
        projected: dict[str, Tensor],
        hop_indices: HopIndices,
    ) -> tuple[
        list[list[Tensor]] | None,
        list[Tensor] | None,
        list[Tensor] | None,
    ]:
        """Gather projected features using hop index structures.

        Parameters
        ----------
        projected : dict[str, Tensor]
            Per-node-type projected features ``{ntype: (N_total, d_model)}``.
        hop_indices : HopIndices
            Index structures from ``HopExtractor.extract()``.

        Returns
        -------
        tuple
            ``(contextual_hop_neighbors, temporal_hop_neighbors, temporal_hop_deltas)``
            matching the shapes the transformer expects.
        """
        d = self.config.d_model

        # --- Contextual ---
        contextual: list[list[Tensor]] | None = None
        if hop_indices.contextual_indices:
            contextual = []
            for path_idx in range(len(hop_indices.contextual_indices)):
                path_tensors: list[Tensor] = []
                for hop_idx in range(len(hop_indices.contextual_indices[path_idx])):
                    indices = hop_indices.contextual_indices[path_idx][hop_idx]  # (B, N)
                    mask = hop_indices.contextual_masks[path_idx][hop_idx]  # (B, N)
                    ntype = hop_indices.contextual_node_types[path_idx][hop_idx]

                    if ntype in projected:
                        feats = projected[ntype]  # (total_nodes, d)
                        gathered = feats[indices]  # (B, N, d)
                    else:
                        # Node type not projected — use zeros
                        gathered = torch.zeros(
                            *indices.shape, d,
                            device=indices.device,
                            dtype=next(iter(projected.values())).dtype,
                        )

                    # Zero out padded positions
                    gathered = gathered * mask.unsqueeze(-1).float()
                    path_tensors.append(gathered)
                contextual.append(path_tensors)

        # --- Temporal ---
        temporal_neighbors: list[Tensor] | None = None
        temporal_deltas: list[Tensor] | None = None
        if hop_indices.temporal_indices is not None:
            temporal_neighbors = []
            for hop_idx in range(len(hop_indices.temporal_indices)):
                indices = hop_indices.temporal_indices[hop_idx]  # (B, N)
                mask = hop_indices.temporal_masks[hop_idx]  # (B, N)

                # Temporal nodes are always admissions
                if "admission" in projected:
                    feats = projected["admission"]
                    gathered = feats[indices]
                else:
                    gathered = torch.zeros(
                        *indices.shape, d,
                        device=indices.device,
                        dtype=next(iter(projected.values())).dtype,
                    )

                gathered = gathered * mask.unsqueeze(-1).float()
                temporal_neighbors.append(gathered)

            temporal_deltas = hop_indices.temporal_deltas

        return contextual, temporal_neighbors, temporal_deltas
