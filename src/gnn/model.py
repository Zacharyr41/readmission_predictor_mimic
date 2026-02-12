"""Full TD4DD-adapted model assembly."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData

from src.gnn.diffusion import DiffusionModule
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

    def forward(
        self,
        batch_data: HeteroData,
        aux_edge_indices: list[Tensor] | None = None,
        *,
        contextual_hop_neighbors: list[list[Tensor]] | None = None,
        temporal_hop_neighbors: list[Tensor] | None = None,
        temporal_hop_deltas: list[Tensor] | None = None,
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
        batch_size = getattr(batch_data, "batch_size", target_emb.shape[0])
        target_emb = target_emb[:batch_size]

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
        probabilities = torch.sigmoid(logits)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "h_hhgat": h_hhgat,
            "h_diff": h_diff,
            "L_diff": L_diff,
            "attention_info": attention_info,
        }
