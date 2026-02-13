"""Hop-structured neighbor index extraction from sampled batches.

Provides lightweight index structures (``HopIndices``) describing which
nodes belong to which hop for each seed node.  Used by ``Trainer`` to
bridge the gap between ``NeighborLoader`` output and the structured
``(B, N, d_model)`` tensors the transformer branch expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch_geometric.data import HeteroData


@dataclass
class HopIndices:
    """Lightweight index structures for hop-organized neighbor gathering.

    All index tensors use **batch-local** node indices (matching the
    local ordering inside a ``NeighborLoader`` batch).
    """

    contextual_indices: list[list[Tensor]] = field(default_factory=list)
    """``[num_paths][k_hops]`` → ``(B, max_N)`` long."""

    contextual_masks: list[list[Tensor]] = field(default_factory=list)
    """``[num_paths][k_hops]`` → ``(B, max_N)`` bool."""

    contextual_node_types: list[list[str]] = field(default_factory=list)
    """``[num_paths][k_hops]`` → node type string to gather projected features from."""

    temporal_indices: list[Tensor] | None = None
    """``[k_hops_temporal]`` → ``(B, max_N)`` long."""

    temporal_masks: list[Tensor] | None = None
    """``[k_hops_temporal]`` → ``(B, max_N)`` bool."""

    temporal_deltas: list[Tensor] | None = None
    """``[k_hops_temporal]`` → ``(B, max_N)`` float day-gap values."""


class HopExtractor:
    """Extracts hop-organized neighbor indices from a NeighborLoader batch.

    Parameters
    ----------
    contextual_edge_sequences : list[list[tuple[str, str, str]]]
        One sequence per contextual meta-path.  Each sequence is a list
        of edge-type triples describing the alternating hops.
    temporal_edge_types : list[tuple[str, str, str]]
        Edge types used for temporal neighbor lookup (e.g. followed_by +
        rev_followed_by).
    k_hops_contextual : int
        Number of hops per contextual path.
    k_hops_temporal : int
        Number of hops for the temporal track.
    neighbors_per_hop_contextual : int
        Max neighbors retained per hop (contextual).
    neighbors_per_hop_temporal : int
        Max neighbors retained per hop (temporal).
    """

    def __init__(
        self,
        contextual_edge_sequences: list[list[tuple[str, str, str]]],
        temporal_edge_types: list[tuple[str, str, str]],
        k_hops_contextual: int,
        k_hops_temporal: int,
        neighbors_per_hop_contextual: int,
        neighbors_per_hop_temporal: int,
    ) -> None:
        self.contextual_edge_sequences = contextual_edge_sequences
        self.temporal_edge_types = temporal_edge_types
        self.k_hops_contextual = k_hops_contextual
        self.k_hops_temporal = k_hops_temporal
        self.neighbors_per_hop_contextual = neighbors_per_hop_contextual
        self.neighbors_per_hop_temporal = neighbors_per_hop_temporal

    def extract(self, batch: HeteroData, batch_size: int) -> HopIndices:
        """Extract hop indices from a sampled batch.

        Parameters
        ----------
        batch : HeteroData
            A batch produced by ``NeighborLoader``.
        batch_size : int
            Number of seed (target) nodes in the batch.

        Returns
        -------
        HopIndices
            Lightweight index structures for gathering projected features.
        """
        ctx_indices: list[list[Tensor]] = []
        ctx_masks: list[list[Tensor]] = []
        ctx_ntypes: list[list[str]] = []

        for seq in self.contextual_edge_sequences:
            hi, hm, hn = self._extract_contextual_path(
                batch, batch_size, seq
            )
            ctx_indices.append(hi)
            ctx_masks.append(hm)
            ctx_ntypes.append(hn)

        tmp_indices, tmp_masks, tmp_deltas = self._extract_temporal(
            batch, batch_size
        )

        return HopIndices(
            contextual_indices=ctx_indices,
            contextual_masks=ctx_masks,
            contextual_node_types=ctx_ntypes,
            temporal_indices=tmp_indices,
            temporal_masks=tmp_masks,
            temporal_deltas=tmp_deltas,
        )

    # ── Contextual path extraction ────────────────────────────────────────

    def _extract_contextual_path(
        self,
        batch: HeteroData,
        batch_size: int,
        edge_sequence: list[tuple[str, str, str]],
    ) -> tuple[list[Tensor], list[Tensor], list[str]]:
        """Walk a single contextual meta-path, recording indices per hop."""
        N = self.neighbors_per_hop_contextual
        hop_indices: list[Tensor] = []
        hop_masks: list[Tensor] = []
        hop_node_types: list[str] = []

        # Each seed starts with its own frontier
        frontiers: list[set[int]] = [{i} for i in range(batch_size)]

        for k, edge_type in enumerate(edge_sequence):
            _, _, dst_type = edge_type
            hop_node_types.append(dst_type)

            adj = self._build_adjacency(batch, edge_type)

            batch_idx = torch.zeros(batch_size, N, dtype=torch.long)
            batch_mask = torch.zeros(batch_size, N, dtype=torch.bool)
            new_frontiers: list[set[int]] = [set() for _ in range(batch_size)]

            for i in range(batch_size):
                neighbors: set[int] = set()
                for f_node in frontiers[i]:
                    if f_node in adj:
                        neighbors.update(adj[f_node])

                neighbor_list = list(neighbors)
                if len(neighbor_list) > N:
                    neighbor_list = neighbor_list[:N]

                actual = len(neighbor_list)
                if actual > 0:
                    batch_idx[i, :actual] = torch.tensor(
                        neighbor_list, dtype=torch.long
                    )
                    batch_mask[i, :actual] = True

                new_frontiers[i] = set(neighbor_list)

            frontiers = new_frontiers
            hop_indices.append(batch_idx)
            hop_masks.append(batch_mask)

        return hop_indices, hop_masks, hop_node_types

    # ── Temporal extraction ───────────────────────────────────────────────

    def _extract_temporal(
        self,
        batch: HeteroData,
        batch_size: int,
    ) -> tuple[list[Tensor] | None, list[Tensor] | None, list[Tensor] | None]:
        """Extract temporal hop indices with day-gap deltas."""
        if not self.temporal_edge_types or self.k_hops_temporal == 0:
            return None, None, None

        N = self.neighbors_per_hop_temporal
        frontiers: list[set[int]] = [{i} for i in range(batch_size)]

        hop_indices: list[Tensor] = []
        hop_masks: list[Tensor] = []
        hop_deltas: list[Tensor] = []

        for _k in range(self.k_hops_temporal):
            # Build adjacency combining all temporal edge types
            adj: dict[int, list[tuple[int, float]]] = {}
            for et in self.temporal_edge_types:
                if et not in batch.edge_types:
                    continue
                ei = batch[et].edge_index
                ea = None
                store = batch[et]
                if hasattr(store, "edge_attr") and store.edge_attr is not None:
                    ea = store.edge_attr
                for idx in range(ei.shape[1]):
                    s = ei[0, idx].item()
                    d = ei[1, idx].item()
                    delta = ea[idx].view(-1)[0].item() if ea is not None else 0.0
                    adj.setdefault(s, []).append((d, delta))

            batch_idx = torch.zeros(batch_size, N, dtype=torch.long)
            batch_mask = torch.zeros(batch_size, N, dtype=torch.bool)
            batch_delta = torch.zeros(batch_size, N, dtype=torch.float)
            new_frontiers: list[set[int]] = [set() for _ in range(batch_size)]

            for i in range(batch_size):
                # Collect unique neighbors with their deltas
                neighbors: dict[int, float] = {}
                for f_node in frontiers[i]:
                    if f_node in adj:
                        for d, delta in adj[f_node]:
                            if d not in neighbors:
                                neighbors[d] = delta

                neighbor_list = list(neighbors.items())
                if len(neighbor_list) > N:
                    neighbor_list = neighbor_list[:N]

                actual = len(neighbor_list)
                if actual > 0:
                    nodes, deltas = zip(*neighbor_list)
                    batch_idx[i, :actual] = torch.tensor(
                        nodes, dtype=torch.long
                    )
                    batch_mask[i, :actual] = True
                    batch_delta[i, :actual] = torch.tensor(
                        deltas, dtype=torch.float
                    )

                new_frontiers[i] = {n for n, _ in neighbor_list}

            frontiers = new_frontiers
            hop_indices.append(batch_idx)
            hop_masks.append(batch_mask)
            hop_deltas.append(batch_delta)

        return hop_indices, hop_masks, hop_deltas

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_adjacency(
        batch: HeteroData,
        edge_type: tuple[str, str, str],
    ) -> dict[int, list[int]]:
        """Build a source → [destinations] adjacency dict for an edge type."""
        adj: dict[int, list[int]] = {}
        if edge_type not in batch.edge_types:
            return adj
        ei = batch[edge_type].edge_index
        for idx in range(ei.shape[1]):
            s = ei[0, idx].item()
            d = ei[1, idx].item()
            adj.setdefault(s, []).append(d)
        return adj
