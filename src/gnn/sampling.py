"""Dual-track k-hop neighbor sampling with auxiliary subgraph construction.

Provides:
- Auxiliary subgraphs (target-to-target similarity via shared clinical entities)
  for the diffusion module.
- Dual-track neighbor sampling with separate configs for contextual and
  temporal edge types.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterator

import scipy.sparse as sp
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from src.gnn.view_adapter import GraphViewConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_edge_type(
    view_data: HeteroData, relation: str
) -> tuple[str, str, str]:
    """Find the unique edge type whose middle element matches *relation*.

    Raises:
        ValueError: If zero or more than one edge type matches.
    """
    matches = [
        (s, r, d) for s, r, d in view_data.edge_types if r == relation
    ]
    if len(matches) == 0:
        raise ValueError(
            f"No edge type with relation '{relation}' found in view_data. "
            f"Available: {view_data.edge_types}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple edge types match relation '{relation}': {matches}"
        )
    return matches[0]


# ──────────────────────────────────────────────────────────────────────────────
# Subgraph density check
# ──────────────────────────────────────────────────────────────────────────────


def check_subgraph_density(
    edge_index: Tensor, num_nodes: int
) -> dict[str, object]:
    """Compute density metrics for a subgraph among *num_nodes* nodes.

    Returns a dict with keys: ``density``, ``num_edges``, ``category``,
    ``recommendation``.
    """
    num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0

    if num_nodes <= 1:
        density = 0.0
    else:
        density = num_edges / (num_nodes * (num_nodes - 1))

    if density < 0.01:
        category = "too_sparse"
        recommendation = (
            "Auxiliary graph is very sparse (<1%). Consider relaxing "
            "meta-path constraints or using additional entity types."
        )
    elif density <= 0.10:
        category = "optimal"
        recommendation = "Density is in the optimal range (1-10%)."
    elif density <= 0.30:
        category = "dense"
        recommendation = (
            "Auxiliary graph is fairly dense (10-30%). Monitor memory "
            "usage during diffusion."
        )
    else:
        category = "too_dense"
        recommendation = (
            "Auxiliary graph is very dense (>30%). Consider stricter "
            "filtering or higher similarity thresholds."
        )

    return {
        "density": density,
        "num_edges": num_edges,
        "category": category,
        "recommendation": recommendation,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Auxiliary subgraphs
# ──────────────────────────────────────────────────────────────────────────────


def build_auxiliary_subgraphs(
    view_data: HeteroData,
    target_node_type: str,
    meta_paths: list[list[str]],
) -> list[Tensor]:
    """Build target-to-target adjacency matrices from 2-hop meta-paths.

    Each meta-path ``[rel_fwd, rel_bwd]`` produces a binary adjacency among
    target nodes that share at least one intermediate entity.

    Args:
        view_data: Task-specific HeteroData produced by ``GraphViewAdapter``.
        target_node_type: Node type carrying prediction labels.
        meta_paths: List of 2-element relation name lists.

    Returns:
        List of ``[2, E]`` long tensors — one per meta-path.
    """
    results: list[Tensor] = []

    for path in meta_paths:
        assert len(path) == 2, (
            f"Only 2-hop meta-paths supported, got {len(path)}: {path}"
        )

        rel_fwd, rel_bwd = path

        # Resolve relation names to full edge-type tuples
        fwd_type = _resolve_edge_type(view_data, rel_fwd)
        bwd_type = _resolve_edge_type(view_data, rel_bwd)

        # First leg: target → entity
        fwd_ei = view_data[fwd_type].edge_index
        n_target = view_data[target_node_type].num_nodes
        # Infer entity count from the destination of the forward edge
        _entity_type = fwd_type[2]
        n_entity = view_data[_entity_type].num_nodes

        # Build sparse incidence matrix M [n_target × n_entity]
        rows = fwd_ei[0].cpu().numpy()
        cols = fwd_ei[1].cpu().numpy()
        data = [1.0] * len(rows)
        M = sp.csr_matrix((data, (rows, cols)), shape=(n_target, n_entity))

        # G = binarize(M @ M^T)
        G = M @ M.T
        G = (G > 0).astype(float)
        G = sp.lil_matrix(G)
        G.setdiag(0)
        G = G.tocoo()
        G.eliminate_zeros()

        edge_index = torch.tensor(
            [G.row.tolist(), G.col.tolist()], dtype=torch.long
        )

        # Log density
        info = check_subgraph_density(edge_index, n_target)
        logger.info(
            "Auxiliary subgraph [%s]: %d edges, density=%.4f (%s)",
            "→".join(path),
            info["num_edges"],
            info["density"],
            info["category"],
        )
        if info["category"] == "too_dense":
            logger.warning(
                "Auxiliary subgraph [%s] is too dense (%.1f%%). %s",
                "→".join(path),
                info["density"] * 100,
                info["recommendation"],
            )
        elif info["category"] == "too_sparse":
            logger.warning(
                "Auxiliary subgraph [%s] is too sparse (%.4f%%). %s",
                "→".join(path),
                info["density"] * 100,
                info["recommendation"],
            )

        results.append(edge_index)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Sampling configs
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SamplingTrackConfig:
    """Configuration for a single sampling track (contextual or temporal)."""

    edge_types: list[tuple[str, str, str]] = field(default_factory=list)
    k_hops: int = 4
    neighbors_per_hop: int = 32
    weighting: str = "uniform"
    recency_decay_alpha: float = 0.01


@dataclass
class DualSamplingConfig:
    """Configuration for dual-track neighbor sampling."""

    contextual: SamplingTrackConfig = field(
        default_factory=lambda: SamplingTrackConfig(
            k_hops=4, neighbors_per_hop=32, weighting="uniform"
        )
    )
    temporal: SamplingTrackConfig = field(
        default_factory=lambda: SamplingTrackConfig(
            k_hops=2, neighbors_per_hop=8, weighting="recency"
        )
    )

    @classmethod
    def from_view_config(
        cls, view_config: GraphViewConfig, view_data: HeteroData
    ) -> DualSamplingConfig:
        """Derive sampling config from a ``GraphViewConfig`` and view data.

        - **Contextual track**: uses meta-path edge types if provided,
          otherwise all non-temporal edge types from the view.
        - **Temporal track**: ``followed_by`` + reverse if
          ``include_temporal_track`` is True, otherwise empty.
        """
        target = view_config.target_node_type

        # --- Contextual track ---
        if view_config.meta_paths:
            ctx_edge_types: list[tuple[str, str, str]] = []
            for path in view_config.meta_paths:
                for rel in path:
                    try:
                        et = _resolve_edge_type(view_data, rel)
                        if et not in ctx_edge_types:
                            ctx_edge_types.append(et)
                    except ValueError:
                        pass
        else:
            temporal_relations = {"followed_by", "rev_followed_by"}
            ctx_edge_types = [
                (s, r, d)
                for s, r, d in view_data.edge_types
                if r not in temporal_relations
            ]

        contextual = SamplingTrackConfig(
            edge_types=ctx_edge_types,
            k_hops=4,
            neighbors_per_hop=32,
            weighting="uniform",
        )

        # --- Temporal track ---
        if view_config.include_temporal_track:
            temporal_types: list[tuple[str, str, str]] = []
            fb = (target, "followed_by", target)
            rev_fb = (target, "rev_followed_by", target)
            if fb in view_data.edge_types:
                temporal_types.append(fb)
            if rev_fb in view_data.edge_types:
                temporal_types.append(rev_fb)
            temporal = SamplingTrackConfig(
                edge_types=temporal_types,
                k_hops=2,
                neighbors_per_hop=8,
                weighting="recency",
            )
        else:
            temporal = SamplingTrackConfig(
                edge_types=[],
                k_hops=2,
                neighbors_per_hop=8,
                weighting="recency",
            )

        return cls(contextual=contextual, temporal=temporal)


# ──────────────────────────────────────────────────────────────────────────────
# Recency pre-filtering
# ──────────────────────────────────────────────────────────────────────────────


def _topk_recency_filter(
    edge_index: Tensor, edge_attr: Tensor, k: int
) -> tuple[Tensor, Tensor]:
    """Per-source-node top-K filtering by smallest ``edge_attr`` value.

    For each source node, keeps only the *k* edges with the smallest
    attribute values (smallest gap = most recent).

    Args:
        edge_index: ``[2, E]`` edge index tensor.
        edge_attr: ``[E, 1]`` or ``[E]`` edge attribute tensor.
        k: Maximum number of edges to keep per source node.

    Returns:
        Filtered ``(edge_index, edge_attr)`` with the same shapes but
        potentially fewer edges.
    """
    attr_flat = edge_attr.view(-1)
    src_nodes = edge_index[0]
    unique_srcs = src_nodes.unique()

    keep_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)

    for src in unique_srcs:
        src_mask = src_nodes == src.item()
        src_indices = torch.where(src_mask)[0]
        src_attrs = attr_flat[src_indices]

        if len(src_indices) <= k:
            keep_mask[src_indices] = True
        else:
            _, topk_local = torch.topk(src_attrs, k, largest=False)
            keep_mask[src_indices[topk_local]] = True

    filtered_ei = edge_index[:, keep_mask]
    filtered_attr = edge_attr[keep_mask]

    return filtered_ei, filtered_attr


# ──────────────────────────────────────────────────────────────────────────────
# Dual-track sampler
# ──────────────────────────────────────────────────────────────────────────────


class DualTrackSampler:
    """Dual-track neighbor sampler for contextual and temporal edges.

    Creates two ``NeighborLoader`` instances per split — one for each track.
    The contextual loader samples clinical entity edges, while the temporal
    loader samples ``followed_by`` edges with optional recency pre-filtering.

    Args:
        view_data: Task-specific HeteroData from ``GraphViewAdapter.apply()``.
        config: ``DualSamplingConfig`` with per-track settings.
        batch_size: Number of target nodes per batch.
    """

    def __init__(
        self,
        view_data: HeteroData,
        config: DualSamplingConfig,
        batch_size: int = 64,
    ) -> None:
        self.view_data = view_data
        self.config = config
        self.batch_size = batch_size

        # Detect target node type (node type with train_mask)
        self.target_node_type = self._detect_target_node_type()

        # Build unified num_neighbors dict combining both tracks
        self._num_neighbors = self._build_unified_num_neighbors()

        # Pre-filter temporal data if recency weighting is active
        self._temporal_data = self._prepare_temporal_data()

    def _detect_target_node_type(self) -> str:
        for nt in self.view_data.node_types:
            store = self.view_data[nt]
            if hasattr(store, "train_mask") and store.train_mask is not None:
                return nt
        raise ValueError("No node type with train_mask found in view_data")

    def _build_unified_num_neighbors(
        self,
    ) -> dict[tuple[str, str, str], list[int]]:
        """Build a single ``num_neighbors`` dict combining both tracks.

        Contextual edge types get ``[neighbors_per_hop_ctx] * k_hops_ctx``
        (padded to ``max_hops`` with zeros).  Temporal edge types get
        ``[neighbors_per_hop_tmp] * k_hops_tmp`` padded likewise.
        All other edge types get ``[0] * max_hops``.
        """
        ctx = self.config.contextual
        tmp = self.config.temporal
        max_hops = max(ctx.k_hops, tmp.k_hops)

        ctx_active = set(tuple(et) for et in ctx.edge_types)
        tmp_active = set(tuple(et) for et in tmp.edge_types)

        result: dict[tuple[str, str, str], list[int]] = {}
        for et in self.view_data.edge_types:
            if et in ctx_active:
                ns = [ctx.neighbors_per_hop] * ctx.k_hops
                ns += [0] * (max_hops - ctx.k_hops)
            elif et in tmp_active:
                ns = [tmp.neighbors_per_hop] * tmp.k_hops
                ns += [0] * (max_hops - tmp.k_hops)
            else:
                ns = [0] * max_hops
            result[et] = ns
        return result

    def _prepare_temporal_data(self) -> HeteroData:
        """Create a copy of view_data with recency-filtered temporal edges."""
        track = self.config.temporal
        if not track.edge_types or track.weighting != "recency":
            return self.view_data

        import copy
        filtered = copy.copy(self.view_data)
        # We need a proper deep-ish copy of edge stores
        filtered = self.view_data.clone()

        k = track.neighbors_per_hop * track.k_hops

        for et in track.edge_types:
            if et in filtered.edge_types:
                store = filtered[et]
                if (
                    hasattr(store, "edge_attr")
                    and store.edge_attr is not None
                ):
                    new_ei, new_attr = _topk_recency_filter(
                        store.edge_index, store.edge_attr, k
                    )
                    filtered[et].edge_index = new_ei
                    filtered[et].edge_attr = new_attr

        return filtered

    def _make_loader(
        self,
        data: HeteroData,
        num_neighbors: dict[tuple[str, str, str], list[int]],
        mask_attr: str,
        shuffle: bool = False,
    ) -> NeighborLoader | None:
        """Create a NeighborLoader for the given mask, or None if no edges."""
        # Check if any active neighbors
        has_active = any(
            any(n > 0 for n in ns)
            for ns in num_neighbors.values()
        )
        if not has_active:
            return None

        target_store = data[self.target_node_type]
        if not hasattr(target_store, mask_attr):
            return None
        mask = getattr(target_store, mask_attr)
        if mask is None or mask.sum() == 0:
            return None

        input_nodes = (self.target_node_type, mask)

        return NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            input_nodes=input_nodes,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def _get_loader(
        self, mask_attr: str, shuffle: bool = False
    ) -> NeighborLoader | None:
        """Create a single unified NeighborLoader for both tracks."""
        return self._make_loader(
            self._temporal_data, self._num_neighbors, mask_attr, shuffle
        )

    def get_train_loader(self) -> Iterator[HeteroData]:
        """Return an iterator over training batches."""
        loader = self._get_loader("train_mask", shuffle=True)
        return iter(loader) if loader is not None else iter([])

    def get_val_loader(self) -> Iterator[HeteroData]:
        """Return an iterator over validation batches."""
        loader = self._get_loader("val_mask", shuffle=False)
        return iter(loader) if loader is not None else iter([])

    def get_test_loader(self) -> Iterator[HeteroData]:
        """Return an iterator over test batches."""
        loader = self._get_loader("test_mask", shuffle=False)
        return iter(loader) if loader is not None else iter([])

    def __len__(self) -> int:
        """Number of batches per epoch (based on training set)."""
        target_store = self.view_data[self.target_node_type]
        if hasattr(target_store, "train_mask") and target_store.train_mask is not None:
            n_train = int(target_store.train_mask.sum().item())
        else:
            n_train = target_store.num_nodes
        return math.ceil(n_train / self.batch_size)
