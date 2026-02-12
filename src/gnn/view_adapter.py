"""Configurable graph view adapters for task-specific subgraph selection.

Produces read-only, task-specific subgraphs from the full HeteroData graph
exported by :mod:`src.gnn.graph_export`.  Different prediction tasks need
different "views" — for 30-day readmission we collapse ICUStay/ICUDay
intermediates into shortcut edges; for ICU mortality we keep them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SHORTCUT_NAMES: dict[str, str] = {
    "drug": "prescribed",
    "lab": "lab_result",
    "vital": "vital_sign",
    "microbiology": "micro_result",
}

# Full hierarchical edge chain from admission down to clinical event nodes.
# Diagnosis is excluded — it connects directly to admission.
_FULL_CHAIN: list[tuple[str, str, str]] = [
    ("admission", "contains_icu_stay", "icu_stay"),
    ("icu_stay", "has_icu_day", "icu_day"),
    ("icu_day", "has_event", "{entity}"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphViewConfig:
    """Configuration for a task-specific graph view.

    Attributes:
        target_node_type: The node type that carries prediction labels.
        label_key: Attribute name on the target node for the primary label.
        active_entity_types: Clinical entity types to include in the view.
        collapse_rules: Maps intermediate node types to handling mode
            (``"collapse"`` merges them into shortcut edges).
        meta_paths: Reserved for future meta-path-based message passing.
        include_temporal_track: Whether to include ``followed_by`` edges.
    """

    target_node_type: str = "admission"
    label_key: str = "readmitted_30d"
    active_entity_types: list[str] = field(
        default_factory=lambda: ["drug", "diagnosis"]
    )
    collapse_rules: dict[str, str] = field(
        default_factory=lambda: {"icu_stay": "collapse", "icu_day": "collapse"}
    )
    meta_paths: list[list[str]] = field(default_factory=list)
    include_temporal_track: bool = True

    @classmethod
    def readmission_default(cls) -> GraphViewConfig:
        """Minimal view for 30-day readmission: drugs + diagnoses only."""
        return cls(
            target_node_type="admission",
            label_key="readmitted_30d",
            active_entity_types=["drug", "diagnosis"],
            collapse_rules={"icu_stay": "collapse", "icu_day": "collapse"},
            include_temporal_track=True,
        )

    @classmethod
    def readmission_extended(cls) -> GraphViewConfig:
        """Extended view for readmission: all five clinical entity types."""
        return cls(
            target_node_type="admission",
            label_key="readmitted_30d",
            active_entity_types=[
                "drug", "diagnosis", "lab", "vital", "microbiology",
            ],
            collapse_rules={"icu_stay": "collapse", "icu_day": "collapse"},
            include_temporal_track=True,
        )

    @classmethod
    def icu_mortality(cls) -> GraphViewConfig:
        """ICU-level view preserving the full stay→day hierarchy."""
        return cls(
            target_node_type="icu_stay",
            label_key="icu_mortality",
            active_entity_types=[
                "drug", "diagnosis", "lab", "vital", "microbiology",
            ],
            collapse_rules={"icu_day": "traverse"},
            include_temporal_track=False,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────────────────────

class GraphViewAdapter:
    """Produces a task-specific HeteroData view from the full graph.

    The adapter is **read-only** — the original HeteroData is never modified.
    All tensors in the returned view are ``.clone()``d copies.

    Args:
        config: A :class:`GraphViewConfig` describing the desired view.
    """

    def __init__(self, config: GraphViewConfig) -> None:
        self.config = config

    # ── Chain resolution ─────────────────────────────────────────────────

    def _resolve_chain(
        self, entity_type: str
    ) -> list[tuple[str, str, str]]:
        """Return the edge chain from target_node_type to *entity_type*.

        - **diagnosis** connects directly: ``(admission, has_diagnosis, diagnosis)``.
        - All others traverse through intermediates: admission → icu_stay →
          icu_day → entity.
        - When ``target_node_type="icu_stay"`` the first hop is trimmed.
        """
        if entity_type == "diagnosis":
            return [("admission", "has_diagnosis", "diagnosis")]

        chain = [
            (src, rel, dst.replace("{entity}", entity_type))
            for src, rel, dst in _FULL_CHAIN
        ]

        # If target is icu_stay, skip the admission→icu_stay hop.
        if self.config.target_node_type == "icu_stay":
            chain = chain[1:]

        return chain

    # ── Edge composition (explicit join) ─────────────────────────────────

    @staticmethod
    def get_edge_chain(
        full_data: HeteroData,
        chain: list[tuple[str, str, str]],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compose a chain of edges into a single shortcut edge_index.

        For a chain ``A→B→C→D`` the result is the set of ``(a, d)`` pairs
        reachable by traversing all three hops.  Edge attributes are
        propagated from the **last** edge in the chain; duplicated
        ``(src, dst)`` pairs are deduplicated with mean-aggregated attrs.

        Returns:
            ``(edge_index [2, E'], edge_attr [E', F] or None)``
        """
        first_src, first_rel, first_dst = chain[0]
        ei = full_data[first_src, first_rel, first_dst].edge_index
        # current_pairs: list of (src, current_dst)
        current_src = ei[0].tolist()
        current_dst = ei[1].tolist()

        last_attr = None
        if hasattr(full_data[first_src, first_rel, first_dst], "edge_attr"):
            last_attr = full_data[first_src, first_rel, first_dst].edge_attr

        for hop_src, hop_rel, hop_dst in chain[1:]:
            hop_ei = full_data[hop_src, hop_rel, hop_dst].edge_index
            hop_attr_store = full_data[hop_src, hop_rel, hop_dst]
            hop_attr = None
            if (
                hasattr(hop_attr_store, "edge_attr")
                and hop_attr_store.edge_attr is not None
            ):
                hop_attr = hop_attr_store.edge_attr

            # Build inverted index: next_src_node → [(next_dst_node, edge_idx)]
            inv: dict[int, list[tuple[int, int]]] = {}
            for idx in range(hop_ei.shape[1]):
                s = hop_ei[0, idx].item()
                d = hop_ei[1, idx].item()
                inv.setdefault(s, []).append((d, idx))

            new_src: list[int] = []
            new_dst: list[int] = []
            new_edge_indices: list[int] = []  # indices into hop's edges

            for orig_s, cur_d in zip(current_src, current_dst):
                if cur_d in inv:
                    for next_d, edge_idx in inv[cur_d]:
                        new_src.append(orig_s)
                        new_dst.append(next_d)
                        new_edge_indices.append(edge_idx)

            current_src = new_src
            current_dst = new_dst

            # Track attr from last hop
            if hop_attr is not None:
                last_attr = hop_attr
                # Remap to the joined indices
                last_attr_indices = new_edge_indices
            else:
                last_attr = None
                last_attr_indices = None

        if not current_src:
            return torch.zeros(2, 0, dtype=torch.long), None

        # Deduplicate (src, dst) pairs, mean-aggregate attrs
        pair_attrs: dict[tuple[int, int], list[torch.Tensor]] = {}
        has_attr = last_attr is not None

        for i, (s, d) in enumerate(zip(current_src, current_dst)):
            key = (s, d)
            if has_attr:
                # Use last_attr_indices if we have a remapped set, otherwise i
                attr_idx = (
                    last_attr_indices[i]
                    if last_attr_indices is not None
                    else i
                )
                pair_attrs.setdefault(key, []).append(
                    last_attr[attr_idx].clone()
                )
            else:
                pair_attrs.setdefault(key, [])

        src_list: list[int] = []
        dst_list: list[int] = []
        attr_list: list[torch.Tensor] = []

        for (s, d), attrs in pair_attrs.items():
            src_list.append(s)
            dst_list.append(d)
            if has_attr and attrs:
                attr_list.append(torch.stack(attrs).mean(dim=0))

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.stack(attr_list) if attr_list else None

        return edge_index, edge_attr

    # ── Main apply ───────────────────────────────────────────────────────

    def apply(self, full_data: HeteroData) -> HeteroData:
        """Produce a task-specific view from *full_data*.

        The returned :class:`HeteroData` is a standalone copy — the original
        graph is not modified.
        """
        cfg = self.config
        view = HeteroData()

        # 1. Copy target node store
        target = cfg.target_node_type
        self._copy_node(full_data, view, target)
        # Copy labels and masks
        target_store = full_data[target]
        for attr in ("y", "y_60d", "train_mask", "val_mask", "test_mask"):
            if hasattr(target_store, attr) and getattr(target_store, attr) is not None:
                setattr(view[target], attr, getattr(target_store, attr).clone())

        # 2. Copy active entity nodes
        for entity in cfg.active_entity_types:
            self._copy_node(full_data, view, entity)

        # 3. Process each active entity's chain
        for entity in cfg.active_entity_types:
            chain = self._resolve_chain(entity)
            self._process_chain(full_data, view, entity, chain)

        # 4. Temporal track (followed_by)
        if cfg.include_temporal_track:
            fb_key = ("admission", "followed_by", "admission")
            if fb_key in full_data.edge_types:
                fb_store = full_data[fb_key]
                view[fb_key].edge_index = fb_store.edge_index.clone()
                if (
                    hasattr(fb_store, "edge_attr")
                    and fb_store.edge_attr is not None
                ):
                    view[fb_key].edge_attr = fb_store.edge_attr.clone()

        # 5. Reverse edges
        self._add_reverse_edges(view)

        return view

    # ── Internals ────────────────────────────────────────────────────────

    @staticmethod
    def _copy_node(
        full_data: HeteroData, view: HeteroData, node_type: str
    ) -> None:
        """Copy node features (``x``) for *node_type* into the view."""
        store = full_data[node_type]
        if hasattr(store, "x") and store.x is not None:
            view[node_type].x = store.x.clone()
        else:
            view[node_type].num_nodes = store.num_nodes

    def _process_chain(
        self,
        full_data: HeteroData,
        view: HeteroData,
        entity_type: str,
        chain: list[tuple[str, str, str]],
    ) -> None:
        """Handle collapse / traverse logic for a single entity chain."""
        cfg = self.config

        if len(chain) == 1:
            # Direct edge (e.g. diagnosis) — just copy it.
            src, rel, dst = chain[0]
            if (src, rel, dst) in full_data.edge_types:
                store = full_data[src, rel, dst]
                view[src, rel, dst].edge_index = store.edge_index.clone()
                if (
                    hasattr(store, "edge_attr")
                    and store.edge_attr is not None
                ):
                    view[src, rel, dst].edge_attr = store.edge_attr.clone()
            return

        # Build ordered segments: ("collapse", [edges]) or ("traverse", [edge]).
        # A collapse run accumulates edges whose destination is "collapse".
        # The run ends (inclusive) when we reach a non-collapsed destination
        # (either a traverse intermediate or the final entity).
        segments: list[tuple[str, list[tuple[str, str, str]]]] = []
        collapse_buf: list[tuple[str, str, str]] = []

        for i, edge in enumerate(chain):
            _src, _rel, dst = edge
            is_last = i == len(chain) - 1

            dst_collapsed = (
                not is_last
                and dst in cfg.collapse_rules
                and cfg.collapse_rules[dst] == "collapse"
            )

            if dst_collapsed:
                collapse_buf.append(edge)
            else:
                # dst is not collapsed (traverse intermediate or entity)
                if collapse_buf:
                    collapse_buf.append(edge)
                    segments.append(("collapse", list(collapse_buf)))
                    collapse_buf = []
                else:
                    segments.append(("traverse", [edge]))

        for seg_type, edges in segments:
            if seg_type == "collapse":
                shortcut_dst = edges[-1][2]
                shortcut_src = edges[0][0]

                if shortcut_dst == entity_type:
                    shortcut_rel = SHORTCUT_NAMES.get(
                        entity_type, f"has_{entity_type}"
                    )
                else:
                    # Shortcut to a traverse intermediate (e.g. icu_day)
                    shortcut_rel = edges[-1][1]
                    self._copy_node(full_data, view, shortcut_dst)

                shortcut_key = (shortcut_src, shortcut_rel, shortcut_dst)
                edge_index, edge_attr = self.get_edge_chain(full_data, edges)
                if edge_index.shape[1] > 0:
                    view[shortcut_key].edge_index = edge_index
                    if edge_attr is not None:
                        view[shortcut_key].edge_attr = edge_attr
            else:
                # Traverse — copy intermediate nodes + original edges.
                for src, rel, dst in edges:
                    if dst != entity_type:
                        self._copy_node(full_data, view, dst)
                    if (src, rel, dst) in full_data.edge_types:
                        store = full_data[src, rel, dst]
                        view[src, rel, dst].edge_index = store.edge_index.clone()
                        if (
                            hasattr(store, "edge_attr")
                            and store.edge_attr is not None
                        ):
                            view[src, rel, dst].edge_attr = store.edge_attr.clone()

    @staticmethod
    def _add_reverse_edges(view: HeteroData) -> None:
        """Add ``rev_*`` reverse for every forward edge in the view."""
        edge_types = list(view.edge_types)
        for src, rel, dst in edge_types:
            ei = view[src, rel, dst].edge_index
            rev_rel = f"rev_{rel}"
            view[dst, rev_rel, src].edge_index = torch.stack(
                [ei[1], ei[0]], dim=0
            ).clone()
            if (
                hasattr(view[src, rel, dst], "edge_attr")
                and view[src, rel, dst].edge_attr is not None
            ):
                view[dst, rev_rel, src].edge_attr = (
                    view[src, rel, dst].edge_attr.clone()
                )
