"""Tests for src.gnn.view_adapter — GraphViewAdapter and GraphViewConfig."""

import copy

import pytest
import torch
from torch_geometric.data import HeteroData

from src.gnn.view_adapter import (
    SHORTCUT_NAMES,
    GraphViewAdapter,
    GraphViewConfig,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mini_hetero_data() -> HeteroData:
    """Manually constructed HeteroData (no RDF dependency).

    Topology:
        2 admissions, 2 icu_stays, 4 icu_days, 3 drugs, 2 diagnoses

    Edges:
        adm0 → stay0
        stay0 → day0, day1
        day0 → drug0          (Δt = 1.0)
        day1 → drug1          (Δt = 2.0)

        adm1 → stay1
        stay1 → day2, day3
        day2 → drug0          (Δt = 3.0)
        day3 → drug2          (Δt = 4.0)

        adm0 → diag0
        adm1 → diag1

        adm0 → (followed_by) → adm1  (gap = 15.0)

    After collapsing icu_stay + icu_day:
        adm0 → drug0 (via day0), adm0 → drug1 (via day1)
        adm1 → drug0 (via day2), adm1 → drug2 (via day3)
        = 4 shortcut edges
    """
    data = HeteroData()

    # ── Node features ────────────────────────────────────────────────────
    data["admission"].x = torch.randn(2, 5)
    data["admission"].y = torch.tensor([1.0, 0.0])
    data["admission"].y_60d = torch.tensor([1.0, 1.0])
    data["admission"].train_mask = torch.tensor([True, False])
    data["admission"].val_mask = torch.tensor([False, True])
    data["admission"].test_mask = torch.tensor([False, False])

    data["icu_stay"].x = torch.randn(2, 2)
    data["icu_day"].x = torch.randn(4, 2)
    data["drug"].x = torch.randn(3, 768)
    data["diagnosis"].x = torch.randn(2, 768)

    # ── Structural edges ─────────────────────────────────────────────────
    # admission → contains_icu_stay → icu_stay
    data["admission", "contains_icu_stay", "icu_stay"].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )

    # icu_stay → has_icu_day → icu_day
    data["icu_stay", "has_icu_day", "icu_day"].edge_index = torch.tensor(
        [[0, 0, 1, 1], [0, 1, 2, 3]], dtype=torch.long
    )

    # icu_day → has_event → drug  (with Δt edge_attr)
    data["icu_day", "has_event", "drug"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 0, 2]], dtype=torch.long
    )
    data["icu_day", "has_event", "drug"].edge_attr = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0]]
    )

    # admission → has_diagnosis → diagnosis
    data["admission", "has_diagnosis", "diagnosis"].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )

    # admission → followed_by → admission
    data["admission", "followed_by", "admission"].edge_index = torch.tensor(
        [[0], [1]], dtype=torch.long
    )
    data["admission", "followed_by", "admission"].edge_attr = torch.tensor(
        [[15.0]]
    )

    # ── Reverse edges (mirror of graph_export._add_reverse_edges) ────────
    for src, rel, dst in list(data.edge_types):
        ei = data[src, rel, dst].edge_index
        rev_rel = f"rev_{rel}"
        data[dst, rev_rel, src].edge_index = torch.stack([ei[1], ei[0]], dim=0)
        if (
            hasattr(data[src, rel, dst], "edge_attr")
            and data[src, rel, dst].edge_attr is not None
        ):
            data[dst, rev_rel, src].edge_attr = data[src, rel, dst].edge_attr.clone()

    return data


# ──────────────────────────────────────────────────────────────────────────────
# 1. Config presets
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigPresets:
    def test_readmission_default_target(self):
        cfg = GraphViewConfig.readmission_default()
        assert cfg.target_node_type == "admission"

    def test_readmission_default_active_entities(self):
        cfg = GraphViewConfig.readmission_default()
        assert set(cfg.active_entity_types) == {"drug", "diagnosis"}

    def test_readmission_default_collapse_rules(self):
        cfg = GraphViewConfig.readmission_default()
        assert cfg.collapse_rules == {
            "icu_stay": "collapse",
            "icu_day": "collapse",
        }

    def test_readmission_default_temporal(self):
        cfg = GraphViewConfig.readmission_default()
        assert cfg.include_temporal_track is True

    def test_readmission_extended_all_entities(self):
        cfg = GraphViewConfig.readmission_extended()
        assert set(cfg.active_entity_types) == {
            "drug", "diagnosis", "lab", "vital", "microbiology",
        }

    def test_icu_mortality_target(self):
        cfg = GraphViewConfig.icu_mortality()
        assert cfg.target_node_type == "icu_stay"
        assert cfg.label_key == "icu_mortality"

    def test_icu_mortality_traverse(self):
        cfg = GraphViewConfig.icu_mortality()
        assert cfg.collapse_rules == {"icu_day": "traverse"}

    def test_icu_mortality_no_temporal(self):
        cfg = GraphViewConfig.icu_mortality()
        assert cfg.include_temporal_track is False


# ──────────────────────────────────────────────────────────────────────────────
# 2. Collapse shortcut edges
# ──────────────────────────────────────────────────────────────────────────────

class TestCollapseShortcutEdges:
    def test_shortcut_edges_exist(self, mini_hetero_data):
        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        shortcut_key = ("admission", "prescribed", "drug")
        assert shortcut_key in view.edge_types

    def test_shortcut_edge_count(self, mini_hetero_data):
        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        ei = view["admission", "prescribed", "drug"].edge_index
        assert ei.shape[1] == 4

    def test_intermediate_nodes_absent(self, mini_hetero_data):
        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        assert "icu_stay" not in view.node_types
        assert "icu_day" not in view.node_types

    def test_shortcut_src_dst_values(self, mini_hetero_data):
        """Verify the specific (admission, drug) pairs in the shortcut."""
        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        ei = view["admission", "prescribed", "drug"].edge_index
        pairs = set(zip(ei[0].tolist(), ei[1].tolist()))
        expected = {(0, 0), (0, 1), (1, 0), (1, 2)}
        assert pairs == expected

    def test_diagnosis_direct_edge_preserved(self, mini_hetero_data):
        """Diagnosis edges are direct and should be kept as-is."""
        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        assert ("admission", "has_diagnosis", "diagnosis") in view.edge_types
        ei = view["admission", "has_diagnosis", "diagnosis"].edge_index
        assert ei.shape[1] == 2


# ──────────────────────────────────────────────────────────────────────────────
# 3. Temporal attr propagation
# ──────────────────────────────────────────────────────────────────────────────

class TestTemporalAttrPropagation:
    def test_shortcut_carries_delta_t(self, mini_hetero_data):
        """Shortcut edges should carry Δt from the last hop (icu_day→drug)."""
        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        attr = view["admission", "prescribed", "drug"].edge_attr
        assert attr is not None
        assert attr.shape[1] == 1

    def test_delta_t_values(self, mini_hetero_data):
        """Each shortcut edge should have the Δt from its source icu_day→drug edge.

        Unique (adm, drug) pairs and their Δt values:
            (0, 0) → 1.0  (day0→drug0)
            (0, 1) → 2.0  (day1→drug1)
            (1, 0) → 3.0  (day2→drug0)
            (1, 2) → 4.0  (day3→drug2)
        """
        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        ei = view["admission", "prescribed", "drug"].edge_index
        attr = view["admission", "prescribed", "drug"].edge_attr

        pair_to_dt = {}
        for i in range(ei.shape[1]):
            pair = (ei[0, i].item(), ei[1, i].item())
            pair_to_dt[pair] = attr[i, 0].item()

        assert pair_to_dt[(0, 0)] == pytest.approx(1.0)
        assert pair_to_dt[(0, 1)] == pytest.approx(2.0)
        assert pair_to_dt[(1, 0)] == pytest.approx(3.0)
        assert pair_to_dt[(1, 2)] == pytest.approx(4.0)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Traverse mode
# ──────────────────────────────────────────────────────────────────────────────

class TestTraverseMode:
    def test_traverse_preserves_icu_day(self, mini_hetero_data):
        """With icu_day="traverse", icu_day nodes should be preserved."""
        cfg = GraphViewConfig(
            target_node_type="admission",
            label_key="readmitted_30d",
            active_entity_types=["drug"],
            collapse_rules={"icu_stay": "collapse", "icu_day": "traverse"},
            include_temporal_track=False,
        )
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        assert "icu_day" in view.node_types

    def test_traverse_keeps_original_edges(self, mini_hetero_data):
        """Traverse mode should keep the original icu_day→drug edges."""
        cfg = GraphViewConfig(
            target_node_type="admission",
            label_key="readmitted_30d",
            active_entity_types=["drug"],
            collapse_rules={"icu_stay": "collapse", "icu_day": "traverse"},
            include_temporal_track=False,
        )
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        assert ("icu_day", "has_event", "drug") in view.edge_types
        ei = view["icu_day", "has_event", "drug"].edge_index
        assert ei.shape[1] == 4

    def test_traverse_creates_shortcut_for_collapsed(self, mini_hetero_data):
        """Collapsing icu_stay produces admission→icu_day shortcut,
        while icu_day→drug is kept as a traverse edge."""
        cfg = GraphViewConfig(
            target_node_type="admission",
            label_key="readmitted_30d",
            active_entity_types=["drug"],
            collapse_rules={"icu_stay": "collapse", "icu_day": "traverse"},
            include_temporal_track=False,
        )
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        # icu_stay is collapsed — not in the view
        assert "icu_stay" not in view.node_types

        # Shortcut: admission→has_icu_day→icu_day (collapsing icu_stay)
        assert ("admission", "has_icu_day", "icu_day") in view.edge_types
        ei = view["admission", "has_icu_day", "icu_day"].edge_index
        # 2 admissions × 2 days each = 4 shortcut edges
        assert ei.shape[1] == 4


# ──────────────────────────────────────────────────────────────────────────────
# 5. Active entity filtering
# ──────────────────────────────────────────────────────────────────────────────

class TestActiveEntityFiltering:
    def test_only_active_entities_present(self, mini_hetero_data):
        """Config with active=["drug"] should exclude diagnosis."""
        cfg = GraphViewConfig(
            target_node_type="admission",
            label_key="readmitted_30d",
            active_entity_types=["drug"],
            collapse_rules={"icu_stay": "collapse", "icu_day": "collapse"},
            include_temporal_track=False,
        )
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        assert "drug" in view.node_types
        assert "diagnosis" not in view.node_types

    def test_no_diagnosis_edges(self, mini_hetero_data):
        """Diagnosis edges should be absent when diagnosis is not active."""
        cfg = GraphViewConfig(
            target_node_type="admission",
            label_key="readmitted_30d",
            active_entity_types=["drug"],
            collapse_rules={"icu_stay": "collapse", "icu_day": "collapse"},
            include_temporal_track=False,
        )
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        for src, rel, dst in view.edge_types:
            assert "diagnosis" not in (src, dst)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Labels and masks preserved
# ──────────────────────────────────────────────────────────────────────────────

class TestLabelsAndMasksPreserved:
    def test_y_preserved(self, mini_hetero_data):
        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        assert torch.equal(view["admission"].y, mini_hetero_data["admission"].y)

    def test_y_60d_preserved(self, mini_hetero_data):
        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        assert torch.equal(
            view["admission"].y_60d, mini_hetero_data["admission"].y_60d
        )

    def test_masks_preserved(self, mini_hetero_data):
        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        assert torch.equal(
            view["admission"].train_mask,
            mini_hetero_data["admission"].train_mask,
        )
        assert torch.equal(
            view["admission"].val_mask,
            mini_hetero_data["admission"].val_mask,
        )
        assert torch.equal(
            view["admission"].test_mask,
            mini_hetero_data["admission"].test_mask,
        )

    def test_labels_are_cloned(self, mini_hetero_data):
        """Modifying the view's labels should not affect the original."""
        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        original_y = mini_hetero_data["admission"].y.clone()
        view["admission"].y[0] = 999.0
        assert torch.equal(mini_hetero_data["admission"].y, original_y)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Reverse edges generated
# ──────────────────────────────────────────────────────────────────────────────

class TestReverseEdgesGenerated:
    def test_every_forward_has_reverse(self, mini_hetero_data):
        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        forward = [
            (s, r, d) for s, r, d in view.edge_types if not r.startswith("rev_")
        ]
        for src, rel, dst in forward:
            rev_key = (dst, f"rev_{rel}", src)
            assert rev_key in view.edge_types, f"Missing reverse for ({src},{rel},{dst})"

    def test_reverse_edge_count_matches(self, mini_hetero_data):
        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        forward = [
            (s, r, d) for s, r, d in view.edge_types if not r.startswith("rev_")
        ]
        for src, rel, dst in forward:
            fwd_n = view[src, rel, dst].edge_index.shape[1]
            rev_n = view[dst, f"rev_{rel}", src].edge_index.shape[1]
            assert fwd_n == rev_n

    def test_reverse_edge_swapped(self, mini_hetero_data):
        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        forward = [
            (s, r, d) for s, r, d in view.edge_types if not r.startswith("rev_")
        ]
        for src, rel, dst in forward:
            fwd_ei = view[src, rel, dst].edge_index
            rev_ei = view[dst, f"rev_{rel}", src].edge_index
            assert torch.equal(fwd_ei[0], rev_ei[1])
            assert torch.equal(fwd_ei[1], rev_ei[0])

    def test_reverse_attr_cloned(self, mini_hetero_data):
        """Reverse edges should carry cloned edge_attr."""
        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        # The followed_by edge has edge_attr
        if ("admission", "followed_by", "admission") in view.edge_types:
            fwd_attr = view[
                "admission", "followed_by", "admission"
            ].edge_attr
            rev_attr = view[
                "admission", "rev_followed_by", "admission"
            ].edge_attr
            assert torch.equal(fwd_attr, rev_attr)


# ──────────────────────────────────────────────────────────────────────────────
# 8. Original unmodified
# ──────────────────────────────────────────────────────────────────────────────

class TestOriginalUnmodified:
    def test_node_types_unchanged(self, mini_hetero_data):
        original_node_types = set(mini_hetero_data.node_types)
        original_edge_types = set(mini_hetero_data.edge_types)

        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        _ = adapter.apply(mini_hetero_data)

        assert set(mini_hetero_data.node_types) == original_node_types
        assert set(mini_hetero_data.edge_types) == original_edge_types

    def test_tensor_values_unchanged(self, mini_hetero_data):
        """Snapshot admission features before apply; assert unchanged after."""
        snapshot = mini_hetero_data["admission"].x.clone()

        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        view = adapter.apply(mini_hetero_data)

        # Mutate view to prove independence
        view["admission"].x[0, 0] = -999.0

        assert torch.equal(mini_hetero_data["admission"].x, snapshot)

    def test_edge_index_unchanged(self, mini_hetero_data):
        """Snapshot edge indices before apply; assert unchanged after."""
        key = ("icu_day", "has_event", "drug")
        snapshot = mini_hetero_data[key].edge_index.clone()

        cfg = GraphViewConfig.readmission_default()
        adapter = GraphViewAdapter(cfg)
        _ = adapter.apply(mini_hetero_data)

        assert torch.equal(mini_hetero_data[key].edge_index, snapshot)
