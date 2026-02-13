"""Tests for src.gnn.sampling — auxiliary subgraphs and dual-track sampling."""

import logging
import math

import pytest
import torch
from torch_geometric.data import HeteroData

from src.gnn.sampling import (
    DualSamplingConfig,
    DualTrackSampler,
    SamplingTrackConfig,
    _resolve_edge_type,
    _topk_recency_filter,
    build_auxiliary_subgraphs,
    check_subgraph_density,
)
from src.gnn.view_adapter import GraphViewConfig


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def aux_view_data() -> HeteroData:
    """Small HeteroData for auxiliary subgraph tests.

    4 admissions, 3 drugs.
    Edges (admission → prescribed → drug):
        adm0 → drug0
        adm1 → drug0
        adm1 → drug2
        adm2 → drug2
        adm3 → drug2

    Expected auxiliary edges (admissions sharing drugs):
        0↔1 (share drug0)
        1↔2 (share drug2)
        1↔3 (share drug2)
        2↔3 (share drug2)
    = 8 directed edges (4 undirected pairs)
    """
    data = HeteroData()

    data["admission"].x = torch.randn(4, 5)
    data["drug"].x = torch.randn(3, 768)

    # admission → prescribed → drug
    data["admission", "prescribed", "drug"].edge_index = torch.tensor(
        [[0, 1, 1, 2, 3], [0, 0, 2, 2, 2]], dtype=torch.long
    )
    data["admission", "prescribed", "drug"].edge_attr = torch.ones(5, 1)

    # drug → rev_prescribed → admission (reverse)
    data["drug", "rev_prescribed", "admission"].edge_index = torch.tensor(
        [[0, 0, 2, 2, 2], [0, 1, 1, 2, 3]], dtype=torch.long
    )
    data["drug", "rev_prescribed", "admission"].edge_attr = torch.ones(5, 1)

    return data


@pytest.fixture
def view_data() -> HeteroData:
    """Larger HeteroData for sampler tests.

    10 admissions, 5 drugs, 4 diagnoses.
    Includes train/val/test masks and temporal edges.
    """
    data = HeteroData()

    # ── Nodes ─────────────────────────────────────────────────────────────
    data["admission"].x = torch.randn(10, 8)
    data["admission"].y = torch.randint(0, 2, (10,)).float()
    data["admission"].train_mask = torch.tensor(
        [True, True, True, True, True, True, False, False, False, False]
    )
    data["admission"].val_mask = torch.tensor(
        [False, False, False, False, False, False, True, True, False, False]
    )
    data["admission"].test_mask = torch.tensor(
        [False, False, False, False, False, False, False, False, True, True]
    )

    data["drug"].x = torch.randn(5, 768)
    data["diagnosis"].x = torch.randn(4, 768)

    # ── Contextual edges ──────────────────────────────────────────────────
    # admission → prescribed → drug
    src = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dst = [0, 1, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    data["admission", "prescribed", "drug"].edge_index = torch.tensor(
        [src, dst], dtype=torch.long
    )
    data["admission", "prescribed", "drug"].edge_attr = torch.rand(
        len(src), 1
    )

    # drug → rev_prescribed → admission
    data["drug", "rev_prescribed", "admission"].edge_index = torch.tensor(
        [dst, src], dtype=torch.long
    )
    data["drug", "rev_prescribed", "admission"].edge_attr = torch.rand(
        len(src), 1
    )

    # admission → has_diagnosis → diagnosis
    diag_src = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    diag_dst = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    data["admission", "has_diagnosis", "diagnosis"].edge_index = torch.tensor(
        [diag_src, diag_dst], dtype=torch.long
    )
    data["admission", "has_diagnosis", "diagnosis"].edge_attr = torch.rand(
        len(diag_src), 1
    )

    # diagnosis → rev_has_diagnosis → admission
    data[
        "diagnosis", "rev_has_diagnosis", "admission"
    ].edge_index = torch.tensor(
        [diag_dst, diag_src], dtype=torch.long
    )
    data[
        "diagnosis", "rev_has_diagnosis", "admission"
    ].edge_attr = torch.rand(len(diag_src), 1)

    # ── Temporal edges ────────────────────────────────────────────────────
    # admission → followed_by → admission (chain: 0→1→2→...→9)
    fb_src = list(range(9))
    fb_dst = list(range(1, 10))
    fb_attr = [float(i * 5 + 10) for i in range(9)]  # gap days
    data["admission", "followed_by", "admission"].edge_index = torch.tensor(
        [fb_src, fb_dst], dtype=torch.long
    )
    data["admission", "followed_by", "admission"].edge_attr = torch.tensor(
        fb_attr, dtype=torch.float
    ).unsqueeze(1)

    # admission → rev_followed_by → admission
    data[
        "admission", "rev_followed_by", "admission"
    ].edge_index = torch.tensor(
        [fb_dst, fb_src], dtype=torch.long
    )
    data[
        "admission", "rev_followed_by", "admission"
    ].edge_attr = torch.tensor(fb_attr, dtype=torch.float).unsqueeze(1)

    return data


# ──────────────────────────────────────────────────────────────────────────────
# 1. Build auxiliary subgraphs
# ──────────────────────────────────────────────────────────────────────────────


class TestBuildAuxiliarySubgraphs:
    def test_correct_edges_from_drug_sharing(self, aux_view_data):
        """Admissions sharing drugs should form the expected edges."""
        results = build_auxiliary_subgraphs(
            aux_view_data,
            target_node_type="admission",
            meta_paths=[["prescribed", "rev_prescribed"]],
        )
        assert len(results) == 1
        edge_index = results[0]

        pairs = set(
            zip(edge_index[0].tolist(), edge_index[1].tolist())
        )
        # Expected undirected pairs: 0↔1, 1↔2, 1↔3, 2↔3
        expected = {
            (0, 1), (1, 0),
            (1, 2), (2, 1),
            (1, 3), (3, 1),
            (2, 3), (3, 2),
        }
        assert pairs == expected

    def test_no_self_loops(self, aux_view_data):
        results = build_auxiliary_subgraphs(
            aux_view_data,
            target_node_type="admission",
            meta_paths=[["prescribed", "rev_prescribed"]],
        )
        edge_index = results[0]
        assert not torch.any(edge_index[0] == edge_index[1]).item()

    def test_edge_count(self, aux_view_data):
        results = build_auxiliary_subgraphs(
            aux_view_data,
            target_node_type="admission",
            meta_paths=[["prescribed", "rev_prescribed"]],
        )
        # 4 undirected pairs = 8 directed edges
        assert results[0].shape[1] == 8

    def test_index_bounds(self, aux_view_data):
        results = build_auxiliary_subgraphs(
            aux_view_data,
            target_node_type="admission",
            meta_paths=[["prescribed", "rev_prescribed"]],
        )
        edge_index = results[0]
        n_target = aux_view_data["admission"].num_nodes
        assert edge_index.max().item() < n_target
        assert edge_index.min().item() >= 0

    def test_multiple_meta_paths(self, view_data):
        """Multiple meta-paths should produce multiple adjacency tensors."""
        results = build_auxiliary_subgraphs(
            view_data,
            target_node_type="admission",
            meta_paths=[
                ["prescribed", "rev_prescribed"],
                ["has_diagnosis", "rev_has_diagnosis"],
            ],
        )
        assert len(results) == 2
        for r in results:
            assert r.shape[0] == 2
            assert r.dtype == torch.long

    def test_rejects_non_2hop_path(self, aux_view_data):
        with pytest.raises(AssertionError, match="Only 2-hop"):
            build_auxiliary_subgraphs(
                aux_view_data,
                target_node_type="admission",
                meta_paths=[["prescribed"]],
            )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Check subgraph density
# ──────────────────────────────────────────────────────────────────────────────


class TestCheckSubgraphDensity:
    def test_too_sparse(self):
        """2 edges among 100 nodes → very sparse."""
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        result = check_subgraph_density(ei, 100)
        assert result["category"] == "too_sparse"
        assert result["num_edges"] == 2

    def test_optimal(self):
        """Build a graph with density in 1-10% range."""
        # 10 nodes, max edges = 90.  5 edges → density = 5/90 ≈ 5.5%
        ei = torch.tensor(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long
        )
        result = check_subgraph_density(ei, 10)
        assert result["category"] == "optimal"

    def test_dense(self):
        """Build a graph with density in 10-30%."""
        # 5 nodes, max edges = 20.  5 edges → density = 25%
        ei = torch.tensor(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long
        )
        result = check_subgraph_density(ei, 5)
        assert result["category"] == "dense"

    def test_too_dense(self):
        """Build a nearly-complete graph."""
        # 4 nodes, max edges = 12.  8 edges → density = 66.7%
        src = [0, 0, 1, 1, 2, 2, 3, 3]
        dst = [1, 2, 0, 2, 0, 1, 0, 1]
        ei = torch.tensor([src, dst], dtype=torch.long)
        result = check_subgraph_density(ei, 4)
        assert result["category"] == "too_dense"

    def test_single_node(self):
        """N <= 1 should gracefully return density 0."""
        ei = torch.zeros(2, 0, dtype=torch.long)
        result = check_subgraph_density(ei, 1)
        assert result["density"] == 0.0

    def test_dict_keys(self):
        ei = torch.tensor([[0], [1]], dtype=torch.long)
        result = check_subgraph_density(ei, 5)
        assert set(result.keys()) == {
            "density", "num_edges", "category", "recommendation"
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Density warnings
# ──────────────────────────────────────────────────────────────────────────────


class TestDensityWarnings:
    def test_too_dense_warning(self, aux_view_data, caplog):
        """A very dense auxiliary subgraph should emit a WARNING."""
        # With 4 nodes and 8 edges: density = 8/12 ≈ 66.7% → too_dense
        # But our fixture gives 8 edges among 4 nodes which is too_dense.
        # Actually: 4 nodes, 8 directed edges, max=12 → 66.7%
        with caplog.at_level(logging.WARNING, logger="src.gnn.sampling"):
            build_auxiliary_subgraphs(
                aux_view_data,
                target_node_type="admission",
                meta_paths=[["prescribed", "rev_prescribed"]],
            )
        assert any("too dense" in r.message for r in caplog.records)

    def test_too_sparse_warning(self, caplog):
        """A very sparse auxiliary subgraph should emit a WARNING."""
        # Build a view with many nodes but very few shared entities
        data = HeteroData()
        data["admission"].x = torch.randn(100, 5)
        data["drug"].x = torch.randn(50, 768)

        # Only adm0 and adm1 share drug0
        data["admission", "prescribed", "drug"].edge_index = torch.tensor(
            [[0, 1], [0, 0]], dtype=torch.long
        )
        data["admission", "prescribed", "drug"].edge_attr = torch.ones(2, 1)
        data["drug", "rev_prescribed", "admission"].edge_index = torch.tensor(
            [[0, 0], [0, 1]], dtype=torch.long
        )
        data["drug", "rev_prescribed", "admission"].edge_attr = torch.ones(
            2, 1
        )

        with caplog.at_level(logging.WARNING, logger="src.gnn.sampling"):
            build_auxiliary_subgraphs(
                data,
                target_node_type="admission",
                meta_paths=[["prescribed", "rev_prescribed"]],
            )
        assert any("too sparse" in r.message for r in caplog.records)


# ──────────────────────────────────────────────────────────────────────────────
# 4. DualSamplingConfig from view config
# ──────────────────────────────────────────────────────────────────────────────


class TestDualSamplingConfigFromViewConfig:
    def test_meta_path_contextual_track(self, view_data):
        """When meta_paths are set, contextual track should use those edges."""
        cfg = GraphViewConfig(
            target_node_type="admission",
            label_key="readmitted_30d",
            active_entity_types=["drug", "diagnosis"],
            meta_paths=[["prescribed", "rev_prescribed"]],
            include_temporal_track=True,
        )
        dual = DualSamplingConfig.from_view_config(cfg, view_data)

        ctx_rels = {r for _, r, _ in dual.contextual.edge_types}
        assert "prescribed" in ctx_rels
        assert "rev_prescribed" in ctx_rels

    def test_no_meta_paths_uses_all_non_temporal(self, view_data):
        """Without meta_paths, all non-temporal edges should be contextual."""
        cfg = GraphViewConfig(
            target_node_type="admission",
            label_key="readmitted_30d",
            active_entity_types=["drug", "diagnosis"],
            meta_paths=[],
            include_temporal_track=True,
        )
        dual = DualSamplingConfig.from_view_config(cfg, view_data)

        ctx_rels = {r for _, r, _ in dual.contextual.edge_types}
        assert "followed_by" not in ctx_rels
        assert "rev_followed_by" not in ctx_rels
        assert "prescribed" in ctx_rels

    def test_temporal_track_includes_followed_by(self, view_data):
        cfg = GraphViewConfig(
            target_node_type="admission",
            label_key="readmitted_30d",
            active_entity_types=["drug"],
            meta_paths=[],
            include_temporal_track=True,
        )
        dual = DualSamplingConfig.from_view_config(cfg, view_data)

        tmp_rels = {r for _, r, _ in dual.temporal.edge_types}
        assert "followed_by" in tmp_rels
        assert "rev_followed_by" in tmp_rels

    def test_temporal_track_disabled(self, view_data):
        cfg = GraphViewConfig(
            target_node_type="admission",
            label_key="readmitted_30d",
            active_entity_types=["drug"],
            meta_paths=[],
            include_temporal_track=False,
        )
        dual = DualSamplingConfig.from_view_config(cfg, view_data)
        assert dual.temporal.edge_types == []

    def test_contextual_defaults(self, view_data):
        cfg = GraphViewConfig.readmission_default()
        dual = DualSamplingConfig.from_view_config(cfg, view_data)
        assert dual.contextual.k_hops == 4
        assert dual.contextual.neighbors_per_hop == 32
        assert dual.contextual.weighting == "uniform"

    def test_temporal_defaults(self, view_data):
        cfg = GraphViewConfig.readmission_default()
        dual = DualSamplingConfig.from_view_config(cfg, view_data)
        assert dual.temporal.k_hops == 2
        assert dual.temporal.neighbors_per_hop == 8
        assert dual.temporal.weighting == "recency"


# ──────────────────────────────────────────────────────────────────────────────
# 5. DualTrackSampler
# ──────────────────────────────────────────────────────────────────────────────


class TestDualTrackSampler:
    def test_constructor_succeeds(self, view_data):
        cfg = GraphViewConfig.readmission_default()
        dual_cfg = DualSamplingConfig.from_view_config(cfg, view_data)
        sampler = DualTrackSampler(view_data, dual_cfg, batch_size=4)
        assert sampler.target_node_type == "admission"

    def test_unified_num_neighbors_contextual_active(self, view_data):
        """Active contextual edges get non-zero neighbors in early hops."""
        cfg = GraphViewConfig.readmission_default()
        dual_cfg = DualSamplingConfig.from_view_config(cfg, view_data)
        sampler = DualTrackSampler(view_data, dual_cfg, batch_size=4)

        nn = sampler._num_neighbors
        prescribed_key = ("admission", "prescribed", "drug")
        # contextual edges should be active (at least some hops > 0)
        assert any(n > 0 for n in nn[prescribed_key])

    def test_unified_num_neighbors_temporal_padded(self, view_data):
        """Temporal edge types get padded lists: [8, 8, 0, 0] for 2 tmp hops within 4 ctx hops."""
        cfg = GraphViewConfig.readmission_default()
        dual_cfg = DualSamplingConfig.from_view_config(cfg, view_data)
        sampler = DualTrackSampler(view_data, dual_cfg, batch_size=4)

        nn = sampler._num_neighbors
        fb_key = ("admission", "followed_by", "admission")
        # temporal hops = 2 active, padded to length 4
        assert len(nn[fb_key]) == 4
        assert nn[fb_key][0] > 0  # first 2 hops active
        assert nn[fb_key][1] > 0
        assert nn[fb_key][2] == 0  # padded zeros
        assert nn[fb_key][3] == 0

    def test_unified_loader_has_both_edge_types(self, view_data):
        """A single batch should contain both contextual and temporal edges."""
        cfg = GraphViewConfig.readmission_default()
        dual_cfg = DualSamplingConfig.from_view_config(cfg, view_data)
        sampler = DualTrackSampler(view_data, dual_cfg, batch_size=4)

        batch = next(iter(sampler.get_train_loader()))
        edge_rels = {r for _, r, _ in batch.edge_types}
        assert "prescribed" in edge_rels or "rev_prescribed" in edge_rels
        assert "followed_by" in edge_rels or "rev_followed_by" in edge_rels

    def test_len_matches_expected(self, view_data):
        """__len__ should be ceil(n_train / batch_size)."""
        cfg = GraphViewConfig.readmission_default()
        dual_cfg = DualSamplingConfig.from_view_config(cfg, view_data)
        batch_size = 4
        sampler = DualTrackSampler(view_data, dual_cfg, batch_size=batch_size)

        n_train = int(view_data["admission"].train_mask.sum().item())
        expected = math.ceil(n_train / batch_size)
        assert len(sampler) == expected

    def test_train_loader_yields_heterodata(self, view_data):
        """One batch from the train loader should be a HeteroData."""
        cfg = GraphViewConfig.readmission_default()
        dual_cfg = DualSamplingConfig.from_view_config(cfg, view_data)
        sampler = DualTrackSampler(view_data, dual_cfg, batch_size=4)

        loader = sampler.get_train_loader()
        batch = next(iter(loader))
        assert isinstance(batch, HeteroData)

    def test_batch_has_expected_node_types(self, view_data):
        """Batch should contain admission and at least some entity types."""
        cfg = GraphViewConfig.readmission_default()
        dual_cfg = DualSamplingConfig.from_view_config(cfg, view_data)
        sampler = DualTrackSampler(view_data, dual_cfg, batch_size=4)

        loader = sampler.get_train_loader()
        batch = next(iter(loader))
        assert "admission" in batch.node_types

    def test_batch_size_respected(self, view_data):
        """Number of input nodes in a batch should not exceed batch_size."""
        cfg = GraphViewConfig.readmission_default()
        dual_cfg = DualSamplingConfig.from_view_config(cfg, view_data)
        batch_size = 4
        sampler = DualTrackSampler(view_data, dual_cfg, batch_size=batch_size)

        loader = sampler.get_train_loader()
        batch = next(iter(loader))
        # batch_size attr on the batch tells how many seed nodes
        assert batch["admission"].batch_size <= batch_size

    def test_val_and_test_loaders_work(self, view_data):
        """Val and test loaders should also produce batches."""
        cfg = GraphViewConfig.readmission_default()
        dual_cfg = DualSamplingConfig.from_view_config(cfg, view_data)
        sampler = DualTrackSampler(view_data, dual_cfg, batch_size=4)

        val_batch = next(iter(sampler.get_val_loader()))
        assert isinstance(val_batch, HeteroData)

        test_batch = next(iter(sampler.get_test_loader()))
        assert isinstance(test_batch, HeteroData)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Recency pre-filtering
# ──────────────────────────────────────────────────────────────────────────────


class TestRecencyPreFiltering:
    def test_keeps_topk_smallest(self):
        """With k=2, only 2 smallest-attr edges per source should remain."""
        # Source 0 has 4 edges with attrs [10, 1, 5, 3]
        edge_index = torch.tensor(
            [[0, 0, 0, 0], [1, 2, 3, 4]], dtype=torch.long
        )
        edge_attr = torch.tensor([[10.0], [1.0], [5.0], [3.0]])

        filtered_ei, filtered_attr = _topk_recency_filter(
            edge_index, edge_attr, k=2
        )

        assert filtered_ei.shape[1] == 2
        # Should keep edges to node 2 (attr=1) and node 4 (attr=3)
        pairs = set(
            zip(filtered_ei[0].tolist(), filtered_ei[1].tolist())
        )
        assert (0, 2) in pairs  # attr = 1.0
        assert (0, 4) in pairs  # attr = 3.0

    def test_multiple_sources(self):
        """Each source independently gets top-k filtering."""
        edge_index = torch.tensor(
            [[0, 0, 0, 1, 1, 1], [1, 2, 3, 4, 5, 6]], dtype=torch.long
        )
        edge_attr = torch.tensor([[10.0], [1.0], [5.0], [8.0], [2.0], [6.0]])

        filtered_ei, filtered_attr = _topk_recency_filter(
            edge_index, edge_attr, k=1
        )

        assert filtered_ei.shape[1] == 2  # 1 per source
        pairs = set(
            zip(filtered_ei[0].tolist(), filtered_ei[1].tolist())
        )
        assert (0, 2) in pairs  # src=0, smallest attr=1.0
        assert (1, 5) in pairs  # src=1, smallest attr=2.0

    def test_k_greater_than_edges(self):
        """If k >= num_edges for a source, all edges are kept."""
        edge_index = torch.tensor(
            [[0, 0], [1, 2]], dtype=torch.long
        )
        edge_attr = torch.tensor([[3.0], [1.0]])

        filtered_ei, filtered_attr = _topk_recency_filter(
            edge_index, edge_attr, k=10
        )

        assert filtered_ei.shape[1] == 2

    def test_preserves_attr_shape(self):
        """Filtered edge_attr should have same column count as input."""
        edge_index = torch.tensor(
            [[0, 0, 0], [1, 2, 3]], dtype=torch.long
        )
        edge_attr = torch.tensor([[10.0], [1.0], [5.0]])

        _, filtered_attr = _topk_recency_filter(
            edge_index, edge_attr, k=2
        )

        assert filtered_attr.shape[1] == 1

    def test_attr_values_match_kept_edges(self):
        """The filtered attrs should correspond to the kept edges."""
        edge_index = torch.tensor(
            [[0, 0, 0], [1, 2, 3]], dtype=torch.long
        )
        edge_attr = torch.tensor([[10.0], [1.0], [5.0]])

        filtered_ei, filtered_attr = _topk_recency_filter(
            edge_index, edge_attr, k=2
        )

        # Map edge → attr for verification
        for i in range(filtered_ei.shape[1]):
            dst = filtered_ei[1, i].item()
            attr_val = filtered_attr[i, 0].item()
            if dst == 2:
                assert attr_val == 1.0
            elif dst == 3:
                assert attr_val == 5.0


# ──────────────────────────────────────────────────────────────────────────────
# 7. Edge type resolver
# ──────────────────────────────────────────────────────────────────────────────


class TestResolveEdgeType:
    def test_finds_unique_match(self, aux_view_data):
        result = _resolve_edge_type(aux_view_data, "prescribed")
        assert result == ("admission", "prescribed", "drug")

    def test_raises_on_no_match(self, aux_view_data):
        with pytest.raises(ValueError, match="No edge type"):
            _resolve_edge_type(aux_view_data, "nonexistent")

    def test_raises_on_multiple_matches(self):
        """If two edge types share a relation name, should raise."""
        data = HeteroData()
        data["a"].num_nodes = 2
        data["b"].num_nodes = 2
        data["c"].num_nodes = 2
        data["a", "rel", "b"].edge_index = torch.tensor(
            [[0], [0]], dtype=torch.long
        )
        data["a", "rel", "c"].edge_index = torch.tensor(
            [[0], [0]], dtype=torch.long
        )
        with pytest.raises(ValueError, match="Multiple"):
            _resolve_edge_type(data, "rel")
