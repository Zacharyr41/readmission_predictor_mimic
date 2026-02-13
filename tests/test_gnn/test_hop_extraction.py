"""Tests for src.gnn.hop_extraction — hop-structured neighbor index extraction."""

import torch
from torch_geometric.data import HeteroData

from src.gnn.hop_extraction import HopExtractor, HopIndices


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_batch(batch_size: int = 2) -> HeteroData:
    """Build a small HeteroData resembling a NeighborLoader batch.

    Graph structure (batch_size=2, seeds are admission 0 and 1):
        admission 0 --prescribed--> drug 0, drug 1
        admission 1 --prescribed--> drug 1, drug 2
        drug 0 --rev_prescribed--> admission 0, admission 2
        drug 1 --rev_prescribed--> admission 0, admission 1
        drug 2 --rev_prescribed--> admission 1, admission 3

        admission 0 --has_diagnosis--> diagnosis 0
        admission 1 --has_diagnosis--> diagnosis 1
        diagnosis 0 --rev_has_diagnosis--> admission 0, admission 2
        diagnosis 1 --rev_has_diagnosis--> admission 1

        admission 0 --followed_by--> admission 1 (gap=5.0)
        admission 1 --followed_by--> admission 2 (gap=10.0)
        admission 1 --rev_followed_by--> admission 0 (gap=5.0)
        admission 2 --rev_followed_by--> admission 1 (gap=10.0)
    """
    data = HeteroData()
    n_adm = max(batch_size + 2, 4)
    data["admission"].x = torch.randn(n_adm, 8)
    data["drug"].x = torch.randn(3, 16)
    data["diagnosis"].x = torch.randn(2, 16)

    # Contextual edges: prescribed / rev_prescribed
    data["admission", "prescribed", "drug"].edge_index = torch.tensor(
        [[0, 0, 1, 1], [0, 1, 1, 2]], dtype=torch.long
    )
    data["drug", "rev_prescribed", "admission"].edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [0, 2, 0, 1, 1, 3]], dtype=torch.long
    )

    # Contextual edges: has_diagnosis / rev_has_diagnosis
    data["admission", "has_diagnosis", "diagnosis"].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    data["diagnosis", "rev_has_diagnosis", "admission"].edge_index = torch.tensor(
        [[0, 0, 1], [0, 2, 1]], dtype=torch.long
    )

    # Temporal edges
    data["admission", "followed_by", "admission"].edge_index = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    )
    data["admission", "followed_by", "admission"].edge_attr = torch.tensor(
        [[5.0], [10.0]]
    )
    data["admission", "rev_followed_by", "admission"].edge_index = torch.tensor(
        [[1, 2], [0, 1]], dtype=torch.long
    )
    data["admission", "rev_followed_by", "admission"].edge_attr = torch.tensor(
        [[5.0], [10.0]]
    )

    return data


def _drug_edge_sequence(k_hops: int = 2) -> list[tuple[str, str, str]]:
    """Alternating edge sequence for the drug meta-path."""
    seq = [
        ("admission", "prescribed", "drug"),
        ("drug", "rev_prescribed", "admission"),
    ]
    full = []
    for i in range(k_hops):
        full.append(seq[i % 2])
    return full


def _diagnosis_edge_sequence(k_hops: int = 2) -> list[tuple[str, str, str]]:
    """Alternating edge sequence for the diagnosis meta-path."""
    seq = [
        ("admission", "has_diagnosis", "diagnosis"),
        ("diagnosis", "rev_has_diagnosis", "admission"),
    ]
    full = []
    for i in range(k_hops):
        full.append(seq[i % 2])
    return full


# ──────────────────────────────────────────────────────────────────────────────
# Contextual path tests
# ──────────────────────────────────────────────────────────────────────────────


class TestSingleSeedSingleHop:
    def test_indices_shape(self):
        """1 seed, 1 hop → shape (1, N), mask has correct True count."""
        batch = _make_batch(batch_size=1)
        extractor = HopExtractor(
            contextual_edge_sequences=[_drug_edge_sequence(k_hops=1)],
            temporal_edge_types=[],
            k_hops_contextual=1,
            k_hops_temporal=0,
            neighbors_per_hop_contextual=8,
            neighbors_per_hop_temporal=4,
        )
        result = extractor.extract(batch, batch_size=1)

        assert len(result.contextual_indices) == 1  # 1 path
        assert len(result.contextual_indices[0]) == 1  # 1 hop

        idx = result.contextual_indices[0][0]
        mask = result.contextual_masks[0][0]
        assert idx.shape == (1, 8)
        assert mask.shape == (1, 8)
        # Admission 0 prescribed drugs 0,1 → 2 neighbors
        assert mask[0].sum().item() == 2
        assert result.contextual_node_types[0][0] == "drug"


class TestMultiHopAlternatingTypes:
    def test_hop_types_alternate(self):
        """2-hop path: hop 0 = drug indices, hop 1 = admission indices."""
        batch = _make_batch(batch_size=1)
        extractor = HopExtractor(
            contextual_edge_sequences=[_drug_edge_sequence(k_hops=2)],
            temporal_edge_types=[],
            k_hops_contextual=2,
            k_hops_temporal=0,
            neighbors_per_hop_contextual=8,
            neighbors_per_hop_temporal=4,
        )
        result = extractor.extract(batch, batch_size=1)

        assert result.contextual_node_types[0][0] == "drug"
        assert result.contextual_node_types[0][1] == "admission"

        # Hop 0: admission 0 → drugs {0, 1}
        assert result.contextual_masks[0][0][0].sum().item() == 2
        # Hop 1: drugs {0,1} → rev_prescribed → admissions
        # drug 0 → adm {0, 2}, drug 1 → adm {0, 1}
        # Union: {0, 1, 2}
        assert result.contextual_masks[0][1][0].sum().item() == 3


class TestPadding:
    def test_fewer_neighbors_than_budget(self):
        """Fewer neighbors than N → padded indices, correct mask."""
        batch = _make_batch(batch_size=1)
        extractor = HopExtractor(
            contextual_edge_sequences=[_drug_edge_sequence(k_hops=1)],
            temporal_edge_types=[],
            k_hops_contextual=1,
            k_hops_temporal=0,
            neighbors_per_hop_contextual=16,
            neighbors_per_hop_temporal=4,
        )
        result = extractor.extract(batch, batch_size=1)

        idx = result.contextual_indices[0][0]  # (1, 16)
        mask = result.contextual_masks[0][0]  # (1, 16)
        assert idx.shape[1] == 16
        # Only 2 actual neighbors, rest should be padding (index 0, mask False)
        assert mask[0].sum().item() == 2
        assert (~mask[0]).sum().item() == 14


class TestTruncation:
    def test_more_neighbors_than_budget(self):
        """More neighbors than N → exactly N neighbors returned."""
        batch = _make_batch(batch_size=1)
        extractor = HopExtractor(
            contextual_edge_sequences=[_drug_edge_sequence(k_hops=1)],
            temporal_edge_types=[],
            k_hops_contextual=1,
            k_hops_temporal=0,
            neighbors_per_hop_contextual=1,  # truncate to 1
            neighbors_per_hop_temporal=4,
        )
        result = extractor.extract(batch, batch_size=1)

        mask = result.contextual_masks[0][0]
        assert mask[0].sum().item() == 1


class TestBatchSizeGt1:
    def test_multiple_seeds(self):
        """Multiple seeds: shape (B, N) correct for each hop."""
        batch = _make_batch(batch_size=2)
        extractor = HopExtractor(
            contextual_edge_sequences=[_drug_edge_sequence(k_hops=1)],
            temporal_edge_types=[],
            k_hops_contextual=1,
            k_hops_temporal=0,
            neighbors_per_hop_contextual=8,
            neighbors_per_hop_temporal=4,
        )
        result = extractor.extract(batch, batch_size=2)

        idx = result.contextual_indices[0][0]
        assert idx.shape == (2, 8)
        # Seed 0: drugs {0, 1}, Seed 1: drugs {1, 2}
        assert result.contextual_masks[0][0][0].sum().item() == 2
        assert result.contextual_masks[0][0][1].sum().item() == 2


class TestBatchSize1:
    def test_batch_dim_preserved(self):
        """batch_size=1 still produces (1, N) tensors."""
        batch = _make_batch(batch_size=1)
        extractor = HopExtractor(
            contextual_edge_sequences=[_drug_edge_sequence(k_hops=1)],
            temporal_edge_types=[],
            k_hops_contextual=1,
            k_hops_temporal=0,
            neighbors_per_hop_contextual=8,
            neighbors_per_hop_temporal=4,
        )
        result = extractor.extract(batch, batch_size=1)
        assert result.contextual_indices[0][0].shape[0] == 1
        assert result.contextual_masks[0][0].shape[0] == 1


class TestTwoContextualPaths:
    def test_drugs_and_diagnoses(self):
        """Two paths (drugs + diagnoses) extracted separately."""
        batch = _make_batch(batch_size=2)
        extractor = HopExtractor(
            contextual_edge_sequences=[
                _drug_edge_sequence(k_hops=2),
                _diagnosis_edge_sequence(k_hops=2),
            ],
            temporal_edge_types=[],
            k_hops_contextual=2,
            k_hops_temporal=0,
            neighbors_per_hop_contextual=8,
            neighbors_per_hop_temporal=4,
        )
        result = extractor.extract(batch, batch_size=2)

        assert len(result.contextual_indices) == 2
        assert result.contextual_node_types[0] == ["drug", "admission"]
        assert result.contextual_node_types[1] == ["diagnosis", "admission"]


class TestMissingEdgeType:
    def test_graceful_empty_hop(self):
        """Missing edge type → empty hop (all-zero indices, all-False mask)."""
        batch = _make_batch(batch_size=1)
        fake_seq = [("admission", "nonexistent_rel", "fake_type")]
        extractor = HopExtractor(
            contextual_edge_sequences=[fake_seq],
            temporal_edge_types=[],
            k_hops_contextual=1,
            k_hops_temporal=0,
            neighbors_per_hop_contextual=8,
            neighbors_per_hop_temporal=4,
        )
        result = extractor.extract(batch, batch_size=1)
        assert result.contextual_masks[0][0].sum().item() == 0


# ──────────────────────────────────────────────────────────────────────────────
# Temporal tests
# ──────────────────────────────────────────────────────────────────────────────


class TestTemporalHopIndices:
    def test_temporal_neighbors_are_admissions(self):
        """followed_by neighbors are admission indices."""
        batch = _make_batch(batch_size=2)
        extractor = HopExtractor(
            contextual_edge_sequences=[],
            temporal_edge_types=[
                ("admission", "followed_by", "admission"),
                ("admission", "rev_followed_by", "admission"),
            ],
            k_hops_contextual=0,
            k_hops_temporal=1,
            neighbors_per_hop_contextual=8,
            neighbors_per_hop_temporal=4,
        )
        result = extractor.extract(batch, batch_size=2)

        assert result.temporal_indices is not None
        assert len(result.temporal_indices) == 1
        assert result.temporal_indices[0].shape == (2, 4)
        # Seed 0: followed_by → {1}, rev_followed_by = {} → {1}
        # Seed 1: followed_by → {2}, rev_followed_by → {0} → {0, 2}
        assert result.temporal_masks[0][0].sum().item() >= 1
        assert result.temporal_masks[0][1].sum().item() >= 1


class TestTemporalDeltasFromEdgeAttr:
    def test_day_gap_values_extracted(self):
        """Day-gap values extracted and padded correctly."""
        batch = _make_batch(batch_size=2)
        extractor = HopExtractor(
            contextual_edge_sequences=[],
            temporal_edge_types=[
                ("admission", "followed_by", "admission"),
            ],
            k_hops_contextual=0,
            k_hops_temporal=1,
            neighbors_per_hop_contextual=8,
            neighbors_per_hop_temporal=4,
        )
        result = extractor.extract(batch, batch_size=2)

        assert result.temporal_deltas is not None
        assert len(result.temporal_deltas) == 1

        deltas = result.temporal_deltas[0]  # (2, 4)
        mask = result.temporal_masks[0]  # (2, 4)

        # Seed 0 → adm 1 with gap 5.0
        valid_deltas_0 = deltas[0][mask[0]]
        assert valid_deltas_0.numel() >= 1
        assert 5.0 in valid_deltas_0.tolist()

        # Seed 1 → adm 2 with gap 10.0
        valid_deltas_1 = deltas[1][mask[1]]
        assert valid_deltas_1.numel() >= 1
        assert 10.0 in valid_deltas_1.tolist()


class TestTemporalDisabled:
    def test_no_temporal_returns_none(self):
        """Empty temporal edge types → all temporal outputs are None."""
        batch = _make_batch(batch_size=2)
        extractor = HopExtractor(
            contextual_edge_sequences=[_drug_edge_sequence(k_hops=1)],
            temporal_edge_types=[],
            k_hops_contextual=1,
            k_hops_temporal=0,
            neighbors_per_hop_contextual=8,
            neighbors_per_hop_temporal=4,
        )
        result = extractor.extract(batch, batch_size=2)
        assert result.temporal_indices is None
        assert result.temporal_masks is None
        assert result.temporal_deltas is None
