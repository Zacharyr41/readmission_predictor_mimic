"""Tests for src.gnn.transformer — Transformer attention backbone."""

import torch

from src.gnn.transformer import (
    DualTrackTransformer,
    HopTransformerEncoder,
    MetaPathTransformer,
    temporal_encoding,
)

B, D, N, K = 8, 128, 32, 4


# ──────────────────────────────────────────────────────────────────────────────
# 1. Temporal encoding
# ──────────────────────────────────────────────────────────────────────────────


class TestTemporalEncoding:
    def test_output_shape_and_range(self):
        """100 random deltas → shape (100, 128), values in [-1, 1], unique rows."""
        torch.manual_seed(42)
        deltas = torch.rand(100) * 365
        pe = temporal_encoding(deltas, d_model=D)

        assert pe.shape == (100, D)
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0
        # Different deltas should produce different rows
        assert not torch.equal(pe[0], pe[1])

    def test_determinism(self):
        """Same input twice → identical output."""
        deltas = torch.tensor([1.0, 7.0, 30.0])
        pe1 = temporal_encoding(deltas, d_model=D)
        pe2 = temporal_encoding(deltas, d_model=D)
        assert torch.equal(pe1, pe2)


# ──────────────────────────────────────────────────────────────────────────────
# 2. HopTransformerEncoder
# ──────────────────────────────────────────────────────────────────────────────


class TestHopTransformerEncoder:
    def test_forward_shape(self):
        """(8, 128) target + (8, 32, 128) neighbors → (8, 128) output."""
        torch.manual_seed(42)
        enc = HopTransformerEncoder(d_model=D, dropout=0.0)
        target = torch.randn(B, D)
        neighbors = torch.randn(B, N, D)

        out = enc(target, neighbors)
        assert out.shape == (B, D)

    def test_with_temporal_deltas(self):
        """Adding temporal deltas changes the output (eval mode for determinism)."""
        torch.manual_seed(42)
        enc = HopTransformerEncoder(d_model=D, dropout=0.0)
        enc.eval()
        target = torch.randn(B, D)
        neighbors = torch.randn(B, N, D)
        deltas = torch.rand(B, N) * 30

        out_no_delta = enc(target, neighbors)
        out_with_delta = enc(target, neighbors, temporal_deltas=deltas)

        assert out_with_delta.shape == (B, D)
        assert not torch.equal(out_no_delta, out_with_delta)

    def test_gradient_flow(self):
        """Gradients should flow to W_Q, W_K, W_V and encoder params."""
        torch.manual_seed(42)
        enc = HopTransformerEncoder(d_model=D, dropout=0.0)
        target = torch.randn(B, D)
        neighbors = torch.randn(B, N, D)

        out = enc(target, neighbors)
        out.sum().backward()

        for name in ["W_Q", "W_K", "W_V"]:
            param = getattr(enc, name).weight
            assert param.grad is not None, f"{name}.weight.grad is None"

        # Also check that neighbor_encoder got gradients
        encoder_params = list(enc.neighbor_encoder.parameters())
        assert any(
            p.grad is not None for p in encoder_params
        ), "No gradients in neighbor_encoder"


# ──────────────────────────────────────────────────────────────────────────────
# 3. MetaPathTransformer
# ──────────────────────────────────────────────────────────────────────────────


class TestMetaPathTransformer:
    def test_forward_shape(self):
        """4 hops × 32 neighbors → (8, 128); info has expected keys/shapes."""
        torch.manual_seed(42)
        mpt = MetaPathTransformer(k_hops=K, d_model=D, dropout=0.0)
        target = torch.randn(B, D)
        hop_neighbors = [torch.randn(B, N, D) for _ in range(K)]

        out, info = mpt(target, hop_neighbors)

        assert out.shape == (B, D)
        assert len(info["hop_attentions"]) == K
        assert info["hierarchical_weights"].shape == (B, K)

    def test_hierarchical_weights_sum_to_one(self):
        """Softmax hierarchical weights should sum to 1.0 per sample."""
        torch.manual_seed(42)
        mpt = MetaPathTransformer(k_hops=K, d_model=D, dropout=0.0)
        target = torch.randn(B, D)
        hop_neighbors = [torch.randn(B, N, D) for _ in range(K)]

        _, info = mpt(target, hop_neighbors)
        weights = info["hierarchical_weights"]

        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# 4. DualTrackTransformer
# ──────────────────────────────────────────────────────────────────────────────


class TestDualTrackTransformer:
    def test_forward_shape(self):
        """2 contextual + 1 temporal → (8, 128)."""
        torch.manual_seed(42)
        dtt = DualTrackTransformer(
            num_contextual_paths=2,
            k_hops_contextual=K,
            k_hops_temporal=2,
            d_model=D,
            dropout=0.0,
        )
        target = torch.randn(B, D)
        ctx = [
            [torch.randn(B, N, D) for _ in range(K)]
            for _ in range(2)
        ]
        tmp_neighbors = [torch.randn(B, N, D) for _ in range(2)]
        tmp_deltas = [torch.rand(B, N) * 30 for _ in range(2)]

        out, info = dtt(target, ctx, tmp_neighbors, tmp_deltas)

        assert out.shape == (B, D)
        assert len(info["contextual_infos"]) == 2
        assert info["temporal_info"] is not None

    def test_without_temporal(self):
        """k_hops_temporal=0, pass None for temporal → (8, 128), temporal_info is None."""
        torch.manual_seed(42)
        dtt = DualTrackTransformer(
            num_contextual_paths=2,
            k_hops_contextual=K,
            k_hops_temporal=0,
            d_model=D,
            dropout=0.0,
        )
        target = torch.randn(B, D)
        ctx = [
            [torch.randn(B, N, D) for _ in range(K)]
            for _ in range(2)
        ]

        out, info = dtt(target, ctx, None, None)

        assert out.shape == (B, D)
        assert info["temporal_info"] is None
        assert dtt.temporal_transformer is None

    def test_parameter_count(self):
        """Total parameters should be between 500K and 5M (~2.27M expected)."""
        dtt = DualTrackTransformer(
            num_contextual_paths=2,
            k_hops_contextual=K,
            k_hops_temporal=2,
            d_model=D,
            dropout=0.0,
        )
        total = sum(p.numel() for p in dtt.parameters())
        assert 500_000 < total < 5_000_000, f"Unexpected param count: {total}"
