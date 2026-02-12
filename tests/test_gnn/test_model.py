"""Tests for src.gnn.model — TD4DD model assembly."""

import torch
from torch_geometric.data import HeteroData

from src.gnn.model import ModelConfig, TD4DDModel

B, D, N, K = 8, 128, 32, 4


def _make_config(**overrides) -> ModelConfig:
    defaults = dict(
        feat_dims={"admission": 64, "drug": 32, "diagnosis": 48},
        d_model=D,
        num_contextual_paths=2,
        k_hops_contextual=K,
        k_hops_temporal=2,
        nhead=4,
        dropout=0.0,
        use_transformer=True,
        use_diffusion=True,
        use_temporal_encoding=True,
        diffusion_T=50,
        diffusion_ddim_steps=5,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_batch(batch_size: int) -> HeteroData:
    data = HeteroData()
    data["admission"].x = torch.randn(batch_size, 64)
    data["drug"].x = torch.randn(batch_size * 2, 32)
    data["diagnosis"].x = torch.randn(batch_size * 2, 48)
    return data


def _make_transformer_inputs(batch_size, d, n_neighbors, k_hops, num_paths, k_temporal):
    contextual = [
        [torch.randn(batch_size, n_neighbors, d) for _ in range(k_hops)]
        for _ in range(num_paths)
    ]
    temporal_neighbors = (
        [torch.randn(batch_size, n_neighbors, d) for _ in range(k_temporal)]
        if k_temporal > 0
        else None
    )
    temporal_deltas = (
        [torch.rand(batch_size, n_neighbors) * 30 for _ in range(k_temporal)]
        if k_temporal > 0
        else None
    )
    return contextual, temporal_neighbors, temporal_deltas


def _make_aux_edges(batch_size, E=30):
    return [torch.randint(0, batch_size, (2, E)) for _ in range(2)]


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestTD4DDModel:
    def test_forward_shape(self):
        """Both branches on → dict keys, logits (B, 1), probs in [0, 1]."""
        torch.manual_seed(42)
        config = _make_config()
        model = TD4DDModel(config)
        model.eval()

        batch = _make_batch(B)
        ctx, tmp_n, tmp_d = _make_transformer_inputs(B, D, N, K, 2, 2)
        aux = _make_aux_edges(B)

        out = model(
            batch,
            aux,
            contextual_hop_neighbors=ctx,
            temporal_hop_neighbors=tmp_n,
            temporal_hop_deltas=tmp_d,
        )

        assert set(out.keys()) == {
            "logits",
            "probabilities",
            "h_hhgat",
            "h_diff",
            "L_diff",
            "attention_info",
        }
        assert out["logits"].shape == (B, 1)
        assert out["probabilities"].min() >= 0.0
        assert out["probabilities"].max() <= 1.0

    def test_transformer_only(self):
        """use_diffusion=False → h_diff is None, L_diff is 0."""
        torch.manual_seed(42)
        config = _make_config(use_diffusion=False)
        model = TD4DDModel(config)
        model.eval()

        batch = _make_batch(B)
        ctx, tmp_n, tmp_d = _make_transformer_inputs(B, D, N, K, 2, 2)

        out = model(
            batch,
            contextual_hop_neighbors=ctx,
            temporal_hop_neighbors=tmp_n,
            temporal_hop_deltas=tmp_d,
        )

        assert out["h_diff"] is None
        assert out["L_diff"].item() == 0.0
        assert out["h_hhgat"] is not None

    def test_diffusion_only(self):
        """use_transformer=False → h_hhgat is None."""
        torch.manual_seed(42)
        config = _make_config(use_transformer=False)
        model = TD4DDModel(config)
        model.eval()

        batch = _make_batch(B)
        aux = _make_aux_edges(B)

        out = model(batch, aux)

        assert out["h_hhgat"] is None
        assert out["h_diff"] is not None

    def test_both_branches(self):
        """Both on → h_hhgat and h_diff both populated."""
        torch.manual_seed(42)
        config = _make_config()
        model = TD4DDModel(config)
        model.train()

        batch = _make_batch(B)
        ctx, tmp_n, tmp_d = _make_transformer_inputs(B, D, N, K, 2, 2)
        aux = _make_aux_edges(B)

        out = model(
            batch,
            aux,
            contextual_hop_neighbors=ctx,
            temporal_hop_neighbors=tmp_n,
            temporal_hop_deltas=tmp_d,
        )

        assert out["h_hhgat"] is not None
        assert out["h_diff"] is not None

    def test_fusion_lambda_learnable(self):
        """forward + backward → lambda_param.grad is not None."""
        torch.manual_seed(42)
        config = _make_config()
        model = TD4DDModel(config)
        model.train()

        batch = _make_batch(B)
        ctx, tmp_n, tmp_d = _make_transformer_inputs(B, D, N, K, 2, 2)
        aux = _make_aux_edges(B)

        out = model(
            batch,
            aux,
            contextual_hop_neighbors=ctx,
            temporal_hop_neighbors=tmp_n,
            temporal_hop_deltas=tmp_d,
        )
        out["logits"].sum().backward()

        assert model.lambda_param.grad is not None

    def test_gradient_flow_e2e(self):
        """Proj[admission] + classifier should have gradients after backward."""
        torch.manual_seed(42)
        config = _make_config()
        model = TD4DDModel(config)
        model.train()

        batch = _make_batch(B)
        ctx, tmp_n, tmp_d = _make_transformer_inputs(B, D, N, K, 2, 2)
        aux = _make_aux_edges(B)

        out = model(
            batch,
            aux,
            contextual_hop_neighbors=ctx,
            temporal_hop_neighbors=tmp_n,
            temporal_hop_deltas=tmp_d,
        )
        (out["logits"].sum() + out["L_diff"]).backward()

        assert model.proj["admission"].weight.grad is not None
        assert model.classifier.weight.grad is not None

    def test_save_load_roundtrip(self):
        """Save state_dict → new model → load → same output."""
        torch.manual_seed(42)
        config = _make_config(use_diffusion=False)
        model = TD4DDModel(config)
        model.eval()

        batch = _make_batch(B)
        ctx, tmp_n, tmp_d = _make_transformer_inputs(B, D, N, K, 2, 2)

        with torch.no_grad():
            out1 = model(
                batch,
                contextual_hop_neighbors=ctx,
                temporal_hop_neighbors=tmp_n,
                temporal_hop_deltas=tmp_d,
            )

        state = model.state_dict()
        model2 = TD4DDModel(config)
        model2.load_state_dict(state)
        model2.eval()

        with torch.no_grad():
            out2 = model2(
                batch,
                contextual_hop_neighbors=ctx,
                temporal_hop_neighbors=tmp_n,
                temporal_hop_deltas=tmp_d,
            )

        assert torch.equal(out1["logits"], out2["logits"])
