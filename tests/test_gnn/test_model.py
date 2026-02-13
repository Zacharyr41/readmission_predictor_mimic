"""Tests for src.gnn.model — TD4DD model assembly."""

import torch
from torch_geometric.data import HeteroData

from src.gnn.hop_extraction import HopIndices
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


# ──────────────────────────────────────────────────────────────────────────────
# HopIndices-based forward tests
# ──────────────────────────────────────────────────────────────────────────────


def _make_hop_indices(batch_size, d, n_neighbors, k_hops, num_paths, k_temporal):
    """Build HopIndices with random valid indices."""
    # Total nodes per type in a fake batch
    total_admission = batch_size + 10
    total_drug = batch_size * 2
    total_diagnosis = batch_size * 2

    node_counts = {
        "admission": total_admission,
        "drug": total_drug,
        "diagnosis": total_diagnosis,
    }

    # Contextual: alternating drug/admission types per hop
    ctx_indices: list[list[torch.Tensor]] = []
    ctx_masks: list[list[torch.Tensor]] = []
    ctx_ntypes: list[list[str]] = []
    for _p in range(num_paths):
        p_indices = []
        p_masks = []
        p_ntypes = []
        for k in range(k_hops):
            ntype = "drug" if k % 2 == 0 else "admission"
            max_idx = node_counts[ntype]
            idx = torch.randint(0, max_idx, (batch_size, n_neighbors))
            mask = torch.ones(batch_size, n_neighbors, dtype=torch.bool)
            # Make last few positions padded
            mask[:, -2:] = False
            p_indices.append(idx)
            p_masks.append(mask)
            p_ntypes.append(ntype)
        ctx_indices.append(p_indices)
        ctx_masks.append(p_masks)
        ctx_ntypes.append(p_ntypes)

    # Temporal
    tmp_indices = None
    tmp_masks = None
    tmp_deltas = None
    if k_temporal > 0:
        tmp_indices = []
        tmp_masks = []
        tmp_deltas = []
        for _k in range(k_temporal):
            idx = torch.randint(0, total_admission, (batch_size, n_neighbors))
            mask = torch.ones(batch_size, n_neighbors, dtype=torch.bool)
            mask[:, -2:] = False
            delta = torch.rand(batch_size, n_neighbors) * 30
            tmp_indices.append(idx)
            tmp_masks.append(mask)
            tmp_deltas.append(delta)

    return HopIndices(
        contextual_indices=ctx_indices,
        contextual_masks=ctx_masks,
        contextual_node_types=ctx_ntypes,
        temporal_indices=tmp_indices,
        temporal_masks=tmp_masks,
        temporal_deltas=tmp_deltas,
    )


def _make_batch_with_extra(batch_size: int) -> HeteroData:
    """HeteroData with extra nodes to simulate NeighborLoader output."""
    data = HeteroData()
    total_adm = batch_size + 10
    data["admission"].x = torch.randn(total_adm, 64)
    data["drug"].x = torch.randn(batch_size * 2, 32)
    data["diagnosis"].x = torch.randn(batch_size * 2, 48)
    data.batch_size = batch_size
    return data


class TestForwardWithHopIndices:
    def test_hop_indices_activates_transformer(self):
        """Providing HopIndices instead of explicit tensors → h_hhgat is not None."""
        torch.manual_seed(42)
        config = _make_config()
        model = TD4DDModel(config)
        model.eval()

        batch = _make_batch_with_extra(B)
        hop_idx = _make_hop_indices(B, D, N, K, 2, 2)
        aux = _make_aux_edges(B)

        out = model(batch, aux, hop_indices=hop_idx)

        assert out["h_hhgat"] is not None
        assert out["h_diff"] is not None
        assert out["logits"].shape == (B, 1)

    def test_backward_compat_explicit_tensors(self):
        """Existing explicit tensor interface still works (no regressions)."""
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
        assert out["h_hhgat"] is not None

    def test_explicit_tensors_take_precedence(self):
        """When both explicit tensors and hop_indices are given, explicit wins."""
        torch.manual_seed(42)
        config = _make_config(use_diffusion=False)
        model = TD4DDModel(config)
        model.eval()

        batch = _make_batch_with_extra(B)
        ctx, tmp_n, tmp_d = _make_transformer_inputs(B, D, N, K, 2, 2)
        hop_idx = _make_hop_indices(B, D, N, K, 2, 2)

        # Both provided — explicit should win (auto-gather not triggered)
        out = model(
            batch,
            contextual_hop_neighbors=ctx,
            temporal_hop_neighbors=tmp_n,
            temporal_hop_deltas=tmp_d,
            hop_indices=hop_idx,
        )
        assert out["h_hhgat"] is not None


class TestGatherShapes:
    def test_gather_produces_correct_shapes(self):
        """_gather_hop_neighbors produces correct (B, N, d) shapes."""
        torch.manual_seed(42)
        config = _make_config()
        model = TD4DDModel(config)

        batch = _make_batch_with_extra(B)
        hop_idx = _make_hop_indices(B, D, N, K, 2, 2)

        # Manually project
        projected = {}
        for ntype in batch.node_types:
            if ntype in model.proj and hasattr(batch[ntype], "x"):
                projected[ntype] = model.proj[ntype](batch[ntype].x)

        ctx, tmp, tmp_d = model._gather_hop_neighbors(projected, hop_idx)

        assert ctx is not None
        assert len(ctx) == 2  # num_paths
        for path in ctx:
            assert len(path) == K  # k_hops
            for hop_tensor in path:
                assert hop_tensor.shape == (B, N, D)

        assert tmp is not None
        assert len(tmp) == 2  # k_temporal
        for t in tmp:
            assert t.shape == (B, N, D)


class TestMaskZerosPadding:
    def test_padded_positions_zero(self):
        """Padded positions (mask=False) have zero features."""
        torch.manual_seed(42)
        config = _make_config()
        model = TD4DDModel(config)

        batch = _make_batch_with_extra(B)
        hop_idx = _make_hop_indices(B, D, N, K, 2, 2)

        projected = {}
        for ntype in batch.node_types:
            if ntype in model.proj and hasattr(batch[ntype], "x"):
                projected[ntype] = model.proj[ntype](batch[ntype].x)

        ctx, _, _ = model._gather_hop_neighbors(projected, hop_idx)

        # Check first path, first hop — last 2 positions are masked
        mask = hop_idx.contextual_masks[0][0]  # (B, N)
        gathered = ctx[0][0]  # (B, N, D)
        # Where mask is False, features should be zero
        padded = gathered[~mask]
        assert torch.all(padded == 0.0)


class TestGradientFlowThroughGather:
    def test_gradients_reach_proj(self):
        """Gradients flow through gather back to self.proj weights."""
        torch.manual_seed(42)
        config = _make_config(use_diffusion=False)
        model = TD4DDModel(config)
        model.train()

        batch = _make_batch_with_extra(B)
        hop_idx = _make_hop_indices(B, D, N, K, 2, 2)

        out = model(batch, hop_indices=hop_idx)
        out["logits"].sum().backward()

        assert model.proj["admission"].weight.grad is not None
        assert model.proj["drug"].weight.grad is not None
