"""Tests for src.gnn.diffusion — DDPM cross-view diffusion module."""

from unittest.mock import patch

import torch

from src.gnn.diffusion import (
    AuxiliaryGCNEncoder,
    CrossViewDenoiser,
    DiffusionModule,
    NoiseSchedule,
)

N, IN_DIM, HIDDEN, E = 20, 64, 128, 50


# ──────────────────────────────────────────────────────────────────────────────
# 1. NoiseSchedule
# ──────────────────────────────────────────────────────────────────────────────


class TestNoiseSchedule:
    def test_alpha_bars_monotonically_decreasing(self):
        """Alpha bars should start near 1, strictly decrease, and end near 0."""
        torch.manual_seed(42)
        ns = NoiseSchedule(T=100, beta_start=1e-4, beta_end=0.02)

        ab = ns.alpha_bars
        assert ab[0].item() > 0.99, "alpha_bars[0] should be ≈ 1 - beta_start"
        assert ab[-1].item() < 0.5, "alpha_bars[-1] should decay significantly"
        # Strictly decreasing
        diffs = ab[1:] - ab[:-1]
        assert (diffs < 0).all(), "alpha_bars should be strictly decreasing"

    def test_forward_diffusion_at_t0_recovers_h0(self):
        """At t=0, h_t ≈ h_0 (minimal noise)."""
        torch.manual_seed(42)
        ns = NoiseSchedule(T=100)
        h_0 = torch.randn(N, HIDDEN)
        t = torch.tensor(0)

        h_t, _ = ns.forward_diffusion(h_0, t)
        assert torch.allclose(h_t, h_0, atol=0.05)

    def test_forward_diffusion_at_tmax_is_noisy(self):
        """At t=T-1, h_t should be very different from h_0."""
        torch.manual_seed(42)
        ns = NoiseSchedule(T=100)
        h_0 = torch.randn(N, HIDDEN)
        t = torch.tensor(99)

        h_t, _ = ns.forward_diffusion(h_0, t)
        assert not torch.allclose(h_t, h_0, atol=0.5)
        # Variance should be approximately 1.0 (pure noise regime)
        assert 0.5 < h_t.var().item() < 2.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. AuxiliaryGCNEncoder
# ──────────────────────────────────────────────────────────────────────────────


class TestAuxiliaryGCNEncoder:
    def test_encoder_output_shape(self):
        """Input (20, 64), edge (2, 50) → output (20, 128)."""
        torch.manual_seed(42)
        enc = AuxiliaryGCNEncoder(in_dim=IN_DIM, hidden_dim=HIDDEN)
        x = torch.randn(N, IN_DIM)
        edge_index = torch.randint(0, N, (2, E))

        out = enc(x, edge_index)
        assert out.shape == (N, HIDDEN)

    def test_encoder_gradient_flow(self):
        """Gradients should reach conv1 and conv2 weights."""
        torch.manual_seed(42)
        enc = AuxiliaryGCNEncoder(in_dim=IN_DIM, hidden_dim=HIDDEN)
        x = torch.randn(N, IN_DIM)
        edge_index = torch.randint(0, N, (2, E))

        out = enc(x, edge_index)
        out.sum().backward()

        assert enc.conv1.lin.weight.grad is not None, "conv1.lin.weight.grad is None"
        assert enc.conv2.lin.weight.grad is not None, "conv2.lin.weight.grad is None"


# ──────────────────────────────────────────────────────────────────────────────
# 3. CrossViewDenoiser
# ──────────────────────────────────────────────────────────────────────────────


class TestCrossViewDenoiser:
    def test_denoiser_output_shape(self):
        """h_t_self (20, 128), h_t_other (20, 128), t=50 → (20, 128)."""
        torch.manual_seed(42)
        den = CrossViewDenoiser(hidden_dim=HIDDEN)
        h_self = torch.randn(N, HIDDEN)
        h_other = torch.randn(N, HIDDEN)
        t = torch.tensor(50)
        edge_index = torch.randint(0, N, (2, E))

        out = den(h_self, h_other, t, edge_index)
        assert out.shape == (N, HIDDEN)


# ──────────────────────────────────────────────────────────────────────────────
# 4. DiffusionModule
# ──────────────────────────────────────────────────────────────────────────────


class TestDiffusionModule:
    def test_training_step_shapes(self):
        """h_diff shape (20, 128), L_diff is scalar > 0."""
        torch.manual_seed(42)
        dm = DiffusionModule(in_dim=IN_DIM, hidden_dim=HIDDEN)
        x = torch.randn(N, IN_DIM)
        edges = [torch.randint(0, N, (2, E)), torch.randint(0, N, (2, E))]

        h_diff, L_diff = dm.training_step(x, edges)

        assert h_diff.shape == (N, HIDDEN)
        assert L_diff.ndim == 0
        assert L_diff.item() > 0

    def test_inference_determinism(self):
        """Same seed → identical inference results."""
        dm = DiffusionModule(in_dim=IN_DIM, hidden_dim=HIDDEN)
        dm.eval()
        x = torch.randn(N, IN_DIM)
        edges = [torch.randint(0, N, (2, E)), torch.randint(0, N, (2, E))]

        torch.manual_seed(42)
        result1 = dm.inference(x, edges)

        torch.manual_seed(42)
        result2 = dm.inference(x, edges)

        assert torch.equal(result1, result2)

    def test_ddim_step_count(self):
        """Denoiser forward should be called 2× per step (one per view)."""
        torch.manual_seed(42)
        ddim_steps = 10
        dm = DiffusionModule(in_dim=IN_DIM, hidden_dim=HIDDEN, ddim_steps=ddim_steps)
        dm.eval()
        x = torch.randn(N, IN_DIM)
        edges = [torch.randint(0, N, (2, E)), torch.randint(0, N, (2, E))]

        call_count = 0
        original_forward = dm.denoiser.forward

        def counting_forward(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_forward(*args, **kwargs)

        with patch.object(dm.denoiser, "forward", side_effect=counting_forward):
            dm.inference(x, edges)

        expected = ddim_steps * 2  # 2 views per step
        assert call_count == expected, f"Expected {expected} calls, got {call_count}"

    def test_loss_decreases(self):
        """Training on a small graph for 50 steps should reduce loss."""
        torch.manual_seed(42)
        n_nodes = 5
        # Fully connected edges
        src = torch.arange(n_nodes).repeat_interleave(n_nodes)
        dst = torch.arange(n_nodes).repeat(n_nodes)
        edge_index = torch.stack([src, dst])
        edges = [edge_index, edge_index]

        dm = DiffusionModule(in_dim=IN_DIM, hidden_dim=HIDDEN, T=50)
        optimizer = torch.optim.Adam(dm.parameters(), lr=1e-3)
        x = torch.randn(n_nodes, IN_DIM)

        losses = []
        for _ in range(50):
            _, loss = dm.training_step(x, edges)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )
