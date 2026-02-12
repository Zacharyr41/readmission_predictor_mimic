"""Tests for src.gnn.losses — modular loss functions."""

import torch
from torch import nn

from src.gnn.losses import (
    LossConfig,
    build_classification_loss,
    compute_total_loss,
    sigmoid_focal_loss,
)


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestLosses:
    def test_bce_pos_weight(self):
        """80/20 imbalance → pos_weight ≈ 4.0."""
        config = LossConfig(cls_loss_type="bce")
        labels = torch.cat([torch.zeros(80), torch.ones(20)])
        loss_fn = build_classification_loss(config, labels)

        assert hasattr(loss_fn, "pos_weight")
        assert abs(loss_fn.pos_weight.item() - 4.0) < 0.01

    def test_focal_loss_shape(self):
        """Scalar output, > 0."""
        torch.manual_seed(42)
        pred = torch.randn(16)
        target = torch.randint(0, 2, (16,)).float()
        loss = sigmoid_focal_loss(pred, target)

        assert loss.ndim == 0
        assert loss.item() > 0

    def test_focal_loss_behavior(self):
        """Easy (logits=10) vs hard (logits=0.1) for label=1; easy < hard."""
        target = torch.ones(16)
        easy_loss = sigmoid_focal_loss(torch.full((16,), 10.0), target)
        hard_loss = sigmoid_focal_loss(torch.full((16,), 0.1), target)

        assert easy_loss < hard_loss

    def test_label_smoothing(self):
        """smoothing=0.1: 0→0.05, 1→0.95."""
        config = LossConfig(cls_loss_type="bce_smoothed", label_smoothing=0.1)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        loss_fn = build_classification_loss(config, labels)

        logits = torch.zeros(4)
        targets = torch.tensor([0.0, 1.0, 0.0, 1.0])

        loss_smooth = loss_fn(logits, targets)

        # Manually compute with smoothed targets: 0→0.05, 1→0.95
        smoothed = targets * 0.9 + 0.05  # [0.05, 0.95, 0.05, 0.95]
        pw = torch.tensor([1.0])  # 2 pos, 2 neg → pw = 1.0
        manual = nn.BCEWithLogitsLoss(pos_weight=pw)(logits, smoothed)

        assert torch.allclose(loss_smooth, manual, atol=1e-6)

    def test_compute_total_loss(self):
        """total = cls + diff_weight * diff, all keys present."""
        config = LossConfig(diff_weight=0.5)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        cls_fn = build_classification_loss(config, labels)

        logits = torch.randn(4)
        L_diff = torch.tensor(2.0)

        result = compute_total_loss(cls_fn, logits, labels, L_diff, config)

        assert set(result.keys()) == {"total", "cls", "diff"}
        expected_total = result["cls"] + 0.5 * L_diff
        assert torch.allclose(result["total"], expected_total)

    def test_auto_pos_weight(self):
        """10 pos / 90 neg → pos_weight ≈ 9.0."""
        config = LossConfig(cls_loss_type="bce")
        labels = torch.cat([torch.zeros(90), torch.ones(10)])
        loss_fn = build_classification_loss(config, labels)

        assert abs(loss_fn.pos_weight.item() - 9.0) < 0.01
