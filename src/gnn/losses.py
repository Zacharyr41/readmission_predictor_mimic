"""Modular loss functions for binary classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


@dataclass
class LossConfig:
    """Configuration for loss computation."""

    cls_loss_type: str = "bce"
    pos_weight: float | None = None
    label_smoothing: float = 0.05
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    diff_weight: float = 1.0


def sigmoid_focal_loss(
    pred: Tensor,
    target: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Sigmoid focal loss for imbalanced binary classification."""
    try:
        from torchvision.ops import sigmoid_focal_loss as _tv_focal

        return _tv_focal(pred, target, alpha=alpha, gamma=gamma, reduction="mean")
    except ImportError:
        pass

    p = torch.sigmoid(pred)
    ce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * ((1 - p_t) ** gamma) * ce
    return loss.mean()


def build_classification_loss(
    config: LossConfig, labels: Tensor
) -> Callable[[Tensor, Tensor], Tensor]:
    """Build a classification loss function based on config.

    Parameters
    ----------
    config : LossConfig
        Loss configuration.
    labels : Tensor
        Training labels, used to auto-compute pos_weight if not specified.

    Returns
    -------
    Callable[[Tensor, Tensor], Tensor]
        Loss function taking (logits, targets) -> scalar loss.
    """
    if config.pos_weight is not None:
        pw = config.pos_weight
    else:
        n_pos = labels.sum().float()
        n_neg = (labels.numel() - n_pos).float()
        pw = (n_neg / n_pos).item() if n_pos > 0 else 1.0

    if config.cls_loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))

    if config.cls_loss_type == "focal":

        def focal_fn(logits: Tensor, targets: Tensor) -> Tensor:
            return sigmoid_focal_loss(
                logits, targets, config.focal_alpha, config.focal_gamma
            )

        return focal_fn

    if config.cls_loss_type == "bce_smoothed":
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
        eps = config.label_smoothing

        def smoothed_fn(logits: Tensor, targets: Tensor) -> Tensor:
            smoothed = targets * (1 - eps) + 0.5 * eps
            return criterion(logits, smoothed)

        return smoothed_fn

    raise ValueError(f"Unknown cls_loss_type: {config.cls_loss_type!r}")


def compute_total_loss(
    cls_loss_fn: Callable[[Tensor, Tensor], Tensor],
    logits: Tensor,
    labels: Tensor,
    L_diff: Tensor,
    config: LossConfig,
) -> dict[str, Tensor]:
    """Compute total loss combining classification and diffusion losses.

    Returns
    -------
    dict[str, Tensor]
        Keys: total, cls, diff
    """
    cls_loss = cls_loss_fn(logits, labels)
    diff_loss = config.diff_weight * L_diff
    total = cls_loss + diff_loss
    return {"total": total, "cls": cls_loss, "diff": diff_loss}
