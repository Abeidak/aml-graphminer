"""Loss functions for the extremely imbalanced laundering labels (≈ 0.1 % positives)."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import GNNConfig

LossFn = Callable[[Tensor, Tensor], Tensor]


def binary_focal_loss_with_logits(
    logits: Tensor, targets: Tensor, gamma: float = 2.0, alpha: float = 0.25
) -> Tensor:
    """Focal loss (Lin et al., 2017) for binary targets given as logits.

    ``alpha`` weights the positive class, ``(1 - p_t) ** gamma`` down-weights easy examples.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t) ** gamma * bce).mean()


def build_criterion(config: GNNConfig) -> LossFn:
    """Loss selected by ``config.loss`` (``"bce"`` with ``pos_weight`` or ``"focal"``).

    The returned function is device-agnostic: constants are created on the logits' device.
    """
    if config.loss == "focal":
        gamma, alpha = config.focal_gamma, config.focal_alpha

        def focal(logits: Tensor, targets: Tensor) -> Tensor:
            return binary_focal_loss_with_logits(logits, targets, gamma, alpha)

        return focal

    weight = float(config.pos_weight)

    def weighted_bce(logits: Tensor, targets: Tensor) -> Tensor:
        pos_weight = torch.full((), weight, device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    return weighted_bce


__all__ = ["LossFn", "binary_focal_loss_with_logits", "build_criterion"]
