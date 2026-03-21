"""
Loss functions for graph-level prediction tasks.

Covers all five benchmarks used in the paper:
- ZINC: L1 loss (MAE) for regression
- ogbg-molhiv: Binary cross-entropy for classification
- ogbg-molpcba: Binary cross-entropy with missing labels for multi-task
- Peptides-func: Binary cross-entropy for multi-label classification
- Peptides-struct: L1 loss (MAE) for regression
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MAELoss(nn.Module):
    """
    Mean Absolute Error loss for regression tasks (ZINC, Peptides-struct).

    Args:
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(pred.squeeze(-1), target.float(), reduction=self.reduction)


class BinaryCELoss(nn.Module):
    """
    Binary cross-entropy loss for single-label binary classification (ogbg-molhiv).

    Args:
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            pred.squeeze(-1), target.float(), reduction=self.reduction
        )


class MultiTaskBinaryCELoss(nn.Module):
    """
    Multi-task binary cross-entropy loss with missing-label handling.

    Used for ogbg-molpcba (128 bioassays) and Peptides-func (10 classes)
    where some labels may be missing (represented as NaN or -1).

    Invalid labels are masked out before loss computation following the
    OGB evaluation protocol.

    Args:
        ignore_value: Scalar sentinel value for missing labels. Default: float('nan').
    """

    def __init__(self, ignore_value: float = float("nan")) -> None:
        super().__init__()
        self.ignore_value = ignore_value

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: Logits, shape (batch, num_tasks).
            target: Binary labels with missing values, shape (batch, num_tasks).

        Returns:
            Scalar loss averaged over valid (non-missing) label positions.
        """
        target = target.float()
        if self.ignore_value != self.ignore_value:  # NaN check
            valid_mask = ~torch.isnan(target)
        else:
            valid_mask = target != self.ignore_value

        if not valid_mask.any():
            return pred.new_zeros(1).squeeze()

        loss = F.binary_cross_entropy_with_logits(
            pred[valid_mask], target[valid_mask], reduction="mean"
        )
        return loss


def build_loss(task: str) -> nn.Module:
    """
    Build the appropriate loss function for a given benchmark task.

    Args:
        task: One of 'zinc', 'molhiv', 'molpcba', 'peptides_func', 'peptides_struct'.

    Returns:
        Configured loss module.

    Raises:
        ValueError: If task is not recognized.
    """
    task = task.lower()
    if task in ("zinc", "peptides_struct"):
        return MAELoss()
    elif task == "molhiv":
        return BinaryCELoss()
    elif task in ("molpcba", "peptides_func"):
        return MultiTaskBinaryCELoss()
    else:
        raise ValueError(
            f"Unknown task '{task}'. Must be one of: "
            "zinc, molhiv, molpcba, peptides_func, peptides_struct"
        )
