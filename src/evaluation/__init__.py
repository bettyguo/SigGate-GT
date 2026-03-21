"""Evaluation utilities for SigGate-GT."""

from siggate_gt.evaluation.evaluator import Evaluator
from siggate_gt.evaluation.metrics import (
    compute_attention_entropy,
    compute_average_precision,
    compute_mad,
    compute_mae,
    compute_roc_auc,
    get_evaluator,
    paired_t_test,
)


__all__ = [
    "Evaluator",
    "get_evaluator",
    "compute_mae",
    "compute_roc_auc",
    "compute_average_precision",
    "compute_attention_entropy",
    "compute_mad",
    "paired_t_test",
]
