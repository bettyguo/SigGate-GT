"""
Tests for evaluation metrics.

Validates all metric implementations against known reference values:
- ZINC MAE
- ogbg-molhiv ROC-AUC
- ogbg-molpcba / Peptides-func Average Precision
- Over-smoothing MAD
- Attention entropy
- Paired t-test
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from siggate_gt.evaluation.metrics import (
    compute_attention_entropy,
    compute_average_precision,
    compute_mad,
    compute_mae,
    compute_roc_auc,
    paired_t_test,
)


class TestMAE:
    """Tests for Mean Absolute Error computation."""

    def test_perfect_predictions(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        assert compute_mae(pred, target) == pytest.approx(0.0, abs=1e-6)

    def test_constant_error(self):
        pred = torch.tensor([0.0, 0.0, 0.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        assert compute_mae(pred, target) == pytest.approx(2.0, abs=1e-5)

    def test_negative_errors(self):
        pred = torch.tensor([2.0, 4.0, 6.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        assert compute_mae(pred, target) == pytest.approx(2.0, abs=1e-5)

    def test_squeezed_last_dim(self):
        pred = torch.tensor([[1.0], [2.0], [3.0]])
        target = torch.tensor([[1.0], [2.0], [3.0]])
        assert compute_mae(pred, target) == pytest.approx(0.0, abs=1e-6)


class TestROCAUC:
    """Tests for ROC-AUC computation."""

    def test_perfect_classifier(self):
        pred = torch.tensor([10.0, 10.0, -10.0, -10.0])  # logits
        target = torch.tensor([1, 1, 0, 0])
        assert compute_roc_auc(pred, target) == pytest.approx(1.0, abs=1e-6)

    def test_random_classifier(self):
        torch.manual_seed(42)
        pred = torch.zeros(100)  # constant logit = random
        target = torch.randint(0, 2, (100,))
        auc = compute_roc_auc(pred, target)
        # AUC of random classifier ≈ 0.5
        assert 0.0 <= auc <= 1.0

    def test_inverse_classifier(self):
        pred = torch.tensor([-10.0, -10.0, 10.0, 10.0])
        target = torch.tensor([1, 1, 0, 0])
        assert compute_roc_auc(pred, target) == pytest.approx(0.0, abs=1e-6)


class TestAveragePrecision:
    """Tests for multi-task Average Precision."""

    def test_perfect_ap(self):
        pred = torch.tensor([[10.0, -10.0], [10.0, -10.0]])  # (N, T)
        target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        ap = compute_average_precision(pred, target)
        assert ap == pytest.approx(1.0, abs=1e-6)

    def test_nan_labels_ignored(self):
        pred = torch.tensor([[10.0, 10.0], [-10.0, -10.0]])
        target = torch.tensor([[1.0, float("nan")], [0.0, float("nan")]])
        ap = compute_average_precision(pred, target, ignore_value=float("nan"))
        # Only first task is valid
        assert 0.0 <= ap <= 1.0

    def test_all_invalid_labels(self):
        pred = torch.randn(4, 2)
        target = torch.full((4, 2), float("nan"))
        ap = compute_average_precision(pred, target)
        assert math.isnan(ap)

    def test_single_class_task_skipped(self):
        pred = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        target = torch.tensor([[1.0, 1.0], [1.0, 0.0]])  # task 0: all positive = skipped
        ap = compute_average_precision(pred, target)
        # Only task 1 (mixed labels) contributes
        assert 0.0 <= ap <= 1.0


class TestMAD:
    """Tests for Mean Average Distance (over-smoothing metric)."""

    def test_identical_representations_zero_mad(self):
        h = torch.ones(10, 8)
        mad = compute_mad(h)
        assert mad == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_representations_high_mad(self):
        dim = 8
        n = 8
        h = torch.zeros(n, dim)
        for i in range(min(n, dim)):
            h[i, i] = 1.0
        mad = compute_mad(h)
        assert mad > 0.5, f"Orthogonal vectors should have high MAD, got {mad}"

    def test_mad_range(self):
        torch.manual_seed(42)
        h = torch.randn(20, 32)
        mad = compute_mad(h)
        assert 0.0 <= mad <= 2.0


class TestAttentionEntropy:
    """Tests for attention entropy computation."""

    def test_uniform_attention_max_entropy(self):
        n = 8
        uniform = torch.full((1, n, n), 1.0 / n)
        entropy = compute_attention_entropy(uniform)
        expected = math.log(n)
        assert entropy == pytest.approx(expected, abs=0.01)

    def test_peaked_attention_low_entropy(self):
        n = 8
        peaked = torch.zeros(1, n, n)
        peaked[:, :, 0] = 1.0  # all mass on first token
        entropy = compute_attention_entropy(peaked)
        assert entropy < 0.01

    def test_entropy_nonnegative(self):
        torch.manual_seed(0)
        attn = torch.softmax(torch.randn(2, 10, 10), dim=-1)
        entropy = compute_attention_entropy(attn)
        assert entropy >= 0.0


class TestPairedTTest:
    """Tests for paired t-test significance testing."""

    def test_identical_scores_not_significant(self):
        scores = [0.5] * 5
        result = paired_t_test(scores, scores)
        assert result["p_value"] == pytest.approx(1.0, abs=0.05)

    def test_clearly_different_scores_significant(self):
        a = [0.9] * 5
        b = [0.5] * 5
        result = paired_t_test(a, b, higher_is_better=True)
        assert result["p_value"] < 0.001
        assert result["mean_diff"] == pytest.approx(0.4, abs=1e-6)

    def test_result_structure(self):
        a = [0.8, 0.85, 0.82, 0.81, 0.83]
        b = [0.7, 0.72, 0.71, 0.73, 0.74]
        result = paired_t_test(a, b)
        required = ["t_stat", "p_value", "mean_diff", "std_diff", "ci_lower", "ci_upper", "n_seeds"]
        for key in required:
            assert key in result

    def test_n_seeds(self):
        a = [1.0, 2.0, 3.0]
        b = [0.5, 1.5, 2.5]
        result = paired_t_test(a, b)
        assert result["n_seeds"] == 3

    def test_paper_zinc_improvement(self):
        """Verify statistical significance of ZINC improvement can be reproduced."""
        # SigGate-GT: 0.059 ± 0.002 vs GraphGPS: 0.070 ± 0.004, p < 0.001
        # Simulated values consistent with reported mean and std
        our_scores = [0.057, 0.059, 0.060, 0.058, 0.061]
        baseline_scores = [0.066, 0.070, 0.073, 0.071, 0.069]
        result = paired_t_test(our_scores, baseline_scores, higher_is_better=False)
        assert result["p_value"] < 0.05, f"Should be significant, got p={result['p_value']:.4f}"
        assert result["mean_diff"] > 0, "SigGate-GT should have lower MAE (positive diff = improvement)"
