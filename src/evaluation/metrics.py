"""
Evaluation metrics for all five benchmarks.

Implements the evaluation protocols used in the paper:
- ZINC: Mean Absolute Error (lower is better)
- ogbg-molhiv: ROC-AUC (higher is better)
- ogbg-molpcba: Average Precision (AP) across 128 tasks (higher is better)
- Peptides-func: Average Precision (AP) across 10 classes (higher is better)
- Peptides-struct: Mean Absolute Error (lower is better)

Statistical testing utilities for paired t-tests across multiple seeds
are also provided.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_mae(pred: Tensor, target: Tensor) -> float:
    """
    Compute Mean Absolute Error for regression.

    Args:
        pred: Predictions, shape (N,) or (N, 1).
        target: Ground-truth values, shape (N,) or (N, 1).

    Returns:
        MAE as a Python float.
    """
    pred = pred.squeeze(-1).cpu().float()
    target = target.squeeze(-1).cpu().float()
    return float(torch.mean(torch.abs(pred - target)).item())


def compute_roc_auc(pred: Tensor, target: Tensor) -> float:
    """
    Compute ROC-AUC for binary classification (ogbg-molhiv protocol).

    Applies sigmoid to raw logits before computing AUC.

    Args:
        pred: Raw logits, shape (N,) or (N, 1).
        target: Binary labels {0, 1}, shape (N,) or (N, 1).

    Returns:
        ROC-AUC as a Python float in [0, 1].

    Raises:
        ValueError: If only one class is present in target.
    """
    pred_np = torch.sigmoid(pred.squeeze(-1)).cpu().numpy()
    target_np = target.squeeze(-1).cpu().long().numpy()
    return float(roc_auc_score(target_np, pred_np))


def compute_average_precision(
    pred: Tensor,
    target: Tensor,
    ignore_value: float = float("nan"),
) -> float:
    """
    Compute mean Average Precision (AP) for multi-task binary classification.

    Follows the OGB evaluation protocol: tasks where all labels are missing
    or contain only one class are excluded.

    Args:
        pred: Logits, shape (N, num_tasks).
        target: Binary labels with possible NaN/missing, shape (N, num_tasks).
        ignore_value: Sentinel for missing labels.

    Returns:
        Mean AP over valid tasks, as a Python float.
    """
    pred_sigmoid = torch.sigmoid(pred).cpu().numpy()  # (N, T)
    target_np = target.cpu().numpy().astype(float)   # (N, T)

    ap_list: list[float] = []
    num_tasks = target_np.shape[1]

    for task_idx in range(num_tasks):
        t = target_np[:, task_idx]
        p = pred_sigmoid[:, task_idx]

        if ignore_value != ignore_value:  # NaN
            valid = ~np.isnan(t)
        else:
            valid = t != ignore_value

        if valid.sum() == 0:
            continue

        t_valid = t[valid]
        p_valid = p[valid]

        # Skip if only one class present
        if len(np.unique(t_valid)) < 2:
            continue

        ap_list.append(average_precision_score(t_valid, p_valid))

    if not ap_list:
        return float("nan")
    return float(np.mean(ap_list))


# ---------------------------------------------------------------------------
# Benchmark-specific evaluators
# ---------------------------------------------------------------------------

def evaluate_zinc(pred: Tensor, target: Tensor) -> dict[str, float]:
    """Evaluate on ZINC (MAE metric)."""
    return {"mae": compute_mae(pred, target)}


def evaluate_molhiv(pred: Tensor, target: Tensor) -> dict[str, float]:
    """Evaluate on ogbg-molhiv (ROC-AUC metric)."""
    return {"rocauc": compute_roc_auc(pred, target)}


def evaluate_molpcba(pred: Tensor, target: Tensor) -> dict[str, float]:
    """Evaluate on ogbg-molpcba (Average Precision metric)."""
    return {"ap": compute_average_precision(pred, target)}


def evaluate_peptides_func(pred: Tensor, target: Tensor) -> dict[str, float]:
    """Evaluate on Peptides-func (Average Precision metric)."""
    return {"ap": compute_average_precision(pred, target, ignore_value=float("nan"))}


def evaluate_peptides_struct(pred: Tensor, target: Tensor) -> dict[str, float]:
    """Evaluate on Peptides-struct (MAE metric)."""
    return {"mae": compute_mae(pred, target)}


def get_evaluator(task: str):
    """
    Return the evaluation function for a given task.

    Args:
        task: Benchmark identifier. One of:
              'zinc', 'molhiv', 'molpcba', 'peptides_func', 'peptides_struct'.

    Returns:
        Callable that takes (pred, target) and returns a dict of metrics.
    """
    evaluators = {
        "zinc": evaluate_zinc,
        "molhiv": evaluate_molhiv,
        "molpcba": evaluate_molpcba,
        "peptides_func": evaluate_peptides_func,
        "peptides_struct": evaluate_peptides_struct,
    }
    if task not in evaluators:
        raise ValueError(
            f"Unknown task '{task}'. Choose from: {list(evaluators.keys())}"
        )
    return evaluators[task]


# ---------------------------------------------------------------------------
# Attention entropy analysis
# ---------------------------------------------------------------------------

def compute_attention_entropy(
    attn_weights: Tensor,
    eps: float = 1e-9,
) -> float:
    """
    Compute mean per-row attention entropy H_bar.

    H_bar = -1/n * sum_i sum_j A_ij * log(A_ij)

    This measures how diffuse vs. concentrated attention distributions are.
    Higher entropy indicates more uniform attention (less collapsed).

    Args:
        attn_weights: Attention weight matrix, shape (batch, num_nodes, num_nodes)
                      or (num_nodes, num_nodes). Values must be in [0, 1] and
                      each row should sum to ~1 (softmax output).
        eps: Small constant for numerical stability in log.

    Returns:
        Mean per-row entropy as a float.
    """
    if attn_weights.dim() == 2:
        attn_weights = attn_weights.unsqueeze(0)

    # Clamp for numerical stability
    attn = attn_weights.clamp(min=eps)
    entropy = -(attn * attn.log()).sum(dim=-1)  # (batch, num_nodes)
    return float(entropy.mean().item())


# ---------------------------------------------------------------------------
# Over-smoothing: Mean Average Distance (MAD)
# ---------------------------------------------------------------------------

def compute_mad(node_features: Tensor, batch: Tensor | None = None) -> float:
    """
    Compute Mean Average Distance (MAD) of node representations.

    MAD measures how distinct node representations are from each other.
    Higher MAD indicates less over-smoothing (more diverse representations).

    MAD = mean over all node pairs (i, j) of cosine_distance(h_i, h_j)
    where cosine_distance = 1 - cosine_similarity.

    Args:
        node_features: Node feature matrix, shape (num_nodes, hidden_dim).
        batch: Optional node-to-graph mapping for per-graph computation.
               If None, MAD is computed globally.

    Returns:
        Mean Average Distance as a float.
    """
    if batch is None:
        return _mad_single_graph(node_features)

    # Per-graph MAD, then averaged
    num_graphs = int(batch.max().item()) + 1
    mad_values: list[float] = []
    for g in range(num_graphs):
        mask = batch == g
        h_g = node_features[mask]
        if h_g.shape[0] < 2:
            continue
        mad_values.append(_mad_single_graph(h_g))
    if not mad_values:
        return 0.0
    return float(np.mean(mad_values))


def _mad_single_graph(h: Tensor) -> float:
    """Compute MAD for a single graph's node features."""
    h = torch.nn.functional.normalize(h, p=2, dim=-1)  # unit vectors
    cos_sim = torch.mm(h, h.t())  # (N, N)
    cos_dist = 1.0 - cos_sim      # cosine distance

    # Exclude self-distances (diagonal)
    n = h.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=h.device)
    mad = cos_dist[mask].mean()
    return float(mad.item())


# ---------------------------------------------------------------------------
# Statistical significance: paired t-test
# ---------------------------------------------------------------------------

def paired_t_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    higher_is_better: bool = True,
) -> dict[str, float]:
    """
    Perform a paired t-test to compare two methods across seeds.

    Tests the null hypothesis that the mean difference is zero.
    Reports the t-statistic, p-value, mean improvement, and 95% CI.

    Args:
        scores_a: Metric values for method A (e.g., proposed model) per seed.
        scores_b: Metric values for method B (e.g., baseline) per seed.
        higher_is_better: If True, positive difference = improvement for A.

    Returns:
        Dictionary with keys: t_stat, p_value, mean_diff, std_diff,
        ci_lower, ci_upper, n_seeds.
    """
    from scipy import stats  # type: ignore[import]

    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    diff = a - b if higher_is_better else b - a

    n = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    stderr = std_diff / math.sqrt(n)

    t_stat, p_value = stats.ttest_rel(a, b)

    # 95% confidence interval
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lower = mean_diff - t_crit * stderr
    ci_upper = mean_diff + t_crit * stderr

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_seeds": n,
    }
