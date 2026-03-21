"""
SigGate Multi-Head Self-Attention.

Implements element-wise sigmoid gating on the attention output for each
attention head. Each gate is a learned, input-dependent projection that
modulates the standard scaled dot-product attention output per dimension,
allowing heads to selectively suppress uninformative connections.

Gate equation per head k:
    head_k = softmax(Q_k K_k^T / sqrt(d_k)) V_k  ⊙  σ(H W^g_k + b^g_k)
where W^g_k ∈ R^{d×d_k}, b^g_k ∈ R^{d_k}, initialized so that
σ(b^g_k) ≈ 0.62 (bias = 0.5).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SigGateMultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with per-head element-wise sigmoid output gating.

    For each head k the gate is computed as:
        g_k = σ(H W^g_k + b^g_k)   ∈ (0, 1)^{n × d_k}
    and the gated head output is:
        head_k^gated = (softmax(Q_k K_k^T / sqrt(d_k)) V_k) ⊙ g_k

    The final output is:
        SigGate-MHSA(H) = Concat(head_1^gated, ..., head_K^gated) W^O

    Args:
        embed_dim: Total embedding dimension d.
        num_heads: Number of attention heads K. embed_dim must be divisible by num_heads.
        dropout: Dropout probability on attention weights.
        gate_bias_init: Initial value for gate bias (scalar). Gate starts at σ(gate_bias_init).
        batch_first: If True, input/output tensors have shape (batch, seq, embed_dim).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        gate_bias_init: float = 0.5,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.scale = math.sqrt(self.head_dim)

        # Standard QKV and output projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Per-head sigmoid gate projections: one W^g_k per head
        # Stored as a single weight matrix of shape (num_heads * head_dim, embed_dim)
        # so that gate for all heads can be computed in one matmul then split.
        self.gate_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_gate_parameters(gate_bias_init)

    def _reset_gate_parameters(self, gate_bias_init: float) -> None:
        """Initialize gate projection weights and biases."""
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_bias_init)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Input node features.
               Shape: (batch, seq_len, embed_dim) if batch_first=True.
            attn_mask: Additive attention mask of shape (seq_len, seq_len)
                or (batch * num_heads, seq_len, seq_len).
            key_padding_mask: Boolean mask of shape (batch, seq_len).
                True positions are ignored.

        Returns:
            output: Gated attention output, same shape as x.
            attn_weights: Averaged attention weights, shape (batch, seq_len, seq_len).
        """
        if not self.batch_first:
            x = x.transpose(0, 1)

        batch_size, seq_len, _ = x.shape

        # ------------------------------------------------------------------ #
        # Compute Q, K, V
        # ------------------------------------------------------------------ #
        q = self.q_proj(x)  # (B, N, d)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, num_heads, N, head_dim)
        def split_heads(t: Tensor) -> Tensor:
            return t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # ------------------------------------------------------------------ #
        # Scaled dot-product attention
        # ------------------------------------------------------------------ #
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, N, N)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                scores = scores + attn_mask.unsqueeze(0)

        if key_padding_mask is not None:
            # (B, 1, 1, N) -> broadcast over heads and query positions
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # (B, H, N, N)

        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Standard attention output: (B, H, N, head_dim)
        attn_out = torch.matmul(attn_weights, v)

        # ------------------------------------------------------------------ #
        # Sigmoid gate: per-head, per-dimension, input-dependent
        # g_k = σ(H W^g_k + b^g_k)
        # ------------------------------------------------------------------ #
        # gate_proj maps (B, N, d) -> (B, N, d), then split heads
        gate = torch.sigmoid(self.gate_proj(x))  # (B, N, d)
        gate = gate.view(batch_size, seq_len, self.num_heads, self.head_dim)
        gate = gate.transpose(1, 2)  # (B, H, N, head_dim)

        # Element-wise gating: the sigmoid gate modulates attention output
        attn_out = attn_out * gate  # (B, H, N, head_dim)

        # ------------------------------------------------------------------ #
        # Merge heads and output projection
        # ------------------------------------------------------------------ #
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_out)

        if not self.batch_first:
            output = output.transpose(0, 1)

        # Return averaged attention weights for analysis/visualization
        avg_attn = attn_weights.mean(dim=1)  # (B, N, N)
        return output, avg_attn

    def get_gate_statistics(self, x: Tensor) -> dict[str, Tensor]:
        """
        Compute gate activation statistics for analysis.

        Args:
            x: Input features, shape (batch, seq_len, embed_dim).

        Returns:
            Dictionary with per-head gate mean, std, fraction < 0.1, fraction > 0.9.
        """
        with torch.no_grad():
            gate = torch.sigmoid(self.gate_proj(x))  # (B, N, d)
            gate = gate.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
            # gate: (B, N, H, head_dim)

            stats: dict[str, Tensor] = {
                "mean": gate.mean(dim=(0, 1, 3)),       # (H,)
                "std": gate.std(dim=(0, 1, 3)),          # (H,)
                "frac_below_01": (gate < 0.1).float().mean(dim=(0, 1, 3)),
                "frac_above_09": (gate > 0.9).float().mean(dim=(0, 1, 3)),
                "overall_mean": gate.mean(),
                "overall_std": gate.std(),
            }
        return stats
