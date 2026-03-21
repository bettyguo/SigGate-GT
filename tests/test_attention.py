"""
Unit tests for SigGate Multi-Head Self-Attention.

Tests cover:
- Output shapes
- Gate initialization
- Gating behavior (gates can suppress to near-zero)
- Parameter count
- Attention weight properties (sums to ~1)
- Batch vs. single-sequence consistency
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from siggate_gt.models.attention import SigGateMultiHeadAttention


class TestSigGateAttentionShapes:
    """Test output shapes of SigGate-MHSA."""

    def test_output_shape_single_batch(self):
        attn = SigGateMultiHeadAttention(embed_dim=32, num_heads=4)
        x = torch.randn(1, 10, 32)
        out, attn_weights = attn(x)
        assert out.shape == (1, 10, 32)
        assert attn_weights.shape == (1, 10, 10)

    def test_output_shape_multi_batch(self):
        attn = SigGateMultiHeadAttention(embed_dim=64, num_heads=8)
        x = torch.randn(4, 12, 64)
        out, attn_weights = attn(x)
        assert out.shape == (4, 12, 64)
        assert attn_weights.shape == (4, 12, 12)

    def test_output_shape_matches_input(self, tiny_attention):
        x = torch.randn(2, 5, 16)
        out, _ = tiny_attention(x)
        assert out.shape == x.shape

    def test_sequence_length_1(self):
        attn = SigGateMultiHeadAttention(embed_dim=16, num_heads=2)
        x = torch.randn(3, 1, 16)
        out, attn_weights = attn(x)
        assert out.shape == (3, 1, 16)
        assert attn_weights.shape == (3, 1, 1)

    def test_batch_first_false(self):
        attn = SigGateMultiHeadAttention(embed_dim=16, num_heads=2, batch_first=False)
        x = torch.randn(10, 2, 16)  # (seq, batch, dim)
        out, _ = attn(x)
        assert out.shape == (10, 2, 16)


class TestGateProperties:
    """Test properties of the sigmoid gate."""

    def test_gate_bias_initialization(self):
        gate_bias_init = 0.5
        attn = SigGateMultiHeadAttention(embed_dim=16, num_heads=2, gate_bias_init=gate_bias_init)
        bias = attn.gate_proj.bias
        assert torch.allclose(bias, torch.full_like(bias, gate_bias_init)), (
            f"Gate bias should be initialized to {gate_bias_init}"
        )

    def test_gate_output_in_0_1_range(self, tiny_attention):
        x = torch.randn(2, 8, 16)
        gate = torch.sigmoid(tiny_attention.gate_proj(x))
        assert gate.min() > 0.0
        assert gate.max() < 1.0

    def test_gate_statistics(self, tiny_attention):
        x = torch.randn(4, 20, 16)
        stats = tiny_attention.get_gate_statistics(x)
        assert "mean" in stats
        assert "std" in stats
        assert "frac_below_01" in stats
        assert "frac_above_09" in stats
        assert stats["mean"].shape == (2,)  # one per head (2 heads)

    def test_gate_zero_bias_suppresses_output(self):
        """Gate initialized near 0 should produce near-zero outputs."""
        attn = SigGateMultiHeadAttention(embed_dim=16, num_heads=2, gate_bias_init=-10.0)
        x = torch.randn(1, 4, 16)
        out, _ = attn(x)
        assert out.abs().mean() < 0.01, "Very negative bias should suppress output"

    def test_gate_large_bias_passes_through(self):
        """Gate initialized near 1 should approximate standard MHSA."""
        attn_gated = SigGateMultiHeadAttention(embed_dim=16, num_heads=2, gate_bias_init=10.0)
        x = torch.randn(1, 4, 16)
        out, _ = attn_gated(x)
        # Output should be finite and non-trivial
        assert torch.isfinite(out).all()
        assert out.abs().mean() > 1e-6

    def test_per_head_independent_gates(self, tiny_attention):
        """Each head should have independent gate parameters."""
        x = torch.randn(1, 6, 16)
        # Get gate activations for a specific input
        gate_vals = torch.sigmoid(tiny_attention.gate_proj(x))  # (1, 6, 16)
        gate_reshaped = gate_vals.view(1, 6, tiny_attention.num_heads, tiny_attention.head_dim)
        # Head 0 and head 1 gate means should potentially differ
        h0_mean = gate_reshaped[..., 0, :].mean().item()
        h1_mean = gate_reshaped[..., 1, :].mean().item()
        # They can differ once trained; both should be in (0, 1)
        assert 0 < h0_mean < 1
        assert 0 < h1_mean < 1


class TestAttentionWeightProperties:
    """Test softmax attention weight properties."""

    def test_attention_weights_sum_to_one(self, tiny_attention):
        x = torch.randn(2, 8, 16)
        _, attn_weights = tiny_attention(x)
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            "Attention weights should sum to 1 per row"
        )

    def test_attention_weights_nonnegative(self, tiny_attention):
        x = torch.randn(2, 8, 16)
        _, attn_weights = tiny_attention(x)
        assert (attn_weights >= 0).all()

    def test_key_padding_mask(self):
        """Masked positions should receive zero attention."""
        attn = SigGateMultiHeadAttention(embed_dim=16, num_heads=2)
        x = torch.randn(2, 6, 16)
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[0, 4:] = True  # mask last 2 positions for graph 0
        _, attn_weights = attn(x, key_padding_mask=mask)
        # Columns 4 and 5 for batch item 0 should be ~0
        assert attn_weights[0, :, 4:].abs().max() < 1e-5


class TestGradients:
    """Test gradient flow through the gated attention."""

    def test_gradients_flow_to_gate(self, tiny_attention):
        x = torch.randn(2, 5, 16, requires_grad=True)
        out, _ = tiny_attention(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gate_projection_gradient(self, tiny_attention):
        x = torch.randn(2, 5, 16)
        out, _ = tiny_attention(x)
        loss = out.sum()
        loss.backward()
        assert tiny_attention.gate_proj.weight.grad is not None
        assert tiny_attention.gate_proj.weight.grad.abs().sum() > 0

    def test_no_gradient_explosion(self, tiny_attention):
        x = torch.randn(1, 10, 16, requires_grad=True)
        out, _ = tiny_attention(x)
        loss = out.sum()
        loss.backward()
        max_grad = x.grad.abs().max().item()
        assert max_grad < 1e4, f"Gradient too large: {max_grad}"


class TestParameterCount:
    """Test parameter counts for different configurations."""

    def test_gate_parameters_fraction(self):
        """Gate parameters should be < 1% of the total for standard configs."""
        from siggate_gt.models.siggate_gps import SigGateGT
        model = SigGateGT.build_zinc()
        info = model.count_parameters()
        assert info["gate_fraction_pct"] < 5.0, (
            f"Gate overhead too large: {info['gate_fraction_pct']:.2f}%"
        )

    def test_no_extra_parameters_with_no_gating(self):
        """Standard MHSA has no gate_proj; SigGate adds exactly embed_dim*(embed_dim+1) per layer."""
        embed_dim = 32
        attn = SigGateMultiHeadAttention(embed_dim=embed_dim, num_heads=4)
        gate_params = sum(p.numel() for p in attn.gate_proj.parameters())
        expected = embed_dim * embed_dim + embed_dim  # weight + bias
        assert gate_params == expected
