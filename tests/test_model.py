"""
Unit tests for SigGate-GT model.

Tests cover:
- Forward pass shapes for all five benchmark tasks
- Build classmethod correctness
- Parameter counting
- No NaN/Inf in outputs
- Gradient flow
- Readout modes
"""

from __future__ import annotations

import pytest
import torch

from siggate_gt.models.siggate_gps import SigGateGT


class TestSigGateGTForwardPass:
    """Test forward pass of the full model."""

    def test_zinc_forward_single_graph(self, small_graph, device):
        model = SigGateGT.build_zinc().to(device)
        model.eval()

        data = small_graph.to(device)
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float()
        pe = data.pe.float()

        out = model(x=x, edge_index=edge_index, edge_attr=edge_attr, pe=pe, batch=None)
        assert out.shape == (1, 1), f"Expected (1, 1), got {out.shape}"
        assert torch.isfinite(out).all()

    def test_zinc_forward_batched(self, graph_batch, device):
        model = SigGateGT.build_zinc().to(device)
        model.eval()

        data = graph_batch.to(device)
        out = model(
            x=data.x.float(),
            edge_index=data.edge_index,
            edge_attr=data.edge_attr.float(),
            pe=data.pe.float(),
            batch=data.batch,
        )
        assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"
        assert torch.isfinite(out).all()

    def test_ogb_forward(self, graph_batch, device):
        model = SigGateGT.build_ogb(out_dim=1).to(device)
        model.eval()
        data = graph_batch.to(device)
        out = model(
            x=data.x.float(),
            edge_index=data.edge_index,
            edge_attr=data.edge_attr.float(),
            pe=data.pe.float(),
            batch=data.batch,
        )
        assert out.shape == (4, 1)
        assert torch.isfinite(out).all()

    def test_molpcba_forward_128_tasks(self, graph_batch, device):
        model = SigGateGT.build_ogb(out_dim=128).to(device)
        model.eval()
        data = graph_batch.to(device)
        out = model(
            x=data.x.float(),
            edge_index=data.edge_index,
            edge_attr=data.edge_attr.float(),
            pe=data.pe.float(),
            batch=data.batch,
        )
        assert out.shape == (4, 128)

    def test_lrgb_forward(self, graph_batch, device):
        model = SigGateGT.build_lrgb(out_dim=10).to(device)
        model.eval()
        data = graph_batch.to(device)
        out = model(
            x=data.x.float(),
            edge_index=data.edge_index,
            edge_attr=data.edge_attr.float(),
            pe=data.pe.float(),
            batch=data.batch,
        )
        assert out.shape == (4, 10)

    def test_forward_without_pe(self, graph_batch, device):
        """Model should work with pe=None by skipping PE concatenation."""
        model = SigGateGT(
            in_dim=9,
            edge_in_dim=3,
            pe_dim=0,
            hidden_dim=16,
            num_layers=2,
            num_heads=2,
            out_dim=1,
        ).to(device)
        model.eval()
        data = graph_batch.to(device)
        out = model(
            x=data.x.float(),
            edge_index=data.edge_index,
            edge_attr=data.edge_attr.float(),
            pe=None,
            batch=data.batch,
        )
        assert out.shape == (4, 1)

    def test_forward_without_edge_attr(self, device):
        """Model should handle missing edge features."""
        from tests.conftest import make_small_graph
        import torch
        graph = make_small_graph(edge_dim=0)
        graph.edge_attr = None

        model = SigGateGT(
            in_dim=9, edge_in_dim=3, pe_dim=32,
            hidden_dim=16, num_layers=1, num_heads=2, out_dim=1
        ).to(device)
        model.eval()
        out = model(
            x=graph.x.float().to(device),
            edge_index=graph.edge_index.to(device),
            edge_attr=None,
            pe=graph.pe.float().to(device),
            batch=None,
        )
        assert out.shape == (1, 1)


class TestSigGateGTBuildMethods:
    """Test build classmethods."""

    def test_build_zinc(self):
        model = SigGateGT.build_zinc()
        assert model.hidden_dim == 64
        assert model.num_layers == 10
        info = model.count_parameters()
        assert info["total"] > 0

    def test_build_ogb(self):
        model = SigGateGT.build_ogb()
        assert model.hidden_dim == 256
        assert model.num_layers == 5

    def test_build_lrgb(self):
        model = SigGateGT.build_lrgb()
        assert model.hidden_dim == 128
        assert model.num_layers == 10

    def test_zinc_parameter_budget(self):
        """ZINC model should be within ~500K parameter budget."""
        model = SigGateGT.build_zinc()
        info = model.count_parameters()
        assert info["total"] <= 550_000, (
            f"ZINC model exceeds 550K params: {info['total']}"
        )


class TestParameterCounting:
    """Test parameter counting utility."""

    def test_count_parameters_structure(self, tiny_siggate_model):
        info = tiny_siggate_model.count_parameters()
        required_keys = [
            "total", "gate_projection", "gate_fraction_pct",
            "node_encoder", "edge_encoder", "gps_layers", "output_head",
        ]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    def test_total_equals_sum_of_parts(self, tiny_siggate_model):
        info = tiny_siggate_model.count_parameters()
        # Total should equal all trainable parameters
        manual_count = sum(
            p.numel() for p in tiny_siggate_model.parameters() if p.requires_grad
        )
        assert info["total"] == manual_count

    def test_gate_overhead_small(self):
        model = SigGateGT.build_zinc()
        info = model.count_parameters()
        assert info["gate_fraction_pct"] < 5.0


class TestGradientFlow:
    """Test gradient flow through the full model."""

    def test_full_model_gradient_flow(self, graph_batch, device):
        model = SigGateGT.build_zinc().to(device)
        data = graph_batch.to(device)

        out = model(
            x=data.x.float(),
            edge_index=data.edge_index,
            edge_attr=data.edge_attr.float(),
            pe=data.pe.float(),
            batch=data.batch,
        )
        loss = out.sum()
        loss.backward()

        # All parameters should have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"NaN gradient for {name}"

    def test_gate_projection_receives_gradient(self, graph_batch, device):
        model = SigGateGT.build_zinc().to(device)
        data = graph_batch.to(device)
        out = model(
            x=data.x.float(),
            edge_index=data.edge_index,
            edge_attr=data.edge_attr.float(),
            pe=data.pe.float(),
            batch=data.batch,
        )
        loss = out.sum()
        loss.backward()

        for layer_idx, layer in enumerate(model.layers):
            gate_grad = layer.global_attn.gate_proj.weight.grad
            assert gate_grad is not None, f"No gate gradient at layer {layer_idx}"
            assert gate_grad.abs().sum() > 0, f"Zero gate gradient at layer {layer_idx}"


class TestReadoutModes:
    """Test different graph-level readout strategies."""

    @pytest.mark.parametrize("readout", ["mean", "sum"])
    def test_readout_modes(self, graph_batch, device, readout):
        model = SigGateGT(
            in_dim=9, edge_in_dim=3, pe_dim=32,
            hidden_dim=16, num_layers=1, num_heads=2, out_dim=1,
            readout=readout,
        ).to(device)
        data = graph_batch.to(device)
        out = model(
            x=data.x.float(),
            edge_index=data.edge_index,
            edge_attr=data.edge_attr.float(),
            pe=data.pe.float(),
            batch=data.batch,
        )
        assert out.shape == (4, 1)
        assert torch.isfinite(out).all()
