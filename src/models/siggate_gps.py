"""
SigGate-GT: Full model definition.

Assembles the complete SigGate-GT model as a stack of SigGate-GPS layers
with input projection, positional encoding injection, and graph-level pooling
for graph classification/regression tasks.

Architecture overview:
    Input: node features x, edge features e, positional encodings pe
    1. Input projection: x_0 = Linear(x || pe)
    2. L × SigGate-GPS layers: local MPNN + SigGate-MHSA + FFN
    3. Graph-level readout: mean/sum pooling
    4. Output head: Linear(h_graph) -> prediction
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from siggate_gt.models.layers import SigGateGPSLayer


class SigGateGT(nn.Module):
    """
    SigGate Graph Transformer (SigGate-GT).

    A stack of SigGate-GPS layers for graph-level prediction tasks.
    Supports molecular property prediction benchmarks:
    - ZINC (regression, MAE)
    - OGB molecular (classification/multi-label, AUC / AP)
    - LRGB Peptides (classification, regression)

    Args:
        in_dim: Input node feature dimension.
        edge_in_dim: Input edge feature dimension.
        pe_dim: Positional encoding dimension (LapPE + RWSE concatenated).
        hidden_dim: Hidden embedding dimension d.
        num_layers: Number of SigGate-GPS layers L.
        num_heads: Number of attention heads K per layer.
        out_dim: Output dimension (number of prediction targets).
        ffn_multiplier: FFN hidden dim = ffn_multiplier * hidden_dim.
        dropout: Dropout rate.
        attn_dropout: Attention dropout rate.
        gate_bias_init: Initial gate bias for sigmoid gates.
        readout: Graph-level readout mode ('mean' or 'sum').
    """

    def __init__(
        self,
        in_dim: int,
        edge_in_dim: int,
        pe_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        out_dim: int,
        ffn_multiplier: int = 2,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        gate_bias_init: float = 0.5,
        readout: str = "mean",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.readout = readout

        # Node input projection (features + positional encodings)
        self.node_encoder = nn.Linear(in_dim + pe_dim, hidden_dim)

        # Edge input projection
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)

        # Stack of SigGate-GPS layers
        self.layers = nn.ModuleList([
            SigGateGPSLayer(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_multiplier * hidden_dim,
                edge_dim=hidden_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
                gate_bias_init=gate_bias_init,
            )
            for _ in range(num_layers)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        pe: Tensor | None = None,
        batch: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features, shape (total_nodes, in_dim).
            edge_index: Edge connectivity, shape (2, num_edges).
            edge_attr: Edge features, shape (num_edges, edge_in_dim).
            pe: Positional encodings (LapPE + RWSE), shape (total_nodes, pe_dim).
            batch: Node-to-graph assignment, shape (total_nodes,).

        Returns:
            Graph-level predictions, shape (num_graphs, out_dim).
        """
        # ------------------------------------------------------------------ #
        # 1. Input encoding
        # ------------------------------------------------------------------ #
        if pe is not None:
            x = torch.cat([x, pe], dim=-1)  # (N, in_dim + pe_dim)

        h = self.node_encoder(x)  # (N, hidden_dim)

        if edge_attr is not None:
            edge_h = self.edge_encoder(edge_attr)  # (E, hidden_dim)
        else:
            edge_h = None

        # ------------------------------------------------------------------ #
        # 2. SigGate-GPS layers
        # ------------------------------------------------------------------ #
        for layer in self.layers:
            h, edge_h = layer(h, edge_index, edge_h, batch)

        # ------------------------------------------------------------------ #
        # 3. Graph-level readout
        # ------------------------------------------------------------------ #
        h_graph = self._readout(h, batch)  # (num_graphs, hidden_dim)

        # ------------------------------------------------------------------ #
        # 4. Prediction
        # ------------------------------------------------------------------ #
        return self.output_head(h_graph)

    def _readout(self, x: Tensor, batch: Tensor | None) -> Tensor:
        """
        Aggregate node representations to graph level.

        Args:
            x: Node features, shape (total_nodes, hidden_dim).
            batch: Node-to-graph assignment, shape (total_nodes,).
                   None = single graph.

        Returns:
            Graph features, shape (num_graphs, hidden_dim).
        """
        if batch is None:
            if self.readout == "mean":
                return x.mean(dim=0, keepdim=True)
            return x.sum(dim=0, keepdim=True)

        num_graphs = int(batch.max().item()) + 1
        out = x.new_zeros(num_graphs, x.shape[1])

        if self.readout == "mean":
            counts = x.new_zeros(num_graphs).scatter_add_(
                0, batch, x.new_ones(x.shape[0])
            )
            out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
            out = out / (counts.unsqueeze(1) + 1e-8)
        else:
            out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)

        return out

    def count_parameters(self) -> dict[str, int]:
        """
        Count trainable parameters, broken down by component.

        Returns:
            Dictionary with parameter counts per component.
        """
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        gate_params = sum(
            count(layer.global_attn.gate_proj)
            for layer in self.layers
        )
        total = count(self)

        return {
            "total": total,
            "gate_projection": gate_params,
            "gate_fraction_pct": round(100.0 * gate_params / total, 2),
            "node_encoder": count(self.node_encoder),
            "edge_encoder": count(self.edge_encoder),
            "gps_layers": count(self.layers),
            "output_head": count(self.output_head),
        }

    @classmethod
    def build_zinc(
        cls,
        in_dim: int = 28,
        edge_in_dim: int = 4,
        pe_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 10,
        num_heads: int = 8,
        out_dim: int = 1,
        dropout: float = 0.0,
        gate_bias_init: float = 0.5,
    ) -> "SigGateGT":
        """
        Instantiate SigGate-GT with ZINC benchmark configuration.

        Hyperparameters match the paper's 500K parameter budget setting:
        10 layers, dim 64, 8 heads, 2000 epochs.
        """
        return cls(
            in_dim=in_dim,
            edge_in_dim=edge_in_dim,
            pe_dim=pe_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            out_dim=out_dim,
            ffn_multiplier=2,
            dropout=dropout,
            gate_bias_init=gate_bias_init,
            readout="mean",
        )

    @classmethod
    def build_ogb(
        cls,
        in_dim: int = 9,
        edge_in_dim: int = 3,
        pe_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 5,
        num_heads: int = 8,
        out_dim: int = 1,
        dropout: float = 0.1,
        gate_bias_init: float = 0.5,
    ) -> "SigGateGT":
        """
        Instantiate SigGate-GT with OGB molecular benchmark configuration.

        Hyperparameters match the paper's OGB setting:
        5 layers, dim 256, 8 heads, 100 epochs.
        """
        return cls(
            in_dim=in_dim,
            edge_in_dim=edge_in_dim,
            pe_dim=pe_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            out_dim=out_dim,
            ffn_multiplier=2,
            dropout=dropout,
            gate_bias_init=gate_bias_init,
            readout="mean",
        )

    @classmethod
    def build_lrgb(
        cls,
        in_dim: int = 9,
        edge_in_dim: int = 7,
        pe_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 10,
        num_heads: int = 8,
        out_dim: int = 1,
        dropout: float = 0.0,
        gate_bias_init: float = 0.5,
    ) -> "SigGateGT":
        """
        Instantiate SigGate-GT with LRGB Peptides configuration.

        Hyperparameters match the paper's LRGB setting:
        10 layers, dim 128, 8 heads, 200 epochs.
        """
        return cls(
            in_dim=in_dim,
            edge_in_dim=edge_in_dim,
            pe_dim=pe_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            out_dim=out_dim,
            ffn_multiplier=2,
            dropout=dropout,
            gate_bias_init=gate_bias_init,
            readout="mean",
        )
