"""
SigGate-GT layer components.

Implements the SigGate-GPS layer that combines:
  - Local message passing (GatedGCN-style)
  - Global SigGate-MHSA
  - Position-wise feed-forward network
  - Layer normalization and residual connections

This follows the GraphGPS modular design:
    h_i^(l) = MLP( h_i^(l-1) + MPNN(·) + SigGate-MHSA(·) )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from siggate_gt.models.attention import SigGateMultiHeadAttention


# ---------------------------------------------------------------------------
# Feed-forward network
# ---------------------------------------------------------------------------

class FeedForwardNetwork(nn.Module):
    """
    Two-layer position-wise FFN with GELU activation.

    Args:
        embed_dim: Input and output dimension.
        ffn_dim: Hidden dimension (typically 2× or 4× embed_dim).
        dropout: Dropout probability.
    """

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Local MPNN (GatedGCN-style)
# ---------------------------------------------------------------------------

class GatedGCNLayer(nn.Module):
    """
    Simplified GatedGCN message-passing layer for local neighborhood aggregation.

    Implements edge-level sigmoid gating as used in the GraphGPS local MPNN:
        e_{ij} = σ(h_i W_s + h_j W_t + e_{ij} W_e)
        m_i    = sum_j (e_{ij} ⊙ h_j W_m)  /  (sum_j e_{ij} + ε)
        h_i'   = BN( h_i W_r + m_i )

    For graphs without explicit edge features, edge features are set to zeros.

    Args:
        node_dim: Node feature dimension.
        edge_dim: Edge feature dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self, node_dim: int, edge_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Node projections
        self.W_src = nn.Linear(node_dim, edge_dim)
        self.W_dst = nn.Linear(node_dim, edge_dim)
        self.W_edge = nn.Linear(edge_dim, edge_dim)
        self.W_msg = nn.Linear(node_dim, node_dim)
        self.W_res = nn.Linear(node_dim, node_dim)

        # Edge update
        self.edge_out = nn.Linear(edge_dim, edge_dim)

        self.bn_node = nn.BatchNorm1d(node_dim)
        self.bn_edge = nn.BatchNorm1d(edge_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Node features, shape (num_nodes, node_dim).
            edge_index: Graph connectivity, shape (2, num_edges).
                        Row 0 = source nodes, row 1 = destination nodes.
            edge_attr: Edge features, shape (num_edges, edge_dim).
                       If None, zeros are used.

        Returns:
            x_out: Updated node features, shape (num_nodes, node_dim).
            edge_out: Updated edge features, shape (num_edges, edge_dim).
        """
        src, dst = edge_index[0], edge_index[1]  # (E,)
        num_nodes = x.shape[0]

        if edge_attr is None:
            edge_attr = x.new_zeros(edge_index.shape[1], self.edge_dim)

        # Edge-level sigmoid gating
        gate = torch.sigmoid(
            self.W_src(x[src]) + self.W_dst(x[dst]) + self.W_edge(edge_attr)
        )  # (E, edge_dim)

        # Message: gated neighbor features
        msg_val = self.W_msg(x[src]) * gate  # (E, node_dim)

        # Aggregation (mean with normalization)
        # gate_sum shape: (num_nodes, node_dim)
        gate_sum = x.new_zeros(num_nodes, msg_val.shape[1]).scatter_add_(
            0, dst.unsqueeze(1).expand_as(msg_val), gate
        ) + 1e-6

        agg = x.new_zeros(num_nodes, msg_val.shape[1]).scatter_add_(
            0, dst.unsqueeze(1).expand_as(msg_val), msg_val
        )
        agg = agg / gate_sum

        # Node update with residual
        x_out = self.bn_node(self.W_res(x) + agg)
        x_out = F.relu(x_out)
        x_out = self.dropout(x_out)

        # Edge update
        edge_new = self.bn_edge(self.edge_out(gate))
        edge_new = F.relu(edge_new)

        return x_out, edge_new


# ---------------------------------------------------------------------------
# SigGate-GPS Layer
# ---------------------------------------------------------------------------

class SigGateGPSLayer(nn.Module):
    """
    A single SigGate-GPS layer combining local MPNN and global SigGate-MHSA.

    For a batch of graphs packed into a single feature matrix:
        h_i^(l) = LayerNorm( FFN( h_i + MPNN(h) + SigGate-MHSA(h) ) )

    The global attention operates over all nodes in a single graph
    (batch dimension is per-graph during the forward pass).

    Args:
        embed_dim: Node embedding dimension d.
        num_heads: Number of attention heads K.
        ffn_dim: Feed-forward hidden dimension (default: 2 * embed_dim).
        edge_dim: Edge feature dimension for the local MPNN.
        dropout: Dropout probability.
        attn_dropout: Dropout on attention weights.
        gate_bias_init: Initial gate bias (σ(gate_bias_init) ≈ 0.62 when 0.5).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int | None = None,
        edge_dim: int | None = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        gate_bias_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        if ffn_dim is None:
            ffn_dim = 2 * embed_dim
        if edge_dim is None:
            edge_dim = embed_dim

        # Local MPNN (GatedGCN)
        self.local_mpnn = GatedGCNLayer(embed_dim, edge_dim, dropout=dropout)

        # Global SigGate-MHSA
        self.global_attn = SigGateMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            gate_bias_init=gate_bias_init,
            batch_first=True,
        )

        # FFN
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout=dropout)

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            x: Node features, shape (total_nodes, embed_dim).
            edge_index: Edge connectivity, shape (2, num_edges).
            edge_attr: Edge features, shape (num_edges, edge_dim).
            batch: Node-to-graph assignment vector, shape (total_nodes,).
                   Required for global attention (separates graphs in a batch).
            attn_mask: Optional additive attention mask.

        Returns:
            x_out: Updated node features, shape (total_nodes, embed_dim).
            edge_attr_out: Updated edge features, shape (num_edges, edge_dim).
        """
        # ------------------------------------------------------------------ #
        # 1. Local MPNN
        # ------------------------------------------------------------------ #
        x_mpnn, edge_attr_out = self.local_mpnn(x, edge_index, edge_attr)
        x = self.norm1(x + self.dropout(x_mpnn))

        # ------------------------------------------------------------------ #
        # 2. Global SigGate-MHSA (per-graph, using batch vector)
        # ------------------------------------------------------------------ #
        x_attn = self._global_attention(x, batch, attn_mask)
        x = self.norm2(x + self.dropout(x_attn))

        # ------------------------------------------------------------------ #
        # 3. FFN
        # ------------------------------------------------------------------ #
        x = self.norm3(x + self.dropout(self.ffn(x)))

        return x, edge_attr_out

    def _global_attention(
        self,
        x: Tensor,
        batch: Tensor | None,
        attn_mask: Tensor | None,
    ) -> Tensor:
        """
        Apply global SigGate-MHSA across each individual graph in the batch.

        When batch is None (single-graph mode), attention is applied globally
        without padding. When batch is provided, graphs are padded to equal
        sizes and masked before applying attention.

        Args:
            x: Node features, shape (total_nodes, embed_dim).
            batch: Graph assignment, shape (total_nodes,). None = single graph.
            attn_mask: Optional additive attention mask.

        Returns:
            Attention output, shape (total_nodes, embed_dim).
        """
        if batch is None:
            # Single graph: straightforward
            x_3d = x.unsqueeze(0)  # (1, N, d)
            out, _ = self.global_attn(x_3d, attn_mask=attn_mask)
            return out.squeeze(0)

        # Multiple graphs: pad to max graph size
        num_graphs = int(batch.max().item()) + 1
        device = x.device

        # Count nodes per graph
        node_counts = batch.bincount(minlength=num_graphs)  # (G,)
        max_nodes = int(node_counts.max().item())

        # Build padded tensor and key_padding_mask
        x_padded = x.new_zeros(num_graphs, max_nodes, self.embed_dim)
        key_padding_mask = torch.ones(num_graphs, max_nodes, dtype=torch.bool, device=device)

        offset = 0
        for g in range(num_graphs):
            n = int(node_counts[g].item())
            x_padded[g, :n] = x[offset : offset + n]
            key_padding_mask[g, :n] = False  # valid positions
            offset += n

        attn_out, _ = self.global_attn(
            x_padded,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )  # (G, max_nodes, d)

        # Unpack back to (total_nodes, d)
        out = x.new_zeros(x.shape[0], self.embed_dim)
        offset = 0
        for g in range(num_graphs):
            n = int(node_counts[g].item())
            out[offset : offset + n] = attn_out[g, :n]
            offset += n

        return out
