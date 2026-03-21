"""
Shared pytest fixtures for SigGate-GT tests.

Provides lightweight graph data and model instances for fast unit tests.
GPU tests are marked with @pytest.mark.gpu and skipped by default.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def make_small_graph(
    num_nodes: int = 6,
    num_edges: int = 10,
    node_dim: int = 9,
    edge_dim: int = 3,
    pe_dim: int = 32,
    seed: int = 42,
) -> Data:
    """Create a small synthetic graph for testing."""
    torch.manual_seed(seed)
    x = torch.randn(num_nodes, node_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_dim)
    pe = torch.randn(num_nodes, pe_dim)
    y = torch.randn(1, 1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pe=pe, y=y, num_nodes=num_nodes)


def make_batch_of_graphs(
    num_graphs: int = 4,
    num_nodes_per_graph: int = 6,
    num_edges_per_graph: int = 10,
    node_dim: int = 9,
    edge_dim: int = 3,
    pe_dim: int = 32,
) -> Batch:
    """Create a batch of small synthetic graphs."""
    graphs = [
        make_small_graph(
            num_nodes=num_nodes_per_graph,
            num_edges=num_edges_per_graph,
            node_dim=node_dim,
            edge_dim=edge_dim,
            pe_dim=pe_dim,
            seed=i,
        )
        for i in range(num_graphs)
    ]
    return Batch.from_data_list(graphs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_graph() -> Data:
    """Single small graph for unit tests."""
    return make_small_graph()


@pytest.fixture
def graph_batch() -> Batch:
    """Batch of 4 small graphs for batch processing tests."""
    return make_batch_of_graphs()


@pytest.fixture
def device() -> torch.device:
    """CPU device (use device_gpu for GPU tests)."""
    return torch.device("cpu")


@pytest.fixture
def device_gpu() -> torch.device:
    """GPU device, skip if not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def tiny_siggate_model():
    """Tiny SigGate-GT model for fast tests (2 layers, dim 16)."""
    from siggate_gt.models.siggate_gps import SigGateGT
    return SigGateGT(
        in_dim=9,
        edge_in_dim=3,
        pe_dim=32,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        out_dim=1,
    )


@pytest.fixture
def tiny_attention():
    """Tiny SigGate attention module for unit tests."""
    from siggate_gt.models.attention import SigGateMultiHeadAttention
    return SigGateMultiHeadAttention(embed_dim=16, num_heads=2)
