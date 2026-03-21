"""
Dataset and dataloader utilities for graph benchmark datasets.

Supports all five benchmarks used in the paper:
- ZINC (PyG built-in)
- OGB: ogbg-molhiv, ogbg-molpcba (via ogb library)
- LRGB: Peptides-func, Peptides-struct (via PyG LRGBDataset)

Positional encodings (LapPE + RWSE) are computed and attached to
each graph following the GraphGPS preprocessing protocol.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Positional encoding computation
# ---------------------------------------------------------------------------

def compute_laplacian_pe(
    data: Data,
    pe_dim: int = 16,
    norm: bool = True,
) -> Data:
    """
    Compute Laplacian eigenvector positional encodings (LapPE).

    Computes the smallest k=pe_dim non-trivial eigenvectors of the
    symmetric normalized graph Laplacian. Signs are randomized for
    basis invariance (SignNet-style).

    Args:
        data: PyG Data object with edge_index and num_nodes.
        pe_dim: Number of eigenvectors to retain.
        norm: If True, use normalized Laplacian.

    Returns:
        Data with added 'lappe' attribute of shape (num_nodes, pe_dim).
    """
    import numpy as np
    import scipy.sparse as sp  # type: ignore[import]
    from scipy.sparse.linalg import eigsh  # type: ignore[import]

    n = data.num_nodes
    if n is None:
        n = int(data.x.shape[0]) if data.x is not None else 0

    if n <= 2:
        data.lappe = torch.zeros(n, pe_dim)
        return data

    # Build adjacency matrix
    edge_index = data.edge_index.cpu().numpy()
    vals = np.ones(edge_index.shape[1])
    adj = sp.coo_matrix((vals, (edge_index[0], edge_index[1])), shape=(n, n))
    adj = (adj + adj.T).tocsr()
    adj.data = np.ones_like(adj.data)  # remove duplicate weights

    # Compute Laplacian
    if norm:
        degree = np.asarray(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        L = sp.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt
    else:
        degree = np.asarray(adj.sum(axis=1)).flatten()
        D = sp.diags(degree)
        L = D - adj

    # Compute k smallest eigenvectors (skip eigenvalue 0)
    k = min(pe_dim + 1, n - 1)
    if k < 2:
        data.lappe = torch.zeros(n, pe_dim)
        return data

    try:
        eigenvalues, eigenvectors = eigsh(L.astype(float), k=k, which="SM", tol=1e-5)
    except Exception:
        data.lappe = torch.zeros(n, pe_dim)
        return data

    # Sort by eigenvalue and skip trivial (≈0) ones
    sort_idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, sort_idx]

    # Skip eigenvectors with eigenvalue ≈ 0
    nontrivial = np.abs(eigenvalues[sort_idx]) > 1e-5
    eigenvectors = eigenvectors[:, nontrivial]

    # Pad or truncate to pe_dim
    if eigenvectors.shape[1] < pe_dim:
        pad = np.zeros((n, pe_dim - eigenvectors.shape[1]))
        eigenvectors = np.concatenate([eigenvectors, pad], axis=1)
    else:
        eigenvectors = eigenvectors[:, :pe_dim]

    # Random sign flip for basis invariance
    signs = np.random.choice([-1, 1], size=pe_dim)
    eigenvectors = eigenvectors * signs

    data.lappe = torch.from_numpy(eigenvectors.astype(np.float32))
    return data


def compute_rwse(
    data: Data,
    walk_length: int = 16,
) -> Data:
    """
    Compute Random Walk Structural Encoding (RWSE).

    For each node i, computes the probability of landing at i after
    t steps of a random walk starting at i, for t = 1, ..., walk_length.

    Args:
        data: PyG Data object with edge_index and num_nodes.
        walk_length: Number of random walk steps k.

    Returns:
        Data with added 'rwse' attribute of shape (num_nodes, walk_length).
    """
    n = data.num_nodes
    if n is None:
        n = int(data.x.shape[0]) if data.x is not None else 0

    if n == 0:
        data.rwse = torch.zeros(0, walk_length)
        return data

    edge_index = data.edge_index
    device = edge_index.device

    # Build row-normalized adjacency (transition matrix)
    src, dst = edge_index[0], edge_index[1]
    deg = torch.zeros(n, device=device)
    deg.scatter_add_(0, src, torch.ones(src.shape[0], device=device))
    deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))

    # Edge weights = 1 / degree[src]
    edge_weight = deg_inv[src]  # (E,)

    # Sparse transition matrix
    adj = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        size=(n, n),
        device=device,
    ).to_dense()

    # Compute landing probabilities: diag(A^t) for t=1..walk_length
    rw_landing = torch.zeros(n, walk_length, device=device)
    mat = adj.clone()
    for step in range(walk_length):
        rw_landing[:, step] = mat.diagonal()
        mat = mat @ adj

    data.rwse = rw_landing.cpu().float()
    return data


def attach_pe(data: Data, lappe_dim: int = 16, rwse_steps: int = 16) -> Data:
    """
    Compute and attach combined positional encodings to a graph.

    Concatenates LapPE and RWSE into a single 'pe' attribute of shape
    (num_nodes, lappe_dim + rwse_steps).

    Args:
        data: Input PyG Data object.
        lappe_dim: LapPE dimension.
        rwse_steps: RWSE walk length.

    Returns:
        Data with 'pe' attribute.
    """
    data = compute_laplacian_pe(data, pe_dim=lappe_dim)
    data = compute_rwse(data, walk_length=rwse_steps)
    data.pe = torch.cat([data.lappe, data.rwse], dim=-1)
    return data


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_zinc(
    root: str = "dataset/zinc",
    subset: bool = True,
    pe_dim: int = 32,
    pre_transform: Callable[[Data], Data] | None = None,
) -> tuple[Dataset, Dataset, Dataset]:  # type: ignore[type-arg]
    """
    Load ZINC dataset with LapPE + RWSE positional encodings.

    Uses the 12K-molecule subset with 500K parameter budget as in the paper.

    Args:
        root: Root directory for dataset storage.
        subset: If True, load the 12K-molecule subset.
        pe_dim: Total PE dimension (split evenly between LapPE and RWSE).
        pre_transform: Optional additional pre-transform.

    Returns:
        Tuple of (train, val, test) datasets.
    """
    from torch_geometric.datasets import ZINC  # type: ignore[import]

    lappe_dim = pe_dim // 2
    rwse_steps = pe_dim // 2

    def transform(data: Data) -> Data:
        data = attach_pe(data, lappe_dim=lappe_dim, rwse_steps=rwse_steps)
        if pre_transform is not None:
            data = pre_transform(data)
        return data

    train_dataset = ZINC(root=root, subset=subset, split="train", pre_transform=transform)
    val_dataset = ZINC(root=root, subset=subset, split="val", pre_transform=transform)
    test_dataset = ZINC(root=root, subset=subset, split="test", pre_transform=transform)

    log.info(
        f"ZINC (subset={subset}): train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset


def load_ogb(
    name: str,
    root: str = "dataset",
    pe_dim: int = 32,
) -> tuple[Dataset, Dataset, Dataset]:  # type: ignore[type-arg]
    """
    Load OGB molecular benchmark dataset.

    Args:
        name: OGB dataset name, e.g. 'ogbg-molhiv' or 'ogbg-molpcba'.
        root: Root directory for dataset storage.
        pe_dim: Total PE dimension.

    Returns:
        Tuple of (train, val, test) datasets.
    """
    from ogb.graphproppred import PygGraphPropPredDataset  # type: ignore[import]

    lappe_dim = pe_dim // 2
    rwse_steps = pe_dim // 2

    def transform(data: Data) -> Data:
        return attach_pe(data, lappe_dim=lappe_dim, rwse_steps=rwse_steps)

    dataset = PygGraphPropPredDataset(name=name, root=root, pre_transform=transform)
    split_idx = dataset.get_idx_split()

    train_dataset = dataset[split_idx["train"]]
    val_dataset = dataset[split_idx["valid"]]
    test_dataset = dataset[split_idx["test"]]

    log.info(
        f"{name}: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset  # type: ignore[return-value]


def load_lrgb(
    name: str,
    root: str = "dataset",
    pe_dim: int = 32,
) -> tuple[Dataset, Dataset, Dataset]:  # type: ignore[type-arg]
    """
    Load Long Range Graph Benchmark (LRGB) dataset.

    Args:
        name: LRGB dataset name, e.g. 'Peptides-func' or 'Peptides-struct'.
        root: Root directory for dataset storage.
        pe_dim: Total PE dimension.

    Returns:
        Tuple of (train, val, test) datasets.
    """
    from torch_geometric.datasets import LRGBDataset  # type: ignore[import]

    lappe_dim = pe_dim // 2
    rwse_steps = pe_dim // 2

    def transform(data: Data) -> Data:
        return attach_pe(data, lappe_dim=lappe_dim, rwse_steps=rwse_steps)

    train_dataset = LRGBDataset(root=root, name=name, split="train", pre_transform=transform)
    val_dataset = LRGBDataset(root=root, name=name, split="val", pre_transform=transform)
    test_dataset = LRGBDataset(root=root, name=name, split="test", pre_transform=transform)

    log.info(
        f"{name}: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset


def build_dataloaders(
    task: str,
    root: str = "dataset",
    pe_dim: int = 32,
    batch_size_train: int = 32,
    batch_size_eval: int = 64,
    num_workers: int = 4,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:  # type: ignore[type-arg]
    """
    Build train/val/test DataLoaders for a given benchmark task.

    Batch sizes follow the paper's configuration:
    - ZINC: batch 32
    - OGB molhiv: batch 256
    - OGB molpcba: batch 512
    - LRGB: batch 64

    Args:
        task: Benchmark identifier.
        root: Root data directory.
        pe_dim: Total PE dimension (LapPE dim + RWSE steps).
        batch_size_train: Batch size for training.
        batch_size_eval: Batch size for val/test.
        num_workers: DataLoader worker processes.
        seed: Random seed for shuffle generator.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    from siggate_gt.utils.reproducibility import get_generator, get_worker_init_fn

    task_map = {
        "zinc": lambda: load_zinc(root=f"{root}/zinc", pe_dim=pe_dim),
        "molhiv": lambda: load_ogb("ogbg-molhiv", root=root, pe_dim=pe_dim),
        "molpcba": lambda: load_ogb("ogbg-molpcba", root=root, pe_dim=pe_dim),
        "peptides_func": lambda: load_lrgb("Peptides-func", root=root, pe_dim=pe_dim),
        "peptides_struct": lambda: load_lrgb("Peptides-struct", root=root, pe_dim=pe_dim),
    }

    if task not in task_map:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(task_map.keys())}")

    train_ds, val_ds, test_ds = task_map[task]()

    loader_kwargs: dict = {
        "num_workers": num_workers,
        "worker_init_fn": get_worker_init_fn(seed),
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        generator=get_generator(seed),
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size_eval, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size_eval, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
