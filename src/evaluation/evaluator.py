"""
Evaluator for SigGate-GT on graph benchmark datasets.

Provides a unified interface for running evaluation across all five
benchmark tasks, collecting predictions, and computing metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from siggate_gt.evaluation.metrics import (
    compute_attention_entropy,
    compute_mad,
    get_evaluator,
)


log = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates a SigGate-GT model on a graph benchmark dataset.

    Args:
        task: Benchmark task identifier. One of:
              'zinc', 'molhiv', 'molpcba', 'peptides_func', 'peptides_struct'.
        compute_diagnostics: If True, also compute MAD and attention entropy
                             during evaluation (requires model to return them).
    """

    def __init__(self, task: str, compute_diagnostics: bool = False) -> None:
        self.task = task.lower()
        self.compute_diagnostics = compute_diagnostics
        self._metric_fn = get_evaluator(self.task)

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,  # type: ignore[type-arg]
        device: torch.device,
    ) -> dict[str, float]:
        """
        Run full evaluation over a dataset split.

        Args:
            model: The SigGate-GT model (or any graph model with compatible API).
            loader: DataLoader yielding PyG-style Data batches.
            device: Computation device.

        Returns:
            Dictionary of metric names to scalar values.
        """
        model.eval()

        all_preds: list[Tensor] = []
        all_targets: list[Tensor] = []
        all_node_feats: list[Tensor] = []

        for batch in loader:
            batch = batch.to(device)

            # Extract graph data fields
            x = batch.x.float()
            edge_index = batch.edge_index
            edge_attr = getattr(batch, "edge_attr", None)
            if edge_attr is not None:
                edge_attr = edge_attr.float()

            pe = getattr(batch, "pe", None)
            if pe is not None:
                pe = pe.float()

            graph_batch = getattr(batch, "batch", None)
            y = batch.y

            # Forward pass
            pred = model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pe=pe,
                batch=graph_batch,
            )

            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

        pred_all = torch.cat(all_preds, dim=0)
        target_all = torch.cat(all_targets, dim=0)

        metrics = self._metric_fn(pred_all, target_all)

        return metrics

    @torch.no_grad()
    def compute_node_diagnostics(
        self,
        model: nn.Module,
        loader: DataLoader,  # type: ignore[type-arg]
        device: torch.device,
        num_batches: int = 10,
    ) -> dict[str, Any]:
        """
        Compute over-smoothing (MAD) and attention entropy diagnostics.

        This method runs a limited number of batches to collect intermediate
        representations. It hooks into the first and last SigGate-GPS layers
        to capture node features and attention weights.

        Args:
            model: SigGate-GT model.
            loader: DataLoader for diagnostics (train or val split).
            device: Computation device.
            num_batches: Number of mini-batches to use.

        Returns:
            Dictionary with keys: 'mad_per_layer', 'attn_entropy_per_layer'.
        """
        model.eval()

        # Collect node features from each layer via hooks
        layer_outputs: dict[int, list[Tensor]] = {}
        layer_attn: dict[int, list[Tensor]] = {}

        hooks: list[Any] = []
        gps_layers = list(model.layers)

        for layer_idx, layer in enumerate(gps_layers):
            layer_outputs[layer_idx] = []
            layer_attn[layer_idx] = []

            def make_hook(idx: int):
                def hook(module: nn.Module, inp: Any, out: Any) -> None:  # noqa: ANN001
                    if isinstance(out, tuple):
                        layer_outputs[idx].append(out[0].detach().cpu())
                    else:
                        layer_outputs[idx].append(out.detach().cpu())
                return hook

            def make_attn_hook(idx: int):
                def hook(module: nn.Module, inp: Any, out: Any) -> None:  # noqa: ANN001
                    if isinstance(out, tuple) and len(out) == 2:
                        layer_attn[idx].append(out[1].detach().cpu())
                return hook

            hooks.append(layer.register_forward_hook(make_hook(layer_idx)))
            hooks.append(
                layer.global_attn.register_forward_hook(make_attn_hook(layer_idx))
            )

        try:
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= num_batches:
                    break
                batch = batch.to(device)
                x = batch.x.float()
                edge_index = batch.edge_index
                edge_attr = getattr(batch, "edge_attr", None)
                if edge_attr is not None:
                    edge_attr = edge_attr.float()
                pe = getattr(batch, "pe", None)
                if pe is not None:
                    pe = pe.float()
                graph_batch = getattr(batch, "batch", None)
                model(x=x, edge_index=edge_index, edge_attr=edge_attr, pe=pe, batch=graph_batch)
        finally:
            for h in hooks:
                h.remove()

        # Compute diagnostics per layer
        mad_per_layer: dict[int, float] = {}
        entropy_per_layer: dict[int, float] = {}

        for layer_idx in range(len(gps_layers)):
            if layer_outputs[layer_idx]:
                feats = torch.cat(layer_outputs[layer_idx], dim=0)
                mad_per_layer[layer_idx] = compute_mad(feats)

            if layer_attn[layer_idx]:
                attn = torch.cat(layer_attn[layer_idx], dim=0)
                entropy_per_layer[layer_idx] = compute_attention_entropy(attn)

        return {
            "mad_per_layer": mad_per_layer,
            "attn_entropy_per_layer": entropy_per_layer,
        }
