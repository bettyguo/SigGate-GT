"""
Checkpoint management for training resumption.

Handles complete training state serialization including:
- Model weights
- Optimizer state
- LR scheduler state
- RNG states for exact reproducibility
- Training metrics history
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from siggate_gt.utils.reproducibility import get_rng_state, set_rng_state


log = logging.getLogger(__name__)


class CheckpointDict(TypedDict, total=False):
    """Type definition for checkpoint contents."""

    epoch: int
    global_step: int
    best_metric: float
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any] | None
    scaler_state_dict: dict[str, Any] | None
    config: dict[str, Any]
    rng_states: dict[str, Any]
    metrics_history: dict[str, list[float]]
    pytorch_version: str
    cuda_version: str | None
    timestamp: str
    task: str
    seed: int


@dataclass
class CheckpointManager:
    """
    Manages checkpoint saving, loading, and cleanup.

    Args:
        checkpoint_dir: Directory for saving checkpoints.
        max_checkpoints: Maximum number of periodic checkpoints to keep (0 = unlimited).
        metric_name: Metric name to track (e.g., 'mae' or 'rocauc').
        mode: 'min' (e.g., MAE) or 'max' (e.g., AUC) for best-model tracking.
    """

    checkpoint_dir: Path
    max_checkpoints: int = 5
    metric_name: str = "mae"
    mode: str = "min"
    best_metric: float = field(init=False)
    saved_checkpoints: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = float("inf") if self.mode == "min" else float("-inf")

    def is_better(self, metric: float) -> bool:
        """Check whether metric is an improvement over current best."""
        return metric < self.best_metric if self.mode == "min" else metric > self.best_metric

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        global_step: int,
        metric: float | None = None,
        config: dict[str, Any] | None = None,
        scheduler: LRScheduler | None = None,
        scaler: torch.cuda.amp.GradScaler | None = None,
        metrics_history: dict[str, list[float]] | None = None,
        task: str = "",
        seed: int = 0,
    ) -> Path | None:
        """
        Save a checkpoint with complete training state.

        Saves both a timestamped checkpoint and, if metric improves,
        overwrites 'best.pt'. Oldest periodic checkpoints are pruned
        to keep at most max_checkpoints.

        Returns:
            Path to saved checkpoint, or None if save was skipped.
        """
        ckpt: CheckpointDict = {
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": self.best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "config": config or {},
            "rng_states": get_rng_state(),
            "metrics_history": metrics_history or {},
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "seed": seed,
        }

        filename = f"checkpoint_epoch{epoch:04d}.pt"
        path = self.checkpoint_dir / filename
        torch.save(ckpt, path)
        log.info("Saved checkpoint: %s", path)

        self.saved_checkpoints.append(path)
        if self.max_checkpoints > 0:
            while len(self.saved_checkpoints) > self.max_checkpoints:
                old = self.saved_checkpoints.pop(0)
                if old.exists() and old.name != "best.pt":
                    old.unlink()
                    log.debug("Removed old checkpoint: %s", old)

        # Save best checkpoint
        if metric is not None and self.is_better(metric):
            self.best_metric = metric
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(ckpt, best_path)
            log.info("New best %s=%.4f → saved as best.pt", self.metric_name, metric)

        return path

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        scaler: torch.cuda.amp.GradScaler | None = None,
        device: torch.device | None = None,
        restore_rng: bool = False,
    ) -> CheckpointDict:
        """Load the best checkpoint from checkpoint_dir/best.pt."""
        return load_checkpoint(
            path=self.checkpoint_dir / "best.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            restore_rng=restore_rng,
        )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    device: torch.device | None = None,
    restore_rng: bool = False,
) -> CheckpointDict:
    """
    Load a checkpoint and restore model (and optionally optimizer) state.

    Args:
        path: Path to checkpoint file.
        model: Model to restore weights into.
        optimizer: Optimizer to restore state into (optional).
        scheduler: LR scheduler to restore state into (optional).
        scaler: AMP gradient scaler to restore state into (optional).
        device: Device to map tensors to (default: current device).
        restore_rng: If True, restore all RNG states for exact reproducibility.

    Returns:
        The full checkpoint dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device or "cpu"
    ckpt: CheckpointDict = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    log.info("Loaded model weights from %s (epoch %d)", path, ckpt.get("epoch", -1))

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    if restore_rng and "rng_states" in ckpt:
        set_rng_state(ckpt["rng_states"])
        log.info("Restored RNG states from checkpoint.")

    return ckpt
