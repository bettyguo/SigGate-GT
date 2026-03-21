"""
Training entry point for SigGate-GT.

Supports all five benchmark tasks with full reproducibility across 5 seeds.
Configuration is handled via YAML files in configs/ using Hydra.

Usage:
    # Train on ZINC (paper configuration)
    python train.py --config-name=experiment/zinc

    # Train on OGB molhiv
    python train.py --config-name=experiment/ogbg_molhiv

    # Train on LRGB Peptides-func
    python train.py --config-name=experiment/peptides_func

    # Debug run (fast, small model)
    python train.py --config-name=config training=debug

    # Multi-seed sweep
    for seed in 0 1 2 3 4; do
        python train.py --config-name=experiment/zinc seed=$seed
    done
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from siggate_gt.data.dataloader import build_dataloaders
from siggate_gt.evaluation.evaluator import Evaluator
from siggate_gt.models.losses import build_loss
from siggate_gt.models.siggate_gps import SigGateGT
from siggate_gt.utils.checkpoint import CheckpointManager, load_checkpoint
from siggate_gt.utils.reproducibility import print_reproducibility_info, set_seed


log = logging.getLogger(__name__)


def build_model(cfg: DictConfig) -> SigGateGT:
    """Instantiate SigGate-GT from a Hydra model config."""
    model_cfg = cfg.model
    task = cfg.task

    factory = {
        "zinc": SigGateGT.build_zinc,
        "molhiv": SigGateGT.build_ogb,
        "molpcba": SigGateGT.build_ogb,
        "peptides_func": SigGateGT.build_lrgb,
        "peptides_struct": SigGateGT.build_lrgb,
    }

    if task not in factory:
        raise ValueError(
            f"Unknown task '{task}'. Valid tasks: {list(factory.keys())}"
        )

    # Override defaults with config values
    kwargs = {
        "hidden_dim": model_cfg.get("hidden_dim"),
        "num_layers": model_cfg.get("num_layers"),
        "num_heads": model_cfg.get("num_heads"),
        "dropout": model_cfg.get("dropout", 0.0),
        "gate_bias_init": model_cfg.get("gate_bias_init", 0.5),
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if "out_dim" in model_cfg:
        kwargs["out_dim"] = model_cfg.out_dim

    model = factory[task](**kwargs)

    param_info = model.count_parameters()
    log.info(
        "Model parameters: %d total (%.2f%% gate projections)",
        param_info["total"],
        param_info["gate_fraction_pct"],
    )
    return model


def build_optimizer(model: SigGateGT, cfg: DictConfig) -> torch.optim.Optimizer:
    """Build AdamW optimizer from config."""
    opt_cfg = cfg.training.optimizer
    return torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.lr,
        weight_decay=opt_cfg.get("weight_decay", 1e-5),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps=opt_cfg.get("eps", 1e-8),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: DictConfig
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build cosine annealing scheduler from config."""
    sched_cfg = cfg.training.get("scheduler", {})
    sched_name = sched_cfg.get("name", "cosine")

    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs,
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )
    elif sched_name == "none":
        return None
    else:
        log.warning("Unknown scheduler '%s', using None.", sched_name)
        return None


def train_epoch(
    model: SigGateGT,
    loader: torch.utils.data.DataLoader,  # type: ignore[type-arg]
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    max_grad_norm: float = 1.0,
    epoch: int = 0,
) -> float:
    """Run one training epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
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
        y = batch.y

        pred = model(x=x, edge_index=edge_index, edge_attr=edge_attr, pe=pe, batch=graph_batch)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    if scheduler is not None:
        scheduler.step()

    return total_loss / n_batches


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Main training loop.

    Returns the best validation metric (for hyperparameter sweeps).
    """
    OmegaConf.resolve(cfg)
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    seed = cfg.get("seed", 0)
    task = cfg.get("task", "zinc")

    set_seed(seed, deterministic=cfg.get("deterministic", True))
    print_reproducibility_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # W&B initialization
    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    if use_wandb:
        wandb.init(
            project=cfg.wandb.get("project", "siggate-gt"),
            name=f"{task}_seed{seed}",
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[task, f"seed{seed}"],
        )

    # Data
    data_cfg = cfg.data
    train_loader, val_loader, test_loader = build_dataloaders(
        task=task,
        root=data_cfg.get("root", "dataset"),
        pe_dim=data_cfg.get("pe_dim", 32),
        batch_size_train=data_cfg.get("batch_size_train", 32),
        batch_size_eval=data_cfg.get("batch_size_eval", 64),
        num_workers=data_cfg.get("num_workers", 4),
        seed=seed,
    )

    # Model
    model = build_model(cfg).to(device)

    # Loss
    criterion = build_loss(task)

    # Optimizer + Scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Checkpoint manager
    higher_is_better = task in ("molhiv", "molpcba", "peptides_func")
    metric_name = "rocauc" if task == "molhiv" else ("ap" if task in ("molpcba", "peptides_func") else "mae")
    ckpt_manager = CheckpointManager(
        checkpoint_dir=Path("checkpoints") / task / f"seed{seed}",
        max_checkpoints=cfg.training.get("max_checkpoints", 3),
        metric_name=metric_name,
        mode="max" if higher_is_better else "min",
    )

    # Resume from checkpoint if requested
    start_epoch = 0
    if cfg.training.get("resume"):
        ckpt = load_checkpoint(
            path=cfg.training.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            restore_rng=True,
        )
        start_epoch = ckpt["epoch"] + 1
        log.info("Resuming from epoch %d", start_epoch)

    evaluator = Evaluator(task=task)
    best_metric: float = float("inf") if not higher_is_better else float("-inf")
    max_grad_norm = cfg.training.get("max_grad_norm", 1.0)

    # Training loop
    log.info("Starting training: task=%s, seed=%d, epochs=%d", task, seed, cfg.training.epochs)

    for epoch in range(start_epoch, cfg.training.epochs):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            max_grad_norm=max_grad_norm,
            epoch=epoch,
        )

        if (epoch + 1) % cfg.training.get("eval_every_n_epochs", 1) == 0:
            val_metrics = evaluator.evaluate(model, val_loader, device)
            current_metric = val_metrics[metric_name]

            log.info(
                "Epoch %d/%d: train_loss=%.4f, val_%s=%.4f",
                epoch + 1,
                cfg.training.epochs,
                train_loss,
                metric_name,
                current_metric,
            )

            if use_wandb:
                wandb.log(
                    {
                        "train/loss": train_loss,
                        "val/" + metric_name: current_metric,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    }
                )

            is_best = (
                current_metric < best_metric if not higher_is_better
                else current_metric > best_metric
            )
            if is_best:
                best_metric = current_metric

            if (epoch + 1) % cfg.training.get("save_every_n_epochs", 10) == 0 or is_best:
                ckpt_manager.save(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=epoch * len(train_loader),
                    metric=current_metric,
                    config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
                    scheduler=scheduler,
                    task=task,
                    seed=seed,
                )

    # Final test evaluation
    log.info("Loading best checkpoint for final test evaluation...")
    best_ckpt = ckpt_manager.load_best(model=model, device=device)
    test_metrics = evaluator.evaluate(model, test_loader, device)

    log.info("Test metrics (%s, seed=%d): %s", task, seed, test_metrics)

    if use_wandb:
        for k, v in test_metrics.items():
            wandb.summary[f"test/{k}"] = v
        wandb.summary[f"best_val/{metric_name}"] = best_metric
        wandb.finish()

    return best_metric


if __name__ == "__main__":
    main()
