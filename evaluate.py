"""
Evaluation entry point for SigGate-GT.

Loads a trained checkpoint and reports test-set metrics.
Also supports multi-seed aggregation (mean ± std) and paired t-tests.

Usage:
    # Evaluate a single checkpoint
    python evaluate.py --task zinc --checkpoint checkpoints/zinc/seed0/best.pt

    # Multi-seed evaluation with significance testing vs. GraphGPS
    python evaluate.py --task zinc --seeds 0 1 2 3 4

    # Evaluate on all datasets
    python evaluate.py --all-datasets --seeds 0 1 2 3 4

    # Compute over-smoothing and attention entropy diagnostics
    python evaluate.py --task zinc --checkpoint checkpoints/zinc/seed0/best.pt --diagnostics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from siggate_gt.data.dataloader import build_dataloaders
from siggate_gt.evaluation.evaluator import Evaluator
from siggate_gt.evaluation.metrics import paired_t_test
from siggate_gt.models.siggate_gps import SigGateGT
from siggate_gt.utils.checkpoint import load_checkpoint
from siggate_gt.utils.reproducibility import set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# Paper results for significance testing reference
# ogbg-molhiv uses ROC-AUC, ogbg-molpcba uses AP, ZINC/Peptides-struct use MAE,
# Peptides-func uses AP
PAPER_RESULTS: dict[str, dict[str, float]] = {
    "zinc": {"mae": 0.059},
    "molhiv": {"rocauc": 82.47},
    "molpcba": {"ap": 29.84},
    "peptides_func": {"ap": 0.6947},
    "peptides_struct": {"mae": 0.2431},
}

TASK_METRIC: dict[str, str] = {
    "zinc": "mae",
    "molhiv": "rocauc",
    "molpcba": "ap",
    "peptides_func": "ap",
    "peptides_struct": "mae",
}

HIGHER_IS_BETTER: dict[str, bool] = {
    "zinc": False,
    "molhiv": True,
    "molpcba": True,
    "peptides_func": True,
    "peptides_struct": False,
}

TASK_CONFIGS: dict[str, dict] = {
    "zinc": {
        "hidden_dim": 64, "num_layers": 10, "num_heads": 8,
        "batch_size_eval": 32, "pe_dim": 32,
    },
    "molhiv": {
        "hidden_dim": 256, "num_layers": 5, "num_heads": 8,
        "batch_size_eval": 256, "pe_dim": 32,
    },
    "molpcba": {
        "hidden_dim": 256, "num_layers": 5, "num_heads": 8,
        "batch_size_eval": 512, "pe_dim": 32,
    },
    "peptides_func": {
        "hidden_dim": 128, "num_layers": 10, "num_heads": 8,
        "batch_size_eval": 64, "pe_dim": 32,
    },
    "peptides_struct": {
        "hidden_dim": 128, "num_layers": 10, "num_heads": 8,
        "batch_size_eval": 64, "pe_dim": 32,
    },
}

TASK_OUTDIM: dict[str, int] = {
    "zinc": 1,
    "molhiv": 1,
    "molpcba": 128,
    "peptides_func": 10,
    "peptides_struct": 11,
}


def load_model_from_checkpoint(
    task: str,
    checkpoint_path: str | Path,
    device: torch.device,
) -> SigGateGT:
    """Load a SigGate-GT model from a checkpoint file."""
    cfg = TASK_CONFIGS[task]
    out_dim = TASK_OUTDIM[task]

    factory = {
        "zinc": SigGateGT.build_zinc,
        "molhiv": SigGateGT.build_ogb,
        "molpcba": SigGateGT.build_ogb,
        "peptides_func": SigGateGT.build_lrgb,
        "peptides_struct": SigGateGT.build_lrgb,
    }

    model = factory[task](
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        out_dim=out_dim,
    ).to(device)

    load_checkpoint(path=checkpoint_path, model=model, device=device)
    return model


def evaluate_single(
    task: str,
    checkpoint: str,
    data_root: str = "dataset",
    diagnostics: bool = False,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate a single checkpoint on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed, deterministic=True)

    cfg = TASK_CONFIGS[task]
    _, val_loader, test_loader = build_dataloaders(
        task=task,
        root=data_root,
        pe_dim=cfg["pe_dim"],
        batch_size_train=cfg["batch_size_eval"],
        batch_size_eval=cfg["batch_size_eval"],
        seed=seed,
    )

    model = load_model_from_checkpoint(task, checkpoint, device)
    evaluator = Evaluator(task=task, compute_diagnostics=diagnostics)

    test_metrics = evaluator.evaluate(model, test_loader, device)
    log.info("Test results (%s): %s", task, test_metrics)

    if diagnostics:
        diag = evaluator.compute_node_diagnostics(model, val_loader, device)
        log.info("MAD per layer: %s", diag["mad_per_layer"])
        log.info("Attention entropy per layer: %s", diag["attn_entropy_per_layer"])
        test_metrics.update(
            {f"mad_layer{k}": v for k, v in diag["mad_per_layer"].items()}
        )
        test_metrics.update(
            {f"entropy_layer{k}": v for k, v in diag["attn_entropy_per_layer"].items()}
        )

    return test_metrics


def evaluate_multi_seed(
    task: str,
    checkpoint_dir: str = "checkpoints",
    seeds: list[int] | None = None,
    data_root: str = "dataset",
) -> dict[str, float | str]:
    """
    Evaluate across multiple seeds and report mean ± std.

    Matches the paper's protocol: 5 seeds (0–4), report mean ± std.
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    metric_name = TASK_METRIC[task]
    per_seed_values: list[float] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TASK_CONFIGS[task]

    for seed in seeds:
        ckpt_path = Path(checkpoint_dir) / task / f"seed{seed}" / "best.pt"
        if not ckpt_path.exists():
            log.warning("Checkpoint not found for seed %d: %s", seed, ckpt_path)
            continue

        set_seed(seed, deterministic=True)
        _, _, test_loader = build_dataloaders(
            task=task,
            root=data_root,
            pe_dim=cfg["pe_dim"],
            batch_size_train=cfg["batch_size_eval"],
            batch_size_eval=cfg["batch_size_eval"],
            seed=seed,
        )

        model = load_model_from_checkpoint(task, ckpt_path, device)
        evaluator = Evaluator(task=task)
        metrics = evaluator.evaluate(model, test_loader, device)
        val = metrics[metric_name]
        per_seed_values.append(val)
        log.info("Seed %d: %s=%.4f", seed, metric_name, val)

    if not per_seed_values:
        log.error("No checkpoints found for task %s.", task)
        sys.exit(1)

    mean_val = float(np.mean(per_seed_values))
    std_val = float(np.std(per_seed_values))

    result: dict[str, float | str] = {
        "task": task,
        "metric": metric_name,
        "mean": mean_val,
        "std": std_val,
        "seeds_evaluated": len(per_seed_values),
        "formatted": f"{mean_val:.4f} ± {std_val:.4f}",
        "per_seed": str(per_seed_values),
    }
    return result


def run_significance_test(
    task: str,
    our_scores: list[float],
    baseline_scores: list[float],
) -> None:
    """
    Run and print paired t-test between our method and a baseline.

    The paper reports p-values from paired t-tests over 5 seeds
    against GraphGPS.
    """
    metric_name = TASK_METRIC[task]
    higher_better = HIGHER_IS_BETTER[task]

    result = paired_t_test(our_scores, baseline_scores, higher_is_better=higher_better)

    print("\n" + "=" * 60)
    print(f"Paired t-test: SigGate-GT vs. baseline on {task} ({metric_name})")
    print("=" * 60)
    print(f"  Our scores:      {our_scores}")
    print(f"  Baseline scores: {baseline_scores}")
    print(f"  Mean diff:       {result['mean_diff']:+.4f} ({'↑' if higher_better else '↓'} better)")
    print(f"  t-statistic:     {result['t_stat']:.3f}")
    print(f"  p-value:         {result['p_value']:.4f}", end="  ")
    if result["p_value"] < 0.01:
        print("(**)")
    elif result["p_value"] < 0.05:
        print("(*)")
    else:
        print("(n.s.)")
    print(f"  95% CI:          [{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}]")
    print(f"  N seeds:         {result['n_seeds']}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SigGate-GT on graph benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_METRIC.keys()),
        default="zinc",
        help="Benchmark task to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a single checkpoint file.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Root directory for multi-seed evaluation.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Seeds to evaluate (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="Root directory for datasets.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Evaluate on all five benchmark tasks.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Compute MAD and attention entropy diagnostics.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write results to a JSON file.",
    )

    args = parser.parse_args()

    tasks = list(TASK_METRIC.keys()) if args.all_datasets else [args.task]
    all_results: dict[str, dict] = {}

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Evaluating: {task}")
        print(f"{'='*60}")

        if args.checkpoint is not None:
            result = evaluate_single(
                task=task,
                checkpoint=args.checkpoint,
                data_root=args.data_root,
                diagnostics=args.diagnostics,
                seed=args.seeds[0],
            )
        else:
            result = evaluate_multi_seed(
                task=task,
                checkpoint_dir=args.checkpoint_dir,
                seeds=args.seeds,
                data_root=args.data_root,
            )

        all_results[task] = result  # type: ignore[assignment]

        metric = TASK_METRIC[task]
        if isinstance(result.get("mean"), float):
            print(f"\nResults: {metric} = {result['formatted']}")
        else:
            print(f"\nResults: {metric} = {result.get(metric, 'N/A'):.4f}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        log.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
