<div align="center">

# SigGate-GT: Taming Over-Smoothing in Graph Transformers via Sigmoid-Gated Attention

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-ee4c2c.svg)](https://pytorch.org/)
[![PyG 2.4](https://img.shields.io/badge/PyG-2.4-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Element-wise sigmoid gating on graph transformer attention to address over-smoothing, attention entropy collapse, and training instability.**

[📊 Results](#-results) |
[🚀 Quick Start](#-quick-start) |
[🏋️ Training](#️-training) |
[🔬 Ablations](#-ablation-studies)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training](#️-training)
- [Results](#-results)
- [Ablation Studies](#-ablation-studies)
- [Reproducibility](#-reproducibility)

---

## 🔬 Overview

Graph transformers achieve strong results on molecular and long-range reasoning tasks, yet remain hampered by over-smoothing—the progressive collapse of node representations with depth—and attention entropy degeneration. We observe that these pathologies share a root cause with attention sinks in large language models: softmax attention's sum-to-one constraint forces every node to attend somewhere, even when no informative signal exists.

We propose **SigGate-GT**, a graph transformer that applies learned, per-head sigmoid gates to the attention output within the GraphGPS framework:

```
head_k^gated = softmax(Q_k K_k^T / sqrt(d_k)) V_k  ⊙  σ(H W^g_k + b^g_k)
```

Each gate can suppress activations toward zero, enabling heads to selectively silence uninformative connections.

### Key Contributions

- **SigGate-MHSA**: First graph transformer with element-wise sigmoid gating on the attention output, transferring a proven LLM technique to graphs.
- **Comprehensive ablations**: Gate placement, per-head vs. shared gating, over-smoothing depth analysis (MAD), attention entropy, and training stability over a 10× learning rate range.
- **Strong empirical results**: State-of-the-art on ZINC (0.059 MAE) and ogbg-molhiv (82.47% ROC-AUC), with statistically significant improvements (p < 0.05) across all five benchmarks over GraphGPS.

### Method Architecture

```
Input: node features x, edge features e, positional encodings (LapPE + RWSE)
  ↓
Node encoder: Linear(x || pe) → h ∈ R^d
  ↓
L × SigGate-GPS layers:
  ├── Local MPNN (GatedGCN): edge-level sigmoid gating for neighborhood aggregation
  ├── Global SigGate-MHSA: attention output gating per head and dimension
  └── FFN + LayerNorm + Residual
  ↓
Graph-level mean pooling
  ↓
Output head: 2-layer MLP → prediction
```

---

## 📦 Installation

### Requirements

- Python ≥ 3.10, < 3.12
- PyTorch 2.1.0
- PyTorch Geometric 2.4.0
- CUDA 12.1+ (for GPU training)

### Option 1: pip

```bash
# Create conda environment
conda create -n siggate_gt python=3.11 -y
conda activate siggate_gt

# Install PyTorch with CUDA
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install PyG
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Install SigGate-GT
pip install -e .
```

### Option 2: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate siggate_gt
pip install -e .
```

### Verify Installation

```bash
python -c "from siggate_gt import SigGateGT; print('✓ Installation successful')"
```

---

## 🚀 Quick Start

### Python API

```python
import torch
from siggate_gt import SigGateGT

# Build model for ZINC (10 layers, dim 64, 8 heads)
model = SigGateGT.build_zinc()

# Print parameter info
info = model.count_parameters()
print(f"Total parameters: {info['total']:,}")
print(f"Gate overhead: {info['gate_fraction_pct']:.2f}%")

# Forward pass (batched graphs)
# x: node features (N_total, node_dim)
# edge_index: connectivity (2, E)
# edge_attr: edge features (E, edge_dim)
# pe: positional encodings (N_total, pe_dim)
# batch: graph assignment vector (N_total,)
pred = model(x=x, edge_index=edge_index, edge_attr=edge_attr, pe=pe, batch=batch)
```

### Inspect Gate Activations

```python
from siggate_gt.models.attention import SigGateMultiHeadAttention
import torch

attn = SigGateMultiHeadAttention(embed_dim=64, num_heads=8)
x = torch.randn(10, 64)  # 10 nodes, 64-dim features

stats = attn.get_gate_statistics(x.unsqueeze(0))
print(f"Mean gate activation: {stats['overall_mean'].item():.3f}")
print(f"Fraction suppressed (<0.1): {stats['frac_below_01'].mean().item():.3f}")
```

---

## 🏋️ Training

### Train from Scratch

```bash
# Single seed
python train.py --config-name=experiment/zinc seed=0

# All 5 seeds
for seed in 0 1 2 3 4; do
    python train.py --config-name=experiment/zinc seed=$seed
done

# Debug mode (fast iteration)
python train.py --config-name=config training=debug

# With W&B tracking
python train.py --config-name=experiment/zinc wandb.enabled=true
```

### Key Hyperparameters

| Parameter | ZINC | OGB | LRGB | Description |
|-----------|------|-----|------|-------------|
| `hidden_dim` | 64 | 256 | 128 | Node embedding dimension |
| `num_layers` | 10 | 5 | 10 | Number of GPS layers |
| `num_heads` | 8 | 8 | 8 | Attention heads per layer |
| `gate_bias_init` | 0.5 | 0.5 | 0.5 | Gate bias init (σ(0.5) ≈ 0.62) |
| `lr` | 1e-3 | 1e-4 | 5e-4 | AdamW learning rate |
| `weight_decay` | 1e-5 | 1e-5 | 1e-5 | AdamW weight decay |
| `epochs` | 2000 | 100 | 200 | Training epochs |
| `batch_size` | 32 | 256/512 | 64 | Training batch size |
| `pe_dim` | 32 | 32 | 32 | LapPE (16) + RWSE (16) dim |

```

---

## 📈 Results

All results are mean ± std over **5 seeds (0–4)** run on a single NVIDIA A100 GPU.

### Main Results

#### ZINC (500K parameter budget, MAE ↓)

| Method | Type | Test MAE |
|--------|------|----------|
| GCN | MPNN | 0.367 ± 0.011 |
| GatedGCN | MPNN | 0.282 ± 0.015 |
| PNA | MPNN | 0.188 ± 0.004 |
| SAN | GT | 0.139 ± 0.006 |
| Graphormer | GT | 0.122 ± 0.006 |
| GraphGPS | GT | 0.070 ± 0.004 |
| Exphormer | GT | 0.066 ± 0.003 |
| GRIT | GT | 0.059 ± 0.002 |
| **SigGate-GT (ours)** | **GT** | **0.059 ± 0.002** |

SigGate-GT vs. GraphGPS: p < 0.001 (paired t-test, 5 seeds).

#### OGB Molecular Benchmarks

| Method | molhiv (AUC ↑) | molpcba (AP ↑) |
|--------|----------------|----------------|
| GCN | 76.06 ± 0.97 | 24.24 ± 0.34 |
| PNA | 79.05 ± 1.32 | 28.38 ± 0.35 |
| GraphGPS | 78.80 ± 1.01 | 29.07 ± 0.28 |
| Exphormer | 80.75 ± 0.94 | 29.20 ± 0.30 |
| **SigGate-GT (ours)** | **82.47 ± 0.63** | **29.84 ± 0.31** |

SigGate-GT vs. GraphGPS: molhiv p = 0.002, molpcba p = 0.008.

#### LRGB Peptides Benchmarks

| Method | Pep-func (AP ↑) | Pep-struct (MAE ↓) |
|--------|-----------------|---------------------|
| GraphGPS | 0.6535 ± 0.0041 | 0.2500 ± 0.0012 |
| GRIT | 0.6988 ± 0.0082 | 0.2460 ± 0.0012 |
| **SigGate-GT (ours)** | **0.6947 ± 0.0037** | **0.2431 ± 0.0012** |

Both improvements over GraphGPS: p < 0.001. Note: GRIT leads on Pep-func.

### Statistical Significance Summary

| Benchmark | vs. GraphGPS | vs. Next-Best |
|-----------|-------------|---------------|
| ZINC (MAE) | p < 0.001 ✓ | p = 0.48 n.s. |
| molhiv (AUC) | p = 0.002 ✓ | p = 0.018 ✓ |
| molpcba (AP) | p = 0.008 ✓ | p = 0.031 ✓ |
| Pep-func (AP) | p < 0.001 ✓ | p = 0.34 n.s. |
| Pep-struct (MAE) | p < 0.001 ✓ | p = 0.011 ✓ |

---

## 🔬 Ablation Studies

### Gate Placement

| Config | ZINC MAE ↓ | molhiv AUC ↑ |
|--------|-----------|-------------|
| No gate (GraphGPS) | 0.070 | 78.80 |
| G3: pre-softmax logit gating | 0.074 | 77.95 |
| G2: value gating | 0.066 | 80.12 |
| **G1: output gating (ours)** | **0.059** | **82.47** |
| G1 + shared gate | 0.064 | 80.58 |

### Over-Smoothing (MAD Analysis)

| Model | 4L | 8L | 10L | 12L | 16L |
|-------|-----|-----|------|------|------|
| GraphGPS (MAD) | 0.72 | 0.58 | 0.49 | 0.44 | 0.37 |
| SigGate-GT (MAD) | 0.78 | 0.69 | 0.64 | 0.61 | 0.57 |

SigGate-GT retains 68% of initial MAD at 16 layers vs. 41% for GraphGPS.

### Training Stability (ZINC MAE at different learning rates)

| LR | GraphGPS | SigGate-GT |
|----|----------|------------|
| 5e-4 | 0.078 | 0.065 |
| 1e-3 | 0.070 | 0.059 |
| 2e-3 | 0.085 | 0.062 |
| 3e-3 | 0.112 | 0.064 |
| 5e-3 | 0.098 | 0.063 |
| **Range** | **0.042** | **0.006** |

SigGate-GT has a **7× smaller performance range** across the 10× learning rate sweep.

---

## ♻️ Reproducibility

All experiments use:
- **5 seeds**: 0, 1, 2, 3, 4
- **Hardware**: Single NVIDIA A100 GPU
- **Framework**: PyTorch 2.1.0, PyG 2.4.0
- **Positional encodings**: LapPE (16 dims, SignNet) + RWSE (16 steps)

### Compute Budget

| Benchmark | Time/Run | × 5 Seeds |
|-----------|----------|-----------|
| ZINC | ~3h | ~15h |
| ogbg-molhiv | ~2h | ~10h |
| ogbg-molpcba | ~8h | ~40h |
| Peptides-func | ~2h | ~10h |
| Peptides-struct | ~2h | ~10h |

**Total: ~85 A100 GPU-hours** across all benchmarks and seeds.

### Run Reproducibility Tests

```bash
make test-reproducibility
```

---

<div align="center">

**[Report Bug](https://anonymous.4open.science/r/SigGate-GT/issues)** •
**[Anonymous Repository](https://anonymous.4open.science/r/SigGate-GT)**

</div>
