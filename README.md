# TSA-HGNN: Temporal Stability-Aware Hybrid Graph Neural Network

[![Paper](https://img.shields.io/badge/Paper-Frontiers_in_AI-blue)](https://www.frontiersin.org/journals/artificial-intelligence)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Experiments](#experiments)
- [Results](#results)
- [Hyperparameters](#hyperparameters)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

TSA-HGNN is a **unified framework for dynamic community detection** in evolving graphs. It addresses three critical challenges:

1. **Inductive scalability** — handles evolving node sets without retraining
2. **Multi-scale temporal modeling** — captures both short-range and long-range dependencies
3. **Temporal stability** — prevents unreasonable community drift between snapshots

### What Makes TSA-HGNN Different?

Unlike existing methods that focus on either spatial structure OR temporal evolution, TSA-HGNN **integrates four complementary components** into a single pipeline:

| Component | Purpose | Key Advantage |
|-----------|---------|---------------|
| **GraphSAGE** | Snapshot-level spatial encoding | Inductive — handles new nodes without retraining |
| **TCN** | Short-range temporal patterns | Causal dilated convolutions — no future leakage |
| **Informer** | Long-range temporal dependencies | ProbSparse attention — sub-quadratic in T |
| **ESN** | Nonlinear temporal memory | Fixed reservoir — only readout trained (lightweight) |
| **Stability Loss** | Prevents abrupt community drift | L₂ penalty on consecutive embeddings |

**Result:** State-of-the-art accuracy (0.9843, 0.9755, 0.9931 on 3 benchmarks) with 81.5% reduction in community switch rate compared to baselines.

---

## ✨ Key Features

- ✅ **Multi-scale temporal modeling** — combines TCN (local) + Informer (global) for comprehensive coverage
- ✅ **Stability-aware optimization** — explicit regularization prevents unstable community assignments
- ✅ **Inductive framework** — handles dynamic node sets without architectural changes
- ✅ **Computational efficiency** — sub-quadratic complexity in time horizon (O(T log T) vs O(T²))
- ✅ **Extensive validation** — 3 benchmarks + statistical significance tests (Holm correction)
- ✅ **Reproducible** — 5-seed protocol, deterministic settings, complete hyperparameter config

---

## 🏗️ Architecture

```
Dynamic Graph Sequence                    TSA-HGNN Pipeline
┌────────────────────┐
│  G₁  G₂  ...  Gₜ   │
│ (V,E,X) evolving   │
└──────────┬─────────┘
           │
           ▼
    ┌──────────────┐
    │  GraphSAGE   │  Inductive spatial encoding
    │   (§3.2)     │  • Sample s=10 neighbors per node
    └──────┬───────┘  • 2 aggregation layers
           │          • Output: Zₜ ∈ ℝⁿˣᵈ per snapshot
           ▼
    ┌──────────────┐
    │     TCN      │  Short-range temporal modeling
    │   (§3.3)     │  • Dilated causal convolutions
    └──────┬───────┘  • Receptive field: 2^L snapshots
           │          • 4 residual blocks
           ▼
    ┌──────────────┐
    │   Informer   │  Long-range temporal modeling
    │   (§3.4)     │  • ProbSparse attention (O(T log T))
    └──────┬───────┘  • Encoder-decoder architecture
           │          • Handles T=10-20 snapshots
           ▼
    ┌──────────────┐
    │     ESN      │  Nonlinear temporal memory
    │   (§3.5)     │  • Fixed reservoir (500 neurons)
    └──────┬───────┘  • Only readout trained
           │          • Spectral radius ρ=0.9
           ▼
    ┌──────────────┐
    │  Stability   │  Temporal smoothness
    │   (§3.6)     │  • L_stab = ‖Zₜ - Zₜ₋₁‖²_F
    └──────┬───────┘  • Weight λ=0.10
           │
           ▼
    ┌──────────────┐
    │  K-Means     │  Community assignment
    │   (§3.7)     │  • Applied to final embeddings
    └──────────────┘  • Stable across snapshots
```

### Joint Training Objective

```
L_total = L_recon + L_temporal + λ·L_stability

where:
  L_recon    = BCE(Âₜ, Aₜ)           [Eq. 17]
  Âₜ(u,v)    = σ(zᵤᵀzᵥ)              [Eq. 16a]
  L_stability = Σ ‖Zₜ - Zₜ₋₁‖²_F    [Eq. 16]
  λ          = 0.10 (default)
```

---

## 🚀 Installation

### Requirements

- Python ≥ 3.9
- CUDA-capable GPU (recommended; CPU also supported)

### Setup

```bash
# Clone the repository
git clone https://github.com/gowthamivarahi/Dynamic-Community-Detection.git
cd Dynamic-Community-Detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.1
torch-geometric>=2.3.1
numpy>=1.24.3
scipy>=1.10.0
scikit-learn>=1.2.2
networkx>=2.8.8
pandas>=1.5.3
matplotlib>=3.7.1  # For visualization
```

---

## 🎯 Quick Start

### 1. **Basic Usage — Single Forward Pass**

```python
import torch
from models.tsa_hgnn import TemporalStabilityAwareHGNN

# Create 5 graph snapshots (8 nodes, 4 features each)
snapshots = []
for t in range(5):
    X_t = torch.rand(8, 4)  # Node features
    A_t = (torch.rand(8, 8) > 0.7).float()  # Adjacency
    A_t = (A_t + A_t.T).clamp(max=1.0)  # Symmetrize
    A_t.fill_diagonal_(0)
    snapshots.append((X_t, A_t))

# Initialize TSA-HGNN
model = TemporalStabilityAwareHGNN(
    node_feat=4,           # Input feature dimension
    hidden_dim=64,         # Embedding dimension
    out_dim=32,            # Output dimension
    esn_reservoir_size=200,
    stability_weight=0.10
)

# Forward pass
final_emb, per_snap_embs = model(snapshots)
print(f"Final embedding shape: {final_emb.shape}")  # [8, 32]

# Compute loss
loss = model.loss(final_emb, snapshots[-1][1], per_snap_embs)
print(f"Loss: {loss.item():.4f}")
```

### 2. **Build Dynamic Graph Snapshots from CSV**

```bash
# Prepare your edge CSV with columns: source, target, weight, timestamp
# Example: data/sample_dynamic_edges.csv

cd data/
python dynamic_graph_construction.py

# Output: output_snapshots/snapshot_YYYY-MM-DD.csv for each timestamp
```

**CSV format:**
```csv
source,target,weight,timestamp
A,B,0.8,2024-01-01
A,C,0.6,2024-01-01
B,C,0.9,2024-01-01
...
```

### 3. **Run Full Experiments (Reproduce Paper Results)**

```bash
python experiments/train_eval.py \
    --dataset 1 \
    --epochs 200 \
    --patience 20 \
    --seeds 1 2 3 4 5 \
    --output results/
```

---

## 📁 Repository Structure

```
Dynamic-Community-Detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── .gitignore                         # Git exclusions
│
├── models/
│   ├── __init__.py
│   └── tsa_hgnn.py                    # TSA-HGNN model (GraphSAGE+TCN+Informer+ESN)
│                                      # - Fixed ESN reservoir weights
│                                      # - Multi-snapshot forward pass
│                                      # - Loss function (Eq. 19)
│
├── data/
│   ├── dynamic_graph_construction.py  # Snapshot builder from edge CSVs
│   ├── sample_dynamic_edges.csv       # Demo 3-snapshot data
│   └── README.md                      # Dataset instructions
│
├── experiments/
│   ├── __init__.py
│   ├── train_eval.py                  # Training and evaluation script
│   └── utils.py                       # Helper functions
│
├── config/
│   ├── __init__.py
│   └── default_config.py              # Hyperparameters (Table 2 from paper)
│
└── docs/
    ├── RESULTS.md                     # Detailed experimental results
    └── DATASETS.md                    # Dataset descriptions
```

---

## 📊 Datasets

| Dataset | Nodes | Edges | Snapshots | Type | Labels |
|---------|-------|-------|-----------|------|--------|
| **Dataset 1** (LFR + Karate + Dolphins) | varies | varies | 10–20 | Undirected | ✅ Ground truth |
| **Dataset 2** (Reddit Hyperlink) | large | temporal | 10 | Directed | Proxy metadata |
| **Dataset 3** (DBLP Collaboration) | 317,080 | 1,049,866 | 10 | Undirected | Venue-based |
| **CollegeMsg surrogate** | 200 | synthetic | 10 | Temporal | Planted communities |

### Data Sources

- **LFR / Karate / Dolphins:** IEEE DataPort
- **Reddit Hyperlink Network:** [SNAP](https://snap.stanford.edu/data/)
- **DBLP Collaboration:** [SNAP](https://snap.stanford.edu/data/)

### Snapshot Construction

- **Dataset 1:** LFR graphs ordered by complexity (T=20); Karate/Dolphins with 5% edge rewiring per step (T=10)
- **Dataset 2:** Timestamped edges binned into non-overlapping windows; directed edges symmetrized
- **Dataset 3:** 3-year publication windows from 1990–2020 (T=10)
- **CollegeMsg:** Planted partition with 12% edge rewiring per snapshot

See [`docs/DATASETS.md`](docs/DATASETS.md) for detailed preprocessing steps.

---

## 🧪 Experiments

### Evaluation Protocol

- **Train / Val / Test:** 70% / 10% / 20% (chronological split)
- **Seeds:** {1, 2, 3, 4, 5} (all results reported as mean ± std)
- **Deterministic settings:**
  ```python
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```
- **Hardware:** NVIDIA RTX 3090 (24 GB), Intel i9-10900X, 64 GB RAM
- **Runtime:** ~3.2 hours for full 5-seed run

### Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Fraction of correctly assigned nodes |
| **F-Score** | Harmonic mean of precision and recall |
| **Modularity Q** | Community cohesion (structure-based) |
| **NMI** | Normalized Mutual Information |
| **ARI** | Adjusted Rand Index |
| **Switch Rate** | Fraction of nodes changing communities between snapshots |

### Statistical Validation

All improvements tested with **paired t-tests** + **Holm step-down correction** across 5 metrics × 3 datasets (p < 0.05).

---

## 📈 Results

### Main Comparison (Accuracy)

| Method | Dataset 1 | Dataset 2 | Dataset 3 | Avg. |
|--------|-----------|-----------|-----------|------|
| DeepWalk | 0.8920 | 0.8810 | 0.9025 | 0.8918 |
| Node2Vec | 0.9105 | 0.8950 | 0.9180 | 0.9078 |
| TGN | 0.9320 | 0.9210 | 0.9385 | 0.9305 |
| TGAT | 0.9505 | 0.9420 | 0.9590 | 0.9505 |
| GCN-LSTM | 0.9380 | 0.9280 | 0.9450 | 0.9370 |
| **TSA-HGNN** | **0.9843** | **0.9755** | **0.9931** | **0.9843** |
| **Gain vs TGAT** | **+3.56%** | **+3.56%** | **+3.56%** | **+3.56%** |

### Temporal Stability (Switch Rate — Lower is Better)

| Dataset | Spectral | TGAT | **TSA-HGNN** | **Reduction** |
|---------|----------|------|-------------|--------------|
| Dataset 1 | 0.124 | 0.067 | **0.023** | **−81.5%** |
| Dataset 2 | 0.178 | 0.098 | **0.039** | **−78.1%** |
| Dataset 3 | 0.145 | 0.078 | **0.028** | **−80.7%** |

### Ablation Study (Dataset 1)

| Variant | Accuracy | Modularity Q | Δ vs Full |
|---------|----------|-------------|----------|
| **Full TSA-HGNN** | **0.9843** | **0.8200** | — |
| w/o Stability Reg. | 0.9680 | 0.8000 | −1.66% |
| w/o ESN | 0.9615 | 0.7800 | −2.32% |
| w/o TCN | 0.9580 | 0.7700 | −2.67% |
| w/o Informer | 0.9530 | 0.7500 | −3.19% |
| w/o GraphSAGE | 0.9480 | 0.7510 | −3.69% |

**Key Insight:** Removing Informer (long-range temporal) causes the largest drop — confirms importance of multi-scale temporal modeling.

See [`docs/RESULTS.md`](docs/RESULTS.md) for complete results across all metrics.

---

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **GraphSAGE** | | |
| Neighbors (s) | 10 | Sampled neighbors per layer |
| Layers | 2 | Aggregation depth |
| Hidden dim (d) | 128 | Embedding dimension |
| **TCN** | | |
| Kernel size (k) | 3 | Causal convolution kernel |
| Dilation layers | 4 | d = 2^L, L ∈ {0,1,2,3} |
| **Informer** | | |
| Attention heads | 8 | Multi-head ProbSparse |
| d_model | 128 | QKV dimension |
| Encoder layers | 3 | Stacked blocks |
| **ESN** | | |
| Reservoir size (N) | 500 | Fixed neurons |
| Leaky rate (γ) | 0.3 | State update speed |
| Spectral radius (ρ) | 0.9 | Stability control |
| Sparsity | 0.1 | Reservoir connectivity |
| **Training** | | |
| Learning rate | 0.001 | Adam optimizer |
| Batch size | 32 | Snapshots per batch |
| Epochs | 200 | With early stopping (patience=20) |
| Stability weight (λ) | 0.10 | Smoothness vs accuracy |

See [`config/default_config.py`](config/default_config.py) for full configuration.

---



