# TSA-HGNN: A Stability-Aware Multi-Scale Temporal Graph Neural Network for Dynamic Community Detection

This repository provides a lightweight and reproducible reference implementation of **TSA-HGNN**, a temporal stability-aware hybrid graph neural network for dynamic community detection in evolving graphs.

The model combines spatial and temporal learning to capture both structural patterns and their evolution across time. It integrates the following components:

- **GraphSAGE** for inductive snapshot-level node embeddings  
- **Temporal Convolutional Network (TCN)** for short-range temporal modeling  
- **Informer-style attention** for long-range temporal dependency learning  
- **Echo State Network (ESN)** for nonlinear temporal memory  

This implementation is intended to support reproducibility and facilitate further research and extensions.

---

## Repository Contents

- `tsa_hgnn.py`  
  Core implementation of the TSA-HGNN model, including a minimal runnable example.

- `dynamic_graph_construction.py`  
  Utility for constructing time-ordered graph snapshots from a timestamped edge list (`dynamic_edges.csv`) and exporting per-snapshot graphs.

---

## Installation

Tested with Python 3.9 or later.

```bash
pip install -r requirements.txt
