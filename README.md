# TSA-HGNN: A Stability-Aware Multi-Scale Temporal Graph Neural Network for Dynamic Community Detection

This repository provides a lightweight, reproducible reference implementation of **TSA-HGNN**, a temporal stability-aware hybrid graph neural architecture for dynamic community detection.

The model integrates:
- **GraphSAGE** (snapshot-level inductive spatial embeddings)
- **TCN** (short-range temporal dynamics)
- **Informer-style attention** (long-horizon dependency modeling)
- **ESN (Echo State Network)** (nonlinear temporal memory)

This code is intended to support reproducibility and future extension of the pipeline described in the manuscript.

---

## Files in this repository

- `tsa_hgnn.py`  
  Single-file TSA-HGNN implementation with a small runnable example.

- `dynamic_graph_construction.py`  
  Constructs time-ordered graph snapshots from a timestamped edge list (`dynamic_edges.csv`) and exports per-snapshot CSV edge lists.

---

## Installation

Tested with Python 3.9+.

```bash
pip install -r requirements.txt