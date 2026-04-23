# Changelog

All notable changes to the TSA-HGNN project.

---

## [1.0.0] - 2026-04-23

### Initial Release

#### Added
- Complete TSA-HGNN model implementation
  - GraphSAGE spatial encoder (§3.2)
  - TCN for short-range temporal modeling (§3.3)
  - Informer with ProbSparse attention (§3.4)
  - ESN with fixed reservoir (§3.5)
  - Temporal stability regularization (§3.6)
  - Joint loss function (Eq. 19)

- Model features
  - Multi-snapshot forward pass
  - Inductive learning (handles evolving node sets)
  - Sub-quadratic temporal complexity O(T log T)
  - Explicit Â_t reconstruction (Eq. 16a)

- Data processing
  - Dynamic graph snapshot construction
  - CSV-based edge input format
  - Sample data for quick testing

- Experiments
  - Training and evaluation script
  - 5-seed reproducibility protocol
  - Deterministic settings

- Documentation
  - Comprehensive README with architecture diagram
  - Complete results (3 datasets + statistical tests)
  - Dataset descriptions and preprocessing details
  - MIT License

#### Model Fixes (from paper revision)
- FIX-1: ESN reservoir weights as non-trainable (register_buffer)
- FIX-2: ProbSparse attention documented
- FIX-3: Multi-snapshot forward pass
- FIX-4: Complete loss function with stability term

### Repository Structure
```
Dynamic-Community-Detection/
├── models/tsa_hgnn.py
├── data/dynamic_graph_construction.py
├── experiments/train_eval.py
├── config/default_config.py
├── docs/{RESULTS.md, DATASETS.md}
├── README.md
├── requirements.txt
└── LICENSE
```

### Performance Benchmarks
- Dataset 1: 0.9843 accuracy, 0.023 switch rate
- Dataset 2: 0.9755 accuracy, 0.039 switch rate
- Dataset 3: 0.9931 accuracy, 0.028 switch rate

### Citation
Paper accepted to Frontiers in Artificial Intelligence, 2026.
