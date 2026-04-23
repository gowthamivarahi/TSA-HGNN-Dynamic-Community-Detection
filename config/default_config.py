"""
TSA-HGNN: Default Hyperparameter Configuration
================================================
All hyperparameters from Table 2 of the paper.

Authors: Gowthami Vusirikkayala, Madhu Viswanatham V
Institution: VIT Vellore
Paper: Frontiers in Artificial Intelligence, 2026
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TSAHGNNConfig:
    """Complete hyperparameter configuration for TSA-HGNN."""

    # ══════════════════════════════════════════════════════════════
    # GraphSAGE Configuration (§3.2)
    # ══════════════════════════════════════════════════════════════
    graphsage_neighbors: int = 10           # Sampled neighbors per node per layer
    graphsage_layers: int = 2               # Number of aggregation layers
    graphsage_hidden_dim: int = 128         # Embedding dimension per node

    # ══════════════════════════════════════════════════════════════
    # TCN Configuration (§3.3)
    # ══════════════════════════════════════════════════════════════
    tcn_kernel_size: int = 3                # Dilated causal convolution kernel
    tcn_dilation_layers: int = 4            # d = 2^L, L ∈ {0,1,2,3}

    # ══════════════════════════════════════════════════════════════
    # Informer Configuration (§3.4)
    # ══════════════════════════════════════════════════════════════
    informer_heads: int = 8                 # Multi-head ProbSparse attention
    informer_d_model: int = 128             # Query/Key/Value dimension
    informer_encoder_layers: int = 3        # Stacked encoder blocks

    # ══════════════════════════════════════════════════════════════
    # ESN Configuration (§3.5)
    # ══════════════════════════════════════════════════════════════
    esn_reservoir_size: int = 500           # Number of reservoir neurons
    esn_leaky_rate: float = 0.3             # Controls state update speed (γ)
    esn_spectral_radius: float = 0.9        # Controls stability of reservoir
    esn_sparsity: float = 0.1               # Fraction of non-zero reservoir weights

    # ══════════════════════════════════════════════════════════════
    # Training Configuration
    # ══════════════════════════════════════════════════════════════
    learning_rate: float = 0.001            # Adam optimizer
    weight_decay: float = 1e-4              # L2 regularization
    epochs: int = 200                       # Maximum training epochs
    patience: int = 20                      # Early stopping patience
    grad_clip: float = 1.0                  # Gradient clipping norm
    batch_size: int = 32                    # Snapshots per batch

    # ══════════════════════════════════════════════════════════════
    # Stability Regularization (§3.6, Eq. 19)
    # ══════════════════════════════════════════════════════════════
    stability_weight: float = 0.10          # λ: smoothness vs accuracy tradeoff

    # ══════════════════════════════════════════════════════════════
    # Evaluation Configuration
    # ══════════════════════════════════════════════════════════════
    n_clusters: int = 5                     # Number of communities (K)
    n_seeds: List[int] = None               # Seeds for reproducibility
    data_seed: int = 42                     # Seed for data generation

    # ══════════════════════════════════════════════════════════════
    # Data Split Configuration
    # ══════════════════════════════════════════════════════════════
    train_ratio: float = 0.70               # Chronological training split
    val_ratio: float = 0.10                 # Validation split
    test_ratio: float = 0.20                # Test split

    # ══════════════════════════════════════════════════════════════
    # Model Dimensions
    # ══════════════════════════════════════════════════════════════
    in_dim: int = 32                        # Input node feature dimension
    out_dim: int = 32                       # Output embedding dimension

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_seeds is None:
            self.n_seeds = [1, 2, 3, 4, 5]
        
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, \
            "Split ratios must sum to 1.0"
        assert 0 < self.stability_weight < 1, \
            "Stability weight λ must be in (0, 1)"
        assert self.esn_spectral_radius < 1, \
            "ESN spectral radius must be < 1 for stability"


# ══════════════════════════════════════════════════════════════════
# Reproducibility Settings
# ══════════════════════════════════════════════════════════════════

DETERMINISTIC_SETTINGS = {
    'torch_deterministic': True,
    'cudnn_benchmark': False,
    'cudnn_deterministic': True,
}


# ══════════════════════════════════════════════════════════════════
# Hardware Configuration
# ══════════════════════════════════════════════════════════════════

HARDWARE_SPECS = {
    'gpu': 'NVIDIA RTX 3090',
    'vram': '24 GB',
    'cpu': 'Intel Core i9-10900X',
    'ram': '64 GB',
}


# ══════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════

def get_default_config() -> TSAHGNNConfig:
    """Get the default configuration used in the paper."""
    return TSAHGNNConfig()


def print_config(config: TSAHGNNConfig):
    """Pretty print configuration."""
    print("=" * 60)
    print("TSA-HGNN Configuration")
    print("=" * 60)
    for field_name, field_value in config.__dict__.items():
        print(f"  {field_name:30s} = {field_value}")
    print("=" * 60)


if __name__ == "__main__":
    cfg = get_default_config()
    print_config(cfg)
