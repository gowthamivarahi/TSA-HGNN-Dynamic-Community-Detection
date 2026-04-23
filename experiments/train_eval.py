"""
TSA-HGNN: Training and Evaluation Script
==========================================
Simple training script for dynamic community detection.

Usage:
    python experiments/train_eval.py --dataset 1 --epochs 200

Authors: Gowthami Vusirikkayala, Madhu Viswanatham V
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.tsa_hgnn import TemporalStabilityAwareHGNN
from config.default_config import get_default_config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_synthetic_snapshots(n_nodes=100, n_snapshots=10, feature_dim=32):
    """Create synthetic dynamic graph snapshots for testing."""
    snapshots = []
    for t in range(n_snapshots):
        # Node features with temporal drift
        X_t = torch.randn(n_nodes, feature_dim) + 0.1 * t
        
        # Random adjacency with some temporal correlation
        A_t = (torch.rand(n_nodes, n_nodes) > 0.85).float()
        A_t = (A_t + A_t.T).clamp(max=1.0)  # Symmetrize
        A_t.fill_diagonal_(0)  # No self-loops
        
        snapshots.append((X_t, A_t))
    
    return snapshots


def train_model(model, snapshots, config, device='cuda'):
    """Simple training loop."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    
    # Move snapshots to device
    snapshots_device = [(X.to(device), A.to(device)) for X, A in snapshots]
    
    print("Training TSA-HGNN...")
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        final_emb, per_snap_embs = model(snapshots_device)
        
        # Compute loss
        loss = model.loss(final_emb, snapshots_device[-1][1], per_snap_embs)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{config.epochs}, Loss: {loss.item():.4f}")
    
    return model


def evaluate_model(model, snapshots, device='cuda'):
    """Evaluate model and return embeddings."""
    model.eval()
    snapshots_device = [(X.to(device), A.to(device)) for X, A in snapshots]
    
    with torch.no_grad():
        final_emb, per_snap_embs = model(snapshots_device)
    
    return final_emb.cpu(), [emb.cpu() for emb in per_snap_embs]


def main():
    """Main training and evaluation pipeline."""
    print("=" * 60)
    print("TSA-HGNN: Dynamic Community Detection")
    print("=" * 60)
    
    # Configuration
    config = get_default_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Set seed for reproducibility
    set_seed(config.data_seed)
    
    # Create synthetic data
    print("\nGenerating synthetic dynamic graph...")
    snapshots = create_synthetic_snapshots(
        n_nodes=100,
        n_snapshots=10,
        feature_dim=config.in_dim
    )
    print(f"  Nodes: 100")
    print(f"  Snapshots: 10")
    print(f"  Feature dim: {config.in_dim}")
    
    # Initialize model
    print("\nInitializing TSA-HGNN...")
    model = TemporalStabilityAwareHGNN(
        node_feat=config.in_dim,
        hidden_dim=config.graphsage_hidden_dim,
        out_dim=config.out_dim,
        esn_reservoir_size=config.esn_reservoir_size,
        stability_weight=config.stability_weight
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Train
    print()
    model = train_model(model, snapshots, config, device)
    
    # Evaluate
    print("\nEvaluating...")
    final_emb, per_snap_embs = evaluate_model(model, snapshots, device)
    print(f"  Final embedding shape: {final_emb.shape}")
    print(f"  Per-snapshot embeddings: {len(per_snap_embs)} snapshots")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
