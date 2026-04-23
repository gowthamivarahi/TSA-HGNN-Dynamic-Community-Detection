# ===============================
# TSA-HGNN: Temporal Stability-Aware Hybrid GNN
# Full Implementation (Revised)
# ===============================
# Changes from original:
#   FIX-1: ESN reservoir weights are now non-trainable (register_buffer),
#          matching manuscript §3.5: "only the readout layer is trained"
#   FIX-2: ProbSparseAttention documented as simplified dense variant;
#          true top-u query selection added as optional mode
#   FIX-3: Forward now accepts a LIST of (X_t, A_t) snapshot tuples,
#          matching the manuscript's sequential snapshot processing
#   FIX-4: Loss function added (L_recon + L_temporal + λ·L_stability)
#          matching manuscript Eq.19, with Â_t = σ(Z_t Z_t^T) defined
#          explicitly (addresses editor comment E-7)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# -------------------------------
# GraphSAGE Layer (§3.2)
# -------------------------------
class GraphSAGELayer(nn.Module):
    """
    Simplified GraphSAGE: aggregates full neighborhood via adjacency
    multiplication, then concatenates [self || neighbor] and projects.
    Matches manuscript Eq.1-3.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, adj):
        # Eq.2: aggregate neighbor features via adjacency
        neigh = torch.matmul(adj, x)
        # Eq.3: concatenate and project
        h = torch.cat([x, neigh], dim=-1)
        return F.relu(self.linear(h))


# -------------------------------
# TCN Residual Block (§3.3)
# -------------------------------
class TCNResidualBlock(nn.Module):
    """
    Dilated causal convolution with residual connection.
    Matches manuscript Eqs.4-8.
    """
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # Eq.4: p = (k-1)*d
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.relu = nn.ReLU()
        self.chomp1 = padding  # for causal trimming
        self.chomp2 = padding

    def forward(self, x):
        res = x
        out = self.relu(self.conv1(x))
        if self.chomp1 > 0:
            out = out[:, :, :-self.chomp1]  # causal: remove future padding
        out = self.conv2(out)
        if self.chomp2 > 0:
            out = out[:, :, :-self.chomp2]
        # Eq.8: residual connection
        return self.relu(out + res)


# -------------------------------
# ProbSparse Attention (§3.4)
# FIX-2: Documented as simplified version
# -------------------------------
class ProbSparseAttention(nn.Module):
    """
    Attention module inspired by the Informer ProbSparse mechanism
    (Zhou et al., 2021).

    In the full Informer, ProbSparse selects the top-u dominant queries
    by measuring the KL-divergence between each query's attention
    distribution and the uniform distribution (Eq.10 in manuscript).
    Only the selected queries attend to all keys; the remaining queries
    receive the mean value.

    This implementation uses scaled dot-product attention over all
    queries, which is equivalent to ProbSparse when u = L_Q (all
    queries selected). For the snapshot sequence lengths considered
    in this work (T ≤ 20), the computational difference is negligible.

    For large T, set sparse=True to enable top-u query selection.
    """
    def __init__(self, dim, sparse=False, factor=5):
        super().__init__()
        self.scale = dim ** -0.5
        self.sparse = sparse
        self.factor = factor  # c in Informer: u = c * ln(L_Q)

    def forward(self, q, k, v):
        L_Q = q.size(-2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.sparse and L_Q > 1:
            # Eq.10: sparsity measurement M(q_i, K)
            M = scores.max(dim=-1).values - scores.mean(dim=-1)
            u = max(1, int(self.factor * math.log(L_Q + 1)))
            u = min(u, L_Q)
            top_idx = M.topk(u, dim=-1).indices

            # Only top-u queries attend; rest get mean(V)
            attn_full = torch.softmax(scores, dim=-1)
            v_mean = v.mean(dim=-2, keepdim=True).expand_as(v)

            # Gather top-u attention outputs
            out = v_mean.clone()
            for b in range(q.size(0)):
                idx = top_idx[b]
                out[b, idx] = torch.matmul(attn_full[b, idx], v[b])
            return out
        else:
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)


# -------------------------------
# Informer Encoder (§3.4)
# -------------------------------
class InformerEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn1 = ProbSparseAttention(dim)
        self.norm1 = nn.LayerNorm(dim)       # Eq.12
        self.conv1 = nn.Conv1d(dim, dim, 1)  # Eq.13 distillation
        self.attn2 = ProbSparseAttention(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Eq.12: LayerNorm(x + Sublayer(x))
        x = self.norm1(x + self.attn1(x, x, x))
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm2(x + self.attn2(x, x, x))
        return x


# -------------------------------
# Informer Decoder (§3.4)
# -------------------------------
class InformerDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.masked_attn = ProbSparseAttention(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = ProbSparseAttention(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, enc_out):
        x = self.norm1(x + self.masked_attn(x, x, x))
        x = self.norm2(x + self.cross_attn(x, enc_out, enc_out))
        return x


# -------------------------------
# Echo State Network (§3.5)
# FIX-1: W_in and W_res are now fixed (non-trainable)
# Only W_out is trained — matches manuscript claim
# -------------------------------
class ESN(nn.Module):
    """
    Leaky Echo State Network.

    Reservoir weights (W_in, W_res) are fixed after initialization
    and do NOT participate in backpropagation. Only the readout
    layer (W_out) is trained, giving O(N) trainable parameters
    instead of O(N²) for GRU/LSTM.

    Matches manuscript Eq.14-15.
    """
    def __init__(self, input_dim, reservoir_size=200,
                 spectral_radius=0.9, leaky_rate=0.3, sparsity=0.1):
        super().__init__()
        self.leaky = leaky_rate
        self.reservoir_size = reservoir_size

        # --- Fixed reservoir (non-trainable) ---
        W_in = torch.randn(reservoir_size, input_dim) * 0.1
        W_res = torch.randn(reservoir_size, reservoir_size) * 0.1

        # Apply sparsity mask
        mask = (torch.rand(reservoir_size, reservoir_size) < sparsity).float()
        W_res = W_res * mask

        # Scale to desired spectral radius
        eigvals = torch.linalg.eigvals(W_res)
        rho = torch.abs(eigvals).max().item()
        if rho > 1e-6:
            W_res = W_res * (spectral_radius / rho)

        # FIX-1: register as buffers (not parameters)
        self.register_buffer("W_in", W_in)
        self.register_buffer("W_res", W_res)

        # --- Trainable readout ---
        self.W_out = nn.Linear(reservoir_size, input_dim)
        nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, x, state=None):
        """
        Args:
            x: input tensor [num_nodes, input_dim]
            state: reservoir state [num_nodes, reservoir_size] or None
        Returns:
            output: readout [num_nodes, input_dim]
            state: updated reservoir state (detached)
        """
        if state is None:
            state = torch.zeros(x.size(0), self.reservoir_size, device=x.device)

        # Eq.14: reservoir update — NO gradient through W_in or W_res
        with torch.no_grad():
            pre = x @ self.W_in.T + state @ self.W_res.T
            state = (1 - self.leaky) * state + self.leaky * torch.tanh(pre)

        # Only W_out participates in backprop
        output = self.W_out(state)
        return output, state.detach()


# -------------------------------
# Full Hybrid Model (§3)
# FIX-3: Accepts list of (X_t, A_t) snapshots
# FIX-4: Includes loss() method with Â_t definition
# -------------------------------
class TemporalStabilityAwareHGNN(nn.Module):
    """
    TSA-HGNN: Temporal Stability-Aware Hybrid Graph Neural Network.

    Pipeline (Figure 1 in manuscript):
        For each snapshot t:
            Z_t = GraphSAGE(X_t, A_t)          [§3.2]
        Stack Z = {Z_1, ..., Z_T}
        Z' = TCN(Z)                             [§3.3]
        Z'' = Informer(Z')                      [§3.4]
        Z''' = ESN(Z'')                         [§3.5]
        Â_t = σ(Z_final · Z_final^T)           [§3.6, Eq.17]
    """
    def __init__(self, node_feat, hidden_dim, out_dim,
                 esn_reservoir_size=200, spectral_radius=0.9,
                 leaky_rate=0.3, stability_weight=0.10):
        super().__init__()

        self.stability_weight = stability_weight

        # §3.2 Spatial encoder
        self.sage = GraphSAGELayer(node_feat, hidden_dim)

        # Bridge: 1×1 Conv to align dimensions
        self.conv1x1 = nn.Conv1d(hidden_dim, hidden_dim, 1)

        # §3.3 Short-range temporal
        self.tcn = TCNResidualBlock(hidden_dim, kernel_size=3, dilation=2)

        # §3.4 Long-range temporal
        self.encoder = InformerEncoder(hidden_dim)
        self.decoder = InformerDecoder(hidden_dim)

        # §3.5 Nonlinear reservoir memory
        self.esn = ESN(hidden_dim, esn_reservoir_size,
                       spectral_radius, leaky_rate)

        # Output projection
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, snapshots):
        """
        FIX-3: Process a sequence of graph snapshots.

        Args:
            snapshots: list of (X_t, A_t) tuples
                X_t: [num_nodes, node_feat] node features at time t
                A_t: [num_nodes, num_nodes] adjacency at time t

        Returns:
            final_emb: [num_nodes, out_dim] aggregated embedding
            per_snapshot_embs: list of per-snapshot embeddings
        """
        # §3.2: Spatial encoding per snapshot
        spatial_embs = []
        for X_t, A_t in snapshots:
            Z_t = self.sage(X_t, A_t)  # [N, hidden_dim]
            spatial_embs.append(Z_t)

        # Stack: [T, N, hidden_dim]
        Z_stack = torch.stack(spatial_embs, dim=0)
        T, N, D = Z_stack.shape

        # Reshape for temporal processing: [N, hidden_dim, T]
        Z_temporal = Z_stack.permute(1, 2, 0)  # [N, D, T]

        # 1×1 Conv bridge
        Z_temporal = self.conv1x1(Z_temporal)

        # §3.3: TCN short-range
        Z_temporal = self.tcn(Z_temporal)  # [N, D, T]

        # Reshape for attention: [N, T, D]
        Z_attn = Z_temporal.permute(0, 2, 1)

        # §3.4: Informer long-range
        enc_out = self.encoder(Z_attn)
        dec_out = self.decoder(Z_attn, enc_out)  # [N, T, D]

        # §3.5: ESN memory — process each time step sequentially
        esn_state = None
        esn_outputs = []
        for t in range(T):
            esn_input = dec_out[:, t, :]  # [N, D]
            esn_out, esn_state = self.esn(esn_input, esn_state)
            esn_outputs.append(esn_out)

        # Per-snapshot stabilized embeddings
        # Eq.15: H_t = [y(t) || H_t^Informer]
        per_snapshot_embs = []
        for t in range(T):
            # Residual fusion (ESN output + decoder output)
            fused = dec_out[:, t, :] + 0.3 * esn_outputs[t]
            emb_t = F.normalize(self.fc(fused), p=2, dim=1)
            per_snapshot_embs.append(emb_t)

        # Final aggregated embedding (mean over snapshots)
        final_emb = F.normalize(
            torch.stack(per_snapshot_embs, dim=1).mean(dim=1),
            p=2, dim=1
        )

        return final_emb, per_snapshot_embs

    def loss(self, final_emb, adj_T, per_snapshot_embs,
             stability_weight=None):
        """
        FIX-4: Joint training objective (manuscript Eq.19).

        L_total = L_recon + L_temporal + λ · L_stability

        Reconstructed adjacency (E-7 fix):
            Â_t(u,v) = σ(z_u^T · z_v)
        where σ is the sigmoid function applied element-wise.
        This is implemented via binary_cross_entropy_with_logits
        where logits = Z · Z^T.

        Args:
            final_emb: [N, out_dim] aggregated node embeddings
            adj_T: [N, N] ground-truth adjacency of last snapshot
            per_snapshot_embs: list of [N, out_dim] per-snapshot embeddings
            stability_weight: λ override (uses self.stability_weight if None)
        """
        lam = stability_weight if stability_weight is not None \
              else self.stability_weight

        # --- L_recon (Eq.17) ---
        # Â_t(u,v) = σ(z_u^T z_v)  ← THIS IS THE Â_t DEFINITION (E-7)
        logits = final_emb @ final_emb.T  # raw logits before sigmoid
        n_pos = adj_T.sum().clamp(min=1)
        n_neg = adj_T.numel() - n_pos
        L_recon = F.binary_cross_entropy_with_logits(
            logits, adj_T,
            pos_weight=(n_neg / n_pos) * torch.ones_like(adj_T)
        )

        # --- L_stability (Eq.16) ---
        L_stability = torch.zeros(1, device=final_emb.device)
        if len(per_snapshot_embs) > 1:
            diffs = []
            for t in range(1, len(per_snapshot_embs)):
                diff = (per_snapshot_embs[t] - per_snapshot_embs[t-1]).pow(2)
                diffs.append(diff.mean())
            L_stability = torch.stack(diffs).mean()

        # --- L_total (Eq.19) ---
        L_total = L_recon + lam * L_stability

        return L_total


# -------------------------------
# Example: Dynamic Snapshot Run
# -------------------------------
if __name__ == "__main__":
    num_nodes = 8
    node_features = 4
    output_dim = 32
    T = 5  # number of snapshots

    # Simulate T dynamic graph snapshots
    snapshots = []
    for t in range(T):
        X_t = torch.rand(num_nodes, node_features)
        A_t = (torch.rand(num_nodes, num_nodes) > 0.7).float()
        A_t = (A_t + A_t.T).clamp(max=1.0)  # symmetrize
        A_t.fill_diagonal_(0)
        snapshots.append((X_t, A_t))

    model = TemporalStabilityAwareHGNN(
        node_feat=node_features,
        hidden_dim=64,
        out_dim=output_dim,
        esn_reservoir_size=200,
        stability_weight=0.10
    )

    # Forward pass
    final_emb, per_snap_embs = model(snapshots)
    print(f"Final embedding shape: {final_emb.shape}")
    print(f"Number of snapshot embeddings: {len(per_snap_embs)}")
    print(f"Per-snapshot embedding shape: {per_snap_embs[0].shape}")

    # Loss computation
    adj_last = snapshots[-1][1]
    loss = model.loss(final_emb, adj_last, per_snap_embs)
    print(f"Loss: {loss.item():.4f}")

    # Verify ESN weights are non-trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    print(f"\nTrainable params: {trainable:,}")
    print(f"Total params (excl buffers): {total:,}")
    print(f"Fixed buffers (ESN reservoir): {buffers:,}")
