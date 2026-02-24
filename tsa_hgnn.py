# ===============================
# Temporal Stability-Aware Hybrid GNN
# Single-Cell Full Implementation
# ===============================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------------
# GraphSAGE Layer
# -------------------------------
class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, adj):
        neigh = torch.matmul(adj, x)
        h = torch.cat([x, neigh], dim=-1)
        return F.relu(self.linear(h))


# -------------------------------
# TCN Residual Block
# -------------------------------
class TCNResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=dilation, dilation=dilation
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + res)


# -------------------------------
# ProbSparse Attention (Informer)
# -------------------------------
class ProbSparseAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


# -------------------------------
# Informer Encoder
# -------------------------------
class InformerEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn1 = ProbSparseAttention(dim)
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.attn2 = ProbSparseAttention(dim)
        self.conv2 = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        x = self.attn1(x, x, x)
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = self.attn2(x, x, x)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        return x


# -------------------------------
# Informer Decoder
# -------------------------------
class InformerDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.masked_attn = ProbSparseAttention(dim)
        self.cross_attn = ProbSparseAttention(dim)

    def forward(self, x, enc_out):
        x = self.masked_attn(x, x, x)
        x = self.cross_attn(x, enc_out, enc_out)
        return x


# -------------------------------
# Echo State Network (ESN)
# -------------------------------
class ESN(nn.Module):
    def __init__(self, input_dim, reservoir_size=200):
        super().__init__()
        self.W_in = nn.Linear(input_dim, reservoir_size)
        self.W_res = nn.Linear(reservoir_size, reservoir_size, bias=False)
        self.state = None

    def forward(self, x):
        if self.state is None:
            self.state = torch.zeros(x.size(0), self.W_res.out_features)
        self.state = torch.tanh(self.W_in(x) + self.W_res(self.state))
        return self.state


# -------------------------------
# Full Hybrid Model
# -------------------------------
class TemporalStabilityAwareHGNN(nn.Module):
    def __init__(self, node_feat, hidden_dim, out_dim):
        super().__init__()

        # Spatial
        self.sage = GraphSAGELayer(node_feat, hidden_dim)

        # 1×1 Conv
        self.conv1x1 = nn.Conv1d(hidden_dim, hidden_dim, 1)

        # TCN
        self.tcn = TCNResidualBlock(hidden_dim, dilation=2)

        # Informer
        self.encoder = InformerEncoder(hidden_dim)
        self.decoder = InformerDecoder(hidden_dim)

        # ESN
        self.esn = ESN(hidden_dim)

        # Output
        self.fc = nn.Linear(200, out_dim)

    def forward(self, x, adj):
        # GraphSAGE
        x = self.sage(x, adj)

        # Prepare temporal format
        x = x.unsqueeze(0).transpose(1, 2)

        # 1×1 Conv
        x = self.conv1x1(x)

        # TCN
        x = self.tcn(x)
        x = x.transpose(1, 2)

        # Informer
        enc = self.encoder(x)
        dec = self.decoder(x, enc)

        # ESN memory
        esn_out = self.esn(dec.mean(dim=1))

        return self.fc(esn_out)


# -------------------------------
# Example Dynamic Snapshot Run
# -------------------------------
if __name__ == "__main__":

    num_nodes = 8
    node_features = 4
    output_classes = 3

    # Dummy dynamic graph snapshot
    X = torch.rand(num_nodes, node_features)
    A = torch.eye(num_nodes)  # adjacency matrix

    model = TemporalStabilityAwareHGNN(
        node_feat=node_features,
        hidden_dim=64,
        out_dim=output_classes
    )

    output = model(X, A)
    print("Model Output:", output)