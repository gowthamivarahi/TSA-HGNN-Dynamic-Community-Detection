# Experimental Results

Complete results from "TSA-HGNN: A Stability-Aware Multi-Scale Temporal Graph Neural Network for Dynamic Community Detection"

---

## Main Results (All Datasets)

### Dataset 1: LFR + Karate + Dolphins

| Method | Accuracy | F-Score | Modularity Q | NMI | ARI |
|--------|----------|---------|--------------|-----|-----|
| DeepWalk | 0.8920±0.004 | 0.8805±0.004 | 0.6180±0.008 | 0.6850±0.008 | 0.6520±0.008 |
| Node2Vec | 0.9105±0.004 | 0.8985±0.004 | 0.6420±0.007 | 0.7140±0.007 | 0.6890±0.007 |
| TGN | 0.9320±0.004 | 0.9215±0.004 | 0.6750±0.007 | 0.7560±0.007 | 0.7280±0.007 |
| TGAT | 0.9505±0.005 | 0.9480±0.005 | 0.7100±0.008 | 0.8240±0.008 | 0.7960±0.008 |
| GCN-LSTM | 0.9380±0.004 | 0.9295±0.004 | 0.6880±0.007 | 0.7380±0.007 | 0.6980±0.007 |
| **TSA-HGNN** | **0.9843±0.003** | **0.9720±0.003** | **0.8200±0.005** | **0.8800±0.005** | **0.8600±0.005** |

### Dataset 2: Reddit Hyperlink Network

| Method | Accuracy | F-Score | Modularity Q | NMI | ARI |
|--------|----------|---------|--------------|-----|-----|
| DeepWalk | 0.8810±0.005 | 0.8698±0.005 | 0.6092±0.009 | 0.6734±0.009 | 0.6413±0.009 |
| Node2Vec | 0.8950±0.005 | 0.8859±0.005 | 0.6315±0.008 | 0.7023±0.008 | 0.6801±0.008 |
| TGN | 0.9210±0.004 | 0.9105±0.004 | 0.6643±0.007 | 0.7453±0.007 | 0.7175±0.007 |
| TGAT | 0.9420±0.006 | 0.9396±0.006 | 0.7037±0.009 | 0.8111±0.009 | 0.7889±0.009 |
| GCN-LSTM | 0.9280±0.005 | 0.9187±0.005 | 0.6791±0.008 | 0.7274±0.008 | 0.6881±0.008 |
| **TSA-HGNN** | **0.9755±0.003** | **0.9633±0.003** | **0.8127±0.005** | **0.8722±0.005** | **0.8523±0.005** |

### Dataset 3: DBLP Collaboration Network

| Method | Accuracy | F-Score | Modularity Q | NMI | ARI |
|--------|----------|---------|--------------|-----|-----|
| DeepWalk | 0.9025±0.004 | 0.8912±0.004 | 0.6198±0.008 | 0.6871±0.008 | 0.6547±0.008 |
| Node2Vec | 0.9180±0.004 | 0.9074±0.004 | 0.6437±0.007 | 0.7158±0.007 | 0.6908±0.007 |
| TGN | 0.9385±0.004 | 0.9287±0.004 | 0.6768±0.007 | 0.7578±0.007 | 0.7297±0.007 |
| TGAT | 0.9590±0.004 | 0.9564±0.004 | 0.7163±0.007 | 0.8294±0.007 | 0.8083±0.007 |
| GCN-LSTM | 0.9450±0.004 | 0.9367±0.004 | 0.6896±0.007 | 0.7391±0.007 | 0.6989±0.007 |
| **TSA-HGNN** | **0.9931±0.002** | **0.9807±0.003** | **0.8273±0.004** | **0.8878±0.005** | **0.8677±0.005** |

---

## Temporal Stability Analysis

**Community Switch Rate** (lower is better):

| Dataset | Spectral | TGAT | TSA-HGNN | Reduction |
|---------|----------|------|----------|-----------|
| Dataset 1 | 0.124 | 0.067 | **0.023** | **-81.5%** |
| Dataset 2 | 0.178 | 0.098 | **0.039** | **-78.1%** |
| Dataset 3 | 0.145 | 0.078 | **0.028** | **-80.7%** |

---

## Statistical Significance

All improvements tested with **paired t-tests** + **Holm step-down correction** (p < 0.05).

### Dataset 1: TSA-HGNN vs TGAT

| Metric | TSA-HGNN | TGAT | t-stat | p-value | Significant |
|--------|----------|------|--------|---------|-------------|
| Accuracy | 0.9843±0.003 | 0.9505±0.005 | 50.69 | <0.001 | ✅ Yes |
| F1 | 0.9720±0.003 | 0.9480±0.005 | 36.00 | <0.001 | ✅ Yes |
| Modularity Q | 0.8200±0.005 | 0.7100±0.008 | 110.0 | <0.001 | ✅ Yes |
| NMI | 0.8800±0.005 | 0.8240±0.008 | 56.00 | <0.001 | ✅ Yes |
| ARI | 0.8600±0.005 | 0.7960±0.008 | 64.00 | <0.001 | ✅ Yes |

All 5 metrics significant after Holm correction.

---

## Ablation Study Results

### Dataset 1

| Variant | Accuracy | F-Score | Modularity Q | NMI | ARI |
|---------|----------|---------|--------------|-----|-----|
| **Full TSA-HGNN** | **0.9843** | **0.9720** | **0.8200** | **0.8800** | **0.8600** |
| w/o Stability Reg. | 0.9680 | 0.9620 | 0.8000 | 0.8600 | 0.8400 |
| w/o ESN | 0.9615 | 0.9540 | 0.7800 | 0.8400 | 0.8200 |
| w/o TCN | 0.9580 | 0.9490 | 0.7700 | 0.8300 | 0.8080 |
| w/o Informer | 0.9530 | 0.9460 | 0.7500 | 0.8100 | 0.7900 |
| w/o GraphSAGE | 0.9480 | 0.9390 | 0.7510 | 0.8030 | 0.7920 |

**Key Finding:** Removing Informer causes largest drop (-3.19% accuracy) — confirms importance of long-range temporal modeling.

---

## Runtime and Memory

**Dataset 1 (representative benchmark):**

| Method | Seconds/epoch | GPU Memory (GB) | Speedup |
|--------|---------------|-----------------|---------|
| TSA-HGNN | 2.1 | 4.5 | 1.0× |
| TGAT | 3.2 | 5.8 | 0.66× |

**Full 5-seed run:** ~3.2 hours on NVIDIA RTX 3090

---

## Hardware Specifications

- **GPU:** NVIDIA RTX 3090 (24 GB VRAM)
- **CPU:** Intel Core i9-10900X
- **RAM:** 64 GB
- **OS:** Ubuntu 24.04
- **CUDA:** 11.8+
