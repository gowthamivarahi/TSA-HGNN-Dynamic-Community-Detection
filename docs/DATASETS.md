# Datasets

Detailed information about datasets used in TSA-HGNN experiments.

---

## Overview

| Dataset | Nodes | Edges | Snapshots | Type | Ground Truth |
|---------|-------|-------|-----------|------|--------------|
| Dataset 1 | varies | varies | 10–20 | Undirected | ✅ Yes |
| Dataset 2 | large | temporal | 10 | Directed | Proxy labels |
| Dataset 3 | 317,080 | 1,049,866 | 10 | Undirected | Venue-based |
| CollegeMsg | 200 | synthetic | 10 | Temporal | Planted |

---

## Dataset 1: Complex Network Community Detection

### Components

1. **LFR Synthetic Graphs** (T=20)
   - Controlled degree distribution and community size
   - Mixing parameter μ varies from 0.1 to 0.5
   - Ordered by increasing complexity
   - Used for controlled evaluation

2. **Zachary's Karate Club** (T=10)
   - 34 nodes, 2 known communities
   - 5% edge rewiring per snapshot
   - Protocol from Sattar et al. (2023)

3. **Dolphins Social Network** (T=10)
   - 62 nodes, 2 known communities
   - 5% edge rewiring per snapshot
   - Tests stability under controlled dynamics

### Snapshot Construction

```python
# LFR: Direct sequence
snapshots = [LFR(n=100, mu=0.1), LFR(n=100, mu=0.15), ...]

# Karate/Dolphins: Rewiring protocol
for t in range(10):
    G_new = rewire_edges(G_prev, fraction=0.05)
    snapshots.append(G_new)
```

### Evaluation

- **Labels:** Ground truth communities from original graphs
- **Metrics:** All 5 metrics (Accuracy, F1, Q, NMI, ARI)

---

## Dataset 2: Reddit Hyperlink Network

### Description

- **Source:** Stanford SNAP
- **URL:** https://snap.stanford.edu/data/soc-RedditHyperlinks.html
- **Nodes:** Subreddits
- **Edges:** Directed hyperlinks with timestamps
- **Attributes:** Sentiment, text features

### Snapshot Construction

1. Bin timestamped edges into 10 non-overlapping time windows
2. Symmetrize directed edges: (u→v) becomes (u,v)
3. Aggregate sentiment/text as node attributes
4. Each window = 1 snapshot

### Preprocessing

```python
# Symmetrize adjacency
A_undirected = (A_directed + A_directed.T).clamp(max=1.0)

# Aggregate edge attributes to nodes
node_features[u] = aggregate([edge_attrs for (u,v) in edges])
```

### Evaluation

- **Labels:** Subreddit metadata (proxy labels)
- **Primary metric:** Modularity Q (structure-based)
- **Secondary:** Accuracy, F1, NMI, ARI (with metadata)

---

## Dataset 3: DBLP Collaboration Network

### Description

- **Source:** Stanford SNAP
- **URL:** https://snap.stanford.edu/data/com-DBLP.html
- **Nodes:** 317,080 authors
- **Edges:** 1,049,866 collaborations
- **Temporal range:** 1990–2020

### Snapshot Construction

1. Divide into 10 non-overlapping 3-year windows
2. Edge (u,v) appears in snapshot t if earliest publication falls in that window
3. Venue metadata provides proxy community labels

**Time windows:**
```
Snapshot 1: 1990-1992
Snapshot 2: 1993-1995
...
Snapshot 10: 2018-2020
```

### Evaluation

- **Labels:** Venue-based proxy labels
- **Metrics:** All 5 metrics

---

## CollegeMsg Surrogate

### Description

Synthetic temporal benchmark mimicking messaging network dynamics.

### Configuration

- **Nodes:** 200
- **Communities:** 5 (planted partition)
- **Snapshots:** 10
- **p_in:** 0.12 (intra-community edge probability)
- **p_out:** 0.018 (inter-community edge probability)
- **Edge rewiring:** 12% per snapshot

### Generation Protocol

```python
# Initial planted partition
G_0 = planted_partition(n=200, k=5, p_in=0.12, p_out=0.018)

# Evolve with rewiring
for t in range(1, 10):
    G_t = rewire_edges(G_{t-1}, fraction=0.12)
    
    # Add temporal noise to features
    X_t = community_mean_features + noise(sigma=0.40)
    
    snapshots.append((X_t, A_t))
```

### Purpose

- Tests temporal robustness beyond the 3 main datasets
- Known ground truth for controlled comparison
- Balances drift (12% rewiring) with stability

---

## Data Availability

### Official Sources

- **LFR / Karate / Dolphins:** IEEE DataPort
- **Reddit:** [SNAP Stanford](https://snap.stanford.edu/data/)
- **DBLP:** [SNAP Stanford](https://snap.stanford.edu/data/)

### Preprocessing Scripts

All snapshot construction code is in `data/`:
- `dynamic_graph_construction.py` — General snapshot builder
- `sample_dynamic_edges.csv` — Demo input format

---

## Data Format

### Expected CSV format for custom data:

```csv
source,target,weight,timestamp
nodeA,nodeB,0.8,2024-01-01
nodeA,nodeC,0.6,2024-01-01
...
```

### Generated snapshot format:

```csv
source,target,weight,timestamp
nodeA,nodeB,0.8,2024-01-01
```

One CSV file per snapshot: `snapshot_2024-01-01.csv`

---

## Citation

If you use these datasets, please cite the original sources:

**Reddit Hyperlink Network:**
```bibtex
@inproceedings{kumar2018community,
  title={Community Interaction and Conflict on the Web},
  author={Kumar, Srijan and Hamilton, William L and Leskovec, Jure and Jurafsky, Dan},
  booktitle={WWW},
  year={2018}
}
```

**DBLP:**
```bibtex
@inproceedings{yang2015defining,
  title={Defining and evaluating network communities based on ground-truth},
  author={Yang, Jaewon and Leskovec, Jure},
  booktitle={ICDM},
  year={2015}
}
```
