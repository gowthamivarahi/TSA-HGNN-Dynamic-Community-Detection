import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# -------------------------------
# Load CSV
# -------------------------------
df = pd.read_csv("dynamic_edges.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Output folder
os.makedirs("output_snapshots", exist_ok=True)

snapshots = {}

# -------------------------------
# Create snapshots
# -------------------------------
for time, group in df.groupby('timestamp'):
    G = nx.Graph()
    for _, row in group.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
    snapshots[time] = G

# -------------------------------
# Display Graph & Save CSV
# -------------------------------
for time, G in snapshots.items():

    print(f"\nSnapshot @ {time}")

    # ---- Display Graph ----
    plt.figure(figsize=(5, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=1200,
        node_color="lightblue",
        edge_color="gray"
    )

    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title(f"Graph Snapshot @ {time.date()}")
    plt.show()

    # ---- Save Snapshot CSV ----
    edge_list = []
    for u, v, data in G.edges(data=True):
        edge_list.append({
            "source": u,
            "target": v,
            "weight": data.get("weight", 1.0),
            "timestamp": time
        })

    snapshot_df = pd.DataFrame(edge_list)
    file_name = f"output_snapshots/snapshot_{time.date()}.csv"
    snapshot_df.to_csv(file_name, index=False)

    print(f"Saved: {file_name}")