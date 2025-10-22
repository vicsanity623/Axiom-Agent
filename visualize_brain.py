#!/usr/bin/env python3
"""
Axiom Brain Visualizer (Optimized for Large Brains with Scaled Nodes)
----------------------------------------------------------------------

Visualizes the Axiom agent's ontology graph interactively.
Automatically finds src/axiom/brain/my_agent_brain.json
and exports to visualizations/axiom_brain_network.html
Node sizes scale with connectivity for better readability in large brains.
Just a cool interactive view of the brain.
"""

import json
from pathlib import Path

import networkx as nx
from pyvis.network import Network

# ---------------------------------------------------------------------
# Locate project root and brain file
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
BRAIN_PATH = PROJECT_ROOT / "src" / "axiom" / "brain" / "my_agent_brain.json"

print(f"🔍 Searching for brain file at:\n   {BRAIN_PATH}")

if not BRAIN_PATH.exists():
    raise FileNotFoundError(
        f"❌ Could not find brain file at:\n   {BRAIN_PATH}\n"
        f"Make sure it's saved as 'my_agent_brain.json'.",
    )

# ---------------------------------------------------------------------
# Load brain JSON
# ---------------------------------------------------------------------
with open(BRAIN_PATH, encoding="utf-8") as f:
    brain_data = json.load(f)

nodes = brain_data.get("nodes", [])
edges = brain_data.get("links", []) + brain_data.get("edges", [])

print(f"🧠 Loaded brain file successfully ({len(nodes)} nodes)")
print(f"🔗 Found {len(edges)} edges")

# ---------------------------------------------------------------------
# Build simple undirected graph (faster for large networks)
# ---------------------------------------------------------------------
G = nx.Graph()

# Add nodes
for node in nodes:
    G.add_node(
        node["id"],
        label=node["name"],
        type=node.get("type", "unknown"),
        value=node.get("value", 0.5),
    )

# Add edges (avoid duplicates)
seen_edges = set()
for edge in edges:
    src = edge.get("source")
    dst = edge.get("target")
    if not src or not dst:
        continue
    edge_key = tuple(sorted((src, dst)))  # undirected: order doesn’t matter
    if edge_key in seen_edges:
        continue
    seen_edges.add(edge_key)
    relation = edge.get("relation") or edge.get("name", "related_to")
    G.add_edge(src, dst, label=relation)

# ---------------------------------------------------------------------
# Create visualization
# ---------------------------------------------------------------------
if G.number_of_nodes() == 0:
    raise ValueError("⚠️ No nodes found in brain JSON — nothing to visualize!")

vis = Network(
    height="100%",
    width="100%",
    bgcolor="#0d1117",
    font_color="#ffffff",
    notebook=False,
    directed=False,  # undirected graph
)

vis.barnes_hut(
    gravity=-25000,
    central_gravity=0.3,
    spring_length=180,
    spring_strength=0.015,
)

# Node color mapping
color_map = {
    "noun": "#ffcc00",
    "noun_phrase": "#ffaa00",
    "verb": "#00d8ff",
    "proper_noun": "#90ee90",
    "concept": "#ff66cc",
}

# Compute node sizes based on degree
degrees = dict(G.degree())
min_size, max_size = 15, 35  # min and max node sizes
if degrees:
    min_degree = min(degrees.values())
    max_degree = max(degrees.values())
else:
    min_degree = max_degree = 0


def scale_size(degree):
    if max_degree == min_degree:
        return (min_size + max_size) / 2
    return min_size + (degree - min_degree) / (max_degree - min_degree) * (
        max_size - min_size
    )


# Add nodes to visualization
for node_id, data in G.nodes(data=True):
    node_type = data.get("type", "unknown")
    degree = degrees.get(node_id, 1)
    vis.add_node(
        node_id,
        label=data["label"],
        color=color_map.get(node_type, "#888"),
        title=f"<b>{data['label']}</b><br>Type: {node_type}<br>Value: {data.get('value', 0.5)}<br>Connections: {degree}",
        size=scale_size(degree),
    )

# Add edges to visualization
for src, dst, data in G.edges(data=True):
    vis.add_edge(src, dst, label=data.get("label", "related_to"), color="#888888")

# ---------------------------------------------------------------------
# Export visualization
# ---------------------------------------------------------------------
output_dir = PROJECT_ROOT / "visualizations"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "axiom_brain_network.html"

vis.write_html(str(output_path))

print("✅ Visualization generated successfully!")
print(f"📁 File saved at:\n   {output_path}")
print("🌐 Open it in your browser to explore interactively.")
