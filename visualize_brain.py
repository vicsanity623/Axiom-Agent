"""
Axiom Brain Visualizer (Advanced)
----------------------------------

Visualizes the Axiom agent's knowledge graph with advanced features.
This script is designed to be run from the project root.

Features:
- Correctly locates the brain file using the central config.
- Renders a directed graph to show relationship flow.
- Scales node size based on connectivity.
- Scales edge width based on confidence score.
- Styles negated relationships differently (dashed, red).
- Provides detailed tooltips for both nodes and edges with all properties.
- Includes interactive dropdowns to filter nodes by type.
"""

import json
from pathlib import Path

import networkx as nx
from pyvis.network import Network

try:
    from axiom.config import DEFAULT_BRAIN_FILE
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent
    DEFAULT_BRAIN_FILE = (
        PROJECT_ROOT / "src" / "axiom" / "brain" / "my_agent_brain.json"
    )


def visualize_brain():
    """
    Loads the agent's brain and generates an interactive HTML visualization.
    """
    print(f"🔍 Searching for brain file at:\n   {DEFAULT_BRAIN_FILE}")

    if not DEFAULT_BRAIN_FILE.exists():
        raise FileNotFoundError(
            f"❌ Could not find brain file at:\n   {DEFAULT_BRAIN_FILE}\n"
            f"Please run the agent first to generate a brain file.",
        )

    # ---------------------------------------------------------------------
    # Load brain JSON from the standard networkx node_link_data format
    # ---------------------------------------------------------------------
    with open(DEFAULT_BRAIN_FILE, encoding="utf-8") as f:
        brain_data = json.load(f)

    nodes = brain_data.get("nodes", [])
    edges = brain_data.get("links", [])

    print(f"🧠 Loaded brain file successfully ({len(nodes)} nodes)")
    print(f"🔗 Found {len(edges)} edges")

    if not nodes:
        raise ValueError("⚠️ No nodes found in brain JSON — nothing to visualize!")

    # ---------------------------------------------------------------------
    # Build a directed graph to accurately represent the brain
    # ---------------------------------------------------------------------
    g = nx.MultiDiGraph()

    for node in nodes:
        g.add_node(
            node["id"],
            label=node.get("name", node["id"]),
            node_type=node.get("type", "unknown"),
            properties=node.get("properties", {}),
        )

    for edge in edges:
        g.add_edge(
            edge["source"],
            edge["target"],
            label=edge.get("type", "related_to"),
            weight=edge.get("weight", 0.5),
            properties=edge.get("properties", {}),
        )

    # ---------------------------------------------------------------------
    # Create the Pyvis visualization network
    # ---------------------------------------------------------------------
    vis = Network(
        height="100vh",
        width="100%",
        bgcolor="#0d1117",
        font_color="#ffffff",
        notebook=False,
        directed=True,
    )

    options = {
        "physics": {
            "barnesHut": {
                "gravity": -40000,
                "centralGravity": 0.4,
                "springLength": 200,
                "springConstant": 0.02,
            },
            "maxVelocity": 50,
            "minVelocity": 0.75,
            "stabilization": {
                "enabled": True,
                "iterations": 1000,
                "fit": True,
            },
        },
        "interaction": {
            "tooltipDelay": 200,
            "hideEdgesOnDrag": True,
        },
        "configure": {
            "enabled": True,
            "filter": "nodes",
        },
    }

    vis.set_options(json.dumps(options))

    color_map = {
        "noun": "#ffcc00",
        "noun_phrase": "#ffaa00",
        "verb": "#00d8ff",
        "proper_noun": "#90ee90",
        "concept": "#ff66cc",
        "unknown": "#888888",
    }

    degrees = dict(g.degree())
    min_size, max_size = 15, 40
    min_degree = min(degrees.values()) if degrees else 0
    max_degree = max(degrees.values()) if degrees else 0

    def scale_node_size(degree):
        if max_degree == min_degree:
            return (min_size + max_size) / 2
        return min_size + (degree - min_degree) / (max_degree - min_degree) * (
            max_size - min_size
        )

    for node_id, data in g.nodes(data=True):
        node_type = data.get("node_type", "unknown")
        degree = degrees.get(node_id, 1)

        props_str = json.dumps(data.get("properties", {}), indent=2)
        title_html = (
            f"<b>ID:</b> {node_id}<br>"
            f"<b>Name:</b> {data['label']}<br>"
            f"<b>Type:</b> {node_type}<br>"
            f"<b>Connections:</b> {degree}<br>"
            f"<hr><b>Properties:</b><br><pre>{props_str}</pre>"
        )

        vis.add_node(
            node_id,
            label=data["label"],
            color=color_map.get(node_type, "#888"),
            title=title_html,
            size=scale_node_size(degree),
            group=node_type,
        )

    for u, v, data in g.edges(data=True):
        props = data.get("properties", {})
        confidence = float(props.get("confidence", data.get("weight", 0.5)))
        is_negated = props.get("negated", False)

        edge_width = 1 + (confidence * 4)
        edge_color = "#e06c75" if is_negated else "#555555"
        dashes = True if is_negated else False

        props_str = json.dumps(props, indent=2)
        title_html = (
            f"<b>Type:</b> {data['label']}<br>"
            f"<b>Confidence:</b> {confidence:.2f}<br>"
            f"<hr><b>Properties:</b><br><pre>{props_str}</pre>"
        )

        vis.add_edge(
            u,
            v,
            title=title_html,
            label=data["label"],
            width=edge_width,
            color=edge_color,
            dashes=dashes,
        )

    # ---------------------------------------------------------------------
    # Export visualization to HTML file
    # ---------------------------------------------------------------------
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "axiom_brain_network.html"

    vis.write_html(str(output_path))

    print("✅ Visualization generated successfully!")
    print(f"📁 File saved at:\n   {output_path.resolve()}")
    print("🌐 Open it in your browser to explore interactively.")


if __name__ == "__main__":
    visualize_brain()
