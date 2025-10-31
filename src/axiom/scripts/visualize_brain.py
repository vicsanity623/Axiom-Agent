import json
import logging
from pathlib import Path

import networkx as nx
from pyvis.network import Network

from axiom.config import DEFAULT_BRAIN_FILE

logger = logging.getLogger(__name__)


def visualize_brain():
    """
    Loads the agent's brain and generates a high-performance, interactive
    HTML visualization of its core concepts.
    """
    core_node_threshold = 750
    brain_file_path = DEFAULT_BRAIN_FILE

    print(f"🔍 Searching for brain file at:\n   {brain_file_path}")

    if not brain_file_path.exists():
        raise FileNotFoundError(
            f"❌ Could not find brain file at:\n   {brain_file_path}\n"
            f"Please run the agent first to generate a brain file.",
        )

    with open(brain_file_path, encoding="utf-8") as f:
        brain_data = json.load(f)

    nodes = brain_data.get("nodes", [])
    edges = brain_data.get("links", [])

    print(f"🧠 Loaded brain file successfully ({len(nodes)} nodes, {len(edges)} edges)")

    if not nodes:
        raise ValueError("⚠️ No nodes found in brain JSON — nothing to visualize!")

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

    if len(g.nodes) > core_node_threshold:
        print(
            f"⚠️ Graph is large ({len(g.nodes)} nodes). Culling to the {core_node_threshold} most connected nodes for performance."
        )
        degrees = dict(g.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda item: item[1], reverse=True)
        core_node_ids = {node_id for node_id, _ in sorted_nodes[:core_node_threshold]}

        core_g = nx.MultiDiGraph()
        for node_id in core_node_ids:
            core_g.add_node(node_id, **g.nodes[node_id])

        for u, v, data in g.edges(data=True):
            if u in core_node_ids and v in core_node_ids:
                core_g.add_edge(u, v, **data)

        g = core_g
        print(f"✨ Culled graph now has {len(g.nodes)} nodes and {len(g.edges)} edges.")

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
            "enabled": False,
            "barnesHut": {
                "gravity": -40000,
                "centralGravity": 0.4,
                "springLength": 200,
                "springConstant": 0.02,
            },
            "maxVelocity": 50,
            "minVelocity": 0.75,
            "stabilization": {"enabled": True, "iterations": 1000, "fit": True},
        },
        "interaction": {"tooltipDelay": 200, "hideEdgesOnDrag": True},
        "configure": {"enabled": False},
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
            f"<b>ID:</b> {node_id}<br><b>Name:</b> {data['label']}<br><b>Type:</b> {node_type}<br>"
            f"<b>Connections:</b> {degree}<br><hr><b>Properties:</b><br><pre>{props_str}</pre>"
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
            f"<b>Type:</b> {data['label']}<br><b>Confidence:</b> {confidence:.2f}<br>"
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

    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "axiom_brain_network.html"
    vis.write_html(str(output_path))

    html = output_path.read_text(encoding="utf-8")
    injection = """
    <style>
    #physicsToggle {
        position: absolute; top: 15px; left: 15px; z-index: 1000;
        background: #161b22; color: #c9d1d9; padding: 8px 12px;
        border: 1px solid #30363d; border-radius: 6px; cursor: pointer;
        font-size: 14px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }
    #physicsToggle:hover { background: #21262d; }
    </style>
    <button id="physicsToggle">Enable Physics</button>
    <script type="text/javascript">
        document.addEventListener("DOMContentLoaded", function() {
            const btn = document.getElementById('physicsToggle');
            let physicsEnabled = false;
            btn.onclick = function() {
                physicsEnabled = !physicsEnabled;
                network.setOptions({ physics: { enabled: physicsEnabled } });
                this.innerText = physicsEnabled ? 'Disable Physics' : 'Enable Physics';
            };
        });
    </script>
    """
    html = html.replace("</body>", injection + "\n</body>")
    output_path.write_text(html, encoding="utf-8")

    print("✅ High-performance visualization generated successfully!")
    print(f"📁 File saved at:\n   {output_path.resolve()}")
    print("🌐 Open it in your browser to explore interactively.")


def main():
    """Entry point for the axiom-visualize command."""
    try:
        visualize_brain()
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))


if __name__ == "__main__":
    main()
