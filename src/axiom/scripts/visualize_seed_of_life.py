import json
import logging
import math
from pathlib import Path

import networkx as nx

from axiom.config import DEFAULT_BRAIN_FILE

logger = logging.getLogger(__name__)
CORE_NODE_THRESHOLD = 100000


def seed_of_life_positions(total_nodes: int, radius_step: float = 50):
    """Generate positions with hexagonal 'Seed of Life' symmetry (denser)."""
    positions = {}
    idx = 0
    positions[idx] = {"x": 0, "y": 0}
    idx += 1

    layer = 1
    while idx < total_nodes:
        num_points = 6 * layer
        radius = radius_step * layer
        for i in range(num_points):
            if idx >= total_nodes:
                break
            angle = (2 * math.pi / num_points) * i
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions[idx] = {"x": int(x), "y": int(y)}
            idx += 1
        layer += 1
    return positions


def visualize_seed_of_life():
    brain_file_path = DEFAULT_BRAIN_FILE

    if not brain_file_path.exists():
        raise FileNotFoundError(f"Brain file not found at {brain_file_path}")

    with open(brain_file_path, encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("links", [])

    if not nodes:
        raise ValueError("No nodes found in brain file — did you run axiom-train?")

    logger.info("Loaded %d nodes, %d edges", len(nodes), len(edges))

    g = nx.MultiDiGraph()
    for n in nodes:
        g.add_node(n["id"], **n)
    for e in edges:
        g.add_edge(e["source"], e["target"], **e)

    if len(g.nodes) > CORE_NODE_THRESHOLD:
        degrees = dict(g.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:CORE_NODE_THRESHOLD]
        g = g.subgraph(top_nodes).copy()
        logger.info("Trimmed to %d nodes for performance", len(g.nodes))

    positions = seed_of_life_positions(len(g.nodes), radius_step=160)

    elements = []
    for i, (node_id, data) in enumerate(g.nodes(data=True)):
        pos = positions.get(i, {"x": 0, "y": 0})
        elements.append(
            {
                "data": {"id": node_id, "label": data.get("name", node_id)},
                "position": pos,
            }
        )

    for u, v, data in g.edges(data=True):
        elements.append(
            {"data": {"source": u, "target": v, "label": data.get("type", "")}}
        )

    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "axiom_seed_of_life.html"

    html = BASE_HTML.format(elements=json.dumps(elements))
    output_path.write_text(html, encoding="utf-8")
    print(f"✅ Visualization saved: {output_path.resolve()}")


BASE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Axiom Seed of Life (Dense)</title>
<script src="https://unpkg.com/cytoscape@3.27.0/dist/cytoscape.min.js"></script>
<style>
html, body {{
  margin: 0; padding: 0; width: 100%; height: 100%;
  background: radial-gradient(circle at 50% 50%, #0c0015, #04000a 80%, #000008 100%);
  overflow: hidden;
}}
#cy {{ width: 100%; height: 100%; }}
.label-float {{
  position: absolute;
  color: #fff;
  background: rgba(0,0,0,0.7);
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 14px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  pointer-events: none;
  z-index: 10;
  display: none;
  max-width: 250px;
  word-wrap: break-word;
}}
</style>
</head>
<body>
<div id="cy"></div>
<div id="floatLabel" class="label-float"></div>
<script>
const cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: {elements},
  style: [
    {{ selector: 'node', style: {{
      'background-color': '#7fffd4', 'width': 8, 'height': 8, 'opacity': 0.9
    }} }},
    {{ selector: 'edge', style: {{
      'width': 0.4, 'opacity': 0.2, 'line-style': 'dashed', 'line-color': '#ff69b4'
    }} }}
  ],
  layout: {{ name: 'preset' }},
  minZoom: 0.02, maxZoom: 4,
  // Enable touch gestures for panning and zooming
  zoomingEnabled: true,
  userZoomingEnabled: true,
  panningEnabled: true,
  userPanningEnabled: true,
  boxSelectionEnabled: false,
}});

const floatLabel = document.getElementById('floatLabel');

// Handle tap on nodes
cy.on('tap', 'node', evt => {{
  const node = evt.target;
  const pos = node.renderedPosition();
  floatLabel.innerText = node.data('label');
  floatLabel.style.left = (pos.x + 15) + 'px';
  floatLabel.style.top = (pos.y - 15) + 'px';
  floatLabel.style.display = 'block';
}});

// Handle tap on edges (for mobile)
cy.on('tap', 'edge', evt => {{
    const edge = evt.target;
    const pos = evt.renderedPosition; // Use event position for tap
    floatLabel.innerText = edge.data('label') || `${{edge.source().data('label')}} → ${{edge.target().data('label')}}`;
    floatLabel.style.left = (pos.x + 15) + 'px';
    floatLabel.style.top = (pos.y - 15) + 'px';
    floatLabel.style.display = 'block';
}});

// Hide label when tapping the background
cy.on('tap', evt => {{
  if (evt.target === cy) floatLabel.style.display = 'none';
}});

// Keep mouseover for desktop, but use standard event properties
cy.on('mouseover', 'edge', evt => {{
  const edge = evt.target;
  const pos = evt.renderedPosition;
  floatLabel.innerText = edge.data('label') || `${{edge.source().data('label')}} → ${{edge.target().data('label')}}`;
  floatLabel.style.left = (pos.x + 15) + 'px';
  floatLabel.style.top = (pos.y - 15) + 'px';
  floatLabel.style.display = 'block';
}});
cy.on('mouseout', 'edge', () => floatLabel.style.display = 'none');
</script>
</body>
</html>"""


def main():
    """Entry point for the script, handling setup and errors."""

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        visualize_seed_of_life()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to generate visualization: {e}")
        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()
