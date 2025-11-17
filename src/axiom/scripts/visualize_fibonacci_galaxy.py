import json
import logging
import math
from pathlib import Path

import networkx as nx

from axiom.config import DEFAULT_BRAIN_FILE

logger = logging.getLogger(__name__)
CORE_NODE_THRESHOLD = 100000


def fibonacci_spiral_layout(nodes):
    """Generate (x, y) coordinates following a denser Fibonacci spiral."""
    golden_angle = math.pi * (3 - math.sqrt(5))
    scale = 25
    positions = {}
    for i, node in enumerate(nodes):
        r = scale * math.sqrt(i)
        theta = i * golden_angle
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        positions[node] = (x, y)
    return positions


def visualize_brain_galaxy_fibonacci():
    brain_file_path = DEFAULT_BRAIN_FILE

    with open(brain_file_path, encoding="utf-8") as f:
        brain_data = json.load(f)

    nodes = brain_data.get("nodes", [])
    edges = brain_data.get("links", [])
    g = nx.MultiDiGraph()
    for node in nodes:
        g.add_node(node["id"], **node)
    for edge in edges:
        g.add_edge(edge["source"], edge["target"], **edge)

    if len(g.nodes) > CORE_NODE_THRESHOLD:
        degrees = dict(g.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:CORE_NODE_THRESHOLD]
        g = g.subgraph(top_nodes).copy()

    positions = fibonacci_spiral_layout(list(g.nodes))

    elements = []
    for node_id, data in g.nodes(data=True):
        x, y = positions.get(node_id, (0, 0))
        elements.append(
            {
                "data": {"id": node_id, "label": data.get("name", node_id)},
                "position": {"x": x, "y": y},
            }
        )
    for u, v, data in g.edges(data=True):
        elements.append(
            {"data": {"source": u, "target": v, "label": data.get("type", "")}}
        )

    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "axiom_brain_galaxy_fibonacci.html"
    print(f"✅ Visualization saved: {output_path.resolve()}")

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no" />
<title>Fibonacci Spiral Galaxy (Dense)</title>
<script src="https://unpkg.com/cytoscape@3.27.0/dist/cytoscape.min.js"></script>
<style>
  html, body {{ margin:0; height:100%; background:#000010; overflow:hidden; }}
  #cy {{ width:100vw; height:100vh; }}
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
  elements: {json.dumps(elements)},
  layout: {{ name: 'preset' }},
  style: [
    {{ selector: 'node', style: {{
      'background-color': '#ffd966', 'width': 8, 'height': 8, 'opacity': 0.9
    }} }},
    {{ selector: 'edge', style: {{
      'width': 0.4, 'opacity': 0.05, 'line-style': 'dashed', 'line-color': '#b388ff'
    }} }}
  ],
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
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    logger.info("🌀 Dense Fibonacci spiral galaxy saved at %s", output_path.resolve())


def main():
    """Entry point for the axiom-visualize command."""
    try:
        visualize_brain_galaxy_fibonacci()
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))


if __name__ == "__main__":
    main()
