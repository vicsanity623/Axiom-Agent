import json
import logging
from pathlib import Path
import networkx as nx

logger = logging.getLogger(__name__)
CORE_NODE_THRESHOLD = 100000

brain_file_path: Path
try:
    from axiom.config import DEFAULT_BRAIN_FILE
    brain_file_path = DEFAULT_BRAIN_FILE
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    brain_file_path = PROJECT_ROOT / "src" / "axiom" / "brain" / "my_agent_brain.json"


def visualize_brain_galaxy():
    if not brain_file_path.exists():
        raise FileNotFoundError(f"Brain file not found at {brain_file_path}")

    with open(brain_file_path, encoding="utf-8") as f:
        brain_data = json.load(f)

    nodes = brain_data.get("nodes", [])
    edges = brain_data.get("links", [])
    logger.info("Loaded %d nodes, %d edges", len(nodes), len(edges))

    g = nx.MultiDiGraph()
    for node in nodes:
        g.add_node(node["id"], **node)
    for edge in edges:
        g.add_edge(edge["source"], edge["target"], **edge)

    if len(g.nodes) > CORE_NODE_THRESHOLD:
        degrees = dict(g.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:CORE_NODE_THRESHOLD]
        g = g.subgraph(top_nodes).copy()
        logger.info("Culled to %d nodes for performance", len(g.nodes))

    elements = []
    for node_id, data in g.nodes(data=True):
        elements.append({
            "data": {
                "id": node_id,
                "label": data.get("name", node_id),
                "type": data.get("type", "unknown"),
            }
        })
    for u, v, data in g.edges(data=True):
        props = data.get("properties", {})
        confidence = float(props.get("confidence", data.get("weight", 0.5)))
        neg = props.get("negated", False)
        elements.append({
            "data": {
                "source": u,
                "target": v,
                "label": data.get("type", "related_to"),
                "confidence": confidence,
                "negated": neg,
            }
        })

    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "axiom_brain_galaxy.html"

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Axiom Brain Galaxy</title>
  <script src="https://unpkg.com/cytoscape@3.27.0/dist/cytoscape.min.js"></script>
  <script src="https://unpkg.com/cytoscape-cose-bilkent@4.1.0/cytoscape-cose-bilkent.js"></script>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no" />
  <style>
    html, body {{
      margin: 0; padding: 0;
      width: 100%; height: 100%;
      overflow: hidden;
      background: radial-gradient(circle at 50% 50%, #12002b 0%, #0a0020 30%, #000010 60%, #000000 100%),
                  radial-gradient(circle at 70% 30%, rgba(255,100,200,0.15), transparent 60%),
                  radial-gradient(circle at 30% 70%, rgba(0,150,255,0.15), transparent 60%);
      background-blend-mode: screen;
      color: #fff;
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    #cy {{
      width: 100vw; height: 100vh; display: block;
      touch-action: none;
    }}
    .tooltip {{
      position: absolute; background: rgba(22,27,34,0.9); padding: 6px 8px;
      border-radius: 8px; border: 1px solid #30363d;
      color: #c9d1d9; font-size: 12px;
      display: none; z-index: 1000;
    }}
    #physicsToggle {{
      position: fixed; top: 15px; left: 15px; z-index: 2000;
      background: rgba(22,27,34,0.9);
      color: #c9d1d9; padding: 8px 14px;
      border: 1px solid #30363d; border-radius: 8px;
      cursor: pointer; font-size: 14px;
      -webkit-tap-highlight-color: transparent;
    }}
    #physicsToggle:active {{ background: #21262d; }}
  </style>
</head>
<body>
  <button id="physicsToggle">Enable Galactic Physics</button>
  <div id="cy"></div>
  <div class="tooltip" id="tooltip"></div>

  <script>
    const IS_MOBILE = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    const rawElements = {json.dumps(elements)};
    const batchSize = IS_MOBILE ? 500 : 2000;
    const galaxyScale = IS_MOBILE ? 25000 : 50000;
    let batchIndex = 0;

    // Assign star colors
    const hues = ['#66ccff', '#ff99cc', '#b388ff', '#88ffcc', '#ffd966'];
    rawElements.forEach(el => {{
      if (el.data && !el.data.color) {{
        el.data.color = hues[Math.floor(Math.random() * hues.length)];
      }}
    }});

    const cy = cytoscape({{
      container: document.getElementById('cy'),
      wheelSensitivity: 0.15,
      minZoom: 0.02,
      maxZoom: 4,
      style: [
        {{ selector: 'node', style: {{
          'background-color': 'data(color)',
          'label': 'data(label)',
          'font-size': IS_MOBILE ? 5 : 7,
          'color': '#fff',
          'opacity': 0,
          'width': IS_MOBILE ? 12 : 10,
          'height': IS_MOBILE ? 12 : 10,
          'shadow-blur': 30,
          'shadow-color': 'data(color)',
          'shadow-opacity': 0.9,
          'text-outline-color': '#000',
          'text-outline-width': 1
        }} }},
        {{ selector: 'edge', style: {{
          'width': IS_MOBILE ? 1.2 : 0.6,
          'opacity': 0.25,
          'line-color': '#b088ff',
          'curve-style': 'bezier'
        }} }},
        {{ selector: 'edge[negated=true]', style: {{
          'line-color': '#ff6666',
          'line-style': 'dashed',
          'opacity': 0.35
        }} }}
      ],
      layout: {{ name: 'preset' }}
    }});

    // Tooltip
    const tooltip = document.getElementById('tooltip');
    cy.on('tap', 'node', e => {{
      const n = e.target;
      tooltip.style.display = 'block';
      tooltip.innerHTML = `<b>${{n.data('label')}}</b><br><small>${{n.data('type')}}</small>`;
      tooltip.style.left = (e.originalEvent.touches?.[0]?.clientX || e.originalEvent.clientX) + 'px';
      tooltip.style.top = (e.originalEvent.touches?.[0]?.clientY || e.originalEvent.clientY) + 'px';
    }});
    cy.on('tap', e => {{
      if (e.target === cy) tooltip.style.display = 'none';
    }});

    // Progressive loading
    function loadBatch() {{
      if (batchIndex * batchSize >= rawElements.length) return;
      const batch = rawElements.slice(batchIndex * batchSize, (batchIndex + 1) * batchSize);
      cy.add(batch);
      batchIndex++;

      cy.nodes().positions(n => {{
        return {{
          x: (Math.random() - 0.5) * galaxyScale,
          y: (Math.random() - 0.5) * galaxyScale
        }};
      }});

      cy.nodes().forEach(n => {{
        n.animate({{
          style: {{ opacity: 1 }},
          duration: 700 + Math.random() * 400,
          easing: 'ease-out'
        }});
      }});

      setTimeout(loadBatch, IS_MOBILE ? 1500 : 800);
    }}
    loadBatch();

    // Physics toggle
    let physicsEnabled = false;
    const layoutOpts = {{
      name: 'cose-bilkent',
      animate: true,
      randomize: true,
      fit: false,
      nodeRepulsion: IS_MOBILE ? 100000 : 200000,
      idealEdgeLength: IS_MOBILE ? 250 : 500,
      gravity: IS_MOBILE ? 0.2 : 0.05,
      numIter: 100
    }};

    document.getElementById('physicsToggle').onclick = () => {{
      physicsEnabled = !physicsEnabled;
      if (physicsEnabled) {{
        cy.layout(layoutOpts).run();
      }} else {{
        cy.layout({{ name: 'preset' }}).run();
      }}
      document.getElementById('physicsToggle').innerText =
        physicsEnabled ? 'Disable Physics' : 'Enable Galactic Physics';
    }};
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    logger.info("🌌 Galaxy visualization saved at %s", output_path.resolve())


def main():
    logging.basicConfig(level=logging.INFO)
    visualize_brain_galaxy()


if __name__ == "__main__":
    main()
