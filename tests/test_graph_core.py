from axiom.graph_core import ConceptGraph, ConceptNode, RelationshipEdge


def test_concept_node_creation():
    node = ConceptNode(name="Test Node", node_type="concept", value=0.8)
    assert node.name == "test node"
    assert node.type == "concept"
    assert node.value == 0.8
    assert node.id is not None


def test_concept_node_serialization():
    node = ConceptNode(name="Test Node")
    data = node.to_dict()
    assert data["name"] == "test node"

    node2 = ConceptNode.from_dict(data)
    assert node2.name == node.name
    assert node2.id == node.id


def test_relationship_edge_creation():
    edge = RelationshipEdge(source="id1", target="id2", type="related_to", weight=0.5)
    assert edge.source == "id1"
    assert edge.target == "id2"
    assert edge.type == "related_to"
    assert edge.weight == 0.5


def test_graph_add_node():
    graph = ConceptGraph()
    node = ConceptNode(name="Node A")
    graph.add_node(node)

    retrieved = graph.get_node_by_name("Node A")
    assert retrieved is not None
    assert retrieved.id == node.id

    retrieved_id = graph.get_node_by_id(node.id)
    assert retrieved_id is not None
    assert retrieved_id.name == "node a"


def test_graph_add_edge():
    graph = ConceptGraph()
    node_a = ConceptNode(name="Node A")
    node_b = ConceptNode(name="Node B")

    edge = graph.add_edge(node_a, node_b, "connects_to")
    assert edge is not None

    edges = graph.get_edges_from_node(node_a.id)
    assert len(edges) == 1
    assert edges[0].target == node_b.id


def test_graph_persistence(tmp_path):
    graph = ConceptGraph()
    node = ConceptNode(name="Persistent Node")
    graph.add_node(node)

    save_path = tmp_path / "brain.json"
    graph.save_to_file(save_path)

    loaded_graph = ConceptGraph.load_from_file(save_path)
    assert loaded_graph.get_node_by_name("Persistent Node") is not None


def test_conflicting_edge():
    graph = ConceptGraph()
    node_a = ConceptNode(name="A")
    node_b = ConceptNode(name="B")

    # A -> B (is_a)
    edge1 = graph.add_edge(node_a, node_b, "is_a", weight=0.5)

    # Check conflict
    conflict = graph.find_exclusive_conflict(node_a, "is_a")
    assert conflict is not None
    assert conflict.id == edge1.id

    # Revise conflict
    # New edge A -> C (is_a) with higher confidence
    # edge2 = RelationshipEdge(node_a.id, node_c.id, "is_a", weight=0.9)
    action = graph.revise_conflicting_edge(edge1, 0.9, new_provenance="system")

    assert action == "replaced"
    # Edge1 should be superseded or updated?
    # The logic in revise_conflicting_edge updates existing_edge if new rank > old rank.
    # Wait, revise_conflicting_edge updates the *existing* edge's weight if new is better.
    # It doesn't automatically replace the target unless we handle that logic outside.
    # Let's check what revise_conflicting_edge actually does.
    # It calls update_edge_weight on existing_edge.

    assert edge1.properties["revision_status"] == "replaced"
