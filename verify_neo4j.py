import uuid

from axiom.graph_core import ConceptGraph, ConceptNode


def verify():
    print("Initializing ConceptGraph...")
    graph = ConceptGraph()
    if not graph.driver:
        print("Failed to connect to Neo4j.")
        return

    print("Connected to Neo4j.")

    # Create a test node
    node_id = str(uuid.uuid4())
    node = ConceptNode(name="Verification Node", node_type="test", id_=node_id)
    print(f"Adding node: {node.name} ({node.id})")
    graph.add_node(node)

    # Retrieve it
    print("Retrieving node...")
    retrieved = graph.get_node_by_id(node.id)
    if retrieved:
        print(f"Retrieved node: {retrieved.name} ({retrieved.id})")
        if retrieved.name == "verification node":  # ConceptNode lowercases names
            print("SUCCESS: Node retrieved correctly.")
        else:
            print(
                f"WARNING: Node name mismatch. Expected 'verification node', got '{retrieved.name}'"
            )
    else:
        print("FAILURE: Node not found.")

    # Test Edge
    node2_id = str(uuid.uuid4())
    node2 = ConceptNode(name="Verification Node 2", node_type="test", id_=node2_id)
    graph.add_node(node2)

    print("Adding edge...")
    edge = graph.add_edge(
        node, node2, "connects_to", weight=0.8, properties={"test": True}
    )
    if edge:
        print(f"Added edge: {edge.source} -> {edge.target} ({edge.type})")
    else:
        print("FAILURE: Failed to add edge.")

    print("Retrieving edges...")
    edges = graph.get_edges_from_node(node.id)
    if len(edges) > 0:
        print(f"Found {len(edges)} edges.")
        found = False
        for e in edges:
            if e.target == node2.id and e.properties.get("test"):
                found = True
                break
        if found:
            print("SUCCESS: Edge retrieved correctly.")
        else:
            print("WARNING: Edge mismatch or property missing.")
    else:
        print("FAILURE: No edges found.")

    # Clean up
    print("Cleaning up...")
    with graph.driver.session() as session:
        session.run("MATCH (n:Concept {id: $id}) DETACH DELETE n", id=node.id)
        session.run("MATCH (n:Concept {id: $id}) DETACH DELETE n", id=node2.id)
    print("Cleanup complete.")
    graph.close()


if __name__ == "__main__":
    verify()
