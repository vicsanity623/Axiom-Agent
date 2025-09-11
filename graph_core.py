# graph_core.py

import json
import uuid
import os

class ConceptNode:
    """Represents a single concept or entity in the knowledge graph."""
    def __init__(self, name, node_type="concept", properties=None, value=0.5, activation=0.0, id=None):
        self.id = id or str(uuid.uuid4())
        self.name = name.lower()
        self.type = node_type
        self.properties = properties or {}
        self.value = value
        self.activation = activation

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "type": self.type,
            "properties": self.properties, "value": self.value,
            "activation": self.activation
        }

    @staticmethod
    def from_dict(data):
        return ConceptNode(
            id=data.get("id"), name=data.get("name"), node_type=data.get("type"),
            properties=data.get("properties", {}), value=data.get("value", 0.5),
            activation=data.get("activation", 0.0)
        )

class RelationshipEdge:
    """Represents a directed, weighted relationship between two concepts."""
    def __init__(self, source, target, type, weight=0.5, id=None, properties=None):
        self.id = id or str(uuid.uuid4())
        self.source = source
        self.target = target
        self.type = type
        self.weight = weight
        self.properties = properties or {}

    def to_dict(self):
        return {
            "id": self.id, "source": self.source, "target": self.target,
            "type": self.type, "weight": self.weight,
            "properties": self.properties
        }

    @staticmethod
    def from_dict(data):
        return RelationshipEdge(
            id=data.get("id"), source=data.get("source"),
            target=data.get("target"), type=data.get("type"),
            weight=data.get("weight", 0.5),
            properties=data.get("properties", {})
        )

class ConceptGraph:
    """Manages the collection of nodes and edges that form the knowledge base."""
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.node_by_name = {}
        self.lexical_dictionary = {}

    def add_node(self, node: ConceptNode):
        if node.id not in self.nodes:
            existing_node = self.get_node_by_name(node.name)
            if existing_node:
                return existing_node
            self.nodes[node.id] = node
            self.node_by_name[node.name] = node
        return node

    def get_node_by_name(self, name):
        return self.node_by_name.get(name.lower())
    
    def add_edge(self, source_node, target_node, relation_type, weight=0.5, properties=None):
        """Creates and adds a new edge to the graph."""
        if not all([source_node, target_node]):
            return None
        
        for edge in self.get_edges_from_node(source_node.id):
            if edge.target == target_node.id and edge.type == relation_type:
                edge.weight = max(edge.weight, weight)
                if properties: edge.properties.update(properties)
                return edge

        new_edge = RelationshipEdge(source_node.id, target_node.id, relation_type, weight, properties=properties)
        self.edges[new_edge.id] = new_edge
        return new_edge

    def get_edges_from_node(self, node_id):
        return [edge for edge in self.edges.values() if edge.source == node_id]

    def get_edges_to_node(self, node_id):
        return [edge for edge in self.edges.values() if edge.target == node_id]
        
    def decay_activations(self, decay_rate=0.1):
        for node in self.nodes.values():
            node.activation = max(0.0, node.activation - decay_rate)

    def save_to_file(self, filename):
        graph_data = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()]
        }
        with open(filename, 'w') as f:
            json.dump(graph_data, f, indent=4)
        print(f"Agent brain saved to {filename}")

    @classmethod
    def load_from_file(cls, filename):
        graph = cls()
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    graph_data = json.load(f)
                graph.load_from_dict_data(graph_data) # Use the new helper method
                print(f"Agent brain loaded from {filename}")
            except Exception as e:
                print(f"Error loading brain from {filename}: {e}. Creating a fresh brain.")
        else:
            print(f"No saved brain found at {filename}. Creating a fresh brain.")
        
        graph._seed_universal_concepts_internal()
        return graph

    # --- NEW: The missing `load_from_dict` method ---
    @classmethod
    def load_from_dict(cls, graph_data: dict):
        """Creates a ConceptGraph instance from a dictionary."""
        graph = cls()
        graph.load_from_dict_data(graph_data)
        graph._seed_universal_concepts_internal()
        return graph

    # --- NEW: A helper method to populate the graph from data ---
    def load_from_dict_data(self, graph_data: dict):
        """Populates the current graph instance from a dictionary."""
        for node_data in graph_data.get("nodes", []):
            self.add_node(ConceptNode.from_dict(node_data))
        
        for edge_data in graph_data.get("edges", []):
            edge = RelationshipEdge.from_dict(edge_data)
            self.edges[edge.id] = edge

    def _seed_universal_concepts_internal(self):
        """Seeds essential structural nodes that should always exist."""
        core_concepts = {
            "concept": "meta_category", "action": "meta_category", "attribute": "meta_category",
            "sentiment": "category", "agent": "concept", "user": "concept",
            "positive": "sentiment", "negative": "sentiment", "understand": "action",
            "learn": "action"
        }
        for name, type_ in core_concepts.items():
            if not self.get_node_by_name(name):
                self.add_node(ConceptNode(name, node_type=type_))