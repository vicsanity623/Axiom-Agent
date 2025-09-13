# graph_core.py

import json
import uuid
import os
import networkx as nx

class ConceptNode:
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
    def __init__(self, source, target, type, weight=0.5, id=None, properties=None, access_count=0):
        self.id = id or str(uuid.uuid4())
        self.source = source
        self.target = target
        self.type = type
        self.weight = weight
        self.properties = properties or {}
        self.access_count = access_count

    def to_dict(self):
        return {
            "id": self.id, "source": self.source, "target": self.target,
            "type": self.type, "weight": self.weight,
            "properties": self.properties,
            "access_count": self.access_count
        }

    @staticmethod
    def from_dict(data):
        return RelationshipEdge(
            id=data.get("id"), source=data.get("source"),
            target=data.get("target"), type=data.get("type"),
            weight=data.get("weight", 0.5),
            properties=data.get("properties", {}),
            access_count=data.get("access_count", 0)
        )

class ConceptGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.name_to_id = {}

    def add_node(self, node: ConceptNode):
        if self.get_node_by_name(node.name):
            return self.get_node_by_name(node.name)
        
        self.graph.add_node(node.id, **node.to_dict())
        self.name_to_id[node.name] = node.id
        return node

    def get_node_by_name(self, name):
        node_id = self.name_to_id.get(name.lower())
        if node_id and node_id in self.graph:
            return ConceptNode.from_dict(self.graph.nodes[node_id])
        return None
    
    def add_edge(self, source_node, target_node, relation_type, weight=0.5, properties=None):
        if not all([source_node, target_node]):
            return None
        
        if self.graph.has_edge(source_node.id, target_node.id):
            for key, data in self.graph.get_edge_data(source_node.id, target_node.id).items():
                if data.get('type') == relation_type:
                    data['weight'] = max(data['weight'], weight)
                    if properties: data['properties'].update(properties)
                    return RelationshipEdge.from_dict(data)

        new_edge = RelationshipEdge(source_node.id, target_node.id, relation_type, weight, properties=properties)
        self.graph.add_edge(new_edge.source, new_edge.target, key=new_edge.id, **new_edge.to_dict())
        return new_edge

    def get_edges_from_node(self, node_id):
        if node_id not in self.graph: return []
        return [RelationshipEdge.from_dict(data) for _, _, data in self.graph.out_edges(node_id, data=True)]

    def get_edges_to_node(self, node_id):
        if node_id not in self.graph: return []
        return [RelationshipEdge.from_dict(data) for _, _, data in self.graph.in_edges(node_id, data=True)]
        
    def decay_activations(self, decay_rate=0.1):
        for node_id in self.graph.nodes:
            current_activation = self.graph.nodes[node_id].get('activation', 0.0)
            self.graph.nodes[node_id]['activation'] = max(0.0, current_activation - decay_rate)

    def save_to_file(self, filename):
        graph_data = nx.readwrite.json_graph.node_link_data(self.graph)
        with open(filename, 'w') as f:
            json.dump(graph_data, f, indent=4)
        print(f"Agent brain saved to {filename}")

    @classmethod
    def load_from_file(cls, filename):
        graph_instance = cls()
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    graph_data = json.load(f)

                if "nodes" in graph_data and ("links" in graph_data or "edges" in graph_data):
                    if "edges" in graph_data:
                        graph_data["links"] = graph_data.pop("edges")
                    graph_instance.graph = nx.readwrite.json_graph.node_link_graph(graph_data)
                elif "nodes" in graph_data and "edges" not in graph_data: # Old custom format
                    graph_instance.load_from_old_format(graph_data)
                
                graph_instance.name_to_id = {
                    data['name']: node_id for node_id, data in graph_instance.graph.nodes(data=True) if 'name' in data
                }
                print(f"Agent brain loaded from {filename}")

            except Exception as e:
                print(f"Error loading brain from {filename}: {e}. Creating a fresh brain.")
        else:
            print(f"No saved brain found at {filename}. Creating a fresh brain.")
        
        graph_instance._seed_universal_concepts_internal()
        return graph_instance

    def load_from_old_format(self, graph_data: dict):
        print("  [Graph Engine]: Converting old brain format to new NetworkX engine...")
        for node_data in graph_data.get("nodes", []):
            node = ConceptNode.from_dict(node_data)
            self.graph.add_node(node.id, **node.to_dict())
        
        for edge_data in graph_data.get("edges", []):
            edge = RelationshipEdge.from_dict(edge_data)
            self.graph.add_edge(edge.source, edge.target, key=edge.id, **edge.to_dict())

    def _seed_universal_concepts_internal(self):
        core_concepts = {
            "concept": "meta_category", "action": "meta_category", "attribute": "meta_category",
            "sentiment": "category", "agent": "concept", "user": "concept",
            "positive": "sentiment", "negative": "sentiment", "understand": "action",
            "learn": "action"
        }
        for name, type_ in core_concepts.items():
            if not self.get_node_by_name(name):
                self.add_node(ConceptNode(name, node_type=type_))