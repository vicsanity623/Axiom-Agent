from __future__ import annotations

import json
import os
import uuid

# graph_core.py
from typing import TYPE_CHECKING, TypedDict

import networkx as nx
from networkx.readwrite import json_graph

if TYPE_CHECKING:
    from typing import Self


class ConceptNodeData(TypedDict):
    id: str
    name: str
    type: str
    properties: dict[str, str]
    value: float
    activation: float


class ConceptNode:
    def __init__(
        self,
        name: str,
        node_type: str = "concept",
        properties: dict[str, str] | None = None,
        value: float = 0.5,
        activation: float = 0.0,
        id_: str | None = None,
    ) -> None:
        self.id = id_ or str(uuid.uuid4())
        self.name = name.lower()
        self.type = node_type
        self.properties = properties or {}
        self.value = value
        self.activation = activation

    def to_dict(self) -> ConceptNodeData:
        return ConceptNodeData(
            {
                "id": self.id,
                "name": self.name,
                "type": self.type,
                "properties": self.properties,
                "value": self.value,
                "activation": self.activation,
            },
        )

    @classmethod
    def from_dict(cls, data: ConceptNodeData) -> Self:
        return cls(
            id_=data.get("id"),
            name=data["name"],
            node_type=data["type"],
            properties=data.get("properties", {}),
            value=data.get("value", 0.5),
            activation=data.get("activation", 0.0),
        )


class RelationshipEdgeData(TypedDict):
    id: str
    source: str
    target: str
    type: str
    weight: float
    properties: dict[str, str]
    access_count: int


class RelationshipEdge:
    __slots__ = (
        "id",
        "source",
        "target",
        "type",
        "weight",
        "properties",
        "access_count",
    )

    def __init__(
        self,
        source: str,
        target: str,
        type: str,
        weight: float = 0.5,
        id: str | None = None,
        properties: dict[str, str] | None = None,
        access_count: int = 0,
    ) -> None:
        self.id = id or str(uuid.uuid4())
        self.source = source
        self.target = target
        self.type = type
        self.weight = weight
        self.properties = properties or {}
        self.access_count = access_count

    def to_dict(self) -> RelationshipEdgeData:
        return RelationshipEdgeData(
            {
                "id": self.id,
                "source": self.source,
                "target": self.target,
                "type": self.type,
                "weight": self.weight,
                "properties": self.properties,
                "access_count": self.access_count,
            },
        )

    @classmethod
    def from_dict(cls, data: RelationshipEdgeData) -> Self:
        return cls(
            id=data.get("id"),
            source=data["source"],
            target=data["target"],
            type=data["type"],
            weight=data.get("weight", 0.5),
            properties=data.get("properties", {}),
            access_count=data.get("access_count", 0),
        )


class ConceptGraph:
    __slots__ = ("graph", "name_to_id")

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        # This is the correct name for your lookup dictionary
        self.name_to_id: dict[str, str] = {}

    def add_node(self, node: ConceptNode) -> ConceptNode:
        if existing_node := self.get_node_by_name(node.name):
            return existing_node

        self.graph.add_node(node.id, **node.to_dict())
        # Your original code used node.name, which is correct.
        self.name_to_id[node.name] = node.id
        return node

    def get_node_by_name(self, name: str) -> ConceptNode | None:
        # Your original code correctly used .lower() for case-insensitivity
        node_id = self.name_to_id.get(name.lower())
        if node_id and self.graph.has_node(node_id):
            # Your original `from_dict` usage was slightly wrong, it needs the full data dict
            node_data = self.graph.nodes[node_id]
            # We also need to pass the ID back in since it's not in the to_dict() payload
            node_data["id"] = node_id
            return ConceptNode.from_dict(node_data)
        return None

    def add_edge(
        self,
        source_node: ConceptNode,
        target_node: ConceptNode,
        relation_type: str,
        weight: float = 0.5,
        properties: dict[str, str] | None = None,
    ) -> RelationshipEdge | None:
        if not all([source_node, target_node]):
            return None

        # Check for existing edges to reinforce them
        if self.graph.has_edge(source_node.id, target_node.id):
            for key, data in self.graph.get_edge_data(
                source_node.id,
                target_node.id,
            ).items():
                if data.get("type") == relation_type:
                    data["weight"] = max(data["weight"], weight)
                    if properties:
                        data["properties"].update(properties)
                    # We need to reconstruct the full edge object to return it
                    full_edge_data = data.copy()
                    full_edge_data["source"] = source_node.id
                    full_edge_data["target"] = target_node.id
                    return RelationshipEdge.from_dict(full_edge_data)

        new_edge = RelationshipEdge(
            source_node.id,
            target_node.id,
            relation_type,
            weight,
            properties=properties,
        )
        self.graph.add_edge(
            new_edge.source,
            new_edge.target,
            key=new_edge.id,
            **new_edge.to_dict(),
        )
        return new_edge

    def get_edges_from_node(self, node_id: int) -> list[RelationshipEdge]:
        if not self.graph.has_node(node_id):
            return []
        # Your original from_dict usage was slightly wrong here too.
        # The edge data from networkx doesn't include source/target, so we must add it back.
        edges = []
        for u, v, data in self.graph.out_edges(node_id, data=True):
            full_edge_data = data.copy()
            full_edge_data["source"] = u
            full_edge_data["target"] = v
            edges.append(RelationshipEdge.from_dict(full_edge_data))
        return edges

    def get_edges_to_node(self, node_id: int) -> list[RelationshipEdge]:
        if not self.graph.has_node(node_id):
            return []
        edges = []
        for u, v, data in self.graph.in_edges(node_id, data=True):
            full_edge_data = data.copy()
            full_edge_data["source"] = u
            full_edge_data["target"] = v
            edges.append(RelationshipEdge.from_dict(full_edge_data))
        return edges

    def decay_activations(self, decay_rate: float = 0.1) -> None:
        for node_id in self.graph.nodes:
            current_activation = self.graph.nodes[node_id].get("activation", 0.0)
            self.graph.nodes[node_id]["activation"] = max(
                0.0,
                current_activation - decay_rate,
            )

    # --- SAVING AND LOADING LOGIC ---

    def save_to_file(self, filename: str) -> None:
        # This uses the standard, robust networkx serializer
        graph_data = json_graph.node_link_data(self.graph)
        with open(filename, "w") as f:
            json.dump(graph_data, f, indent=4)
        print(f"Agent brain saved to {filename}")

    @classmethod
    def load_from_dict(cls, data) -> Self:
        """THE NEW, CRITICAL METHOD FOR LOADING FROM .AXM MODELS"""
        instance = cls()

        # --- THIS IS THE FIX ---
        # Before: instance.graph = json_graph.node_link_graph(data)
        # After:
        instance.graph = json_graph.node_link_graph(data, edges="links")

        # Rebuild the name_to_id lookup cache after loading
        instance.name_to_id = {
            data["name"].lower(): node_id
            for node_id, data in instance.graph.nodes(data=True)
            if "name" in data
        }
        print(
            f"   - Brain loaded from dictionary. Nodes: {len(instance.graph.nodes)}, Edges: {len(instance.graph.edges)}",
        )
        return instance

    @classmethod
    def load_from_file(cls, filename: str) -> Self:
        """LOADS FROM A .JSON FILE (USED BY THE TRAINER)"""
        if os.path.exists(filename):
            try:
                with open(filename) as f:
                    graph_data = json.load(f)
                # We simply delegate to the robust load_from_dict method
                return cls.load_from_dict(graph_data)
            except Exception as e:
                print(
                    f"Error loading brain from {filename}: {e}. Creating a fresh brain.",
                )
                return cls()  # Return a fresh instance on error
        else:
            print(f"No saved brain found at {filename}. Creating a fresh brain.")
            return cls()  # Return a fresh instance if no file
