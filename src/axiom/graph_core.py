from __future__ import annotations

import json
import os
import time
import uuid

# graph_core.py
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import networkx as nx
from networkx.readwrite import json_graph

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Self

    from axiom.universal_interpreter import PropertyData, RelationData


class ConceptNodeData(TypedDict):
    id: str
    name: str
    type: str
    properties: PropertyData | None
    value: float
    activation: float


class ConceptNode:
    """Represents a single concept or entity in the knowledge graph."""

    def __init__(
        self,
        name: str,
        node_type: str = "concept",
        properties: PropertyData | None = None,
        value: float = 0.5,
        activation: float = 0.0,
        id_: str | None = None,
    ) -> None:
        """Initialize a new ConceptNode.

        Args:
            name: The primary name of the concept (e.g., "dog", "Paris").
            node_type: The classification of the node (e.g., "noun", "city").
            properties: A dictionary of additional, arbitrary metadata.
            value: A numerical value, reserved for future weighting or logic.
            activation: The current activation level of the node in memory.
            id_: A specific UUID to assign. If None, a new one is generated.
        """
        self.id = id_ or str(uuid.uuid4())
        self.name = name.lower()
        self.type = node_type
        self.value = value
        self.activation = activation
        self.properties: PropertyData = properties or {}

    def to_dict(self) -> ConceptNodeData:
        """Serialize the ConceptNode instance to a dictionary.

        Returns:
            A dictionary representation of the node, suitable for JSON
            serialization.
        """
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
        """Create a new ConceptNode instance from a dictionary.

        Args:
            data: A dictionary containing the node's data.

        Returns:
            A new instance of the ConceptNode class.
        """
        return cls(
            id_=data.get("id"),
            name=data["name"],
            node_type=data["type"],
            properties=data.get("properties"),
            value=data.get("value", 0.5),
            activation=data.get("activation", 0.0),
        )


class RelationshipEdgeData(TypedDict):
    id: str
    source: str
    target: str
    type: str
    weight: float
    properties: PropertyData
    access_count: int


class RelationshipEdge:
    """Represents a single, directed relationship between two ConceptNodes."""

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
        properties: PropertyData | None = None,
        access_count: int = 0,
    ) -> None:
        """Initialize a new RelationshipEdge.

        Args:
            source: The UUID of the source (origin) ConceptNode.
            target: The UUID of the target (destination) ConceptNode.
            type: The semantic type of the relationship (e.g., "is_a").
            weight: The confidence or strength of this fact (0.0 to 1.0).
            id: A specific UUID to assign. If None, a new one is generated.
            properties: A dictionary of additional, arbitrary metadata.
            access_count: A counter for how many times this fact has been
                accessed, used for salience calculations.
        """
        self.id = id or str(uuid.uuid4())
        self.source = source
        self.target = target
        self.type = type
        self.weight = weight
        self.properties = properties or {}
        self.access_count = access_count

    def to_dict(self) -> RelationshipEdgeData:
        """Serialize the RelationshipEdge instance to a dictionary.

        Returns:
            A dictionary representation of the edge, suitable for JSON
            serialization.
        """
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
        """Create a new RelationshipEdge instance from a dictionary.

        Args:
            data: A dictionary containing the edge's data.

        Returns:
            A new instance of the RelationshipEdge class.
        """
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
    """A manager for the agent's knowledge graph, built on NetworkX.

    This class provides a high-level API for interacting with the agent's
    brain. It handles the creation, retrieval, and connection of nodes and
    edges, abstracting away the underlying `networkx.MultiDiGraph`
    implementation. It also maintains a fast lookup table for finding
    nodes by name.
    """

    __slots__ = ("graph", "name_to_id")

    def __init__(self) -> None:
        """Initialize an empty ConceptGraph."""
        self.graph = nx.MultiDiGraph()
        self.name_to_id: dict[str, str] = {}

    def add_node(self, node: ConceptNode) -> ConceptNode:
        """Add a new concept node to the graph if it doesn't already exist.

        This method is idempotent: if a node with the same name already
        exists, it will return the existing node instead of creating a
        duplicate.

        Args:
            node: The `ConceptNode` instance to add to the graph.

        Returns:
            The newly added or pre-existing `ConceptNode`.
        """
        if existing_node := self.get_node_by_name(node.name):
            return existing_node

        self.graph.add_node(node.id, **node.to_dict())
        self.name_to_id[node.name] = node.id
        return node

    def get_node_by_name(self, name: str) -> ConceptNode | None:
        """Find and retrieve a concept node from the graph by its name.

        Uses a fast in-memory dictionary for the name-to-ID lookup before
        retrieving the full node data from the graph.

        Args:
            name: The case-insensitive name of the node to find.

        Returns:
            The corresponding `ConceptNode` instance, or None if not found.
        """
        node_id = self.name_to_id.get(name.lower())
        if node_id and self.graph.has_node(node_id):
            node_data = self.graph.nodes[node_id]
            node_data["id"] = node_id
            return ConceptNode.from_dict(node_data)
        return None

    def get_node_by_id(self, node_id: str) -> ConceptNode | None:
        """Retrieve a single ConceptNode object from the graph by its ID.

        Args:
            node_id: The unique identifier of the node to retrieve.

        Returns:
            The corresponding `ConceptNode` instance, or None if not found.
        """
        if self.graph.has_node(node_id):
            node_data = self.graph.nodes[node_id]
            node_data["id"] = node_id
            return ConceptNode.from_dict(node_data)
        return None

    def get_all_node_names(self) -> list[str]:
        """Retrieve a list of all concept names in the graph.

        This is useful for fuzzy matching, as it provides a complete
        "vocabulary" of known entities to match against.

        Returns:
            A list of strings, where each string is the name of a node.
        """
        return list(self.name_to_id.keys())

    def add_edge(
        self,
        source_node: ConceptNode,
        target_node: ConceptNode,
        relation_type: str,
        weight: float = 0.5,
        properties: PropertyData | None = None,
    ) -> RelationshipEdge | None:
        """Add a directed edge (relationship) between two nodes.

        If an edge with the same source, target, and type already exists,
        this method will update the existing edge's weight (taking the
        maximum of the old and new weights) and merge properties rather
        than creating a duplicate.

        Args:
            source_node: The `ConceptNode` where the relationship originates.
            target_node: The `ConceptNode` where the relationship ends.
            relation_type: The semantic type of the relationship (e.g., "is_a").
            weight: The confidence score for this relationship.
            properties: Additional metadata for the relationship.

        Returns:
            The newly created or updated `RelationshipEdge` instance, or
            None if the source or target node is invalid.
        """
        if not all([source_node, target_node]):
            return None

        if self.graph.has_edge(source_node.id, target_node.id):
            for key, data in self.graph.get_edge_data(
                source_node.id,
                target_node.id,
            ).items():
                if data.get("type") == relation_type:
                    data["weight"] = max(data["weight"], weight)
                    if properties:
                        data["properties"].update(properties)
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

    def get_edges_from_node(self, node_id: str) -> list[RelationshipEdge]:
        """Retrieve all outgoing edges (relationships) from a specific node.

        Args:
            node_id: The unique identifier of the source node.

        Returns:
            A list of `RelationshipEdge` instances originating from the node.
        """
        if not self.graph.has_node(node_id):
            return []
        edges = []
        for u, v, data in self.graph.out_edges(node_id, data=True):
            full_edge_data = data.copy()
            full_edge_data["source"] = u
            full_edge_data["target"] = v
            edges.append(RelationshipEdge.from_dict(full_edge_data))
        return edges

    def get_edges_to_node(self, node_id: str) -> list[RelationshipEdge]:
        """Retrieve all incoming edges (relationships) to a specific node.

        Args:
            node_id: The unique identifier of the target node.

        Returns:
            A list of `RelationshipEdge` instances pointing to the node.
        """
        if not self.graph.has_node(node_id):
            return []
        edges = []
        for u, v, data in self.graph.in_edges(node_id, data=True):
            full_edge_data = data.copy()
            full_edge_data["source"] = u
            full_edge_data["target"] = v
            edges.append(RelationshipEdge.from_dict(full_edge_data))
        return edges

    def get_all_edges(self) -> list[RelationshipEdge]:
        """Retrieve all edges in the graph as RelationshipEdge objects.

        Returns:
            A list of all RelationshipEdge objects in the graph.
        """
        reconstructed_edges = []
        for u, v, data in self.graph.edges(data=True):
            full_data = data.copy()
            full_data["source"] = u
            full_data["target"] = v
            reconstructed_edges.append(RelationshipEdge.from_dict(full_data))
        return reconstructed_edges

    def decay_activations(self, decay_rate: float = 0.1) -> None:
        """Apply a decay function to the activation level of all nodes.

        This method simulates the process of forgetting or reducing the
        short-term "focus" on concepts over time. It is called at the
        beginning of each chat turn.

        Args:
            decay_rate: The amount to subtract from each node's activation.
        """
        for node_id in self.graph.nodes:
            current_activation = self.graph.nodes[node_id].get("activation", 0.0)
            self.graph.nodes[node_id]["activation"] = max(
                0.0,
                current_activation - decay_rate,
            )

    def save_to_file(self, filename: Path | str) -> None:
        """Serialize the entire knowledge graph to a JSON file.

        Uses the NetworkX `node_link_data` format for robust serialization.

        Args:
            filename: The path to the file where the graph will be saved.
        """
        graph_data = json_graph.node_link_data(self.graph, edges="links")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=4)
        print(f"Agent brain saved to {filename}")

    @classmethod
    def load_from_dict(cls, data: dict[str, object]) -> Self:
        """Create a new ConceptGraph instance from a dictionary.

        This method de-serializes a graph from the NetworkX node-link
        format. It also rebuilds the fast name-to-ID lookup table after
        loading. This is the primary method for loading unpacked .axm models.

        Args:
            data: A dictionary containing the node-link graph data.

        Returns:
            A new instance of the ConceptGraph class populated with data.
        """
        instance = cls()

        instance.graph = json_graph.node_link_graph(data, edges="links")

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
    def load_from_file(cls, filename: Path | str) -> Self:
        """Create a new ConceptGraph instance from a JSON file.

        This is a convenience wrapper around `load_from_dict`. It handles
        the file I/O and JSON parsing, including error handling for missing
        or corrupt files. It is the primary method used by training scripts.

        Args:
            filename: The path to the JSON file containing the graph data.

        Returns:
            A new instance of the ConceptGraph class, which will be empty
            if the file could not be loaded.
        """
        if os.path.exists(filename):
            try:
                with open(filename, encoding="utf-8") as f:
                    graph_data = json.load(f)
                return cls.load_from_dict(graph_data)
            except Exception as e:
                print(
                    f"Error loading brain from {filename}: {e}. Creating a fresh brain.",
                )
                return cls()
        else:
            print(f"No saved brain found at {filename}. Creating a fresh brain.")
            return cls()

    def get_conflicting_facts(self, relation: RelationData) -> list[ConceptNode]:
        """
        Return a list of nodes that conflict with the given subject & relation type.
        This is used for exclusive relationships, to generate clarification questions.
        """
        subject_name = relation.get("subject")
        relation_type = (
            relation.get("predicate")
            or relation.get("verb")
            or relation.get("relation")
        )
        if not subject_name or not relation_type:
            return []

        subject_node = self.get_node_by_name(subject_name)
        if not subject_node:
            return []

        conflicts: list[ConceptNode] = []
        for _, v, key, data in self.graph.out_edges(
            subject_node.id,
            keys=True,
            data=True,
        ):
            if data.get("type") == relation_type:
                target_node_data = self.graph.nodes.get(v)
                if target_node_data:
                    conflicts.append(ConceptNode.from_dict(target_node_data))
        return conflicts

    def update_edge_properties(
        self,
        edge: RelationshipEdge,
        updates: dict[str, object],
        merge: bool = True,
    ) -> None:
        """Safely update or merge properties on an existing edge.

        Args:
            edge: The `RelationshipEdge` to update.
            updates: A dictionary of new or updated key-value pairs.
            merge: If True (default), merge into existing properties.
                   If False, replace the entire dictionary.
        """
        if not self.graph.has_edge(edge.source, edge.target, key=edge.id):
            print(
                f"[Graph Core Warning]: Edge {edge.id} not found for property update.",
            )
            return

        edge_data = self.graph.edges[edge.source, edge.target, edge.id]

        if merge:
            edge_data["properties"].update(updates)
        else:
            edge_data["properties"] = updates

        edge_data["properties"]["last_modified"] = time.time()

    def find_exclusive_conflict(
        self,
        subject_node: ConceptNode,
        relation_type: str,
        debug: bool = False,
    ) -> RelationshipEdge | None:
        """Return the strongest non-superseded edge matching this relation type.

        This is used by the CognitiveAgent’s belief revision process.
        Only one 'exclusive' fact per subject-relation type should exist.
        """
        candidates = []
        for edge in self.get_edges_from_node(subject_node.id):
            if edge.type != relation_type:
                continue
            status = edge.properties.get("revision_status", "active")
            if status != "superseded":
                candidates.append(edge)

        if not candidates:
            if debug:
                print(
                    f"[Conflict Check]: No active edges found for '{subject_node.name}' --[{relation_type}]--> ?",
                )
            return None

        winner = max(candidates, key=lambda e: e.weight)
        if debug:
            print(
                f"[Conflict Check]: Found conflict edge {winner.id} (w={winner.weight:.2f}, type={winner.type})",
            )
        return winner

    @staticmethod
    def normalize_provenance_rank(rank_table: dict[str, int], provenance: str) -> int:
        """Return a safe provenance rank value even if missing or malformed."""
        if not provenance:
            return 0
        provenance = provenance.lower().strip()
        return rank_table.get(provenance, 0)

    def compute_confidence_adjustment(
        self,
        current_confidence: float,
        incoming_confidence: float,
        provenance_rank_current: int,
        provenance_rank_new: int,
    ) -> float:
        """Compute a revised edge confidence based on provenance and source reliability."""
        if provenance_rank_new > provenance_rank_current:
            return (incoming_confidence * 0.7) + (current_confidence * 0.3)
        if provenance_rank_new == provenance_rank_current:
            return (incoming_confidence + current_confidence) / 2
        return (current_confidence * 0.8) + (incoming_confidence * 0.2)

    def update_edge_weight(
        self,
        edge: RelationshipEdge,
        new_weight: float,
        provenance: str = "user",
        rank_table: dict[str, int] | None = None,
    ) -> None:
        """Update an edge's confidence weight with provenance-aware adjustment."""
        if not self.graph.has_edge(edge.source, edge.target, key=edge.id):
            print(
                f"[Graph Core Warning]: Could not find edge {edge.id} to update weight.",
            )
            return

        rank_table = rank_table or {
            "user": 1,
            "llm_verified": 2,
            "dictionary": 3,
            "system": 4,
        }

        edge_data = self.graph.edges[edge.source, edge.target, edge.id]
        old_weight = edge_data.get("weight", 0.5)
        old_provenance = edge_data.get("properties", {}).get("provenance", "user")

        rank_current = self.normalize_provenance_rank(rank_table, old_provenance)
        rank_new = self.normalize_provenance_rank(rank_table, provenance)

        adjusted_weight = self.compute_confidence_adjustment(
            old_weight,
            new_weight,
            rank_current,
            rank_new,
        )

        edge_data["weight"] = round(max(0.0, min(1.0, adjusted_weight)), 4)
        edge_data.setdefault("properties", {}).update(
            {
                "provenance": provenance,
                "last_modified": time.time(),
                "confidence_updated": True,
            },
        )

        print(
            f"[Graph Update]: Edge '{edge.type}' confidence updated "
            f"{old_weight:.2f} → {edge_data['weight']:.2f} "
            f"(provenance: {old_provenance} → {provenance})",
        )

    def revise_conflicting_edge(
        self,
        existing_edge: RelationshipEdge,
        new_confidence: float,
        new_provenance: str = "user",
    ) -> str:
        """Decide how to handle a conflict between existing and new edges."""
        rank_table = {
            "user": 1,
            "llm_verified": 2,
            "dictionary": 3,
            "system": 4,
        }

        rank_existing = self.normalize_provenance_rank(
            rank_table,
            existing_edge.properties.get("provenance", "user"),
        )
        rank_new = self.normalize_provenance_rank(rank_table, new_provenance)

        if rank_new > rank_existing:
            self.update_edge_weight(
                existing_edge,
                new_confidence,
                new_provenance,
                rank_table,
            )
            existing_edge.properties["revision_status"] = "replaced"
            return "replaced"

        if rank_new == rank_existing:
            self.update_edge_weight(
                existing_edge,
                new_confidence,
                new_provenance,
                rank_table,
            )
            return "merged"

        existing_edge.properties["revision_status"] = "ignored_lower_provenance"
        return "ignored"
