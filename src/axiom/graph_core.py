from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Any, TypedDict

import networkx as nx

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Self

    from axiom.universal_interpreter import PropertyData, RelationData

logger = logging.getLogger(__name__)


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
        self.id = id_ or str(uuid.uuid4())
        self.name = name.lower()
        self.type = node_type
        self.value = value
        self.activation = activation
        self.properties: PropertyData = properties or {}

    def to_dict(self) -> ConceptNodeData:
        """Serialize the ConceptNode instance to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "properties": self.properties,
            "value": self.value,
            "activation": self.activation,
        }

    @classmethod
    def from_dict(cls, data: ConceptNodeData) -> Self:
        """Create a new ConceptNode instance from a dictionary."""
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
        self.id = id or str(uuid.uuid4())
        self.source = source
        self.target = target
        self.type = type
        self.weight = weight
        self.properties = properties or {}
        self.access_count = access_count

    def to_dict(self) -> RelationshipEdgeData:
        """Serialize the RelationshipEdge instance to a dictionary."""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "weight": self.weight,
            "properties": self.properties,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: RelationshipEdgeData) -> Self:
        """Create a new RelationshipEdge instance from a dictionary."""
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
    """A manager for the agent's knowledge graph, built on NetworkX."""

    __slots__ = ("graph", "name_to_id", "_type_index")

    def __init__(self) -> None:
        """Initialize an empty ConceptGraph."""
        self.graph = nx.MultiDiGraph()
        self.name_to_id: dict[str, str] = {}
        self._type_index: dict[str, set[str]] = defaultdict(set)

    def add_node(self, node: ConceptNode) -> ConceptNode:
        """Add or update a concept node in the graph."""
        if existing_id := self.name_to_id.get(node.name):
            self.graph.nodes[existing_id].update(node.to_dict())
            return self.get_node_by_id(existing_id) or node

        self.graph.add_node(node.id, **node.to_dict())
        self.name_to_id[node.name] = node.id
        self._type_index[node.type].add(node.id)
        return node

    def get_node_by_name(self, name: str) -> ConceptNode | None:
        """Find and retrieve a concept node from the graph by its name."""
        node_id = self.name_to_id.get(name.lower())
        if node_id:
            return self.get_node_by_id(node_id)
        return None

    def get_node_by_id(self, node_id: str) -> ConceptNode | None:
        """Retrieve a single ConceptNode object from the graph by its ID."""
        if self.graph.has_node(node_id):
            node_data = self.graph.nodes[node_id].copy()
            node_data["id"] = node_id
            return ConceptNode.from_dict(node_data)
        return None

    def get_nodes_by_type(self, node_type: str) -> list[ConceptNode]:
        """Efficiently retrieves all nodes of a specific type using the index."""
        node_ids = self._type_index.get(node_type, set())
        return [node for node_id in node_ids if (node := self.get_node_by_id(node_id))]

    def get_all_nodes(self) -> list[ConceptNode]:
        """Retrieve all nodes in the graph as ConceptNode objects."""
        return [
            node
            for node_id in self.graph.nodes
            if (node := self.get_node_by_id(node_id))
        ]

    def get_all_node_names(self) -> list[str]:
        """Retrieve a list of all concept names in the graph."""
        return list(self.name_to_id.keys())

    def add_edge(
        self,
        source_node: ConceptNode,
        target_node: ConceptNode,
        relation_type: str,
        weight: float = 0.5,
        properties: PropertyData | None = None,
    ) -> RelationshipEdge | None:
        """Add a directed edge (relationship) between two nodes."""
        if not all([source_node, target_node]):
            raise ValueError("Source or target node missing")

        self.add_node(source_node)
        self.add_node(target_node)

        if self.graph.has_edge(source_node.id, target_node.id):
            for data in self.graph.get_edge_data(
                source_node.id, target_node.id
            ).values():
                if data.get("type") == relation_type:
                    data["weight"] = max(data["weight"], weight)
                    if properties:
                        data["properties"].update(properties)
                    full_edge_data = data.copy()
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
        """Retrieve all outgoing edges (relationships) from a specific node."""
        if not self.graph.has_node(node_id):
            return []
        edges = []
        for u, v, data in self.graph.out_edges(node_id, data=True):
            full_data = data.copy()
            edges.append(RelationshipEdge.from_dict(full_data))
        return edges

    def get_edges_to_node(self, node_id: str) -> list[RelationshipEdge]:
        """Retrieve all incoming edges (relationships) to a specific node."""
        if not self.graph.has_node(node_id):
            return []
        edges = []
        for u, v, data in self.graph.in_edges(node_id, data=True):
            full_data = data.copy()
            edges.append(RelationshipEdge.from_dict(full_data))
        return edges

    def get_all_edges(self) -> list[RelationshipEdge]:
        """Retrieve all edges in the graph as RelationshipEdge objects."""
        return [
            RelationshipEdge.from_dict(data)
            for _, _, data in self.graph.edges(data=True)
        ]

    def decay_activations(self, decay_rate: float = 0.1) -> None:
        """Apply a decay function to the activation level of all nodes."""
        for node_id in self.graph.nodes:
            current_activation = self.graph.nodes[node_id].get("activation", 0.0)
            self.graph.nodes[node_id]["activation"] = max(
                0.0, current_activation - decay_rate
            )

    def save_to_file(self, filename: Path | str) -> None:
        """Serialize the entire knowledge graph to a JSON file."""
        graph_data = {
            "nodes": [data for _, data in self.graph.nodes(data=True)],
            "links": [data for _, _, data in self.graph.edges(data=True)],
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=4)

    @classmethod
    def load_from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a new ConceptGraph instance from a dictionary."""
        instance = cls()
        nodes = data.get("nodes", [])
        links = data.get("links", [])

        for node_data in nodes:
            node_id = node_data.get("id")
            if not node_id:
                continue
            instance.graph.add_node(node_id, **node_data)
            if "name" in node_data:
                instance.name_to_id[node_data["name"].lower()] = node_id
            if "type" in node_data:
                instance._type_index[node_data["type"]].add(node_id)

        for link_data in links:
            source = link_data.get("source")
            target = link_data.get("target")
            key = link_data.get("id")
            if not all([source, target, key]):
                continue
            instance.graph.add_edge(source, target, key=key, **link_data)

        logger.info(
            "[green]   - Brain loaded from dictionary. Nodes: %d, Edges: %d[/green]",
            len(instance.graph.nodes),
            len(instance.graph.edges),
        )
        return instance

    @classmethod
    def load_from_file(cls, filename: Path | str) -> Self:
        """Create a new ConceptGraph instance from a JSON file."""
        if os.path.exists(filename):
            try:
                with open(filename, encoding="utf-8") as f:
                    graph_data = json.load(f)
                return cls.load_from_dict(graph_data)
            except Exception as e:
                logger.error(
                    "Error loading brain from %s: %s. Creating a fresh brain.",
                    filename,
                    e,
                )
                return cls()
        else:
            logger.info("No saved brain found at %s. Creating a fresh brain.", filename)
            return cls()

    def get_conflicting_facts(self, relation: RelationData) -> list[ConceptNode]:
        """
        Return a list of nodes that conflict with the given subject & relation type.
        This is used for exclusive relationships, to generate clarification questions.
        """
        subject_name_raw = relation.get("subject")
        relation_type = (
            relation.get("predicate")
            or relation.get("verb")
            or relation.get("relation")
        )

        if not isinstance(subject_name_raw, str) or not relation_type:
            return []

        subject_node = self.get_node_by_name(subject_name_raw)
        if not subject_node:
            return []

        conflicts: list[ConceptNode] = []
        for _, v, key, data in self.graph.out_edges(
            subject_node.id,
            keys=True,
            data=True,
        ):
            if data.get("type") == relation_type:
                if target_node := self.get_node_by_id(v):
                    conflicts.append(target_node)
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
            logger.warning(
                "[Graph Core Warning]: Edge %s not found for property update.",
                edge.id,
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
        """Return the strongest non-superseded edge matching this relation type."""
        outgoing = self.get_edges_from_node(subject_node.id)
        incoming = self.get_edges_to_node(subject_node.id)

        candidates = []
        for edge in outgoing + incoming:
            if edge.type != relation_type:
                continue
            status = (edge.properties or {}).get("revision_status", "active")
            if status != "superseded":
                candidates.append(edge)

        if not candidates:
            if debug:
                logger.debug(
                    "[Conflict Check]: No active edges found for '%s' --[%s]--> ?",
                    subject_node.name,
                    relation_type,
                )
            return None

        winner: RelationshipEdge = max(candidates, key=lambda e: e.weight)
        if debug:
            logger.debug(
                "[Conflict Check]: Found conflict edge %s (w=%.2f, type=%s)",
                winner.id,
                winner.weight,
                winner.type,
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
            logger.warning(
                "[Graph Core Warning]: Could not find edge %s to update weight.",
                edge.id,
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

        logger.info(
            "[Graph Update]: Edge '%s' confidence updated "
            "%.2f -> %.2f "
            "(provenance: %s -> %s)",
            edge.type,
            old_weight,
            edge_data["weight"],
            old_provenance,
            provenance,
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
