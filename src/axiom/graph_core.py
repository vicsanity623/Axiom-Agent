from __future__ import annotations

import json
import logging
import os
import time
import typing
import uuid
from typing import TYPE_CHECKING, Any, TypedDict

from neo4j import GraphDatabase

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
    embedding: list[float] | None


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
        embedding: list[float] | None = None,
    ) -> None:
        self.id = id_ or str(uuid.uuid4())
        self.name = name.lower()
        self.type = node_type
        self.value = value
        self.activation = activation
        self.properties: PropertyData = properties or {}
        self.embedding = embedding

    def to_dict(self) -> ConceptNodeData:
        """Serialize the ConceptNode instance to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "properties": self.properties,
            "value": self.value,
            "activation": self.activation,
            "embedding": self.embedding,
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
            embedding=data.get("embedding"),
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

    __slots__ = ("driver",)

    def __init__(self) -> None:
        """Initialize the ConceptGraph with Neo4j connection."""
        self._load_env()
        uri = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USERNAME")
        password = os.environ.get("NEO4J_PASSWORD")

        if not all([uri, user, password]):
            logger.warning(
                "Neo4j credentials not found in environment. Graph will be disabled."
            )
            self.driver = None
        else:
            try:
                # MyPy doesn't know these are not None here
                self.driver = GraphDatabase.driver(uri, auth=(user, password))  # type: ignore
                self.driver.verify_connectivity()
                logger.info("Connected to Neo4j database.")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                self.driver = None

    def _load_env(self) -> None:
        """Load environment variables from .env file."""
        if os.path.exists(".env"):
            with open(".env") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k] = v

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()

    def rename_node_by_id(self, node_id: str, new_name: str) -> None:
        """Renames a node and updates the name-to-ID mapping."""
        if not self.driver:
            return

        query = "MATCH (n:Concept {id: $id}) SET n.name = $name"
        with self.driver.session() as session:
            session.run(query, id=node_id, name=new_name)
            logger.info(f"   - Renamed node {node_id} to '{new_name}'.")

    def get_edges_by_type(self, node_id: str, edge_type: str) -> list[RelationshipEdge]:
        """Gets all outgoing edges of a specific type from a node."""
        if not self.driver:
            return []

        safe_type = edge_type.replace(" ", "_").upper()
        query = f"MATCH (a:Concept {{id: $id}})-[r:`{safe_type}`]->(b:Concept) RETURN a.id, b.id, r"

        edges = []
        with self.driver.session() as session:
            result = session.run(query, id=node_id)
            for record in result:
                r = record["r"]
                r_props = {}
                if "properties" in r and isinstance(r["properties"], str):
                    try:
                        r_props = json.loads(r["properties"])
                    except Exception:
                        pass
                edges.append(
                    RelationshipEdge(
                        source=record["a.id"],
                        target=record["b.id"],
                        type=r.get("type", edge_type),
                        weight=r.get("weight", 0.5),
                        id=r.get("id"),
                        properties=typing.cast("PropertyData | None", r_props),
                    )
                )
        return edges

    def has_relationship(
        self, source_node: ConceptNode, relation_type: str, object_name: str
    ) -> bool:
        """Checks if a specific relationship exists from a source node."""
        if not self.driver:
            return False

        safe_type = relation_type.replace(" ", "_").upper()
        query = (
            f"MATCH (a:Concept {{id: $id}})-[r:`{safe_type}`]->(b:Concept) "
            "WHERE toLower(b.name) = toLower($obj_name) "
            "RETURN count(r) > 0 as exists"
        )
        with self.driver.session() as session:
            result = session.run(query, id=source_node.id, obj_name=object_name)
            record = result.single()
            return record["exists"] if record else False

    def add_node(self, node: ConceptNode) -> ConceptNode:
        """Add or update a concept node in the graph."""
        if not self.driver:
            return node

        # Check for existing node by name to ensure idempotency
        existing_node = self.get_node_by_name(node.name)
        if existing_node:
            node.id = existing_node.id
            # Merge properties
            if existing_node.properties and node.properties:
                existing_node.properties.update(node.properties)
                node.properties = existing_node.properties
            elif existing_node.properties:
                node.properties = existing_node.properties

        props_json = json.dumps(node.properties) if node.properties else "{}"

        query = (
            "MERGE (n:Concept {id: $id}) "
            "ON CREATE SET n.name = $name, n.type = $type, n.value = $value, "
            "n.activation = $activation, n.properties = $properties, n.embedding = $embedding "
            "ON MATCH SET n.name = $name, n.type = $type, n.value = $value, "
            "n.activation = $activation, n.properties = $properties, n.embedding = $embedding "
            "RETURN n"
        )
        with self.driver.session() as session:
            session.run(
                query,
                id=node.id,
                name=node.name,
                type=node.type,
                value=node.value,
                activation=node.activation,
                properties=props_json,
                embedding=node.embedding,
            )
        return node

    def get_node_by_name(self, name: str) -> ConceptNode | None:
        """Find and retrieve a concept node from the graph by its name."""
        if not self.driver:
            return None

        query = "MATCH (n:Concept) WHERE toLower(n.name) = toLower($name) RETURN n"
        with self.driver.session() as session:
            result = session.run(query, name=name)
            record = result.single()
            if record:
                node_data = dict(record["n"])
                if "properties" in node_data and isinstance(
                    node_data["properties"], str
                ):
                    try:
                        node_data["properties"] = json.loads(node_data["properties"])
                    except Exception:
                        node_data["properties"] = {}
                return ConceptNode.from_dict(typing.cast("ConceptNodeData", node_data))
        return None

    def get_nearest_nodes(
        self, query_embedding: list[float], top_k: int = 5, threshold: float = 0.0
    ) -> list[tuple[ConceptNode, float]]:
        """
        Finds the nodes with embeddings most similar to the query embedding.
        Uses a linear scan, which is sufficient for graph sizes < 100k nodes.
        """
        from axiom.vector_store import VectorStore

        store = VectorStore()
        candidates = []

        # Fetch all nodes with embeddings from Neo4j
        # Optimization: We could store embeddings in a vector index in Neo4j
        # but for now we stick to the existing logic of fetching and comparing.
        all_nodes = self.get_all_nodes()

        for node in all_nodes:
            if node.embedding:
                score = store.cosine_similarity(query_embedding, node.embedding)
                if score >= threshold:
                    candidates.append((node, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def get_node_by_id(self, node_id: str) -> ConceptNode | None:
        """Retrieve a single ConceptNode object from the graph by its ID."""
        if not self.driver:
            return None

        query = "MATCH (n:Concept {id: $id}) RETURN n"
        with self.driver.session() as session:
            result = session.run(query, id=node_id)
            record = result.single()
            if record:
                node_data = dict(record["n"])
                if "properties" in node_data and isinstance(
                    node_data["properties"], str
                ):
                    try:
                        node_data["properties"] = json.loads(node_data["properties"])
                    except Exception:
                        node_data["properties"] = {}
                return ConceptNode.from_dict(typing.cast("ConceptNodeData", node_data))
        return None

    def get_nodes_by_type(self, node_type: str) -> list[ConceptNode]:
        """Efficiently retrieves all nodes of a specific type."""
        if not self.driver:
            return []

        query = "MATCH (n:Concept {type: $type}) RETURN n"
        nodes = []
        with self.driver.session() as session:
            result = session.run(query, type=node_type)
            for record in result:
                node_data = dict(record["n"])
                if "properties" in node_data and isinstance(
                    node_data["properties"], str
                ):
                    try:
                        node_data["properties"] = json.loads(node_data["properties"])
                    except Exception:
                        node_data["properties"] = {}
                nodes.append(
                    ConceptNode.from_dict(typing.cast("ConceptNodeData", node_data))
                )
        return nodes

    def get_all_nodes(self) -> list[ConceptNode]:
        """Retrieve all nodes in the graph as ConceptNode objects."""
        if not self.driver:
            return []

        query = "MATCH (n:Concept) RETURN n"
        nodes = []
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                node_data = dict(record["n"])
                if "properties" in node_data and isinstance(
                    node_data["properties"], str
                ):
                    try:
                        node_data["properties"] = json.loads(node_data["properties"])
                    except Exception:
                        node_data["properties"] = {}
                nodes.append(
                    ConceptNode.from_dict(typing.cast("ConceptNodeData", node_data))
                )
        return nodes

    def get_all_node_names(self) -> list[str]:
        """Retrieve a list of all concept names in the graph."""
        if not self.driver:
            return []
        query = "MATCH (n:Concept) RETURN n.name"
        with self.driver.session() as session:
            result = session.run(query)
            return [record["n.name"] for record in result if record["n.name"]]

    def add_edge(
        self,
        source_node: ConceptNode,
        target_node: ConceptNode,
        relation_type: str,
        weight: float = 0.5,
        properties: PropertyData | None = None,
    ) -> RelationshipEdge | None:
        """
        Add a directed edge (relationship) between two nodes.
        """
        if not self.driver:
            return None

        if not all([source_node, target_node]):
            raise ValueError("Source or target node missing")

        self.add_node(source_node)
        self.add_node(target_node)

        # Check for existing edge to merge properties
        # This is expensive but necessary for full compatibility with NetworkX behavior
        existing_edges = self.get_edges_from_node(source_node.id)
        existing_edge = next(
            (
                e
                for e in existing_edges
                if e.target == target_node.id and e.type == relation_type
            ),
            None,
        )

        props = properties or {}
        if existing_edge and existing_edge.properties:
            if props:
                existing_edge.properties.update(props)
                props = existing_edge.properties
            else:
                props = existing_edge.properties

        props_json = json.dumps(props) if props else "{}"
        safe_type = relation_type.replace(" ", "_").upper()
        new_id = str(uuid.uuid4())

        query = (
            f"MATCH (a:Concept {{id: $source}}), (b:Concept {{id: $target}}) "
            f"MERGE (a)-[r:`{safe_type}`]->(b) "
            "ON CREATE SET r.id = $new_id, r.weight = $weight, r.properties = $properties, r.type = $type "
            "ON MATCH SET r.weight = CASE WHEN $weight > r.weight THEN $weight ELSE r.weight END, r.properties = $properties "
            "RETURN r"
        )

        with self.driver.session() as session:
            result = session.run(
                query,
                source=source_node.id,
                target=target_node.id,
                weight=weight,
                properties=props_json,
                new_id=new_id,
                type=relation_type,
            )
            record = result.single()
            if record:
                r = record["r"]
                r_props = {}
                if "properties" in r and isinstance(r["properties"], str):
                    try:
                        r_props = json.loads(r["properties"])
                    except Exception:
                        pass
                # Merge new properties if it was a match?
                # The query didn't update properties on match to avoid overwriting with empty.
                # If we want to merge, we should do it here or in query.
                # For now, let's assume the returned edge is the source of truth.
                return RelationshipEdge(
                    source=source_node.id,
                    target=target_node.id,
                    type=relation_type,
                    weight=r.get("weight", weight),
                    id=r.get("id"),
                    properties=typing.cast("PropertyData | None", r_props),
                )
        return None

    def get_edges_from_node(self, node_id: str) -> list[RelationshipEdge]:
        """Retrieve all outgoing edges (relationships) from a specific node."""
        if not self.driver:
            return []

        query = "MATCH (a:Concept {id: $id})-[r]->(b:Concept) RETURN a.id, b.id, r"
        edges = []
        with self.driver.session() as session:
            result = session.run(query, id=node_id)
            for record in result:
                r = record["r"]
                r_props = {}
                if "properties" in r and isinstance(r["properties"], str):
                    try:
                        r_props = json.loads(r["properties"])
                    except Exception:
                        pass
                edges.append(
                    RelationshipEdge(
                        source=record["a.id"],
                        target=record["b.id"],
                        type=r.get("type", r.type),  # Use stored type or rel type
                        weight=r.get("weight", 0.5),
                        id=r.get("id"),
                        properties=typing.cast("PropertyData | None", r_props),
                    )
                )
        return edges

    def get_edges_to_node(self, node_id: str) -> list[RelationshipEdge]:
        """Retrieve all incoming edges (relationships) to a specific node."""
        if not self.driver:
            return []

        query = "MATCH (a:Concept)-[r]->(b:Concept {id: $id}) RETURN a.id, b.id, r"
        edges = []
        with self.driver.session() as session:
            result = session.run(query, id=node_id)
            for record in result:
                r = record["r"]
                r_props = {}
                if "properties" in r and isinstance(r["properties"], str):
                    try:
                        r_props = json.loads(r["properties"])
                    except Exception:
                        pass
                edges.append(
                    RelationshipEdge(
                        source=record["a.id"],
                        target=record["b.id"],
                        type=r.get("type", r.type),
                        weight=r.get("weight", 0.5),
                        id=r.get("id"),
                        properties=typing.cast("PropertyData | None", r_props),
                    )
                )
        return edges

    def get_all_edges(self) -> list[RelationshipEdge]:
        """Retrieve all edges in the graph as RelationshipEdge objects."""
        if not self.driver:
            return []

        query = "MATCH (a:Concept)-[r]->(b:Concept) RETURN a.id, b.id, r"
        edges = []
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                r = record["r"]
                r_props = {}
                if "properties" in r and isinstance(r["properties"], str):
                    try:
                        r_props = json.loads(r["properties"])
                    except Exception:
                        pass
                edges.append(
                    RelationshipEdge(
                        source=record["a.id"],
                        target=record["b.id"],
                        type=r.get("type", r.type),
                        weight=r.get("weight", 0.5),
                        id=r.get("id"),
                        properties=typing.cast("PropertyData | None", r_props),
                    )
                )
        return edges

    def decay_activations(self, decay_rate: float = 0.1) -> None:
        """Apply a decay function to the activation level of all nodes."""
        if not self.driver:
            return

        query = (
            "MATCH (n:Concept) "
            "SET n.activation = CASE "
            "WHEN n.activation - $decay < 0.0 THEN 0.0 "
            "ELSE n.activation - $decay END"
        )
        with self.driver.session() as session:
            session.run(query, decay=decay_rate)

    def save_to_file(self, filename: Path | str) -> None:
        """Serialize the entire knowledge graph to a JSON file."""
        logger.info("ConceptGraph is now backed by Neo4j. save_to_file is a no-op.")

    @classmethod
    def load_from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a new ConceptGraph instance from a dictionary."""
        logger.info(
            "ConceptGraph is backed by Neo4j. load_from_dict creates a new instance connected to DB."
        )
        return cls()

    @classmethod
    def load_from_file(cls, filename: Path | str) -> Self:
        """Create a new ConceptGraph instance from a JSON file."""
        logger.info(
            "ConceptGraph is backed by Neo4j. load_from_file creates a new instance connected to DB."
        )
        return cls()

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its relationships from the graph."""
        if not self.driver:
            return

        query = "MATCH (n:Concept {id: $id}) DETACH DELETE n"
        with self.driver.session() as session:
            session.run(query, id=node_id)

    def get_node_count(self) -> int:
        """Return the total number of nodes in the graph."""
        if not self.driver:
            return 0

        query = "MATCH (n:Concept) RETURN count(n) as count"
        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            return record["count"] if record else 0

    def get_edge_count(self) -> int:
        """Return the total number of edges in the graph."""
        if not self.driver:
            return 0

        query = "MATCH ()-[r]->() RETURN count(r) as count"
        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            return record["count"] if record else 0

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
        for edge in self.get_edges_from_node(subject_node.id):
            if edge.type == relation_type:
                if target_node := self.get_node_by_id(edge.target):
                    conflicts.append(target_node)
        return conflicts

    def update_edge_properties(
        self,
        edge: RelationshipEdge,
        updates: dict[str, object],
        merge: bool = True,
    ) -> None:
        """Safely update or merge properties on an existing edge."""
        if not self.driver:
            return

        safe_type = edge.type.replace(" ", "_").upper()

        with self.driver.session() as session:
            # Read existing properties
            query_read = f"MATCH (a:Concept {{id: $source}})-[r:`{safe_type}` {{id: $id}}]->(b:Concept {{id: $target}}) RETURN r.properties"
            result = session.run(
                query_read, source=edge.source, target=edge.target, id=edge.id
            )
            record = result.single()

            if not record:
                logger.warning(
                    "[Graph Core Warning]: Edge %s not found for property update.",
                    edge.id,
                )
                return

            current_props = {}
            if record["r.properties"]:
                try:
                    current_props = json.loads(record["r.properties"])
                except Exception:
                    pass

            if merge:
                current_props.update(updates)
            else:
                current_props = updates  # type: ignore

            current_props["last_modified"] = time.time()

            # Write back
            new_props_json = json.dumps(current_props)
            query_write = (
                f"MATCH (a:Concept {{id: $source}})-[r:`{safe_type}` {{id: $id}}]->(b:Concept {{id: $target}}) "
                "SET r.properties = $props"
            )
            session.run(
                query_write,
                source=edge.source,
                target=edge.target,
                id=edge.id,
                props=new_props_json,
            )

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
        if not self.driver:
            return

        rank_table = rank_table or {
            "user": 1,
            "llm_verified": 2,
            "dictionary": 3,
            "system": 4,
        }

        safe_type = edge.type.replace(" ", "_").upper()

        with self.driver.session() as session:
            # Read existing
            query_read = f"MATCH (a:Concept {{id: $source}})-[r:`{safe_type}` {{id: $id}}]->(b:Concept {{id: $target}}) RETURN r.weight, r.properties"
            result = session.run(
                query_read, source=edge.source, target=edge.target, id=edge.id
            )
            record = result.single()

            if not record:
                logger.warning(
                    "[Graph Core Warning]: Could not find edge %s to update weight.",
                    edge.id,
                )
                return

            old_weight = record["r.weight"] or 0.5
            current_props = {}
            if record["r.properties"]:
                try:
                    current_props = json.loads(record["r.properties"])
                except Exception:
                    pass

            old_provenance = current_props.get("provenance", "user")

            rank_current = self.normalize_provenance_rank(rank_table, old_provenance)
            rank_new = self.normalize_provenance_rank(rank_table, provenance)

            adjusted_weight = self.compute_confidence_adjustment(
                old_weight,
                new_weight,
                rank_current,
                rank_new,
            )

            final_weight = round(max(0.0, min(1.0, adjusted_weight)), 4)

            current_props.update(
                {
                    "provenance": provenance,
                    "last_modified": time.time(),
                    "confidence_updated": True,
                }
            )

            new_props_json = json.dumps(current_props)

            # Write back
            query_write = (
                f"MATCH (a:Concept {{id: $source}})-[r:`{safe_type}` {{id: $id}}]->(b:Concept {{id: $target}}) "
                "SET r.weight = $weight, r.properties = $props"
            )
            session.run(
                query_write,
                source=edge.source,
                target=edge.target,
                id=edge.id,
                weight=final_weight,
                props=new_props_json,
            )

            logger.info(
                "[Graph Update]: Edge '%s' confidence updated "
                "%.2f -> %.2f "
                "(provenance: %s -> %s)",
                edge.type,
                old_weight,
                final_weight,
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
            existing_edge.properties["provenance"] = new_provenance
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
