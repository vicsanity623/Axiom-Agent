from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, cast

import pytest

from axiom.graph_core import (
    ConceptGraph,
    ConceptNode,
    ConceptNodeData,
    RelationshipEdge,
)

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.logging import LogCaptureFixture

    from axiom.universal_interpreter import PropertyData


# Fixtures
@pytest.fixture
def graph() -> ConceptGraph:
    """Given: An empty ConceptGraph instance."""
    return ConceptGraph()


@pytest.fixture
def node_a() -> ConceptNode:
    """Given: A sample ConceptNode 'a'."""
    return ConceptNode(name="a", node_type="letter")


@pytest.fixture
def node_b() -> ConceptNode:
    """Given: A sample ConceptNode 'b'."""
    return ConceptNode(name="b", node_type="letter")


@pytest.fixture
def node_c() -> ConceptNode:
    """Given: A sample ConceptNode 'c'."""
    return ConceptNode(name="c", node_type="thing")


@pytest.fixture
def populated_graph(
    graph: ConceptGraph,
    node_a: ConceptNode,
    node_b: ConceptNode,
    node_c: ConceptNode,
) -> ConceptGraph:
    """Given: A ConceptGraph populated with three nodes and two edges."""
    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_c)
    graph.add_edge(node_a, node_b, "is_before")
    graph.add_edge(node_b, node_c, "is_before")
    return graph


class TestConceptNode:
    """Tests for the ConceptNode class."""

    def test_initialization(self) -> None:
        """
        Given: A name and type for a new concept.
        When: A ConceptNode is initialized.
        Then: It has the correct attributes and a valid UUID.
        """
        props = cast("PropertyData", {"source": "test"})
        node = ConceptNode(name="test", node_type="noun", properties=props)
        assert node.name == "test"
        assert node.type == "noun"
        assert node.properties == {"source": "test"}
        assert isinstance(uuid.UUID(node.id), uuid.UUID)

    def test_serialization_deserialization(self) -> None:
        """
        Given: A ConceptNode instance.
        When: It is serialized to a dict and deserialized back.
        Then: The resulting instance has the same data.
        """
        props = cast("PropertyData", {"key": "value"})
        original_node = ConceptNode(
            name="original",
            node_type="test",
            properties=props,
        )
        node_dict = original_node.to_dict()
        rehydrated_node = ConceptNode.from_dict(node_dict)

        assert original_node.id == rehydrated_node.id
        assert original_node.name == rehydrated_node.name
        assert original_node.type == rehydrated_node.type
        assert original_node.properties == rehydrated_node.properties

    def test_from_dict_with_missing_optional_fields(self) -> None:
        """
        Given: A dictionary for a node missing optional fields.
        When: A ConceptNode is created from the dictionary.
        Then: The node is created with default values for the missing fields.
        """
        data: ConceptNodeData = {
            "id": str(uuid.uuid4()),
            "name": "minimal",
            "type": "concept",
            "properties": None,
            "value": 0.5,
            "activation": 0.0,
        }
        node = ConceptNode.from_dict(data)
        assert node.properties == {}
        assert node.value == 0.5
        assert node.activation == 0.0


class TestRelationshipEdge:
    """Tests for the RelationshipEdge class."""

    def test_initialization(self) -> None:
        """
        Given: Source, target, and type for a new relationship.
        When: A RelationshipEdge is initialized.
        Then: It has the correct attributes and a valid UUID.
        """
        props = cast("PropertyData", {"source": "test"})
        edge = RelationshipEdge(
            source="id1",
            target="id2",
            type="connects_to",
            properties=props,
        )
        assert edge.source == "id1"
        assert edge.target == "id2"
        assert edge.type == "connects_to"
        assert edge.properties == {"source": "test"}
        assert isinstance(uuid.UUID(edge.id), uuid.UUID)

    def test_serialization_deserialization(self) -> None:
        """
        Given: A RelationshipEdge instance.
        When: It is serialized to a dict and deserialized back.
        Then: The resulting instance has the same data.
        """
        props = cast("PropertyData", {"key": "value"})
        original_edge = RelationshipEdge(
            source="id1",
            target="id2",
            type="is_a",
            properties=props,
        )
        edge_dict = original_edge.to_dict()
        rehydrated_edge = RelationshipEdge.from_dict(edge_dict)

        assert original_edge.id == rehydrated_edge.id
        assert original_edge.source == rehydrated_edge.source
        assert original_edge.target == rehydrated_edge.target
        assert original_edge.type == rehydrated_edge.type
        assert original_edge.properties == rehydrated_edge.properties


class TestConceptGraph:
    """Tests for the ConceptGraph class."""

    def test_add_and_get_node(self, graph: ConceptGraph, node_a: ConceptNode) -> None:
        """
        Given: An empty graph and a new node.
        When: The node is added to the graph.
        Then: It can be retrieved by name and by ID.
        """
        graph.add_node(node_a)
        retrieved_by_name = graph.get_node_by_name("a")
        assert retrieved_by_name is not None
        assert retrieved_by_name.id == node_a.id

        retrieved_by_id = graph.get_node_by_id(node_a.id)
        assert retrieved_by_id is not None
        assert retrieved_by_id.name == "a"

    def test_add_node_idempotency(
        self,
        graph: ConceptGraph,
        node_a: ConceptNode,
    ) -> None:
        """
        Given: A graph with an existing node.
        When: A new node with the same name is added.
        Then: The original node is returned and no new node is created.
        """
        graph.add_node(node_a)
        new_node_with_same_name = ConceptNode(name="a")
        returned_node = graph.add_node(new_node_with_same_name)

        assert len(graph.graph.nodes) == 1
        assert returned_node.id == node_a.id

    def test_get_nonexistent_node(self, graph: ConceptGraph) -> None:
        """
        Given: An empty graph.
        When: Getting a node by a name or ID that does not exist.
        Then: None is returned.
        """
        assert graph.get_node_by_name("nonexistent") is None
        assert graph.get_node_by_id(str(uuid.uuid4())) is None

    def test_get_nodes_by_type(self, populated_graph: ConceptGraph) -> None:
        """
        Given: A populated graph.
        When: Getting nodes by type.
        Then: The correct list of nodes is returned.
        """
        letter_nodes = populated_graph.get_nodes_by_type("letter")
        assert len(letter_nodes) == 2
        assert {"a", "b"} == {node.name for node in letter_nodes}
        assert populated_graph.get_nodes_by_type("nonexistent_type") == []

    def test_get_all_node_names(self, populated_graph: ConceptGraph) -> None:
        """
        Given: A populated graph.
        When: Getting all node names.
        Then: A list of all names is returned.
        """
        names = populated_graph.get_all_node_names()
        assert sorted(names) == ["a", "b", "c"]

    def test_add_and_get_edge(
        self,
        graph: ConceptGraph,
        node_a: ConceptNode,
        node_b: ConceptNode,
    ) -> None:
        """
        Given: A graph with two nodes.
        When: An edge is added between them.
        Then: The edge can be retrieved from the source and to the target.
        """
        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_edge(node_a, node_b, "connects_to")

        out_edges = graph.get_edges_from_node(node_a.id)
        assert len(out_edges) == 1
        assert out_edges[0].type == "connects_to"
        assert out_edges[0].target == node_b.id

        in_edges = graph.get_edges_to_node(node_b.id)
        assert len(in_edges) == 1
        assert in_edges[0].type == "connects_to"
        assert in_edges[0].source == node_a.id

    def test_add_edge_updates_existing(
        self,
        graph: ConceptGraph,
        node_a: ConceptNode,
        node_b: ConceptNode,
    ) -> None:
        """
        Given: A graph with an existing edge.
        When: An edge with the same source, target, and type is added.
        Then: The existing edge is updated, not duplicated.
        """
        graph.add_node(node_a)
        graph.add_node(node_b)
        props1 = cast("PropertyData", {"a": 1})
        props2 = cast("PropertyData", {"b": 2})
        graph.add_edge(node_a, node_b, "is_a", weight=0.5, properties=props1)
        graph.add_edge(node_a, node_b, "is_a", weight=0.8, properties=props2)

        edges = graph.get_edges_from_node(node_a.id)
        assert len(edges) == 1
        assert edges[0].weight == 0.8  # Weight should be max(0.5, 0.8)
        assert edges[0].properties == {"a": 1, "b": 2}  # Properties should merge

    def test_get_all_edges(self, populated_graph: ConceptGraph) -> None:
        """
        Given: A populated graph.
        When: Getting all edges.
        Then: A list of all edges is returned.
        """
        all_edges = populated_graph.get_all_edges()
        assert len(all_edges) == 2
        assert {edge.type for edge in all_edges} == {"is_before"}

    def test_decay_activations(self, graph: ConceptGraph, node_a: ConceptNode) -> None:
        """
        Given: A node with a positive activation level.
        When: decay_activations is called.
        Then: The node's activation level is reduced but not below zero.
        """
        node_a.activation = 0.5
        graph.add_node(node_a)
        graph.decay_activations(decay_rate=0.2)
        retrieved_node = graph.get_node_by_id(node_a.id)
        assert retrieved_node is not None
        assert retrieved_node.activation == pytest.approx(0.3)

        graph.decay_activations(decay_rate=0.5)
        retrieved_node = graph.get_node_by_id(node_a.id)
        assert retrieved_node is not None
        assert retrieved_node.activation == 0.0

    def test_save_and_load_file(
        self,
        populated_graph: ConceptGraph,
        tmp_path: Path,
    ) -> None:
        """
        Given: A populated graph.
        When: The graph is saved to a file and loaded back.
        Then: The loaded graph is identical to the original.
        """
        file_path = tmp_path / "test_brain.json"
        populated_graph.save_to_file(file_path)
        loaded_graph = ConceptGraph.load_from_file(file_path)

        assert len(loaded_graph.graph.nodes) == len(populated_graph.graph.nodes)
        assert len(loaded_graph.graph.edges) == len(populated_graph.graph.edges)
        assert sorted(loaded_graph.get_all_node_names()) == ["a", "b", "c"]

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """
        Given: A path to a file that does not exist.
        When: load_from_file is called.
        Then: An empty graph is created and a log message is emitted.
        """
        file_path = tmp_path / "nonexistent.json"
        graph = ConceptGraph.load_from_file(file_path)
        assert isinstance(graph, ConceptGraph)
        assert len(graph.graph.nodes) == 0

    def test_load_corrupt_file(
        self,
        tmp_path: Path,
        caplog: LogCaptureFixture,
    ) -> None:
        """
        Given: A path to a corrupt JSON file.
        When: load_from_file is called.
        Then: An empty graph is created and an error is logged.
        """
        file_path = tmp_path / "corrupt.json"
        file_path.write_text("{not json}", encoding="utf-8")
        graph = ConceptGraph.load_from_file(file_path)

        assert isinstance(graph, ConceptGraph)
        assert len(graph.graph.nodes) == 0
        assert "Error loading brain" in caplog.text

    def test_find_exclusive_conflict(self, populated_graph: ConceptGraph) -> None:
        """
        Given: A graph with multiple edges of the same exclusive type.
        When: find_exclusive_conflict is called.
        Then: The edge with the highest weight is returned.
        """
        node_a = populated_graph.get_node_by_name("a")
        node_b = populated_graph.get_node_by_name("b")
        node_c = populated_graph.get_node_by_name("c")
        assert node_a is not None
        assert node_b is not None
        assert node_c is not None

        populated_graph.add_edge(node_a, node_b, "has_name", weight=0.7)
        populated_graph.add_edge(node_a, node_c, "has_name", weight=0.9)

        conflict = populated_graph.find_exclusive_conflict(node_a, "has_name")
        assert conflict is not None
        assert conflict.target == node_c.id
        assert conflict.weight == 0.9

    def test_revise_conflicting_edge(self, graph: ConceptGraph) -> None:
        """
        Given: An existing edge.
        When: revise_conflicting_edge is called with different provenances.
        Then: The correct revision status is returned.
        """
        node1 = graph.add_node(ConceptNode("node1"))
        node2 = graph.add_node(ConceptNode("node2"))
        props = cast("PropertyData", {"provenance": "user"})
        edge = graph.add_edge(
            node1,
            node2,
            "is_a",
            properties=props,
        )
        assert edge is not None

        # New provenance is higher rank
        status = graph.revise_conflicting_edge(edge, 0.9, "system")
        assert status == "replaced"

        # New provenance is same rank
        status = graph.revise_conflicting_edge(edge, 0.8, "system")
        assert status == "merged"

        # New provenance is lower rank
        status = graph.revise_conflicting_edge(edge, 0.7, "user")
        assert status == "ignored"
