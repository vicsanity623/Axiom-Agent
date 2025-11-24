"""
Dream Cycle module for autonomous graph optimization.

This module implements self-optimization routines that run during idle time
to consolidate knowledge, prune weak nodes, and generate new learning hypotheses.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent
    from .graph_core import ConceptNode

logger = logging.getLogger(__name__)


class DreamCycle:
    """Manages autonomous graph optimization during idle time ('dreaming')."""

    def __init__(self, agent: CognitiveAgent) -> None:
        """Initialize dream cycle with agent reference."""
        self.agent = agent
        self.similarity_threshold = 0.9  # For duplicate detection
        self.weak_confidence_threshold = 0.3  # For pruning
        self.sparse_edge_threshold = 2  # Nodes with fewer edges are "sparse"

    def consolidate_duplicates(self) -> int:
        """
        Merge highly similar nodes to reduce redundancy.

        Returns:
            Number of nodes merged
        """
        if not self.agent.vector_store.is_available:
            logger.warning(
                "[Dream Cycle]: VectorStore unavailable. Skipping consolidation."
            )
            return 0

        nodes = list(self.agent.graph.get_all_nodes())
        merged_count = 0
        processed_ids = set()

        for i, node_a in enumerate(nodes):
            if node_a.id in processed_ids:
                continue

            # Get embedding for node_a
            embedding_a = self.agent.vector_store.encode(node_a.name)
            if not embedding_a:
                continue

            for node_b in nodes[i + 1 :]:
                if node_b.id in processed_ids:
                    continue

                embedding_b = self.agent.vector_store.encode(node_b.name)
                if not embedding_b:
                    continue

                similarity = self.agent.vector_store.cosine_similarity(
                    embedding_a, embedding_b
                )

                if similarity > self.similarity_threshold:
                    # Merge node_b into node_a
                    logger.info(
                        f"[Dream Cycle]: Merging '{node_b.name}' into '{node_a.name}' (similarity: {similarity:.2f})"
                    )

                    # Transfer all edges from node_b to node_a
                    edges_b = self.agent.graph.get_edges_from_node(node_b.id)
                    for edge in edges_b:
                        # Create equivalent edge from node_a
                        target_node = self.agent.graph.get_node_by_id(edge.target)
                        if target_node and target_node.id != node_a.id:
                            self.agent.graph.add_edge(
                                node_a,
                                target_node,
                                edge.type,
                                weight=edge.weight,
                                properties=edge.properties,
                            )

                    # Remove node_b
                    self.agent.graph.remove_node(node_b.id)
                    processed_ids.add(node_b.id)
                    merged_count += 1

        logger.info(f"[Dream Cycle]: Consolidated {merged_count} duplicate nodes.")
        return merged_count

    def prune_weak_nodes(self) -> int:
        """
        Remove isolated nodes with low confidence.

        Returns:
            Number of nodes pruned
        """
        nodes = list(self.agent.graph.get_all_nodes())
        pruned_count = 0

        for node in nodes:
            # Check if node is isolated (no edges)
            edges = self.agent.graph.get_edges_from_node(node.id)
            if len(edges) > 0:
                continue

            # Check if node has low confidence
            confidence = node.properties.get("confidence", 1.0)
            if confidence < self.weak_confidence_threshold:
                logger.info(
                    f"[Dream Cycle]: Pruning weak isolated node: '{node.name}' (confidence: {confidence:.2f})"
                )
                self.agent.graph.remove_node(node.id)
                pruned_count += 1

        logger.info(f"[Dream Cycle]: Pruned {pruned_count} weak nodes.")
        return pruned_count

    def hypothesize_tasks(self) -> int:
        """
        Generate learning goals for sparse areas of the graph.

        Returns:
            Number of tasks generated
        """
        nodes = list(self.agent.graph.get_all_nodes())
        sparse_nodes: list[ConceptNode] = []

        for node in nodes:
            edges = self.agent.graph.get_edges_from_node(node.id)
            if len(edges) < self.sparse_edge_threshold:
                sparse_nodes.append(node)

        # Sample up to 10 sparse nodes to avoid overwhelming the queue
        import random

        sampled_nodes = random.sample(sparse_nodes, min(10, len(sparse_nodes)))

        tasks_generated = 0
        for node in sampled_nodes:
            task = f"INVESTIGATE: {node.name}"
            if task not in self.agent.learning_goals:
                self.agent.learning_goals.append(task)
                logger.info(
                    f"[Dream Cycle]: Generated learning task for sparse node: '{node.name}'"
                )
                tasks_generated += 1

        logger.info(f"[Dream Cycle]: Generated {tasks_generated} new learning tasks.")
        return tasks_generated

    def run_dream_phase(self) -> bool:
        """
        Run the complete dream cycle: consolidate, prune, hypothesize.

        Returns:
            True if any changes were made to the graph
        """
        logger.info("[Dream Cycle]: Starting dream phase...")

        merged = self.consolidate_duplicates()
        pruned = self.prune_weak_nodes()
        tasks = self.hypothesize_tasks()

        progress_made = merged > 0 or pruned > 0 or tasks > 0

        if progress_made:
            self.agent.save_brain()
            logger.info(
                f"[Dream Cycle]: Dream phase complete. Merged={merged}, Pruned={pruned}, Tasks={tasks}"
            )
        else:
            logger.info("[Dream Cycle]: No optimization needed.")

        return progress_made
