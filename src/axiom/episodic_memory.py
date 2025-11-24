"""
Episodic Memory module for storing and retrieving conversation history.

This module provides long-term memory for Axiom by indexing conversations
using vector embeddings, enabling recall of previous interactions.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent

logger = logging.getLogger(__name__)


class Episode(TypedDict):
    """Represents a single conversation turn."""

    timestamp: str
    user_query: str
    agent_response: str
    key_entities: list[str]
    embedding: list[float]
    turn_id: int


class EpisodicMemory:
    """Manages long-term conversation memory using vector embeddings."""

    def __init__(self, agent: CognitiveAgent, memory_file: Path | None = None) -> None:
        """Initialize episodic memory with agent and storage."""
        self.agent = agent
        self.vector_store = agent.vector_store
        self.memory_file = memory_file or Path("data/episodic_memory.json")
        self.episodes: list[Episode] = []
        self.turn_counter = 0
        self._load_memory()

    def _load_memory(self) -> None:
        """Load existing episodic memory from disk."""
        if not self.memory_file.exists():
            logger.info("[Episodic Memory]: No existing memory file. Starting fresh.")
            return

        try:
            with self.memory_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                self.episodes = data.get("episodes", [])
                self.turn_counter = data.get("turn_counter", 0)
            logger.info(
                f"[Episodic Memory]: Loaded {len(self.episodes)} episodes from memory."
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"[Episodic Memory]: Failed to load memory: {e}")
            self.episodes = []
            self.turn_counter = 0

    def _save_memory(self) -> None:
        """Save episodic memory to disk."""
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            with self.memory_file.open("w", encoding="utf-8") as f:
                data = {"episodes": self.episodes, "turn_counter": self.turn_counter}
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.error(f"[Episodic Memory]: Failed to save memory: {e}")

    def index_conversation(
        self,
        user_input: str,
        agent_response: str,
        key_entities: list[str] | None = None,
    ) -> None:
        """
        Store a conversation turn with vector embedding.

        Args:
            user_input: The user's query
            agent_response: Axiom's response
            key_entities: Optional list of key entities discussed
        """
        if not self.vector_store.is_available:
            logger.warning(
                "[Episodic Memory]: VectorStore unavailable. Skipping indexing."
            )
            return

        # Create combined text for embedding
        combined_text = f"User: {user_input}\nAxiom: {agent_response}"
        embedding = self.vector_store.encode(combined_text)

        if not embedding:
            logger.warning("[Episodic Memory]: Failed to generate embedding.")
            return

        episode: Episode = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_input,
            "agent_response": agent_response,
            "key_entities": key_entities or [],
            "embedding": embedding,
            "turn_id": self.turn_counter,
        }

        self.episodes.append(episode)
        self.turn_counter += 1
        self._save_memory()

        logger.info(f"[Episodic Memory]: Indexed turn #{self.turn_counter}")

    def recall_similar(self, query: str, top_k: int = 3) -> list[Episode]:
        """
        Retrieve the most relevant past conversations for a query.

        Args:
            query: The search query
            top_k: Number of episodes to return

        Returns:
            List of most similar episodes
        """
        if not self.vector_store.is_available or not self.episodes:
            return []

        query_embedding = self.vector_store.encode(query)
        if not query_embedding:
            return []

        # Calculate similarity scores
        similarities: list[tuple[Episode, float]] = []
        for episode in self.episodes:
            similarity = self.vector_store.cosine_similarity(
                query_embedding, episode["embedding"]
            )
            similarities.append((episode, similarity))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = [ep for ep, _ in similarities[:top_k]]

        logger.info(f"[Episodic Memory]: Recalled {len(results)} similar episodes.")
        return results

    def get_recent_context(self, n: int = 5) -> list[Episode]:
        """
        Get the N most recent conversation turns.

        Args:
            n: Number of recent turns to retrieve

        Returns:
            List of recent episodes
        """
        return self.episodes[-n:] if self.episodes else []

    def clear_memory(self) -> None:
        """Clear all episodic memory (use with caution)."""
        self.episodes = []
        self.turn_counter = 0
        self._save_memory()
        logger.warning("[Episodic Memory]: Memory cleared!")
