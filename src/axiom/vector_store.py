"""
VectorStore module for managing embeddings and semantic search.
This module wraps the sentence-transformers library to provide
vector encoding and similarity calculation for the Neuro-Symbolic architecture.
"""

from __future__ import annotations

import logging
from typing import Any, Self, cast

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)


class VectorStore:
    """Singleton-like manager for the embedding model."""

    _instance: VectorStore | None = None
    _model: Any | None = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_model()
        # Fix 1: Explicitly cast the instance to Self to satisfy MyPy
        return cast("Self", cls._instance)

    def _initialize_model(self) -> None:
        """Initialize the sentence transformer model."""
        if not EMBEDDINGS_AVAILABLE:
            logger.warning(
                "sentence-transformers not installed. Vector capabilities disabled."
            )
            self._model = None
            return

        try:
            logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._model = None

    def encode(self, text: str) -> list[float] | None:
        """Encode a text string into a vector embedding."""
        if not self._model:
            return None
        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            # Fix 2: Cast the 'Any' return type to the specific list[float]
            return cast("list[float]", embedding.tolist())
        except Exception as e:
            logger.error(f"Error encoding text '{text}': {e}")
            return None

    def cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec_a or not vec_b:
            return 0.0

        a = np.array(vec_a)
        b = np.array(vec_b)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    @property
    def is_available(self) -> bool:
        """Check if the embedding model is available."""
        return self._model is not None
