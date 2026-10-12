"""
Embedding providers for generating semantic vector representations.
Supports SentenceTransformer models and deterministic mock providers for offline testing.
"""

import abc
import hashlib
import logging
import math
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(abc.ABC):
    """Abstract base class for text embedding models."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Returns the vector dimensionality produced by the embedding model."""
        pass

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Returns the name or identifier of the embedding model."""
        pass

    @abc.abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embeds a single text string into a vector."""
        pass

    @abc.abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of texts into normalized vectors."""
        pass


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """
    Local embedding provider utilizing compact HuggingFace sentence-transformers.
    Default model: 'all-MiniLM-L6-v2' (approx. 80-90MB download size, 384 dimensions).
    Fast, optimized for CPU inference on laptops.
    """

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self._model = None
        self._dimension = 384

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer embedding model: {self._model_name}")
                self._model = SentenceTransformer(self._model_name)
                # Infer actual dimension
                sample_vec = self._model.encode("hello", normalize_embeddings=True)
                self._dimension = int(sample_vec.shape[0])
            except ImportError as err:
                raise ImportError(
                    "sentence-transformers is not installed. Install it via "
                    "`pip install sentence-transformers`."
                ) from err
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load sentence-transformers model '{self._model_name}': {exc}"
                ) from exc

    @property
    def dimension(self) -> int:
        if self._model is None:
            return 384
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_text(self, text: str) -> List[float]:
        self._load_model()
        if not text or not text.strip():
            return [0.0] * self.dimension
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        if not texts:
            return []
        # Filter empty texts with zero vectors
        processed_texts = [t if (t and t.strip()) else "empty" for t in texts]
        vecs = self._model.encode(
            processed_texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            batch_size=64,
        )
        return vecs.tolist()


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic pseudo-embedding provider for unit tests and offline fixtures.
    Generates unit-norm vectors using MD5 hash seeds, requiring zero network or GPU.
    """

    def __init__(self, dimension: int = 16, model_name: str = "mock-embedding-16d"):
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _vector_from_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self._dimension

        vec: List[float] = []
        for i in range(self._dimension):
            h = hashlib.md5(f"{text}:{i}".encode("utf-8")).hexdigest()
            val = (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
            vec.append(val)

        # Normalize to unit length (L2 norm)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            return [v / norm for v in vec]
        return vec

    def embed_text(self, text: str) -> List[float]:
        return self._vector_from_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._vector_from_text(t) for t in texts]


def get_embedding_provider(
    provider_type: str = "sentence_transformer",
    model_name: Optional[str] = None,
) -> BaseEmbeddingProvider:
    """Factory helper to obtain the requested embedding provider."""
    if provider_type == "mock":
        return MockEmbeddingProvider(dimension=16)
    return SentenceTransformerProvider(model_name=model_name)
