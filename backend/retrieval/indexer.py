"""
Local vector index for storage, loading, and semantic retrieval of transcript chunks.
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.retrieval.embeddings import BaseEmbeddingProvider
from backend.retrieval.models import SearchResult

logger = logging.getLogger(__name__)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class LocalVectorIndex:
    """
    In-memory and file-persisted vector store.
    Stores dense vectors alongside chunk metadata for fast top-k retrieval.
    Designed for straightforward migration to PostgreSQL/pgvector in future stages.
    """

    def __init__(self):
        self.embeddings: List[List[float]] = []
        self.chunks_meta: List[Dict[str, Any]] = []
        self.model_name: Optional[str] = None
        self.dimension: Optional[int] = None
        self.built_at: Optional[str] = None

    @property
    def total_chunks(self) -> int:
        return len(self.chunks_meta)

    def is_empty(self) -> bool:
        return len(self.chunks_meta) == 0

    def build_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embedding_provider: BaseEmbeddingProvider,
    ) -> None:
        """
        Embeds a list of chunk dictionaries and populates the index.
        """
        if not chunks:
            logger.warning("Empty chunks provided to build_from_chunks.")
            self.embeddings = []
            self.chunks_meta = []
            self.model_name = embedding_provider.model_name
            self.dimension = embedding_provider.dimension
            self.built_at = datetime.now(timezone.utc).isoformat()
            return

        logger.info(
            f"Embedding {len(chunks)} chunks with model '{embedding_provider.model_name}' "
            f"(dim={embedding_provider.dimension})..."
        )
        texts = [c.get("content", "") for c in chunks]
        vectors = embedding_provider.embed_batch(texts)

        self.embeddings = vectors
        self.chunks_meta = chunks
        self.model_name = embedding_provider.model_name
        self.dimension = embedding_provider.dimension
        self.built_at = datetime.now(timezone.utc).isoformat()
        logger.info(f"Successfully indexed {len(self.embeddings)} chunks.")

    def save(self, index_dir: Path) -> None:
        """Persists the vector index and metadata to disk."""
        index_dir.mkdir(parents=True, exist_ok=True)

        meta_file = index_dir / "index_meta.json"
        embeddings_file = index_dir / "index_vectors.json"
        chunks_file = index_dir / "index_chunks.jsonl"

        meta_payload = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "total_chunks": self.total_chunks,
            "built_at": self.built_at,
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

        with open(embeddings_file, "w", encoding="utf-8") as f:
            json.dump(self.embeddings, f)

        with open(chunks_file, "w", encoding="utf-8") as f:
            for item in self.chunks_meta:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Index successfully saved to {index_dir}")

    def load(self, index_dir: Path) -> None:
        """Loads the persisted index and metadata from disk."""
        meta_file = index_dir / "index_meta.json"
        embeddings_file = index_dir / "index_vectors.json"
        chunks_file = index_dir / "index_chunks.jsonl"

        if not meta_file.exists() or not embeddings_file.exists() or not chunks_file.exists():
            raise FileNotFoundError(
                f"Index files missing in {index_dir}. Required: index_meta.json, "
                "index_vectors.json, index_chunks.jsonl."
            )

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.model_name = meta.get("model_name")
            self.dimension = meta.get("dimension")
            self.built_at = meta.get("built_at")

            with open(embeddings_file, "r", encoding="utf-8") as f:
                self.embeddings = json.load(f)

            self.chunks_meta = []
            with open(chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.chunks_meta.append(json.loads(line))

            if len(self.embeddings) != len(self.chunks_meta):
                raise ValueError(
                    f"Index corruption: {len(self.embeddings)} vectors but {len(self.chunks_meta)} metadata records."
                )

            logger.info(
                f"Loaded index from {index_dir}: {len(self.chunks_meta)} chunks, "
                f"model '{self.model_name}', dim={self.dimension}."
            )
        except Exception as exc:
            logger.error(f"Failed to load index from {index_dir}: {exc}")
            raise

    def search(
        self,
        query: str,
        embedding_provider: BaseEmbeddingProvider,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Executes semantic search against the index using cosine similarity.
        Returns ranked list of SearchResult objects with traceable source metadata.
        """
        if self.is_empty():
            logger.warning("Search called on an empty index.")
            return []

        if not query or not query.strip():
            logger.warning("Empty query passed to search.")
            return []

        # Validate model compatibility if known
        if self.model_name and embedding_provider.model_name != self.model_name:
            logger.warning(
                f"Query model '{embedding_provider.model_name}' differs from index model '{self.model_name}'. "
                "Results may be inaccurate."
            )

        query_vector = embedding_provider.embed_text(query)

        # Check dimension match
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query embedding dimension ({len(query_vector)}) does not match "
                f"index dimension ({self.dimension})."
            )

        scored_indices: List[Tuple[int, float]] = []
        try:
            import numpy as np

            matrix = np.array(self.embeddings, dtype=np.float32)
            q = np.array(query_vector, dtype=np.float32)

            scores = np.dot(matrix, q)
            scores = np.clip(scores, -1.0, 1.0)

            for idx, score in enumerate(scores):
                s = float(score)
                if min_score is None or s >= min_score:
                    scored_indices.append((idx, s))
        except ImportError:
            for idx, doc_vec in enumerate(self.embeddings):
                sim = cosine_similarity(query_vector, doc_vec)
                if min_score is None or sim >= min_score:
                    scored_indices.append((idx, sim))


        # Sort descending by similarity score
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_indices[:top_k]

        results: List[SearchResult] = []
        for idx, score in top_results:
            meta = self.chunks_meta[idx]
            result = SearchResult(
                chunk_id=meta.get("chunk_id", f"chunk_{idx}"),
                document_id=meta.get("document_id", "unknown_doc"),
                title=meta.get("title", "Untitled"),
                source_type=meta.get("source_type", "unknown"),
                filename=meta.get("filename"),
                source_url=meta.get("source_url"),
                date=meta.get("date"),
                guest=meta.get("guest"),
                content=meta.get("content", ""),
                score=round(score, 4),
            )
            results.append(result)

        return results
