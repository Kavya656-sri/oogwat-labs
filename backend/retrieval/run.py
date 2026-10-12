"""
CLI Diagnostic Entry Point for Retrieval Engine.

Can be run directly via:
    python -m backend.retrieval.run --query "How do startups find product-market fit?" --top-k 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from backend.retrieval.embeddings import (
    BaseEmbeddingProvider,
    SentenceTransformerProvider,
    get_embedding_provider,
)
from backend.retrieval.indexer import LocalVectorIndex
from backend.retrieval.models import SearchResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("retrieval.run")


def load_chunks_from_jsonl(chunks_path: Path) -> List[dict]:
    """Reads processed chunk metadata from chunks.jsonl."""
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Processed chunks file not found at: {chunks_path.resolve()}. "
            "Please run the ingestion pipeline first via `python -m backend.ingestion.run`."
        )

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def get_or_build_index(
    index_dir: Path,
    chunks_path: Path,
    embedding_provider: BaseEmbeddingProvider,
    force_rebuild: bool = False,
) -> LocalVectorIndex:
    """Loads existing index if available; otherwise builds from processed chunks."""
    index = LocalVectorIndex()
    index_meta_file = index_dir / "index_meta.json"

    if index_meta_file.exists() and not force_rebuild:
        logger.info(f"Loading cached vector index from {index_dir.resolve()}...")
        try:
            index.load(index_dir)
            return index
        except Exception as exc:
            logger.warning(f"Could not load cached index ({exc}). Rebuilding fresh index...")

    chunks = load_chunks_from_jsonl(chunks_path)
    logger.info(f"Building vector index for {len(chunks)} chunks...")
    index.build_from_chunks(chunks, embedding_provider)
    index.save(index_dir)
    return index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search transcript chunks using semantic vector retrieval."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Search query or question (e.g. 'How do startups find product-market fit?')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum similarity threshold between -1.0 and 1.0",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="Path to chunks.jsonl file (default: data/processed/chunks.jsonl)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/processed/index"),
        help="Directory to store/load index (default: data/processed/index)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model identifier (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild index even if cached files exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        embedding_provider = get_embedding_provider(
            provider_type="sentence_transformer",
            model_name=args.model,
        )

        index = get_or_build_index(
            index_dir=args.index_dir,
            chunks_path=args.chunks_path,
            embedding_provider=embedding_provider,
            force_rebuild=args.rebuild,
        )

        if not args.query:
            print(f"\nIndex is ready with {index.total_chunks} chunks.")
            print(f"Model: {index.model_name} (dim={index.dimension})")
            print("Specify --query \"Your question here\" to search.")
            return

        print("\n" + "=" * 65)
        print(f"QUERY: \"{args.query}\"")
        print(f"Model: {index.model_name} | Top {args.top_k} results")
        print("=" * 65)

        results: List[SearchResult] = index.search(
            query=args.query,
            embedding_provider=embedding_provider,
            top_k=args.top_k,
            min_score=args.min_score,
        )

        if not results:
            print("No matching transcript passages found.")
            return

        for rank, res in enumerate(results, 1):
            print(f"\n[Result #{rank}] Relevance Score: {res.score:.4f}")
            print(f"Title:       {res.title}")
            print(f"Source Type: {res.source_type.capitalize()}")
            if res.guest:
                print(f"Guest:       {res.guest}")
            if res.date:
                print(f"Date:        {res.date}")
            if res.source_url:
                print(f"Source URL:  {res.source_url}")
            if res.filename:
                print(f"File:        {res.filename}")
            print(f"Chunk ID:    {res.chunk_id}")
            print("-" * 65)
            # Display snippet (first 300 chars if long)
            snippet = res.content.strip()
            if len(snippet) > 350:
                snippet = snippet[:350] + "..."
            print(snippet)

        print("\n" + "=" * 65 + "\n")

    except FileNotFoundError as fnf:
        print(f"ERROR: {fnf}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"RETRIEVAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
