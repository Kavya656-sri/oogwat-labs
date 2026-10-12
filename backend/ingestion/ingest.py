"""
Command-line ingestion script to process the transcript dataset.
Reads raw files read-only and writes normalized documents and chunks to data/processed/.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from backend.ingestion.chunker import chunk_document, validate_chunk_settings
from backend.ingestion.models import TranscriptChunk, TranscriptDocument
from backend.ingestion.parser import load_dataset, load_dataset_with_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ingestion")


def run_ingestion(
    data_dir: Path,
    output_dir: Path,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> Dict:
    """
    Executes the ingestion pipeline:
    1. Validates chunk settings.
    2. Loads and normalizes transcripts with error tracking.
    3. Chunks documents into overlapping segments.
    4. Saves documents.jsonl, chunks.jsonl, and ingestion_report.json to output_dir.
    5. Returns ingestion statistics.
    """
    validate_chunk_settings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    logger.info(f"Starting transcript ingestion from: {data_dir.resolve()}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {data_dir}")

    # 1. Load documents with report
    docs, load_stats = load_dataset_with_report(data_dir)
    if not docs:
        logger.warning("No valid documents found during ingestion.")

    # Count source types
    podcast_count = sum(1 for d in docs if d.source_type == "podcast")
    newsletter_count = sum(1 for d in docs if d.source_type == "newsletter")

    # 2. Chunk documents
    logger.info(
        f"Splitting {len(docs)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})..."
    )
    all_chunks: List[TranscriptChunk] = []
    for doc in docs:
        doc_chunks = chunk_document(
            doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        all_chunks.extend(doc_chunks)

    # 3. Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Save JSONL outputs
    documents_jsonl = output_dir / "documents.jsonl"
    chunks_jsonl = output_dir / "chunks.jsonl"
    report_file = output_dir / "ingestion_report.json"

    logger.info(f"Writing {len(docs)} documents to {documents_jsonl}...")
    with open(documents_jsonl, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.model_dump_json() + "\n")

    logger.info(f"Writing {len(all_chunks)} chunks to {chunks_jsonl}...")
    with open(chunks_jsonl, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.model_dump_json() + "\n")

    # Also keep legacy JSON formats for backwards compatibility
    chunks_file = output_dir / "chunks.json"
    docs_file = output_dir / "documents.json"
    manifest_file = output_dir / "manifest.json"

    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in all_chunks], f, indent=2, ensure_ascii=False)

    doc_summaries = [
        {
            "id": d.id,
            "title": d.title,
            "source_type": d.source_type,
            "filename": d.filename,
            "date": d.date,
            "guest": d.guest,
            "word_count": d.word_count,
            "source_url": d.source_url,
        }
        for d in docs
    ]
    with open(docs_file, "w", encoding="utf-8") as f:
        json.dump(doc_summaries, f, indent=2, ensure_ascii=False)

    avg_chunk_chars = (
        sum(c.char_count for c in all_chunks) // len(all_chunks) if all_chunks else 0
    )
    avg_chunk_words = (
        sum(c.word_count for c in all_chunks) // len(all_chunks) if all_chunks else 0
    )

    report = {
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_directory": str(data_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "discovered_files_count": len(load_stats.get("discovered_files", [])),
        "processed_documents_count": len(docs),
        "total_documents": len(docs),  # backward compatibility
        "total_podcasts": podcast_count,
        "total_newsletters": newsletter_count,
        "total_chunks": len(all_chunks),
        "skipped_files_count": len(load_stats.get("skipped_files", [])),
        "failed_files_count": len(load_stats.get("failed_files", [])),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "avg_chunk_char_count": avg_chunk_chars,
        "avg_chunk_word_count": avg_chunk_words,
        "discovered_files": load_stats.get("discovered_files", []),
        "skipped_files": load_stats.get("skipped_files", []),
        "failed_files": load_stats.get("failed_files", []),
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Ingestion completed successfully!")
    logger.info(f"Ingestion report saved to: {report_file}")
    return report



def main():
    parser = argparse.ArgumentParser(
        description="Ingest Lenny transcripts and generate overlapping citation-ready chunks."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/transcripts"),
        help="Path to raw transcript dataset (default: data/transcripts)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to store processed chunks (default: data/processed)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Target chunk size in characters (default: 1200)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Target chunk overlap in characters (default: 200)",
    )

    args = parser.parse_args()

    try:
        manifest = run_ingestion(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print("\n" + "=" * 50)
        print("INGESTION SUMMARY")
        print("=" * 50)
        print(f"Total Documents:    {manifest['total_documents']}")
        print(f"  - Podcasts:       {manifest['total_podcasts']}")
        print(f"  - Newsletters:    {manifest['total_newsletters']}")
        print(f"Total Chunks:       {manifest['total_chunks']}")
        print(f"Avg Chunk Words:    {manifest['avg_chunk_word_count']}")
        print(f"Output Directory:   {manifest['output_directory']}")
        print("=" * 50)
    except Exception as exc:
        logger.error(f"Ingestion failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
