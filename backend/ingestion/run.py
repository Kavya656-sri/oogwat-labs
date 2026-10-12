"""
CLI Entry Point for Transcript Ingestion Pipeline.

Can be run directly via:
    python -m backend.ingestion.run
"""

import argparse
import logging
import sys
from pathlib import Path

from backend.ingestion.ingest import run_ingestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ingestion.run")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest Lenny's Podcast transcripts and Newsletter posts into normalized chunks."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/transcripts"),
        help="Path to raw transcripts directory (default: data/transcripts)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to output processed data directory (default: data/processed)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Chunk size in characters (default: 1200)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters (default: 200)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Verify input directory exists; exit non-zero on fatal errors
    if not args.data_dir.exists():
        print(f"FATAL ERROR: Input dataset directory does not exist: {args.data_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    try:
        report = run_ingestion(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        print("\n" + "=" * 60)
        print("THE LENNY GROWTH ASSISTANT — INGESTION SUMMARY")
        print("=" * 60)
        print(f"Dataset Directory:       {report['dataset_directory']}")
        print(f"Output Directory:        {report['output_directory']}")
        print(f"Discovered Files:        {report['discovered_files_count']}")
        print(f"Processed Documents:     {report['processed_documents_count']}")
        print(f"  - Podcasts:            {report['total_podcasts']}")
        print(f"  - Newsletters:         {report['total_newsletters']}")
        print(f"Skipped Files:           {report['skipped_files_count']}")
        if report.get("skipped_files"):
            for skip in report["skipped_files"]:
                print(f"    * {skip['path']} ({skip['reason']})")
        print(f"Failed Files:            {report['failed_files_count']}")
        if report.get("failed_files"):
            for fail in report["failed_files"]:
                print(f"    ! {fail['path']} (Error: {fail['error']})")
        print(f"Total Chunks Generated:  {report['total_chunks']}")
        print(f"Avg Chunk Characters:    {report['avg_chunk_char_count']}")
        print(f"Avg Chunk Words:         {report['avg_chunk_word_count']}")
        print(f"Chunk Config:            size={report['chunk_size']}, overlap={report['chunk_overlap']}")
        print("=" * 60)
        print("Generated Artifacts:")
        print(f"  - {args.output_dir / 'documents.jsonl'}")
        print(f"  - {args.output_dir / 'chunks.jsonl'}")
        print(f"  - {args.output_dir / 'ingestion_report.json'}")
        print("=" * 60 + "\n")

    except ValueError as val_err:
        print(f"CONFIGURATION ERROR: {val_err}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"FATAL INGESTION ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
