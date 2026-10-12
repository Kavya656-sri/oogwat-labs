"""
Parser and normalizer for transcript documents and metadata.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from backend.ingestion.models import TranscriptDocument

logger = logging.getLogger(__name__)


def parse_frontmatter_and_content(file_path: Path) -> Tuple[Dict, str]:
    """
    Reads a markdown file and separates YAML frontmatter from body content.
    Gracefully handles empty files, missing frontmatter, and malformed YAML.
    """
    try:
        raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning(f"Could not read file {file_path}: {exc}")
        return {}, ""

    if not raw_text.strip():
        return {}, ""

    # Check for YAML frontmatter block starting at the beginning of the file
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw_text, re.DOTALL)
    if frontmatter_match:
        yaml_content = frontmatter_match.group(1)
        body_content = frontmatter_match.group(2)
        try:
            metadata = yaml.safe_load(yaml_content) or {}
            if not isinstance(metadata, dict):
                metadata = {}
        except yaml.YAMLError as exc:
            logger.warning(f"Malformed YAML frontmatter in {file_path}: {exc}")
            metadata = {}
            body_content = raw_text
        return metadata, body_content

    return {}, raw_text


def normalize_content(text: str) -> str:
    """
    Normalizes transcript text by standardizing line breaks,
    collapsing excessive blank lines, and stripping surrounding whitespace.
    """
    if not text:
        return ""

    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse 3 or more consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def find_dataset_root(base_dir: Path) -> Path:
    """
    Finds the directory containing transcripts and index.json.
    Works whether pointed at data/transcripts/ or data/transcripts/lennys-newsletterpodcastdata-main.
    """
    if (base_dir / "index.json").exists():
        return base_dir

    # Look for subdirectories containing index.json
    for child in base_dir.iterdir():
        if child.is_dir() and (child / "index.json").exists():
            return child

    return base_dir


def load_dataset_with_report(data_dir: Path) -> Tuple[List[TranscriptDocument], Dict[str, Any]]:
    """
    Scans the dataset directory, reads markdown files, normalizes content,
    and returns both valid TranscriptDocument objects and a comprehensive
    statistics report tracking discovered, processed, skipped, and failed files.
    Original files are strictly read-only.
    """
    root_dir = find_dataset_root(data_dir)
    index_file = root_dir / "index.json"
    index_data: Dict[str, Dict] = {}

    # Load index.json metadata if available
    if index_file.exists():
        try:
            with open(index_file, "r", encoding="utf-8", errors="replace") as f:
                raw_index = json.load(f)

            # Map relative normalized filenames to their index metadata
            for source_cat in ["podcasts", "newsletters"]:
                for item in raw_index.get(source_cat, []):
                    rel_name = item.get("filename", "").replace("\\", "/")
                    if rel_name:
                        index_data[rel_name] = {
                            **item,
                            "source_type": "podcast" if source_cat == "podcasts" else "newsletter",
                        }
            logger.info(f"Loaded metadata for {len(index_data)} entries from index.json")
        except Exception as exc:
            logger.warning(f"Failed to load or parse index.json at {index_file}: {exc}")

    documents: List[TranscriptDocument] = []
    discovered_files: List[str] = []
    skipped_files: List[Dict[str, str]] = []
    failed_files: List[Dict[str, str]] = []
    supported_extensions = {".md", ".markdown"}

    # Recursively find all files in the dataset directory
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue

        try:
            rel_path = path.relative_to(root_dir).as_posix()
        except ValueError:
            rel_path = path.name

        discovered_files.append(rel_path)

        # Skip known documentation and license files
        if path.name.lower() in {"license.md", "readme.md"}:
            skipped_files.append({
                "path": rel_path,
                "reason": "Repository documentation/license file",
            })
            continue

        # Skip unsupported extensions
        if path.suffix.lower() not in supported_extensions:
            skipped_files.append({
                "path": rel_path,
                "reason": f"Unsupported file extension ({path.suffix})",
            })
            continue

        try:
            metadata, raw_body = parse_frontmatter_and_content(path)
            content = normalize_content(raw_body)

            # Skip empty files
            if not content:
                logger.warning(f"Skipping empty or whitespace-only transcript: {rel_path}")
                skipped_files.append({
                    "path": rel_path,
                    "reason": "Empty or whitespace-only document body",
                })
                continue

            # Merge metadata with index.json fallback
            idx_meta = index_data.get(rel_path, {})

            # Determine source type (priority: frontmatter -> index.json -> folder name -> unknown)
            folder_hint = "podcast" if "podcast" in rel_path.lower() else ("newsletter" if "newsletter" in rel_path.lower() else "unknown")
            source_type = metadata.get("type") or idx_meta.get("source_type") or folder_hint

            title = (
                metadata.get("title")
                or idx_meta.get("title")
                or path.stem.replace("-", " ").title()
            )
            date = str(metadata.get("date") or idx_meta.get("date") or "") or None
            guest = metadata.get("guest") or idx_meta.get("guest")
            description = metadata.get("description") or idx_meta.get("description")
            subtitle = metadata.get("subtitle") or idx_meta.get("subtitle")
            source_url = metadata.get("post_url") or idx_meta.get("post_url")

            word_count = metadata.get("word_count") or idx_meta.get("word_count")
            if not word_count:
                word_count = len(content.split())

            doc_id = f"{source_type}_{path.stem}".replace(" ", "_")

            # Collect any additional metadata fields
            extra_metadata = {
                k: v
                for k, v in metadata.items()
                if k not in {
                    "title", "date", "type", "guest", "description",
                    "subtitle", "post_url", "word_count",
                }
            }

            doc = TranscriptDocument(
                id=doc_id,
                title=title,
                source_type=source_type,
                filename=rel_path,
                date=date,
                guest=guest,
                description=description,
                subtitle=subtitle,
                source_url=source_url,
                word_count=int(word_count),
                content=content,
                metadata=extra_metadata,
            )
            documents.append(doc)
        except Exception as exc:
            logger.error(f"Failed to process file {rel_path}: {exc}")
            failed_files.append({"path": rel_path, "error": str(exc)})

    report = {
        "discovered_files": discovered_files,
        "discovered_files_count": len(discovered_files),
        "processed_documents_count": len(documents),
        "skipped_files": skipped_files,
        "skipped_files_count": len(skipped_files),
        "failed_files": failed_files,
        "failed_files_count": len(failed_files),
    }

    logger.info(
        f"Discovered: {len(discovered_files)} | Processed: {len(documents)} | "
        f"Skipped: {len(skipped_files)} | Failed: {len(failed_files)}"
    )
    return documents, report


def load_dataset(data_dir: Path) -> List[TranscriptDocument]:
    """
    Scans the dataset directory and returns a list of TranscriptDocument objects.
    Maintains compatibility with callers expecting just the document list.
    """
    documents, _ = load_dataset_with_report(data_dir)
    return documents

