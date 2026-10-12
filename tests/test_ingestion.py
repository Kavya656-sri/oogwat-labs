"""
Unit tests for transcript ingestion, parsing, and chunking.
Uses small synthetic fixtures to run quickly without touching the full dataset.
"""

from pathlib import Path
from backend.ingestion.chunker import chunk_document, split_text_recursive
from backend.ingestion.models import TranscriptDocument
from backend.ingestion.parser import (
    load_dataset,
    normalize_content,
    parse_frontmatter_and_content,
)
from backend.ingestion.ingest import run_ingestion


def test_parse_frontmatter_and_content_valid(tmp_path: Path):
    sample_file = tmp_path / "sample.md"
    sample_file.write_text(
        "---\ntitle: 'Test Episode'\ntype: 'podcast'\nguest: 'Alice'\n---\n\n**Lenny**: Hello Alice!\n\n**Alice**: Great to be here.",
        encoding="utf-8",
    )

    metadata, body = parse_frontmatter_and_content(sample_file)
    assert metadata["title"] == "Test Episode"
    assert metadata["type"] == "podcast"
    assert metadata["guest"] == "Alice"
    assert "**Lenny**: Hello Alice!" in body


def test_parse_frontmatter_missing(tmp_path: Path):
    sample_file = tmp_path / "no_frontmatter.md"
    sample_file.write_text("Just plain markdown text without YAML.", encoding="utf-8")

    metadata, body = parse_frontmatter_and_content(sample_file)
    assert metadata == {}
    assert body.strip() == "Just plain markdown text without YAML."


def test_parse_frontmatter_malformed_yaml(tmp_path: Path):
    sample_file = tmp_path / "bad_yaml.md"
    sample_file.write_text(
        "---\nthis is not: valid: yaml: [broken\n---\nBody text still intact.",
        encoding="utf-8",
    )

    metadata, body = parse_frontmatter_and_content(sample_file)
    assert metadata == {}
    assert "Body text still intact." in body


def test_parse_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.md"
    empty_file.write_text("   \n\n  ", encoding="utf-8")

    metadata, body = parse_frontmatter_and_content(empty_file)
    assert metadata == {}
    assert body == ""


def test_normalize_content():
    messy_text = "Line 1\r\n\r\n\r\n\r\nLine 2\r\n\r\nLine 3   \n"
    normalized = normalize_content(messy_text)
    assert "\r" not in normalized
    assert "\n\n\n" not in normalized
    assert normalized.startswith("Line 1")
    assert normalized.endswith("Line 3")


def test_chunking_preserves_metadata():
    doc = TranscriptDocument(
        id="podcast_test_guest",
        title="Scaling Startups",
        source_type="podcast",
        filename="podcasts/test-guest.md",
        date="2026-01-15",
        guest="Jane Doe",
        source_url="https://example.com/podcast/1",
        content="Paragraph one with insight.\n\n" * 20,
    )

    chunks = chunk_document(doc, chunk_size=150, chunk_overlap=30)
    assert len(chunks) > 1

    # Check each chunk preserves required citation metadata
    for idx, chunk in enumerate(chunks):
        assert chunk.document_id == "podcast_test_guest"
        assert chunk.chunk_id == f"podcast_test_guest_c{idx:04d}"
        assert chunk.title == "Scaling Startups"
        assert chunk.source_type == "podcast"
        assert chunk.source_url == "https://example.com/podcast/1"
        assert chunk.guest == "Jane Doe"
        assert chunk.date == "2026-01-15"
        assert len(chunk.content) > 0


def test_load_dataset_ignores_unsupported_files_and_empty_files(tmp_path: Path):
    # Valid markdown
    (tmp_path / "valid.md").write_text(
        "---\ntitle: Valid Post\ntype: newsletter\n---\nValid newsletter body text here.",
        encoding="utf-8",
    )
    # Empty markdown
    (tmp_path / "empty.md").write_text("", encoding="utf-8")
    # Unsupported file formats
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "document.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "script.py").write_text("print('hello')", encoding="utf-8")

    docs = load_dataset(tmp_path)
    assert len(docs) == 1
    assert docs[0].title == "Valid Post"
    assert docs[0].source_type == "newsletter"


def test_run_ingestion_end_to_end_fixture(tmp_path: Path):
    data_dir = tmp_path / "transcripts"
    output_dir = tmp_path / "processed"
    data_dir.mkdir()

    (data_dir / "episode1.md").write_text(
        "---\ntitle: Ep 1\ntype: podcast\nguest: Bob\n---\n" + ("Insight content. " * 50),
        encoding="utf-8",
    )
    (data_dir / "post1.md").write_text(
        "---\ntitle: Newsletter 1\ntype: newsletter\n---\n" + ("Growth loop strategy. " * 40),
        encoding="utf-8",
    )

    manifest = run_ingestion(
        data_dir=data_dir,
        output_dir=output_dir,
        chunk_size=200,
        chunk_overlap=40,
    )

    assert manifest["total_documents"] == 2
    assert manifest["total_podcasts"] == 1
    assert manifest["total_newsletters"] == 1
    assert manifest["total_chunks"] >= 2
    assert (output_dir / "chunks.json").exists()
    assert (output_dir / "documents.json").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "documents.jsonl").exists()
    assert (output_dir / "chunks.jsonl").exists()
    assert (output_dir / "ingestion_report.json").exists()


def test_chunk_validation_rejects_invalid_config():
    import pytest
    from backend.ingestion.chunker import validate_chunk_settings

    # Non-positive chunk_size
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        validate_chunk_settings(chunk_size=0, chunk_overlap=10)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        validate_chunk_settings(chunk_size=-100, chunk_overlap=10)

    # Negative chunk_overlap
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        validate_chunk_settings(chunk_size=500, chunk_overlap=-1)

    # chunk_overlap >= chunk_size
    with pytest.raises(ValueError, match="strictly less than"):
        validate_chunk_settings(chunk_size=500, chunk_overlap=500)

    with pytest.raises(ValueError, match="strictly less than"):
        validate_chunk_settings(chunk_size=500, chunk_overlap=600)


def test_chunking_prevents_empty_chunks():
    doc = TranscriptDocument(
        id="podcast_empty_test",
        title="Empty Chunk Test",
        source_type="podcast",
        filename="podcasts/empty-test.md",
        content="\n\n\n\n   \n\nParagraph 1.\n\n\n\n\nParagraph 2.\n\n   \n\n",
    )
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.content.strip()) > 0
        assert chunk.filename == "podcasts/empty-test.md"


def test_metadata_fallback_from_index(tmp_path: Path):
    import json
    index_data = {
        "schema_version": "2.0",
        "podcasts": [
            {
                "title": "Fallback Podcast Title",
                "filename": "podcasts/guest-x.md",
                "date": "2025-05-01",
                "guest": "Guest X",
                "post_url": "https://example.com/guest-x",
                "description": "Episode with Guest X",
                "word_count": 500,
            }
        ],
        "newsletters": []
    }
    (tmp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")
    podcasts_dir = tmp_path / "podcasts"
    podcasts_dir.mkdir()
    # File without frontmatter
    (podcasts_dir / "guest-x.md").write_text("Spoken content without frontmatter block.", encoding="utf-8")

    docs = load_dataset(tmp_path)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.title == "Fallback Podcast Title"
    assert doc.guest == "Guest X"
    assert doc.date == "2025-05-01"
    assert doc.source_url == "https://example.com/guest-x"


def test_missing_optional_metadata(tmp_path: Path):
    sample = tmp_path / "minimal.md"
    sample.write_text("Minimal content here without any YAML headers.", encoding="utf-8")

    docs = load_dataset(tmp_path)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.guest is None
    assert doc.subtitle is None
    assert doc.date is None
    assert doc.source_url is None
    assert doc.content == "Minimal content here without any YAML headers."


def test_missing_input_directory_raises(tmp_path: Path):
    import pytest
    missing_dir = tmp_path / "non_existent_folder"
    out_dir = tmp_path / "processed"

    with pytest.raises(FileNotFoundError):
        run_ingestion(data_dir=missing_dir, output_dir=out_dir)


def test_repeated_runs_are_deterministic(tmp_path: Path):
    data_dir = tmp_path / "transcripts"
    output_dir = tmp_path / "processed"
    data_dir.mkdir()

    (data_dir / "test1.md").write_text(
        "---\ntitle: T1\ntype: newsletter\n---\nSome informative paragraphs here.\n\nMore paragraphs.",
        encoding="utf-8",
    )

    run1 = run_ingestion(data_dir, output_dir, chunk_size=50, chunk_overlap=10)
    chunks_file = output_dir / "chunks.jsonl"
    lines_run1 = chunks_file.read_text(encoding="utf-8").strip().split("\n")

    run2 = run_ingestion(data_dir, output_dir, chunk_size=50, chunk_overlap=10)
    lines_run2 = chunks_file.read_text(encoding="utf-8").strip().split("\n")

    assert run1["total_chunks"] == run2["total_chunks"]
    assert lines_run1 == lines_run2


def test_load_dataset_with_report_tracks_skipped_files(tmp_path: Path):
    from backend.ingestion.parser import load_dataset_with_report

    (tmp_path / "license.md").write_text("License terms", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# Readme", encoding="utf-8")
    (tmp_path / "data.csv").write_text("a,b,c", encoding="utf-8")
    (tmp_path / "blank.md").write_text("   \n\n  ", encoding="utf-8")
    (tmp_path / "valid.md").write_text("---\ntitle: Ok\ntype: newsletter\n---\nValid content.", encoding="utf-8")

    docs, report = load_dataset_with_report(tmp_path)
    assert len(docs) == 1
    assert report["processed_documents_count"] == 1
    assert report["skipped_files_count"] == 4
    skipped_paths = [s["path"] for s in report["skipped_files"]]
    assert "license.md" in skipped_paths
    assert "readme.md" in skipped_paths
    assert "data.csv" in skipped_paths
    assert "blank.md" in skipped_paths


def test_cli_run_entry_point_error_handling(monkeypatch, tmp_path: Path):
    import pytest
    from backend.ingestion.run import main

    # Missing directory -> SystemExit(1)
    missing_dir = tmp_path / "missing_dir"
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--data-dir", str(missing_dir)],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1

    # Invalid chunk configuration -> SystemExit(2)
    valid_dir = tmp_path / "valid_dir"
    valid_dir.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--data-dir", str(valid_dir), "--chunk-size", "100", "--chunk-overlap", "200"],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


