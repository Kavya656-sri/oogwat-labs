"""
Unit tests for retrieval engine and local vector index.
Uses deterministic MockEmbeddingProvider to run offline in milliseconds without downloading weights.
"""

from pathlib import Path
import pytest

from backend.retrieval.embeddings import MockEmbeddingProvider
from backend.retrieval.indexer import LocalVectorIndex, cosine_similarity
from backend.retrieval.models import SearchResult


def test_cosine_similarity():
    vec_a = [1.0, 0.0]
    vec_b = [1.0, 0.0]
    vec_c = [0.0, 1.0]
    vec_d = [-1.0, 0.0]

    assert pytest.approx(cosine_similarity(vec_a, vec_b), 0.001) == 1.0
    assert pytest.approx(cosine_similarity(vec_a, vec_c), 0.001) == 0.0
    assert pytest.approx(cosine_similarity(vec_a, vec_d), 0.001) == -1.0


def test_mock_embedding_provider_deterministic():
    provider = MockEmbeddingProvider(dimension=16)
    v1 = provider.embed_text("product-market fit")
    v2 = provider.embed_text("product-market fit")
    v3 = provider.embed_text("enterprise sales")

    assert len(v1) == 16
    assert v1 == v2  # strictly deterministic
    assert v1 != v3  # different texts yield different vectors


def test_index_build_and_search():
    provider = MockEmbeddingProvider(dimension=16)
    index = LocalVectorIndex()

    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc_pmf",
            "title": "Finding Product-Market Fit",
            "source_type": "podcast",
            "filename": "podcasts/pmf.md",
            "source_url": "https://example.com/pmf",
            "date": "2026-01-01",
            "guest": "Super PM",
            "content": "Product-market fit is when retention curves flatten.",
        },
        {
            "chunk_id": "c2",
            "document_id": "doc_sales",
            "title": "Enterprise Sales Guide",
            "source_type": "newsletter",
            "filename": "newsletters/sales.md",
            "source_url": "https://example.com/sales",
            "date": "2025-06-15",
            "guest": None,
            "content": "Closing big deals requires understanding procurement and security reviews.",
        },
    ]

    index.build_from_chunks(chunks, provider)
    assert index.total_chunks == 2
    assert not index.is_empty()

    results = index.search(
        query="retention curves and pmf",
        embedding_provider=provider,
        top_k=2,
    )

    assert len(results) == 2
    top = results[0]
    assert isinstance(top, SearchResult)
    assert top.chunk_id in {"c1", "c2"}
    assert -1.0 <= top.score <= 1.0


def test_top_k_limits():
    provider = MockEmbeddingProvider(dimension=16)
    index = LocalVectorIndex()

    chunks = [
        {
            "chunk_id": f"c_{i}",
            "document_id": f"doc_{i}",
            "title": f"Title {i}",
            "source_type": "podcast",
            "content": f"Content passage number {i}",
        }
        for i in range(10)
    ]

    index.build_from_chunks(chunks, provider)
    res_1 = index.search("Content", provider, top_k=1)
    res_3 = index.search("Content", provider, top_k=3)
    res_20 = index.search("Content", provider, top_k=20)

    assert len(res_1) == 1
    assert len(res_3) == 3
    assert len(res_20) == 10  # Max available is 10


def test_empty_index_handling():
    provider = MockEmbeddingProvider(dimension=16)
    index = LocalVectorIndex()

    # Search on empty index
    results = index.search("Any query", provider, top_k=5)
    assert results == []

    # Build on empty list
    index.build_from_chunks([], provider)
    assert index.is_empty()
    assert index.search("Any query", provider) == []


def test_metadata_and_citation_preservation():
    provider = MockEmbeddingProvider(dimension=16)
    index = LocalVectorIndex()

    chunks = [
        {
            "chunk_id": "chunk_pmf_01",
            "document_id": "doc_pmf_01",
            "title": "PMF Signals",
            "source_type": "podcast",
            "filename": "podcasts/pmf.md",
            "source_url": "https://example.com/podcast/pmf",
            "date": "2026-03-15",
            "guest": "Jane Founder",
            "content": "Users are organically recommending the product to friends.",
        }
    ]

    index.build_from_chunks(chunks, provider)
    results = index.search("organic recommendations", provider, top_k=1)

    assert len(results) == 1
    res = results[0]
    assert res.chunk_id == "chunk_pmf_01"
    assert res.document_id == "doc_pmf_01"
    assert res.title == "PMF Signals"
    assert res.source_type == "podcast"
    assert res.filename == "podcasts/pmf.md"
    assert res.source_url == "https://example.com/podcast/pmf"
    assert res.date == "2026-03-15"
    assert res.guest == "Jane Founder"
    assert res.content == "Users are organically recommending the product to friends."


def test_save_and_load_index(tmp_path: Path):
    provider = MockEmbeddingProvider(dimension=16)
    index = LocalVectorIndex()

    chunks = [
        {
            "chunk_id": "chunk_a",
            "document_id": "doc_a",
            "title": "Alpha",
            "source_type": "podcast",
            "content": "Alpha content snippet",
        },
        {
            "chunk_id": "chunk_b",
            "document_id": "doc_b",
            "title": "Beta",
            "source_type": "newsletter",
            "content": "Beta content snippet",
        },
    ]

    index.build_from_chunks(chunks, provider)
    save_dir = tmp_path / "test_index"
    index.save(save_dir)

    assert (save_dir / "index_meta.json").exists()
    assert (save_dir / "index_vectors.json").exists()
    assert (save_dir / "index_chunks.jsonl").exists()

    loaded_index = LocalVectorIndex()
    loaded_index.load(save_dir)

    assert loaded_index.total_chunks == 2
    assert loaded_index.dimension == 16
    assert loaded_index.model_name == provider.model_name

    res_orig = index.search("test query", provider, top_k=2)
    res_loaded = loaded_index.search("test query", provider, top_k=2)

    assert [r.chunk_id for r in res_orig] == [r.chunk_id for r in res_loaded]
    assert [r.score for r in res_orig] == [r.score for r in res_loaded]


def test_missing_or_corrupt_index_files(tmp_path: Path):
    index = LocalVectorIndex()
    empty_dir = tmp_path / "empty_index"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        index.load(empty_dir)


def test_rebuilding_after_input_changes(tmp_path: Path):
    provider = MockEmbeddingProvider(dimension=16)
    index_dir = tmp_path / "rebuild_index"

    chunks_v1 = [
        {"chunk_id": "c1", "document_id": "d1", "title": "T1", "source_type": "p", "content": "Initial text"}
    ]
    idx1 = LocalVectorIndex()
    idx1.build_from_chunks(chunks_v1, provider)
    idx1.save(index_dir)
    assert idx1.total_chunks == 1

    chunks_v2 = [
        {"chunk_id": "c1", "document_id": "d1", "title": "T1", "source_type": "p", "content": "Initial text"},
        {"chunk_id": "c2", "document_id": "d2", "title": "T2", "source_type": "n", "content": "Added second chunk"},
    ]
    idx2 = LocalVectorIndex()
    idx2.build_from_chunks(chunks_v2, provider)
    idx2.save(index_dir)

    reloaded = LocalVectorIndex()
    reloaded.load(index_dir)
    assert reloaded.total_chunks == 2


def test_invalid_query_and_dimension_mismatch():
    provider_16 = MockEmbeddingProvider(dimension=16)
    provider_32 = MockEmbeddingProvider(dimension=32)

    index = LocalVectorIndex()
    index.build_from_chunks(
        [{"chunk_id": "c1", "document_id": "d1", "title": "T", "source_type": "p", "content": "Text"}],
        provider_16,
    )

    # Empty query returns empty list safely
    assert index.search("", provider_16) == []
    assert index.search("   ", provider_16) == []

    # Dimension mismatch raises ValueError
    with pytest.raises(ValueError, match="dimension"):
        index.search("Text", provider_32)
