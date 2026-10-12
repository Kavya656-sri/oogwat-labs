"""
Unit tests for the AI Chat endpoint.
Uses mocked embeddings and mocked Groq client to test fast and offline without external API calls.
"""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from backend.chat.models import ChatRequest, ChatResponse
from backend.chat.service import ChatService
from backend.main import app
from backend.retrieval.embeddings import MockEmbeddingProvider
from backend.retrieval.indexer import LocalVectorIndex


class MockGroqChoice:
    def __init__(self, content: str):
        self.message = MagicMock()
        self.message.content = content


class MockGroqCompletion:
    def __init__(self, content: str):
        self.choices = [MockGroqChoice(content)]


class MockGroqClient:
    def __init__(self, response_text: str = "According to Lenny, product-market fit requires flattened retention."):
        self.response_text = response_text
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock(return_value=MockGroqCompletion(self.response_text))


@pytest.fixture
def test_index_with_chunks():
    provider = MockEmbeddingProvider(dimension=16)
    index = LocalVectorIndex()
    chunks = [
        {
            "chunk_id": "c_pmf_01",
            "document_id": "doc_pmf",
            "title": "Elena Verna on PMF",
            "source_type": "podcast",
            "filename": "podcasts/elena-verna.md",
            "source_url": "https://www.lennysnewsletter.com/p/elena-verna",
            "date": "2025-12-18",
            "guest": "Elena Verna",
            "content": "Product-market fit is when retention curves flatten out.",
        }
    ]
    index.build_from_chunks(chunks, provider)
    return index, provider


def test_chat_service_grounded_answer(test_index_with_chunks):
    index, provider = test_index_with_chunks
    mock_groq = MockGroqClient("Elena Verna explains that PMF occurs when retention curves flatten.")

    service = ChatService(
        index=index,
        embedding_provider=provider,
        groq_client=mock_groq,
    )

    request = ChatRequest(
        question="How does Elena Verna define PMF?",
        session_id="test-session-1",
        top_k=1,
    )

    response = service.answer_question(request)

    assert isinstance(response, ChatResponse)
    assert response.session_id == "test-session-1"
    assert "Elena Verna" in response.answer
    assert response.evidence_count == 1
    assert len(response.citations) == 1

    citation = response.citations[0]
    assert citation.title == "Elena Verna on PMF"
    assert citation.source_url == "https://www.lennysnewsletter.com/p/elena-verna"
    assert citation.chunk_id == "c_pmf_01"


def test_chat_service_insufficient_evidence():
    provider = MockEmbeddingProvider(dimension=16)
    empty_index = LocalVectorIndex()
    mock_groq = MockGroqClient()

    service = ChatService(
        index=empty_index,
        embedding_provider=provider,
        groq_client=mock_groq,
    )

    request = ChatRequest(question="What is the capital of Mars?")
    response = service.answer_question(request)

    assert "don't have enough information" in response.answer.lower()
    assert response.evidence_count == 0
    assert response.citations == []


def test_chat_service_groq_failure_raises_503(test_index_with_chunks):
    index, provider = test_index_with_chunks

    failing_groq = MagicMock()
    failing_groq.chat.completions.create.side_effect = RuntimeError("Groq rate limit exceeded")

    service = ChatService(
        index=index,
        embedding_provider=provider,
        groq_client=failing_groq,
    )

    request = ChatRequest(question="Any question")
    with pytest.raises(Exception) as exc_info:
        service.answer_question(request)

    assert "503" in str(exc_info.value) or "LLM generation failed" in str(exc_info.value)


def test_fastapi_chat_endpoint_validation():
    client = TestClient(app)

    # Empty question should fail with 422
    res = client.post("/chat", json={"question": ""})
    assert res.status_code == 422


def test_fastapi_chat_endpoint_e2e_with_mock(monkeypatch, test_index_with_chunks):
    from backend.chat.service import chat_service
    index, provider = test_index_with_chunks

    mock_groq = MockGroqClient("Retention is key.")
    monkeypatch.setattr(chat_service, "_index", index)
    monkeypatch.setattr(chat_service, "_embedding_provider", provider)
    monkeypatch.setattr(chat_service, "_groq_client", mock_groq)

    client = TestClient(app)
    res = client.post("/api/chat", json={"question": "What is PMF?", "session_id": "session-42"})

    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == "session-42"
    assert "Retention is key" in data["answer"]
    assert len(data["citations"]) > 0
    assert data["citations"][0]["title"] == "Elena Verna on PMF"
