"""
Data models for the chat endpoint.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming user chat query."""

    question: str = Field(..., min_length=1, description="Question asked by the user")
    session_id: Optional[str] = Field(
        default=None, description="Optional conversation session ID for tracking"
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Maximum number of context chunks to retrieve"
    )


class Citation(BaseModel):
    """Structured citation referencing a retrieved transcript segment."""

    title: str = Field(description="Document title")
    source_type: str = Field(description="podcast or newsletter")
    source_url: Optional[str] = Field(default=None, description="Canonical link to source")
    date: Optional[str] = Field(default=None, description="Publication date")
    guest: Optional[str] = Field(default=None, description="Guest name if podcast")
    chunk_id: str = Field(description="Unique chunk identifier")
    document_id: str = Field(description="Source document identifier")
    relevance_score: float = Field(description="Semantic similarity score")


class ChatResponse(BaseModel):
    """Grounded answer from Lenny Growth Assistant with source citations."""

    answer: str = Field(description="Generated answer grounded in retrieved transcripts")
    session_id: str = Field(description="Session identifier for this conversation")
    citations: List[Citation] = Field(
        default_factory=list, description="List of source passages used as evidence"
    )
    evidence_count: int = Field(description="Number of relevant chunks retrieved")
