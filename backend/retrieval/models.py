"""
Data models for the retrieval engine.
"""

from typing import Optional
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Represents a retrieved transcript chunk ranked by semantic relevance."""

    chunk_id: str = Field(description="Unique identifier of the chunk")
    document_id: str = Field(description="Source document identifier")
    title: str = Field(description="Source document title for citation")
    source_type: str = Field(description="Source type, typically 'podcast' or 'newsletter'")
    filename: Optional[str] = Field(default=None, description="Original relative file path")
    source_url: Optional[str] = Field(default=None, description="Canonical URL for source citation")
    date: Optional[str] = Field(default=None, description="Publication date (YYYY-MM-DD)")
    guest: Optional[str] = Field(default=None, description="Podcast guest name if applicable")
    content: str = Field(description="Text passage of the retrieved chunk")
    score: float = Field(description="Cosine similarity score (0.0 to 1.0)")


class RetrievalConfig(BaseModel):
    """Configuration for semantic retrieval queries."""

    top_k: int = Field(default=5, ge=1, le=50, description="Maximum number of results to return")
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0, description="Minimum relevance threshold")
    model_name: str = Field(default="all-MiniLM-L6-v2", description="HuggingFace model identifier")
