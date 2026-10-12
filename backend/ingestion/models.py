"""
Data models for transcript documents and chunks.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class TranscriptDocument(BaseModel):
    """Represents a normalized transcript document from the dataset."""

    id: str = Field(description="Unique identifier for the document")
    title: str = Field(description="Title of the transcript or newsletter post")
    source_type: str = Field(
        default="unknown",
        description="Source type, typically 'podcast' or 'newsletter'",
    )
    filename: str = Field(description="Original relative path or filename")
    date: Optional[str] = Field(
        default=None, description="Publication date (YYYY-MM-DD)"
    )
    guest: Optional[str] = Field(
        default=None, description="Guest name if podcast episode"
    )
    description: Optional[str] = Field(
        default=None, description="Description or episode summary"
    )
    subtitle: Optional[str] = Field(
        default=None, description="Newsletter subtitle if available"
    )
    source_url: Optional[str] = Field(
        default=None, description="Canonical post URL for citations"
    )
    word_count: Optional[int] = Field(
        default=None, description="Word count reported or computed"
    )
    content: str = Field(description="Normalized transcript or article body")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional original metadata if present"
    )


class TranscriptChunk(BaseModel):
    """Represents an overlapping text chunk derived from a transcript document."""

    chunk_id: str = Field(description="Unique identifier for this chunk")
    document_id: str = Field(description="Parent document identifier")
    chunk_index: int = Field(description="Zero-based index of this chunk")
    content: str = Field(description="Text content of the chunk")
    char_count: int = Field(description="Character count of chunk content")
    word_count: int = Field(description="Word count of chunk content")
    title: str = Field(description="Document title for source citation")
    source_type: str = Field(description="podcast or newsletter")
    filename: Optional[str] = Field(
        default=None, description="Original relative file path of source document"
    )
    source_url: Optional[str] = Field(
        default=None, description="Source URL for citation link"
    )
    date: Optional[str] = Field(
        default=None, description="Publication date for citation"
    )
    guest: Optional[str] = Field(
        default=None, description="Guest name if podcast episode"
    )

