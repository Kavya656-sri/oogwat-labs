"""
Chunking utility to split normalized transcripts into overlapping chunks.
"""

from typing import List
from backend.ingestion.models import TranscriptChunk, TranscriptDocument


def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    """
    Validates chunk size and overlap parameters.
    Raises ValueError if chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )


def split_text_recursive(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Splits text recursively by paragraphs, then sentences/lines, then words,
    producing overlapping chunks respecting chunk_size and chunk_overlap limits.

    Defaults:
    - chunk_size = 1200 characters (~200-250 words): Captures a coherent thought,
      speaker exchange, or newsletter subsection without exceeding dense embedding models' limits.
    - chunk_overlap = 200 characters (~30-40 words): Ensures context at chunk boundaries
      is retained for semantic search without excessive duplication.
    """
    validate_chunk_settings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not text or not text.strip():
        return []

    # If already smaller than chunk_size, return as single chunk
    if len(text) <= chunk_size:
        return [text.strip()]

    # Break into natural paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)

        # If a single paragraph is larger than chunk_size, split by lines/sentences
        if para_len > chunk_size:
            # First flush any accumulated chunk
            if current_chunk:
                joined = "\n\n".join(current_chunk).strip()
                if joined:
                    chunks.append(joined)
                current_chunk = []
                current_length = 0

            # Sub-split long paragraph by single newlines or sentences
            sub_segments = [s.strip() for s in para.split("\n") if s.strip()]
            for seg in sub_segments:
                if len(seg) > chunk_size:
                    # Split long segment by word tokens
                    words = seg.split(" ")
                    sub_buf: List[str] = []
                    sub_len = 0
                    for w in words:
                        if sub_len + len(w) + 1 > chunk_size and sub_buf:
                            chunks.append(" ".join(sub_buf).strip())
                            # Overlap by trailing words
                            overlap_words: List[str] = []
                            overlap_len = 0
                            for ow in reversed(sub_buf):
                                if overlap_len + len(ow) + 1 <= chunk_overlap:
                                    overlap_words.insert(0, ow)
                                    overlap_len += len(ow) + 1
                                else:
                                    break
                            sub_buf = overlap_words + [w]
                            sub_len = sum(len(x) + 1 for x in sub_buf)
                        else:
                            sub_buf.append(w)
                            sub_len += len(w) + 1
                    if sub_buf:
                        chunks.append(" ".join(sub_buf).strip())
                else:
                    if current_length + len(seg) + 2 > chunk_size and current_chunk:
                        chunks.append("\n".join(current_chunk).strip())
                        current_chunk = [seg]
                        current_length = len(seg)
                    else:
                        current_chunk.append(seg)
                        current_length += len(seg) + 1
            continue

        if current_length + para_len + 2 > chunk_size and current_chunk:
            chunk_str = "\n\n".join(current_chunk).strip()
            chunks.append(chunk_str)

            # Build overlap from the end of current_chunk
            overlap_buf: List[str] = []
            overlap_len = 0
            for item in reversed(current_chunk):
                if overlap_len + len(item) + 2 <= chunk_overlap:
                    overlap_buf.insert(0, item)
                    overlap_len += len(item) + 2
                else:
                    break

            current_chunk = overlap_buf + [para]
            current_length = sum(len(x) + 2 for x in current_chunk)
        else:
            current_chunk.append(para)
            current_length += para_len + 2

    if current_chunk:
        chunk_str = "\n\n".join(current_chunk).strip()
        if chunk_str and (not chunks or chunk_str != chunks[-1]):
            chunks.append(chunk_str)

    # Filter out empty strings
    return [c for c in chunks if c.strip()]


def chunk_document(
    doc: TranscriptDocument,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[TranscriptChunk]:
    """
    Takes a normalized TranscriptDocument and produces TranscriptChunk objects,
    preserving all source metadata for downstream citation and retrieval.
    """
    validate_chunk_settings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    raw_chunks = split_text_recursive(
        text=doc.content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: List[TranscriptChunk] = []
    for idx, text_block in enumerate(raw_chunks):
        chunk = TranscriptChunk(
            chunk_id=f"{doc.id}_c{idx:04d}",
            document_id=doc.id,
            chunk_index=idx,
            content=text_block,
            char_count=len(text_block),
            word_count=len(text_block.split()),
            title=doc.title,
            source_type=doc.source_type,
            filename=doc.filename,
            source_url=doc.source_url,
            date=doc.date,
            guest=doc.guest,
        )
        chunks.append(chunk)

    return chunks

