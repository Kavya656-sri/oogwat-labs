"""
RAG chat service for The Lenny Growth Assistant.
Combines semantic retrieval with Groq LLM inference for grounded answers.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import HTTPException

from backend.chat.models import ChatRequest, ChatResponse, Citation
from backend.retrieval.embeddings import BaseEmbeddingProvider, get_embedding_provider
from backend.retrieval.indexer import LocalVectorIndex
from backend.retrieval.models import SearchResult

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are The Lenny Growth Assistant, an AI expert on product management, startup growth, hiring, and leadership.
Your task is to answer the user's question using ONLY the provided excerpts from Lenny's Podcast transcripts and Lenny's Newsletter articles.

STRICT GROUNDING & CITATION RULES:
1. Ground your answer strictly in the facts, advice, frameworks, and quotes directly found in the context excerpts below.
2. If the excerpts do NOT contain sufficient information to answer the question, clearly state:
   "I don't have enough information in Lenny's transcripts to answer this question."
3. Cite the source title and guest/author in your answer when referencing specific advice.
4. When a source URL is available in the context, reference it so the user can explore the full discussion.
5. Do NOT invent facts, fake guest names, fabricated quotes, or invented URLs.
6. Provide concise, clear, and actionable answers in markdown format.
"""


class ChatService:
    """Coordinates retrieval of relevant evidence and grounded LLM generation."""

    def __init__(
        self,
        index: Optional[LocalVectorIndex] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        groq_client: Optional[object] = None,
        index_dir: Optional[Path] = None,
        groq_model: Optional[str] = None,
    ):
        self._index = index
        self._embedding_provider = embedding_provider
        self._groq_client = groq_client
        self._index_dir = index_dir or Path("data/processed/index")
        self._groq_model = groq_model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def _get_embedding_provider(self) -> BaseEmbeddingProvider:
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider("sentence_transformer")
        return self._embedding_provider

    def _get_index(self) -> LocalVectorIndex:
        if self._index is None:
            index = LocalVectorIndex()
            if not self._index_dir.exists():
                raise FileNotFoundError(
                    f"Retrieval index directory not found at {self._index_dir.resolve()}. "
                    "Run `python -m backend.retrieval.run` to create the index first."
                )
            index.load(self._index_dir)
            self._index = index
        return self._index

    def _get_groq_client(self):
        if self._groq_client is not None:
            return self._groq_client

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise HTTPException(
                status_code=500,
                detail="Groq API key not configured. Please set GROQ_API_KEY in your .env file.",
            )

        try:
            from groq import Groq

            self._groq_client = Groq(api_key=api_key)
            return self._groq_client
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize Groq client: {exc}"
            )

    def _build_context_block(self, results: List[SearchResult]) -> str:
        """Formats retrieved chunks into clear context blocks for the prompt."""
        blocks = []
        for idx, res in enumerate(results, 1):
            source_info = f"Source {idx}: [{res.source_type.upper()}] \"{res.title}\""
            if res.guest:
                source_info += f" with {res.guest}"
            if res.date:
                source_info += f" ({res.date})"
            if res.source_url:
                source_info += f" | Link: {res.source_url}"

            block = f"--- {source_info} ---\n{res.content.strip()}"
            blocks.append(block)
        return "\n\n".join(blocks)

    def answer_question(self, request: ChatRequest) -> ChatResponse:
        """Retrieves transcript chunks and generates a grounded response."""
        session_id = request.session_id or str(uuid.uuid4())
        question = request.question.strip()

        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        # 1. Retrieve relevant transcript chunks
        index = self._get_index()
        embedding_provider = self._get_embedding_provider()

        search_results: List[SearchResult] = index.search(
            query=question,
            embedding_provider=embedding_provider,
            top_k=request.top_k,
            min_score=None,
        )

        # 2. Check for empty retrieval or empty index
        if not search_results:
            return ChatResponse(
                answer="I don't have enough information in Lenny's transcripts to answer this question.",
                session_id=session_id,
                citations=[],
                evidence_count=0,
            )

        # 3. Format structured citations
        citations = [
            Citation(
                title=res.title,
                source_type=res.source_type,
                source_url=res.source_url,
                date=res.date,
                guest=res.guest,
                chunk_id=res.chunk_id,
                document_id=res.document_id,
                relevance_score=res.score,
            )
            for res in search_results
        ]

        # 4. Construct grounded prompt
        context_text = self._build_context_block(search_results)
        user_prompt = f"""Transcript Excerpts:
{context_text}

User Question: {question}

Grounded Answer:"""

        # 5. Call Groq
        groq_client = self._get_groq_client()
        try:
            completion = groq_client.chat.completions.create(
                model=self._groq_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            raw_answer = completion.choices[0].message.content
            answer = raw_answer.strip() if raw_answer else ""
        except Exception as exc:
            logger.error(f"Groq generation failed: {exc}")
            raise HTTPException(
                status_code=503, detail=f"LLM generation failed: {str(exc)}"
            )

        return ChatResponse(
            answer=answer,
            session_id=session_id,
            citations=citations,
            evidence_count=len(citations),
        )


# Global singleton instance for application runtime
chat_service = ChatService()
