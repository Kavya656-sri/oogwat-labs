from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="The Lenny Growth Assistant API",
    description="Backend API for The Lenny Growth Assistant chat application",
    version="0.1.0",
)

# Allow CORS for development (useful when connecting the frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """Root endpoint welcoming users and confirming API availability."""
    return {
        "message": "Welcome to The Lenny Growth Assistant API",
        "status": "running",
    }


@app.get("/health")
def health_check():
    """Health check endpoint to verify service status."""
    return {
        "status": "ok",
        "app": "Lenny Growth Assistant",
        "version": "0.1.0",
    }


from backend.chat.models import ChatRequest, ChatResponse
from backend.chat.service import chat_service


@app.post("/chat", response_model=ChatResponse)
@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Answers user questions grounded in Lenny's Podcast transcripts and Newsletter articles.
    Returns response text and structured source citations.
    """
    return chat_service.answer_question(request)

