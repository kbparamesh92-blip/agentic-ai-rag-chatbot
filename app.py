import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Agentic AI RAG Chatbot")


class ChatRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "message": "Agentic AI RAG Chatbot API is running"
    }


@app.post("/chat")
def chat(request: ChatRequest):
    return {
        "query": request.query,
        "final_answer": "Information not available yet.",
        "retrieved_context_chunks": [],
        "confidence_score": 0.0
    }
