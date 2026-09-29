import os
from fastapi import FastAPI
from pydantic import BaseModel

from rag_graph import rag_app

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

    result = rag_app.invoke({
        "query": request.query,
        "retrieved_context_chunks": [],
        "final_answer": "",
        "confidence_score": 0.0
    })

    return {
        "query": request.query,
        "final_answer": result["final_answer"],
        "retrieved_context_chunks": result["retrieved_context_chunks"],
        "confidence_score": result["confidence_score"]
    } 
