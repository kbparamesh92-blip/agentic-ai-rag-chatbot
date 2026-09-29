from typing import TypedDict
from langgraph.graph import StateGraph, END


class RAGState(TypedDict):
    query: str
    retrieved_context_chunks: list
    final_answer: str
    confidence_score: float


def retrieve_node(state: RAGState):
    # Pinecone retrieval will be connected here
    return {
        "retrieved_context_chunks": []
    }


def generate_node(state: RAGState):
    # LLM generation will be connected here
    return {
        "final_answer": "Information not available in the retrieved context.",
        "confidence_score": 0.0
    }


graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

rag_app = graph.compile()
