from typing import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END


class RAGState(TypedDict):
    query: str
    retrieved_context_chunks: list
    final_answer: str
    confidence_score: float


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def retrieve_node(state: RAGState):
    docs = vectorstore.similarity_search(
        state["query"],
        k=4
    )

    chunks = [doc.page_content for doc in docs]

    return {
        "retrieved_context_chunks": chunks
    }


def generate_node(state: RAGState):
    context = "\n\n".join(
        state["retrieved_context_chunks"]
    )

    if not context.strip():
        return {
            "final_answer": "Information not available in the eBook context.",
            "confidence_score": 0.0
        }

    prompt = f"""
You are a grounded RAG assistant.

Answer the user's question ONLY using the provided eBook context.

If the answer is not present in the context, say:
"Information not available in the eBook context."

Context:
{context}

Question:
{state["query"]}
"""

    response = llm.invoke(prompt)

    answer = response.content

    confidence = 0.9

    return {
        "final_answer": answer,
        "confidence_score": confidence
    }


graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

rag_app = graph.compile() 
