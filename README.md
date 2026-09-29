# LangGraph & Pinecone RAG Chatbot

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot for the Agentic AI eBook.

The chatbot is designed to answer questions using information retrieved from the provided Agentic AI eBook.

## Architecture

The system contains the following components:

1. PDF Data Ingestion
2. Text Chunking
3. Vector Embeddings
4. Pinecone Vector Database
5. LangGraph RAG Workflow
6. LLM-based Answer Generation
7. Groundedness / Confidence Scoring
8. FastAPI Interface

## Project Structure

```text
agentic-ai-rag-chatbot/
├── Ebook-Agentic-AI.pdf
├── ingestion.py
├── rag_graph.py
├── app.py
├── requirements.txt
└── README.md
