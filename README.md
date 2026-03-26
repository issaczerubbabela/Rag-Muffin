# Author QA Bot (RAG System)

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Streamlit, and the Gemini API. This application ingests EPUB books and web articles to accurately answer questions based exclusively on the provided texts.

## Architecture

- **Frontend:** Streamlit
- **Orchestration:** LangChain
- **LLM & Embeddings:** Google Gemini 1.5 API (`gemini-1.5-flash` / `models/embedding-001`)
- **Vector Database:** ChromaDB (Local persistent storage)

## Setup Instructions

1. **Install OS Dependencies:** Ensure `pandoc` is installed on your system for EPUB parsing.
2. **Virtual Environment:** ```bash
   python -m venv venv
   source venv/bin/activate
