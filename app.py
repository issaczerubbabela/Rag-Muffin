import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


VECTOR_DB_DIR = "./vector_db"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_CHAT_MODEL = "gemini-flash-latest"


def get_embedding_model() -> str:
    """Resolve embedding model from environment after dotenv has been loaded."""
    return os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_chat_model() -> str:
    """Resolve chat model from environment after dotenv has been loaded."""
    return os.getenv("GEMINI_CHAT_MODEL", DEFAULT_CHAT_MODEL)


def initialize_components():
    """Initialize embeddings, vector store, retriever, and retrieval chain."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your environment or .env file."
        )

    if not Path(VECTOR_DB_DIR).exists():
        raise RuntimeError(
            f"Vector DB not found at {Path(VECTOR_DB_DIR).resolve()}. "
            "Run ingest.py first."
        )

    embedding_model = get_embedding_model()
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    vector_store = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    chat_model = get_chat_model()
    llm = ChatGoogleGenerativeAI(model=chat_model, temperature=0.3)

    system_prompt = (
        "You are an expert assistant deeply intimately familiar with the worldview, theology, "
    "and teachings of the author whose texts are provided below. Your sole purpose is to "
    "answer questions exactly as this author would, acting as a faithful representative of their work.\n\n"
    
    "TONE AND VOICE DIRECTIVES:\n"
    "- Adopt a pastoral, earnest, and direct tone.\n"
    "- Mirror the vocabulary, phrasing, and rhetorical style found in the provided context.\n"
    "- Avoid modern AI-isms, corporate jargon, or overly clinical language. Sound like a human teacher.\n"
    "- When appropriate, emphasize the spiritual or scriptural principles highlighted in the text.\n\n"
    
    "STRICT KNOWLEDGE CONSTRAINTS:\n"
    "1. You must construct your answer using ONLY the information contained in the {context} provided below.\n"
    "2. Do not introduce outside theological interpretations, general biblical knowledge, or viewpoints from other authors, even if they seem relevant.\n"
    "3. If the answer cannot be confidently deduced from the {context}, do not guess. Instead, reply exactly with: 'Based on the texts provided, the author does not directly address this specific question.'\n\n"
    
    "PROVIDED CONTEXT:\n"
    "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


def init_session_state() -> None:
    """Initialize session state for chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_chat_history() -> None:
    """Render existing chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main() -> None:
    st.set_page_config(page_title="RAG Assistant", page_icon=":books:")
    st.title("RAG Assistant")
    st.caption("Ask questions grounded in your EPUB and URL knowledge base.")

    try:
        retrieval_chain = initialize_components()
    except Exception as exc:
        st.error(f"Initialization error: {exc}")
        st.stop()

    init_session_state()
    render_chat_history()

    user_query = st.chat_input("Ask a question about the ingested material...")
    if not user_query:
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            result = retrieval_chain.invoke({"input": user_query})
            answer = result.get("answer", "I do not know based on the provided material.")
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

