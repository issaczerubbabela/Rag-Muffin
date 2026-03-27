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


def get_embedding_model() -> str:
    """Resolve embedding model from environment after dotenv has been loaded."""
    return os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


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

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    system_prompt = (
        "You are an assistant representing the worldview of the author whose texts "
        "you have been provided. Answer the user's question using ONLY the provided "
        "context. If the answer is not contained within the context, state clearly "
        "that you do not know based on the provided material. Do not introduce "
        "outside knowledge.\n\n"
        "Context:\n{context}"
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

