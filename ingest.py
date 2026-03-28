import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredEPubLoader, WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings


DATA_DIR = Path("./data")
URLS_FILE = DATA_DIR / "urls.txt"
VECTOR_DB_DIR = "./vector_db"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_BATCH_SIZE = 20
DEFAULT_BATCH_DELAY_SECONDS = 0.75
DEFAULT_MAX_RETRIES = 8


def get_embedding_model() -> str:
    """Resolve embedding model from environment after dotenv has been loaded."""
    return os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def _parse_retry_delay_seconds(error_text: str) -> float | None:
    """Extract retry delay hints from provider error text."""
    patterns = [
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)s",
        r"retryDelay\s*'?:\s*'([0-9]+)s'",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _is_quota_error(error_text: str) -> bool:
    lowered = error_text.lower()
    return "resource_exhausted" in lowered or "quota" in lowered or "429" in lowered


def validate_environment() -> str:
    """Load and validate required environment variables."""
    print("[1/6] Loading environment variables...")
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your environment or .env file."
        )

    print("Environment variables loaded successfully.")
    return api_key


def load_epub_documents() -> list:
    """Load all EPUB files from the data directory."""
    print("[2/6] Scanning for EPUB files...")
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR.resolve()}")
        return []

    epub_files = sorted(DATA_DIR.glob("*.epub"))
    print(f"Found {len(epub_files)} EPUB file(s).")

    all_docs = []
    for epub_path in epub_files:
        print(f"Loading EPUB: {epub_path.name}")
        loader = UnstructuredEPubLoader(str(epub_path))
        docs = loader.load()
        print(f"  -> Loaded {len(docs)} document section(s) from {epub_path.name}")
        all_docs.extend(docs)

    print(f"Total EPUB documents loaded: {len(all_docs)}")
    return all_docs


def load_web_documents() -> list:
    """Load web content from URLs listed in data/urls.txt."""
    print("[3/6] Loading web URLs from urls.txt...")
    if not URLS_FILE.exists():
        print(f"URLs file not found at {URLS_FILE.resolve()}. Skipping web ingestion.")
        return []

    urls = [
        line.strip()
        for line in URLS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    print(f"Found {len(urls)} URL(s) in urls.txt")

    if not urls:
        return []

    loader = WebBaseLoader(urls)
    docs = loader.load()
    print(f"Loaded {len(docs)} web document(s).")
    return docs


def split_documents(documents: list) -> list:
    """Split documents into retrieval-friendly chunks."""
    print("[4/6] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} source document(s).")
    return chunks


def build_vector_store(chunks: list) -> None:
    """Create and persist the Chroma vector store."""
    embedding_model = get_embedding_model()
    batch_size = int(os.getenv("INGEST_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
    batch_delay_seconds = float(
        os.getenv("INGEST_BATCH_DELAY_SECONDS", str(DEFAULT_BATCH_DELAY_SECONDS))
    )
    max_retries = int(os.getenv("INGEST_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))

    print(f"[5/6] Initializing Google embedding model: {embedding_model}")
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    print("Embedding model ready.")

    print(f"[6/6] Writing vectors to persistent Chroma DB at: {Path(VECTOR_DB_DIR).resolve()}")

    vector_store = None
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    print(
        f"Embedding {len(chunks)} chunk(s) in {total_batches} batch(es); "
        f"batch_size={batch_size}, delay={batch_delay_seconds}s"
    )

    for batch_index in range(total_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, len(chunks))
        batch_docs = chunks[start:end]
        attempt = 0

        while True:
            try:
                print(
                    f"Processing batch {batch_index + 1}/{total_batches} "
                    f"(chunks {start + 1}-{end})"
                )

                if vector_store is None:
                    vector_store = Chroma.from_documents(
                        documents=batch_docs,
                        embedding=embeddings,
                        persist_directory=VECTOR_DB_DIR,
                    )
                else:
                    vector_store.add_documents(batch_docs)

                if batch_index < total_batches - 1 and batch_delay_seconds > 0:
                    time.sleep(batch_delay_seconds)
                break

            except Exception as exc:
                attempt += 1
                error_text = str(exc)
                if not _is_quota_error(error_text) or attempt > max_retries:
                    raise

                retry_after = _parse_retry_delay_seconds(error_text)
                if retry_after is None:
                    retry_after = min(60.0, 2.0**attempt)

                wait_seconds = retry_after + 1.0
                print(
                    "Quota/rate limit reached while embedding batch "
                    f"{batch_index + 1}. Retrying in {wait_seconds:.1f}s "
                    f"(attempt {attempt}/{max_retries})..."
                )
                time.sleep(wait_seconds)

    print("Vector DB creation completed.")


def main() -> int:
    try:
        validate_environment()

        epub_docs = load_epub_documents()
        web_docs = load_web_documents()
        all_docs = epub_docs + web_docs
        print(f"Combined document count: {len(all_docs)}")

        if not all_docs:
            print("No documents found to ingest. Add EPUB files or URLs and try again.")
            return 1

        chunks = split_documents(all_docs)
        if not chunks:
            print("No chunks were produced. Nothing to store.")
            return 1

        build_vector_store(chunks)
        print("Ingestion pipeline finished successfully.")
        return 0

    except Exception as exc:
        print(f"Ingestion failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
