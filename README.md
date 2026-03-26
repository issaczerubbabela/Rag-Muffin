# Author Contextual QA Bot (RAG System)

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Streamlit, and the Gemini API. This application ingests EPUB books and web articles to accurately answer questions based exclusively on the provided texts, acting as a dedicated AI assistant for a specific author's worldview.

## Architecture

- **Frontend UI:** Streamlit
- **Orchestration:** LangChain
- **LLM & Embeddings:** Google Gemini API (`gemini-1.5-flash` for generation / `models/embedding-001` for vectorization)
- **Vector Database:** ChromaDB (Local persistent storage)

## Prerequisites & System Requirements

- Python 3.9+
- Google Gemini API Key
- `pandoc` installed on your host operating system (required for robust EPUB parsing).
  - _Ubuntu/Debian:_ `sudo apt-get install pandoc`
  - _macOS:_ `brew install pandoc`
  - _Windows:_ Install via the official Pandoc installer.

## Setup Instructions

1. **Clone/Setup the Repository:**
   Ensure your project folder has a `/data` directory containing your `.epub` files and a `urls.txt` file (one valid URL per line).

2. **Initialize Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a .env file in the root directory and add your API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## Usage Workflow

**Step 1: Ingest Data and Build the Vector Database**  
Run the ingestion script. This will read all books and URLs, chunk the text, generate embeddings via the Gemini API, and save the resulting database locally to a /vector_db folder. Note: You only need to run this once, or whenever you add new data to the /data folder.

```bash
python ingest.py
```

**Step 2: Launch the Chat Application**  
Start the Streamlit server to interact with your data.

```bash
streamlit run app.py
```

## Deployment Note

Once containerized using Docker, this application is lightweight enough to be deployed to an old laptop running as a local home server. This allows you to access the chatbot from any device on your local network without needing to keep your main development environment active.
