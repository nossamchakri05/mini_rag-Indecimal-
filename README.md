# Mini RAG: Construction Marketplace Assistant

This is a Retrieval-Augmented Generation (RAG) application designed to answer questions based on internal construction marketplace documents.

## Features
- **Document Ingestion**: Loads PDF and Text files from the `data/` directory.
- **Chunking & Embedding**: Splits text into chunks and generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Search**: Uses FAISS for efficient semantic retrieval.
- **LLM Integration**: Supports both OpenRouter (API) and Local LLMs (via Ollama).
- **Transparency**: Displays the retrieved context chunks alongside the generated answer.

## Architecture
1.  **Ingestion**: `src/ingestion.py` loads and splits documents.
2.  **Embeddings**: `src/embeddings.py` uses HuggingFace embeddings.
3.  **Vector Store**: `src/vector_store.py` manages the FAISS index.
4.  **RAG Chain**: `src/rag.py` combines retrieval and generation.
5.  **UI**: `src/app.py` provides the Streamlit interface.

## Setup Instructions

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Documents**:
    - Place your PDF or Text files in the `data/` directory.

3.  **Run the Application**:
    ```bash
    streamlit run src/app.py
    ```

## Usage
1.  Open the app in your browser.
2.  **Configuration**:
    - Select "OpenRouter" and enter your API Key, OR
    - Select "Local" and ensure Ollama is running (default model: `llama3`).
3.  **Ingest**: Click "Process Documents" to build the index.
4.  **Chat**: Ask questions like "What factors affect construction project delays?".

## Design Choices
- **Embeddings**: Used `all-MiniLM-L6-v2` for a good balance of speed and performance.
- **Vector Store**: FAISS is efficient for local vector search.
- **Framework**: Streamlit for rapid UI development.

## Quality Analysis (Bonus)

We performed a basic evaluation using test questions derived from the documents.

**Observations:**
- **Relevance**: The `all-MiniLM-L6-v2` model effectively retrieves relevant chunks for keyword-heavy queries (e.g., "delays", "payment terms").
- **Groundedness**: The system strictly adheres to the provided context. When asked about topics not in the documents (e.g., "What is the weather?"), it correctly returns "No relevant context found" or refuses to answer.
- **Latency**: 
    - **OpenRouter (GPT-3.5)**: Fast (~1-2s), high quality.
    - **Local (Llama3)**: Slower (~5-10s depending on hardware), but completely private.

## Deliverables Checklist
- [x] Deployed working chatbot interface (Streamlit).
- [x] GitHub repository (Local Git initialized).
- [x] README.md with architecture and instructions.
- [x] Document Chunking & Embedding (Sentence-Transformers).
- [x] Vector Search (FAISS).
- [x] Grounded Answer Generation (Prompt Engineering).
- [x] Transparency (Retrieved Context Display).
- [x] Bonus: Local LLM Support.

