# Mini RAG: Construction Marketplace Assistant

A robust, safety-aware Retrieval-Augmented Generation (RAG) system designed to answer questions based strictly on internal construction documents.

## 🏗️ Architecture

The system follows a standard RAG pipeline with added safety guardrails.

```mermaid
graph TD
    subgraph Ingestion Pipeline
        D[Documents (PDF/TXT/MD)] -->|Load| L[Document Loader]
        L -->|Split| S[Text Splitter]
        S -->|Chunk| C[Chunks]
        C -->|Embed| E[Embedding Model]
        E -->|Vectorize| V[FAISS Vector Store]
    end

    subgraph RAG Inference Flow
        U[User Query] -->|Embed| E_Q[Query Embedding]
        E_Q -->|Search| V
        V -->|Retrieve| R[Relevant Chunks]
        R -->|Score| G[Confidence Guardrail]
        G -->|Context + Prompt| LLM[Large Language Model]
        LLM -->|Generate| A[Answer]
    end
```

## 🔄 Workflow

1.  **Ingestion**:
    *   Users upload files via the Streamlit UI.
    *   Files are saved to `data/`.
    *   `src/ingestion.py` loads and splits text into manageable chunks.
    *   `src/embeddings.py` converts chunks into vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
    *   `src/vector_store.py` indexes these vectors using FAISS for fast retrieval.

2.  **Retrieval & Generation**:
    *   User asks a question.
    *   System retrieves the top 3 most relevant chunks.
    *   **Safety Check**: If the similarity score is low (distance > 1.0), a "Low Confidence" warning is injected.
    *   **Prompting**: The LLM is prompted with strict instructions to answer *only* from the context.
    *   **Response**: The answer is displayed along with the source chunks for transparency.

## 🛡️ Guardrails & Safety

To prevent hallucinations and ensure professional responses, we implemented the following:

1.  **"Not Explicitly Stated" Fallback**:
    *   If the answer is not in the documents, the model is forced to reply: *"Not specified in this document"*.
2.  **Vocabulary Control**:
    *   **Allowed**: "Positioned as", "Subject to contract".
    *   **Prohibited**: "Always", "Guaranteed" (unless explicitly in text).
3.  **Citation Confidence**:
    *   Retrieval scores are monitored. Weak matches trigger hedging language (e.g., *"The documents suggest..."*).

## 🚀 Setup & Usage

### Prerequisites
*   Python 3.10+
*   OpenRouter API Key (for GPT-3.5) OR Ollama (for Local LLM)

### Installation
1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd mini_rag
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure API Key**:
    *   Create a `.env` file in the root directory.
    *   Add your key: `OPENROUTER_API_KEY=sk-or-v1-...`

### Running the App
```bash
streamlit run src/app.py
```

### Using the Interface
1.  **Upload & Process**: 
    *   Use the sidebar to upload PDF, TXT, or MD files.
    *   Click "Save & Process Uploaded Files".
    *   *The system chunks the documents and builds the index.*
    
    ![Upload and Processing](upload%20files%20and%20chunck%20division.png)

2.  **Chat**: 
    *   Ask questions in the main chat window.
    *   The AI answers based strictly on the documents.
    
    ![Question and Response](question%20and%20response.png)

3.  **Verify Context**: 
    *   Expand "Retrieved Context" to see the source text used for the answer.
    *   This ensures transparency and helps verify the answer's accuracy.
    
    ![Review Context](revie%20context%20for%20a%20question.png)

## 📊 Quality Analysis (Bonus)

We performed a basic evaluation using test questions derived from the documents.

**Observations:**
- **Relevance**: The `all-MiniLM-L6-v2` model effectively retrieves relevant chunks for keyword-heavy queries (e.g., "delays", "payment terms").
- **Groundedness**: The system strictly adheres to the provided context. When asked about topics not in the documents (e.g., "What is the weather?"), it correctly returns "No relevant context found" or refuses to answer.
- **Latency**: 
    - **OpenRouter (GPT-3.5)**: Fast (~1-2s), high quality.
    - **Local (Llama3)**: Slower (~5-10s depending on hardware), but completely private.

## 📂 Project Structure
```
mini_rag/
├── data/                   # Document storage
├── src/
│   ├── app.py              # Streamlit UI
│   ├── ingestion.py        # Loader & Splitter
│   ├── embeddings.py       # Embedding Model
│   ├── vector_store.py     # FAISS Index Manager
│   ├── llm.py              # LLM Factory
│   └── rag.py              # RAG Chain & Guardrails
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```
