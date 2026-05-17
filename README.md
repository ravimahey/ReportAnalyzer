# Financial Report Analyzer Chat Bot

A **Retrieval-Augmented Generation (RAG)** application built with Streamlit and OpenAI that allows users to upload financial documents (PDF, DOCX, Excel, CSV, TXT) and ask questions about them using natural language. The system stores document embeddings persistently using FAISS, so re-processing is never needed.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Setup Instructions](#setup-instructions)
   - [Prerequisites](#prerequisites)
   - [Local Setup](#local-setup)
   - [Environment Variables](#environment-variables)
7. [Running the Application](#running-the-application)
8. [Usage Guide](#usage-guide)
   - [Uploading Documents](#uploading-documents)
   - [Managing Documents](#managing-documents)
   - [Chatting with Documents](#chatting-with-documents)
9. [Docker Deployment](#docker-deployment)
10. [API Reference](#api-reference)
    - [Document Processor](#document-processor)
    - [Embedding Store](#embedding-store)
    - [LLM Handler](#llm-handler)
11. [Design Decisions](#design-decisions)
12. [Troubleshooting](#troubleshooting)
13. [Future Enhancements](#future-enhancements)

---

## Overview

This application implements a **RAG (Retrieval-Augmented Generation)** pipeline to enable conversational analysis of financial and business documents. Users upload documents, the system chunks them, generates embeddings via OpenAI, stores them in a FAISS vector index, and then uses GPT-4 to answer questions based on the retrieved context.

**Key Workflow:**

```
Upload Document ➔ Chunking ➔ Embedding (OpenAI) ➔ FAISS Index
                                                          │
User Question ──➔ Similarity Search ──➔ Context ──➔ GPT-4 ➔ Answer
```

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │     Streamlit Frontend       │
                    │   (app.py - 3 Tabs)          │
                    └──────────┬──────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌────────────┐     ┌──────────────┐     ┌──────────────┐
   │  Document   │     │  Embedding   │     │     LLM      │
   │  Processor  │────▶│    Store     │────▶│   Handler    │
   │ (chunking)  │     │   (FAISS)    │     │  (OpenAI)    │
   └────────────┘     └──────────────┘     └──────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
   ┌────────────┐     ┌──────────────┐     ┌──────────────┐
   │ uploaded   │     │ vector_store │     │  OpenAI API  │
   │ _docs/     │     │  /indexes    │     │  (GPT-4)     │
   │            │     │  /chunks     │     │              │
   └────────────┘     └──────────────┘     └──────────────┘
```

### Component Breakdown

| Component | File | Responsibility |
|-----------|------|----------------|
| **Frontend** | `app.py` | Streamlit UI with 3 tabs (Upload, Manage, Chat) |
| **Document Processor** | `src/document_processor.py` | Loads files, splits into chunks |
| **Embedding Store** | `src/embedding_store.py` | FAISS vector DB, persistence, similarity search |
| **LLM Handler** | `src/llm_handler.py` | Prompt construction, GPT-4 response generation |

---

## Features

- ✅ **Multi-format Upload** — PDF, DOCX, XLSX, CSV, TXT
- ✅ **Persistent Embeddings** — Once processed, documents are cached. No re-embedding on restart.
- ✅ **Document Management** — View, delete individual docs, or clear all.
- ✅ **RAG-based Q&A** — Ask questions; the system retrieves relevant chunks and answers via GPT-4.
- ✅ **Source Citation** — View the exact document chunks used to generate each answer.
- ✅ **Docker Support** — One-command deployment via Docker Compose.
- ✅ **Greeting Detection** — Conversational responses for greetings; no unnecessary document search.
- ✅ **Fixed Chat Input** — Input box stays at the bottom; chat history scrolls above.
- ✅ **Adaptive Prompt** — System prompt adapts to document type (financial, HR policies, resumes, etc.).

---

## Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | Runtime |
| **Streamlit** | 1.31.0 | Web UI framework |
| **LangChain** | 1.3.1 | RAG pipeline orchestration |
| **OpenAI** | 2.37.0 | Embeddings & GPT-4 |
| **FAISS** | 1.13.2 | Vector similarity search (CPU) |
| **PyPDF** | 3.17.4 | PDF parsing |
| **python-docx** | 1.1.0 | DOCX parsing |
| **openpyxl** | 3.1.2 | Excel parsing |
| **pandas** | 2.2.0 | CSV & data handling |
| **Docker** | — | Containerization |

---

## Project Structure

```
financial_report_analysis/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker build instructions
├── docker-compose.yml         # Multi-service Docker setup
├── .env                       # Environment variables (API keys)
├── .env.example               # Example environment file
├── .dockerignore              # Docker build exclusions
├── README.md                  # This file
├── src/
│   ├── __init__.py            # Package marker
│   ├── document_processor.py  # File loading & chunking
│   ├── embedding_store.py     # FAISS vector store management
│   └── llm_handler.py         # LLM prompt & response generation
├── uploaded_docs/             # Uploaded files (auto-created)
└── vector_store/              # FAISS indexes & metadata (auto-created)
    ├── metadata.json          # Document registry
    ├── main_index/            # Combined FAISS index
    ├── indexes/               # Per-document FAISS indexes
    └── chunks/                # Cached document chunks (.pkl)
```

---

## Setup Instructions

### Prerequisites

- **Python 3.11+** (or Docker)
- **OpenAI API Key** — Sign up at [platform.openai.com](https://platform.openai.com)

### Local Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd financial_report_analysis
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your OpenAI API key:

   ```env
   OPENAI_API_KEY=sk-your-key-here
   ```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |

If not set in `.env`, you can enter the key through the Streamlit sidebar at runtime.

---

## Running the Application

### Local

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Docker

```bash
docker compose up --build
```

Then open **http://localhost:8501**.

---

## Usage Guide

### Uploading Documents

1. Navigate to the **📤 Upload Documents** tab.
2. Drag & drop or select files (PDF, DOCX, XLSX, CSV, TXT).
3. Click **🚀 Process All Files**.
4. The system will:
   - Save the file to `uploaded_docs/`
   - Check if already processed (skips duplicates)
   - Load and chunk the document
   - Generate embeddings via OpenAI
   - Store in FAISS index (merged with existing)

### Managing Documents

1. Navigate to the **📚 Document Manager** tab.
2. View all processed documents with chunk counts and sizes.
3. **Delete** individual documents — removes embeddings and rebuilds the index.
4. **🗑️ Clear All** — wipes the entire vector store.

### Chatting with Documents

1. Navigate to the **💬 Chat with Your Financial Documents** tab.
2. Type a question in the input box at the bottom.
3. The system:
   - Determines if the query is a **greeting** (returns conversational response, no search)
   - Otherwise, performs **similarity search** on the FAISS index
   - Constructs a prompt with retrieved context
   - Sends to **GPT-4** for answer generation
   - Displays the answer with optional **📖 View source context** expander
4. Chat history persists in the session.

### Chat Order

Messages display in chronological order:
```
Q1 (oldest)
A1
Q2
A2
... (newest at bottom, just above the input box)
```

---

## Docker Deployment

### Dockerfile

The `Dockerfile` uses `python:3.11-slim` and:

1. Installs system build dependencies (gcc, g++ for FAISS)
2. Installs Python packages
3. Creates `uploaded_docs/` and `vector_store/` directories
4. Exposes port **8501**
5. Includes a health check
6. Runs Streamlit on `0.0.0.0:8501`

### Docker Compose

```yaml
version: "3.8"
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./uploaded_docs:/app/uploaded_docs   # Persist uploads
      - ./vector_store:/app/vector_store     # Persist embeddings
    env_file:
      - .env
```

**Why bind mounts?** — So document uploads and FAISS indexes survive container restarts.

### Build & Run

```bash
# Build and start
docker compose up --build

# Run in background
docker compose up --build -d

# Stop
docker compose down
```

---

## API Reference

### Document Processor (`src/document_processor.py`)

```python
# Supported file extensions
SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF Document",
    ".csv": "CSV File",
    ".xlsx": "Excel File",
    ".xls": "Excel File",
    ".docx": "Word Document",
    ".doc": "Word Document",
    ".txt": "Text File",
}

# Load a document file into LangChain Documents
load_document(file_path: str) -> List[Document]

# Split documents into overlapping chunks
split_documents(documents, chunk_size=1000, chunk_overlap=200) -> List[Document]

# Save an uploaded Streamlit file to disk
process_uploaded_file(uploaded_file, upload_dir) -> Dict[str, Any]
```

**Chunking Strategy:**
- **Chunk size:** 1000 characters
- **Overlap:** 200 characters (maintains context across chunk boundaries)
- **Separators:** `\n\n`, `\n`, ` `, `""` (recursive)

### Embedding Store (`src/embedding_store.py`)

```python
EmbeddingStoreManager(store_dir="vector_store", embedding_model="text-embedding-ada-002")

# Check if a file has already been processed (by MD5 hash)
is_document_processed(file_path) -> bool

# Process chunks into embeddings and store
process_and_store_document(file_path, file_name, chunks) -> Dict

# Search for similar chunks (returns (Document, score) tuples)
similarity_search(query, k=4) -> List[Tuple[Document, float]]

# Delete a document and rebuild index
delete_document(doc_id) -> bool

# Clear all stored data
clear_all()
```

**Persistence Architecture:**
- **`metadata.json`** — Registers all processed documents with file name, hash, chunk count, and timestamp
- **`indexes/{doc_id}/`** — Per-document FAISS index (for individual deletion)
- **`chunks/{doc_id}.pkl`** — Pickled LangChain Document chunks (for index rebuilds)
- **`main_index/`** — Merged FAISS index used for similarity search

### LLM Handler (`src/llm_handler.py`)

```python
LLMQuestionAnswerer(model_name="gpt-4", temperature=0.3, max_tokens=1024)

# Generate a response based on question and retrieved context
generate_response(question, context_docs) -> str

# Generate a streaming response (yields chunks)
generate_streaming_response(question, context_docs) -> Generator
```

**System Prompt Behavior:**
- **Greetings** ("hi", "hello", "what can you do") — Returns a brief conversational reply without document search
- **Document questions** — Retrieves relevant chunks and answers concisely
- **Missing information** — Simply states "The documents don't contain this information"
- **Adaptive analysis** — Adjusts tone based on document type (financial data, HR policies, resumes)

---

## Design Decisions

### Why FAISS over other vector databases?
- **Lightweight** — No external server needed. Runs in-process.
- **Persistent** — `save_local()` / `load_local()` enables disk-based storage.
- **Fast** — In-memory HNSW/IVF indexes for sub-5ms search on moderate datasets.

### Why Persistent Embeddings?
Re-embedding documents on every restart wastes OpenAI API credits and time. By caching:
- **FAISS index** saved to `vector_store/main_index/`
- **Chunks** saved as `.pkl` files for index rebuilds
- **Metadata** saved as `metadata.json`
- The app loads existing embeddings on startup and only processes new documents.

### Why Merge FAISS indexes?
When a new document is added, the system:
1. Creates a per-document index (`vector_store/indexes/{doc_id}/`)
2. Merges it into the main index (`vector_store/main_index/`)
This allows individual document deletion (by rebuilding the main index from remaining chunks).

### Why `st.chat_input` before `st.container`?
Streamlit renders elements in order. By placing `st.chat_input` after the chat history container, the input box naturally appears at the bottom — no custom CSS required.

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| "No API key configured" | Missing `OPENAI_API_KEY` in `.env` | Add key or enter via sidebar |
| "Error processing file" | Unsupported format or corrupted file | Check file extension; try re-downloading |
| Embeddings already processed | Same file uploaded twice | Delete via Document Manager first |
| Docker: port already in use | Another service on 8501 | Change port in `docker-compose.yml` |
| Slow first query | FAISS index loading from disk | Subsequent queries are cached in memory |
| "allow_dangerous_deserialization" warning | FAISS pickle safety | Expected; FAISS uses pickle by design |

### Clearing Stored Data

**Option 1:** Use the **🗑️ Clear All** button in the Document Manager tab.

**Option 2:** Delete the `vector_store/` directory manually:

```bash
rm -rf vector_store/
```

**Option 3 (Docker):** Delete the bind-mounted volume:

```bash
rm -rf vector_store/ uploaded_docs/
```

---

## Future Enhancements

- [ ] **Multi-user support** — Session-specific vector stores
- [ ] **Streaming responses** — Token-by-token output using `generate_streaming_response()`
- [ ] **Document comparison** — Compare multiple financial reports side-by-side
- [ ] **OCR support** — Extract text from scanned PDFs
- [ ] **Local LLM support** — Ollama / LlamaCpp as alternatives to OpenAI
- [ ] **Web search fallback** — When documents don't contain the answer
- [ ] **Analytics dashboard** — Usage metrics, document statistics
- [ ] **Authentication** — Login system for multi-tenant deployment

---

## License

This project is for internal / educational use. All document data stays local — only embedding vectors and LLM prompts are sent to OpenAI.