# Architecture Documentation

The **Smart Document Assistant** is designed using clean architecture principles, separating the system into logical layers: Presentation, Routing, Business Logic, and Data.

## Layer Overview

```
┌────────────────────────────────────────────────────────┐
│                   Presentation Layer                   │
│         (HTML5 / CSS3 / Vanilla JS Frontend)           │
└───────────────────────────┬────────────────────────────┘
                            │ (JSON HTTP Requests)
                            ▼
┌────────────────────────────────────────────────────────┐
│                     Routing Layer                      │
│            (FastAPI endpoints / API routes)            │
└───────────────────────────┬────────────────────────────┘
                            │ (Invocations)
                            ▼
┌────────────────────────────────────────────────────────┐
│                  Business Logic Layer                  │
│       (Retrieval RAG Pipeline, Document Parsing,       │
│        Text Chunking, Dense Hashing, LLM Grounding)    │
└───────────────────────────┬────────────────────────────┘
                            │ (Read/Write)
                            ▼
┌────────────────────────────────────────────────────────┐
│                       Data Layer                       │
│        (Vector DB / Local JSON DB, Local Files)        │
└────────────────────────────────────────────────────────┘
```

### 1. Presentation Layer
- Contains `index.html`, `css/style.css`, and `js/app.js`.
- Manages user interaction, visual renders (including citations and glassmorphism animation components), and hyperparameter settings (Temperature, Top-P, Top-K).

### 2. Routing Layer
- **Upload Routes** (`upload_routes.py`): Manages document uploads, vector indexing, inventory listing, and clearing the vector database.
- **Query Routes** (`query_routes.py`): Receives natural language questions and hyperparameters, processes RAG requests, and responds with grounding citations.

### 3. Business Logic Layer
- **Document Processor** (`document_processor.py`): Converts documents (PDF, DOC, DOCX, CSV, TXT, JSON, MD, LOG) into unified plain text and partitions pages into overlapping sliding-window chunks.
- **Embeddings** (`embeddings.py`): Generates vector representation embeddings using `SentenceTransformers` or a high-quality zero-dependency TF-IDF subword dense hash vectorizer.
- **Retriever** (`retriever.py`): Matches query embeddings against indexed document vectors using cosine similarity.
- **Prompt Builder** (`prompt_builder.py`): Injects context blocks and constraints into prompt structures.
- **LLM Client** (`llm_client.py`): Dispatches grounding requests to local Ollama (Llama3) for privacy-preserving inference, with automatic local extractive fallback.

### 4. Data Layer
- **ChromaDB**: Native C++ dense vector database.
- **PersistentJsonVectorStore**: Fallback database engine storing vector metadata locally in standard JSON files.
