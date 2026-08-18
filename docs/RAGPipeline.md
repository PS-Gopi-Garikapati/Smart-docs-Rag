# RAG Pipeline Documentation

This document explains the Retrieval-Augmented Generation (RAG) components built into the **Smart Document Assistant**.

## Pipeline Components

```
User Question
     │
     ▼
[embeddings.py] ──► Generate query vector embedding (384-dim)
     │
     ▼
[retriever.py]  ──► Query ChromaDB / Json fallback (Cosine similarity)
     │
     ▼
[prompt_builder.py] ◄── Format retrieved chunk text and append strict grounding constraints
     │
     ▼
[llm_client.py] ──► Call local Ollama (Llama3) for generation (fallback to extractive search if offline)
     │
     ▼
Ground Response (Concise ground truth citations returned to user UI)
```

### 1. Document Extraction & Chunker
- **Processor**: `document_processor.py` reads PDFs, DOC/DOCX, CSVs, TXT, MD, JSON, and LOG files.
- **Chunking**: Splits virtual pages into overlapping segments of `CHUNK_SIZE` characters (default 600) with a `CHUNK_OVERLAP` window (default 100). Snaps boundaries to whitespace or periods to preserve linguistic semantics.

### 2. Embedding Generator
- **Processor**: `embeddings.py` loads `all-MiniLM-L6-v2` via `SentenceTransformers` to convert text chunks into 384-dimensional dense vectors.
- **Fallback Hashing**: If PyTorch fails, it computes character, bigram, and trigram MD5 hashes mapped to a 384-dimensional normalized vector space.

### 3. Vector Database
- **Processor**: `vector_store.py` manages a persistent local collection.
- **Fallback DB**: If ChromaDB fails, it uses `PersistentJsonVectorStore`, executing cosine calculations over serialized vectors.

### 4. Grounded Response Generator
- **System Instructions**: Built via `prompt_builder.py`. Employs rules prohibiting external knowledge usage.
- **Engines**: Dispatched in `llm_client.py`. Uses local Ollama (Llama3) with automatic fallback to the local sentence matching algorithm if offline.
