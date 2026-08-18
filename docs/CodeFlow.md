# Code Flow Documentation

This document traces the program execution flow from client interaction down to data lookup and response compilation.

## 1. Document Upload Flow

```
User (Upload Area)
  │
  ▼  (HTTP POST /api/upload)
[upload_routes.py] 
  │
  ├─► checks file extensions and saves the payload to data/uploads
  │
  ├─► [document_processor.py] extracts texts per virtual page based on extension
  │
  ├─► [document_processor.py] chunks page elements using sliding window parameters
  │
  ├─► [embeddings.py] generates batch vectors of extracted chunks
  │
  └─► [vector_store.py] saves vectors, text payloads, and metadata to ChromaDB or Fallback JSON
```

---

## 2. Question Answering (RAG) Flow

```
User (Query input click / Enter)
  │
  ▼  (HTTP POST /api/query with question & hyperparameters)
[query_routes.py]
  │
  ├─► [retriever.py] coordinates query vector generation and searches matching candidate chunks
  │
  ├─► [prompt_builder.py] formats context blocks and binds system instructions
  │
  ├─► [llm_client.py] triggers Llama client (Ollama SDK)
  │     │
  │     └─► [FALLBACK] If Ollama fails, triggers local extractive search engine
  │
  ▼
API responds with ground truth answer, similarity metrics, and citation sources to UI chat interface
```
