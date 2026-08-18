# Folder Structure Documentation

This document explains the organization and details of each folder and file in the **Smart Document Assistant** workspace.

## Workspace Layout

```
Smart docs/
├── app/
│   ├── __init__.py               # Packages the application folder
│   ├── config.py                 # Application configurations, variables, path initializers
│   ├── modules/
│   │   ├── __init__.py           # Packaging modules
│   │   ├── document_processor.py # Parsing algorithms for CSV, PDF, Word, Plain text
│   │   ├── embeddings.py         # Embedding initialization & dense hashing
│   │   ├── vector_store.py       # ChromaDB interactions and fallback local database
│   │   ├── retriever.py          # Similarity vector lookup coordinator
│   │   ├── prompt_builder.py     # Prompt grounding template constructions
│   │   ├── llm_client.py         # LLM connector and offline local fallback engine
│   │   └── mcp_stub.py           # Model Context Protocol stub integration
│   └── routes/
│       ├── __init__.py           # Packaging router endpoints
│       ├── upload_routes.py      # Endpoints for document management
│       └── query_routes.py       # Endpoints for context-grounded RAG query
├── docs/                         # Developer documentation folder
├── frontend/                     # Glassmorphic client app files
│   ├── index.html                # Main UI layout
│   ├── css/
│   │   └── style.css             # Glassmorphic styles and animations
│   └── js/
│       ├── app.js                # Frontend core bindings and api handlers
│       └── markdown.js           # Client-side markdown formatting rendering engine
├── data/                         # Local database storage directory
│   ├── uploads/                  # Extracted source document storage
│   └── chroma_db/                # ChromaDB vector binary assets / fallback JSON index
├── main.py                       # FastAPI entrypoint and static folder mounting
├── requirements.txt              # Project library dependencies list
├── .env                          # Local credentials file (loaded at runtime)
├── .env.example                  # Environmental keys template
└── README.md                     # General project setup details
```
