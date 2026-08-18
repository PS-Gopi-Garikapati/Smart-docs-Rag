# Smart Document Assistant (RAG)

A production-grade, modular Retrieval-Augmented Generation (RAG) system built with **Python FastAPI**, **SentenceTransformers**, **ChromaDB**, and **local Ollama (Llama3)** for privacy-preserving inference, featuring a glassmorphic **Vanilla HTML/CSS/JS** frontend.

---

## Key Features

- **Document Processing**: Reads multi-format documents (**PDF**, **Word .docx**, **CSV**, **Plain Text .txt**, **Markdown .md**, **JSON**) and extracts clean text content.
- **Text Chunking**: Sliding-window semantic text chunking with metadata tracking (source file, page numbers, character lengths).
- **Dense Vector Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`) generating high-dimensional semantic embeddings.
- **Vector Database**: Persistent local ChromaDB collection storing chunk vectors and metadata.
- **Configurable Hyperparameters**: Real-time frontend control for:
  - **Temperature** (`0.0` to `1.0`)
  - **Top-P (Nucleus Sampling)** (`0.0` to `1.0`)
  - **Top-K Chunks** (`1` to `20`)
- **Strict Prompt Engineering**: System prompt forcing strict reliance on retrieved context. Returns `"The answer is not available in the uploaded documents."` if the answer is missing.
- **Extensible Architecture**: Modular codebase with a dedicated Model Context Protocol (**MCP**) stub interface (`mcp_stub.py`) for future subagent tool integration.
- **Extractive Fallback Engine**: Built-in local extractive RAG mode so the app works seamlessly offline even without an external LLM API key.

---

## Project Structure

```
Smart docs/
├── app/
│   ├── __init__.py
│   ├── config.py                 # App settings, paths, env variables, default bounds
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── document_processor.py # PDF, DOCX, CSV, TXT text extraction and sliding-window chunking
│   │   ├── embeddings.py         # SentenceTransformers embedding generation singleton
│   │   ├── vector_store.py       # ChromaDB persistence, collection index & similarity search
│   │   ├── retriever.py          # Query embedding generation and top-K context retrieval
│   │   ├── prompt_builder.py     # System prompt formatting and context assembly
│   │   ├── llm_client.py         # Ollama (Llama3) / Local extractive fallback with Temp, Top-P, Top-K
│   │   └── mcp_stub.py           # Model Context Protocol (MCP) registry and stubs
│   └── routes/
│       ├── __init__.py
│       ├── upload_routes.py      # Document upload (PDF, DOCX, CSV, TXT), listing, clearing index
│       └── query_routes.py       # RAG Query endpoint accepting question & sliders
├── docs/                         # Developer documentation folder
├── frontend/
│   ├── index.html                # Single page glassmorphic UI layout
│   ├── css/
│   │   └── style.css             # Glassmorphism, animations, sliders, responsive layout
│   └── js/
│       ├── app.js                # UI event bindings, API fetch handlers, chat history
│       └── markdown.js           # Client-side Markdown response renderer
├── data/
│   ├── uploads/                  # Saved uploaded documents
│   └── chroma_db/                # Persistent vector database
├── main.py                       # FastAPI entrypoint and static file server
├── requirements.txt              # Dependency specifications
├── .env.example                  # Environment configuration template
└── README.md                     # Documentation and setup guide


```

---

## Setup & Installation

### 1. Prerequisites
- **Python 3.10+** (Python 3.12 recommended)
- **pip** package manager

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to configure Ollama connection (default settings work for local Ollama):
```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL_NAME=llama3
```
*(Note: Ensure Ollama is running locally. The application uses local Llama3 model for privacy-preserving inference with no external API calls required.)*

---

## Running the Application

Launch the FastAPI application with Uvicorn:

```bash
python main.py
```

Or using Uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser and navigate to:
👉 **`http://localhost:8000`**

---

## User Guide & Workflow

1. **Upload Documents**:
   - Drag & drop **PDF**, **Word (.docx)**, **CSV**, or **Text (.txt, .md)** files into the upload dropzone or click **Select Documents**.
   - The system extracts page text, generates dense vector embeddings, and stores them in ChromaDB.
2. **Adjust Hyperparameters**:
   - Use the **Temperature** slider to control answer creativity/randomness (`0.0` to `1.0`).
   - Use the **Top-P** slider to adjust nucleus sampling (`0.0` to `1.0`).
   - Specify the **Top-K** field to choose how many document chunks to retrieve (default: `3`).
3. **Ask Questions**:
   - Type your question in the text box and press **Enter** or click **Ask Question**.
   - Expand the **View Context Sources** accordion under any response to inspect source citations and similarity scores.

---

## API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the HTML frontend interface |
| `GET` | `/api/health` | Application healthcheck status |
| `POST` | `/api/upload` | Uploads documents (PDF, DOCX, CSV, TXT) and indexes text chunks into ChromaDB |
| `GET` | `/api/documents` | Lists all indexed documents and chunk counts |
| `DELETE` | `/api/clear` | Clears ChromaDB vector store and deletes stored files |
| `POST` | `/api/query` | Executes RAG context retrieval and generates answer |

---

## Future MCP Integration

The application includes `app/modules/mcp_stub.py` which provides an `MCPToolRegistry`.
This structure allows external Model Context Protocol (MCP) clients or subagents to discover registered tools (e.g. document summarization, PDF search, entity extraction) and execute them seamlessly.
