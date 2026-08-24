# Smart Document Assistant (RAG)

A production-grade, modular Retrieval-Augmented Generation (RAG) system built with **Python FastAPI**, **SentenceTransformers**, **ChromaDB**, and **local Ollama (Llama3)** for privacy-preserving inference, featuring a glassmorphic **Vanilla HTML/CSS/JS** frontend.

---

## Key Features

- **Document Processing**: Reads multi-format documents (**PDF**, **Word .docx**, **Plain Text .txt**, **Markdown .md**, **JSON**) and extracts clean text content.
- **Text Chunking**: Sliding-window semantic text chunking with metadata tracking (source file, page numbers, character lengths).
- **Dense Vector Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`) generating high-dimensional semantic embeddings.
- **Vector Database**: Persistent local ChromaDB collection storing chunk vectors and metadata.
- **Configurable Hyperparameters**: Real-time frontend control for:
  - **Temperature** (`0.0` to `1.0`)
  - **Top-P (Nucleus Sampling)** (`0.0` to `1.0`)
  - **Top-K Chunks** (`1` to `20`)
- **Strict Prompt Engineering**: System prompt forcing strict reliance on retrieved context. Returns `"The answer is not available in the uploaded documents."` if the answer is missing.
- **Extractive Fallback Engine**: Built-in local extractive RAG mode so the app works seamlessly offline even without an external LLM API key.


---

## Project Structure

```
Smart docs/
├── app/
│   ├── __init__.py
│   ├── config.py                 # App settings, paths, env variables, default bounds
│   ├── evaluation_set.json       # Versioned evaluation dataset (in-scope, out-of-scope, adversarial)
│   ├── evaluate.py               # Evaluator script to measure retrieval and generation accuracy
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── document_processor.py # PDF, DOCX, CSV, TXT text extraction and sliding-window chunking
│   │   ├── embeddings.py         # SentenceTransformers embedding generation singleton
│   │   ├── vector_store.py       # ChromaDB persistence, collection index & similarity search
│   │   ├── retriever.py          # Query embedding generation and top-K context retrieval
│   │   ├── prompt_builder.py     # System prompt formatting and context assembly
│   │   └── llm_client.py         # Ollama (Llama3) / Local extractive fallback with Temp, Top-P, Top-K
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
├── tests/
│   └── test_rag.py               # Comprehensive pytest suite for RAG validations and rules
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
   - **Upload Hardening**: Files are subject to a **200MB size limit** per file. Filenames are automatically sanitized to prevent path-traversal vulnerabilities.
   - The system extracts page text, generates dense vector embeddings, and stores them in ChromaDB.
2. **Adjust Hyperparameters**:
   - Use the **Temperature** slider to control answer creativity/randomness (`0.0` to `1.0`).
   - Use the **Top-P** slider to adjust nucleus sampling (`0.0` to `1.0`).
   - Specify the **Top-K** field to choose how many document chunks to retrieve (default: `3`).
3. **Ask Questions**:
   - Type your question in the text box and press **Enter** or click **Ask Question**.
   - **Retrieval Confidence Controls**: A similarity threshold gate (`0.35`) is applied. Any retrieved chunks below this threshold are discarded. If no chunks exceed this threshold or if context is missing, the system activates an answerability gate and instantly returns `"I don't have relevant answer for that."` rather than hallucinating.

---

## Evaluation & Testing

### 1. Automated Evaluation Set
A versioned, structured evaluation dataset is located at [`app/evaluation_set.json`](file:///c:/Users/GopiChandGarikapati/Desktop/Smart%20docs/app/evaluation_set.json) containing:
- **In-Scope Cases**: Questions grounded in target documents.
- **Out-of-Scope Cases**: Unrelated or random questions.
- **Adversarial / Unanswerable Cases**: Conflicting statements, fake options, or missing details.

To run the automated retrieval and generation quality check:
```bash
python -m app.evaluate
```
This will produce an evaluation summary report saved to `data/evaluation_report.json` showing retrieval matches, keyword scoring, and response grounding.

### 2. Automated Test Suite
A comprehensive unit and integration test suite is located in [`tests/test_rag.py`](file:///c:/Users/GopiChandGarikapati/Desktop/Smart%20docs/tests/test_rag.py). It tests document extraction, chunk metadata, vector store retrieval accuracy, empty-index behavior, API response schemas, out-of-bounds/insufficient context handling, and upload constraints.

To execute the test suite:
```bash
python -m pytest tests/test_rag.py
```

---

## API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the HTML frontend interface |
| `GET` | `/api/health` | Application healthcheck status |
| `POST` | `/api/upload` | Uploads documents (PDF, DOCX, CSV, TXT), validates sizes/filenames, and indexes chunks |
| `GET` | `/api/documents` | Lists all indexed documents and chunk counts |
| `DELETE` | `/api/clear` | Clears ChromaDB vector store and deletes stored files |
| `POST` | `/api/query` | Executes RAG context retrieval with confidence controls and generates answer |


