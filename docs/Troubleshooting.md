# Troubleshooting Documentation

This guide provides troubleshooting steps for common error states and installation failures.

## Common Failures

### 1. ChromaDB fails to import
- **Error**: `ModuleNotFoundError: No module named 'chroma_db'` or DLL loading errors.
- **Cause**: On Windows systems, ChromaDB relies on native C++ compilers and DLL files.
- **Resolution**:
  - The application includes an **extractive fallback JSON vector database** (`PersistentJsonVectorStore`). It will automatically handle chunk queries if ChromaDB import fails.
  - To fix the underlying native compiler issues, install Visual Studio C++ build tools or install `chromadb` using precompiled packages if available.

### 2. PyTorch or SentenceTransformers DLL import fails
- **Error**: `OSError: [WinError 127] The specified procedure could not be found` when importing `torch` or `sentence_transformers`.
- **Cause**: Incompatible Python, PyTorch version, or missing compiler dependencies.
- **Resolution**:
  - The application will automatically detect this import crash and switch to **zero-dependency TF-IDF / Subword Dense Vector Hashing fallback mode** to continue generating embeddings offline.
  - To clean, execute `pip uninstall torch sentence-transformers` and reinstall stable versions suited for your CPU/GPU hardware.

### 3. Ollama connection error
- **Error**: `RuntimeError: Ollama error: Connection refused` or `Network timeout`.
- **Cause**: Ollama is not running locally or not accessible at `OLLAMA_HOST`.
- **Resolution**:
  - Ensure Ollama is installed and running: `ollama serve`
  - Check `.env` file: `OLLAMA_HOST=http://localhost:11434`
  - The application will automatically switch to **local extractive RAG mode** using sentence matching over document chunks if Ollama is unavailable.
