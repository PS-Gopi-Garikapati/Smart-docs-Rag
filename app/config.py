"""
Configuration Module for Smart Document Assistant.

Defines project-wide configuration settings, including directory paths,
environment variables, vector database settings, default hyperparameters,
and fallback message strings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Base Directory Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
UPLOAD_DIR: Path = DATA_DIR / "uploads"
CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma_db"

# Ensure data storage directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Local LLM Service (Ollama)
# Uses Ollama with Llama3 for local, privacy-preserving inference
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL_NAME: str = os.getenv("OLLAMA_MODEL_NAME", "llama3")

# Server Settings
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# Embedding Model Configuration
# Uses SentenceTransformer model (small, fast, high accuracy for semantic similarity)
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Document Chunking Settings
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

# Vector DB Collection Name
CHROMA_COLLECTION_NAME: str = "smart_document_assistant"

# Default RAG Parameters
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "3"))
DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))
DEFAULT_TOP_P: float = float(os.getenv("DEFAULT_TOP_P", "0.9"))

# Standard Fallback Guardrail Response (Required by specification)
NOT_AVAILABLE_RESPONSE: str = "The answer is not available in the uploaded documents."

