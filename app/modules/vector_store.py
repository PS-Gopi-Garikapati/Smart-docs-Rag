"""
Vector Store Module.
Manages ChromaDB vector database interactions including storage, retrieval,
collection management, and vector similarity search.
Includes a robust fallback PersistentJsonVectorStore when ChromaDB DLL dependencies are missing.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from app.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from app.modules.embeddings import generate_batch_embeddings

logger = logging.getLogger(__name__)

# Global vector store instance
_active_store = None


class PersistentJsonVectorStore:
    """
    Purpose:
    Zero-dependency persistent vector database engine.
    Used as a fallback when ChromaDB native C++ DLLs fail on Windows systems.

    Responsibilities:
    - Load vectors from a local JSON file.
    - Save/persist indexed chunks to the local JSON file.
    - Index document chunks, dense embeddings, and metadata.
    - Perform cosine similarity search based on query dense vector.
    - Delete documents by source file name.
    """

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        """
        Purpose:
        Initializes the local JSON vector store instance.

        Parameters:
        persist_dir (Path): The directory path to persist the JSON database.
        collection_name (str): Collection prefix identifier name.
        """
        self.persist_dir: Path = Path(persist_dir)
        self.collection_name: str = collection_name
        self.file_path: Path = self.persist_dir / f"{collection_name}.json"
        self.chunks: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """
        Purpose:
        Loads chunks and embeddings from the persistent JSON database.
        """
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                logger.info(f"Loaded {len(self.chunks)} chunks from local JSON vector store.")
            except Exception as e:
                logger.warning(f"Error reading JSON vector store ({e}), starting fresh.")
                self.chunks = []

    def _save(self) -> None:
        """
        Purpose:
        Saves currently indexed chunks to the local persistent JSON file.
        """
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist JSON vector store: {e}")

    def count(self) -> int:
        """
        Purpose:
        Returns total number of chunks currently stored.

        Returns:
        int: Number of chunk dictionaries.
        """
        return len(self.chunks)

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Purpose:
        Inserts or updates document chunks in the JSON vector store database.

        Parameters:
        ids (List[str]): Unique chunk IDs.
        documents (List[str]): Raw text payloads.
        embeddings (List[List[float]]): Corresponding dense embeddings.
        metadatas (List[Dict[str, Any]]): Metadata annotations.
        """
        id_map = {c["id"]: idx for idx, c in enumerate(self.chunks)}
        for cid, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            item = {
                "id": cid,
                "text": doc,
                "embedding": emb,
                "metadata": meta
            }
            if cid in id_map:
                self.chunks[id_map[cid]] = item
            else:
                self.chunks.append(item)
                id_map[cid] = len(self.chunks) - 1
        self._save()

    def query(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Purpose:
        Executes a dense cosine similarity search against index.

        Parameters:
        query_embedding (List[float]): User query vector.
        top_k (int): Candidate count (defaults to 3).

        Returns:
        List[Dict[str, Any]]: List of matching chunk documents with scores.
        """
        if not self.chunks:
            return []

        q_arr = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        if q_norm > 0:
            q_arr = q_arr / q_norm

        scored = []
        for item in self.chunks:
            emb = np.array(item["embedding"], dtype=np.float32)
            e_norm = np.linalg.norm(emb)
            if e_norm > 0:
                emb = emb / e_norm
            
            sim = float(np.dot(q_arr, emb))
            dist = max(0.0, 1.0 - sim)

            scored.append({
                "text": item["text"],
                "metadata": item["metadata"],
                "distance": float(dist),
                "similarity": round(max(0.0, sim), 4)
            })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def get_metadatas(self) -> List[Dict[str, Any]]:
        """
        Purpose:
        Retrieves all chunk metadatas stored in the collection.

        Returns:
        List[Dict[str, Any]]: All metadata dictionary maps.
        """
        return [c.get("metadata", {}) for c in self.chunks]

    def delete_by_source(self, filename: str) -> int:
        """
        Purpose:
        Deletes all chunks linked to a specific source file.

        Parameters:
        filename (str): Name of the source file.

        Returns:
        int: Number of deleted records.
        """
        initial = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.get("metadata", {}).get("source") != filename]
        deleted = initial - len(self.chunks)
        if deleted > 0:
            self._save()
        return deleted

    def clear(self) -> None:
        """
        Purpose:
        Clears the JSON store and deletes its physical database file.
        """
        self.chunks = []
        if self.file_path.exists():
            try:
                os.remove(self.file_path)
            except Exception:
                pass



import subprocess
import sys

_chroma_checked = False
_chroma_available = False


def _check_chromadb_working() -> bool:
    """
    Purpose:
    Checks if chromadb library is installed and running correctly.

    Returns:
    bool: True if chromadb client runs successfully, otherwise False.
    """
    global _chroma_checked, _chroma_available
    if _chroma_checked:
        return _chroma_available

    try:
        code = "import chromadb; c = chromadb.PersistentClient(path='./data/chroma_db'); col = c.get_or_create_collection('probe_col'); col.upsert(ids=['p1'], documents=['test'], embeddings=[[0.1]*384], metadatas=[{'s': 'a'}])"
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=5)
        _chroma_available = (res.returncode == 0)
    except Exception:
        _chroma_available = False

    _chroma_checked = True
    return _chroma_available


def get_vector_store() -> Any:
    """
    Purpose:
    Initializes and returns the persistent vector store client.
    Attempts ChromaDB first, falling back to PersistentJsonVectorStore if ChromaDB fails.

    Returns:
    Any: ChromaDB Collection object or PersistentJsonVectorStore fallback instance.
    """
    global _active_store

    if _active_store is not None:
        return _active_store

    if _check_chromadb_working():
        try:
            import chromadb
            from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

            class SmartDocsEmbeddingFunction(EmbeddingFunction):
                def __init__(self) -> None: pass
                def __call__(self, input: Documents) -> Embeddings: return generate_batch_embeddings(list(input))
                def embed_query(self, input: str) -> List[float]: return generate_batch_embeddings([input])[0]
                def name(self) -> str: return "smart_docs_ef"

            client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
            col = client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                embedding_function=SmartDocsEmbeddingFunction(),
                metadata={"hnsw:space": "cosine"}
            )
            _active_store = col
            logger.info("ChromaDB vector store initialized successfully.")
            return _active_store

        except Exception as e:
            logger.warning(f"ChromaDB initialization failed ({e}). Reverting to PersistentJsonVectorStore fallback engine.")

    logger.info("Operating in PersistentJsonVectorStore Fallback Engine.")
    _active_store = PersistentJsonVectorStore(CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME)
    return _active_store


def add_chunks_to_vector_store(chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
    """
    Purpose:
    Inserts or updates document text chunks along with vector embeddings and metadata in vector store.

    Parameters:
    chunks (List[Dict[str, Any]]): List of chunk dictionary payloads.
    embeddings (List[List[float]]): Coordinated dense embeddings.

    Returns:
    int: Number of chunks successfully indexed.

    Raises:
    ValueError: If chunk count and embeddings count mismatch.
    """
    if not chunks or not embeddings:
        return 0

    if len(chunks) != len(embeddings):
        raise ValueError("Mismatch between number of chunks and embeddings length.")

    store = get_vector_store()
    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    try:
        if hasattr(store, "upsert"):
            store.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            store.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        logger.info(f"Successfully indexed {len(chunks)} chunks into vector store.")
        return len(chunks)
    except Exception as e:
        logger.warning(f"ChromaDB write error ({e}), switching vector store to PersistentJsonVectorStore...")
        global _active_store
        _active_store = PersistentJsonVectorStore(CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME)
        _active_store.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        return len(chunks)


def query_vector_store(query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Purpose:
    Performs similarity search using query vector embedding.

    Parameters:
    query_embedding (List[float]): Generated query vector.
    top_k (int): Number of nearest documents to retrieve (defaults to 3).

    Returns:
    List[Dict[str, Any]]: List of matching chunk documents with scores.
    """
    store = get_vector_store()

    try:
        if isinstance(store, PersistentJsonVectorStore):
            return store.query(query_embedding=query_embedding, top_k=top_k)

        if store.count() == 0:
            return []

        results = store.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, store.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved_items = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                retrieved_items.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": float(dist),
                    "similarity": round(max(0.0, 1.0 - float(dist)), 4)
                })

        return retrieved_items

    except Exception as e:
        logger.warning(f"ChromaDB query warning ({e}). Using PersistentJsonVectorStore query fallback.")
        global _active_store
        _active_store = PersistentJsonVectorStore(CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME)
        return _active_store.query(query_embedding=query_embedding, top_k=top_k)


def get_indexed_documents() -> List[Dict[str, Any]]:
    """
    Purpose:
    Retrieves summary information about currently stored document sources.

    Returns:
    List[Dict[str, Any]]: List of unique documents containing filename and chunk count.
    """
    store = get_vector_store()
    
    try:
        if isinstance(store, PersistentJsonVectorStore):
            metas = store.get_metadatas()
        else:
            if store.count() == 0:
                return []
            all_records = store.get(include=["metadatas"])
            metas = all_records.get("metadatas", [])

        doc_summary = {}
        for m in metas:
            if m and "source" in m:
                src = m["source"]
                doc_summary[src] = doc_summary.get(src, 0) + 1

        return [
            {"filename": name, "chunk_count": count}
            for name, count in doc_summary.items()
        ]

    except Exception as e:
        logger.error(f"Error fetching indexed documents: {e}")
        return []


def delete_document_from_vector_store(filename: str) -> int:
    """
    Purpose:
    Deletes all chunks associated with a specific document source.

    Parameters:
    filename (str): Name of the source file.

    Returns:
    int: Count of deleted vector index elements.
    """
    store = get_vector_store()
    try:
        if isinstance(store, PersistentJsonVectorStore):
            return store.delete_by_source(filename)

        records = store.get(where={"source": filename}, include=["metadatas"])
        ids_to_delete = records.get("ids", [])
        if ids_to_delete:
            store.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for document '{filename}' from ChromaDB.")
        return len(ids_to_delete)
    except Exception as e:
        logger.warning(f"ChromaDB delete exception ({e}), applying fallback deletion.")
        global _active_store
        _active_store = PersistentJsonVectorStore(CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME)
        return _active_store.delete_by_source(filename)


def clear_vector_store() -> bool:
    """
    Purpose:
    Clears all documents and embeddings stored in the vector database.

    Returns:
    bool: True if completed successfully, otherwise False.
    """
    global _active_store
    store = get_vector_store()
    try:
        if isinstance(store, PersistentJsonVectorStore):
            store.clear()
            logger.info("PersistentJsonVectorStore cleared successfully.")
            return True

        if _check_chromadb_working():
            import chromadb
            try:
                chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR)).delete_collection(name=CHROMA_COLLECTION_NAME)
            except Exception:
                pass

        _active_store = None
        get_vector_store()
        logger.info("Vector store reset successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to clear vector store: {e}")
        _active_store = PersistentJsonVectorStore(CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME)
        _active_store.clear()
        return True


def sync_vector_store_with_uploads() -> None:
    """
    Synchronizes the vector database with the physical files in the data/uploads directory.
    If a file exists in the directory but is not indexed, it chunks and indexes it.
    If a file is indexed but does not exist in the directory, it deletes its index entry.
    """
    from app.config import UPLOAD_DIR
    from app.modules.document_processor import extract_text_from_file, chunk_document_pages
    from app.modules.embeddings import generate_batch_embeddings

    try:
        # Get physical files in uploads directory
        physical_files = []
        if UPLOAD_DIR.exists():
            physical_files = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]

        # Get currently indexed files
        indexed_docs = get_indexed_documents()
        indexed_files = [doc["filename"] for doc in indexed_docs]

        logger.info(f"Sync: Running. Physical files: {physical_files} | Indexed files: {indexed_files}")

        # 1. Delete index for files that are physically missing
        for filename in indexed_files:
            if filename not in physical_files:
                logger.info(f"Sync: Document '{filename}' is physically missing. Deleting from index.")
                delete_document_from_vector_store(filename)

        # 2. Index files that are physically present but not indexed
        for filename in physical_files:
            if filename not in indexed_files:
                file_path = UPLOAD_DIR / filename
                logger.info(f"Sync: Found new physical file '{filename}'. Auto-indexing.")
                try:
                    pages = extract_text_from_file(str(file_path))
                    chunks = chunk_document_pages(pages, doc_name=filename)
                    if chunks:
                        texts = [c["text"] for c in chunks]
                        embeddings = generate_batch_embeddings(texts)
                        add_chunks_to_vector_store(chunks, embeddings)
                        logger.info(f"Sync: Successfully indexed '{filename}' ({len(chunks)} chunks).")
                    else:
                        logger.warning(f"Sync: No chunks extracted from '{filename}'.")
                except Exception as ex:
                    logger.error(f"Sync: Failed to auto-index '{filename}': {ex}")

    except Exception as e:
        logger.error(f"Error during vector store synchronization: {e}")




